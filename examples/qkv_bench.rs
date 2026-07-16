use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_transformers::{
    models::{quantized_qwen3::RotaryEmbedding, with_tracing::QMatMul},
    quantized_var_builder::VarBuilder,
};
use std::{path::PathBuf, time::Instant};

const H: usize = 3072;
const Q: usize = 6144;
const KV: usize = 1024;
const HEAD_DIM: usize = 128;
const ROPE_DIM: usize = 64;

fn elapsed_ms<F>(device: &Device, iterations: usize, mut f: F) -> Result<f64>
where
    F: FnMut() -> candle_core::Result<()>,
{
    device.synchronize()?;
    let start = Instant::now();
    for _ in 0..iterations {
        f()?;
    }
    device.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1e3 / iterations as f64)
}

fn main() -> Result<()> {
    let model_dir = PathBuf::from("/storage/models/minimax-m2.7-gguf/UD-Q4_K_XL");
    let mut shards = std::fs::read_dir(model_dir)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "gguf"))
        .collect::<Vec<_>>();
    shards.sort();
    let device = Device::new_cuda(0)?;
    unsafe { device.as_cuda_device()?.disable_event_tracking() };
    let vb = VarBuilder::from_gguf_selected(&shards, &["blk.0.".to_owned()], &device)?.pp("blk.0");
    let qw = vb.pp("attn_q").get((Q, H), "weight")?;
    let kw = vb.pp("attn_k").get((KV, H), "weight")?;
    let vw = vb.pp("attn_v").get((KV, H), "weight")?;
    let q_norm = vb.pp("attn_q_norm").get(Q, "weight")?.dequantize(&device)?;
    let k_norm = vb
        .pp("attn_k_norm")
        .get(KV, "weight")?
        .dequantize(&device)?;
    println!(
        "dtypes q={:?} k={:?} v={:?}",
        qw.dtype(),
        kw.dtype(),
        vw.dtype()
    );
    let qmat = QMatMul::from_weights(qw.clone())?;
    let kmat = QMatMul::from_weights(kw.clone())?;
    let vmat = QMatMul::from_weights(vw.clone())?;
    let x = Tensor::randn(0f32, 1f32, (1, 1, H), &device)?.to_dtype(DType::F32)?;

    let q_ref = qmat.forward(&x)?;
    let k_ref = kmat.forward(&x)?;
    let v_ref = vmat.forward(&x)?;
    let (q, k, v) = candle_nn::fused_qkv::q8_0_decode(&x, &qw, &kw, &vw)?;
    device.synchronize()?;
    for (name, reference, candidate) in [
        ("q", q_ref.clone(), q),
        ("k", k_ref.clone(), k),
        ("v", v_ref.clone(), v),
    ] {
        let max = (&reference - &candidate)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        println!("{name} max_abs={max:.9}");
    }

    let q_ref_f16 = candle_nn::ops::rms_norm(&q_ref, &q_norm, 1e-6)?.to_dtype(DType::F16)?;
    let k_ref_f16 = candle_nn::ops::rms_norm(&k_ref, &k_norm, 1e-6)?.to_dtype(DType::F16)?;
    let v_ref_f16 = v_ref.to_dtype(DType::F16)?;
    let (q, k, v) =
        candle_nn::fused_qkv::q8_0_decode_rmsnorm_f16(&x, &qw, &kw, &vw, &q_norm, &k_norm, 1e-6)?;
    device.synchronize()?;
    for (name, reference, candidate) in [
        ("q_norm_f16", q_ref_f16.clone(), q.clone()),
        ("k_norm_f16", k_ref_f16.clone(), k.clone()),
        ("v_f16", v_ref_f16, v),
    ] {
        let max = (&reference - &candidate)?
            .abs()?
            .to_dtype(DType::F32)?
            .max_all()?
            .to_scalar::<f32>()?;
        println!("{name} max_abs={max:.9}");
    }

    let rope = RotaryEmbedding::new(DType::F32, ROPE_DIM, 196_608, 5_000_000., &device)?;
    let position = 4097;
    let q_shaped = q_ref_f16
        .reshape((1, 1, Q / HEAD_DIM, HEAD_DIM))?
        .transpose(1, 2)?
        .contiguous()?;
    let k_shaped = k_ref_f16
        .reshape((1, 1, KV / HEAD_DIM, HEAD_DIM))?
        .transpose(1, 2)?
        .contiguous()?;
    let (qr, kr) = rope.apply(
        &q_shaped.narrow(3, 0, ROPE_DIM)?,
        &k_shaped.narrow(3, 0, ROPE_DIM)?,
        position,
    )?;
    let q_rope_ref = Tensor::cat(&[qr, q_shaped.narrow(3, ROPE_DIM, HEAD_DIM - ROPE_DIM)?], 3)?;
    let k_rope_ref = Tensor::cat(&[kr, k_shaped.narrow(3, ROPE_DIM, HEAD_DIM - ROPE_DIM)?], 3)?;
    let (cos, sin) = rope.cos_sin_tensors();
    let (q_rope, k_rope) = candle_nn::fused_qkv::partial_rope_f16(
        &q_ref_f16, &k_ref_f16, cos, sin, position, HEAD_DIM, ROPE_DIM,
    )?;
    let q_rope = q_rope
        .reshape((1, 1, Q / HEAD_DIM, HEAD_DIM))?
        .transpose(1, 2)?
        .contiguous()?;
    let k_rope = k_rope
        .reshape((1, 1, KV / HEAD_DIM, HEAD_DIM))?
        .transpose(1, 2)?
        .contiguous()?;
    device.synchronize()?;
    for (name, reference, candidate) in [
        ("q_rope", q_rope_ref, q_rope),
        ("k_rope", k_rope_ref, k_rope),
    ] {
        let max = (&reference - &candidate)?
            .abs()?
            .to_dtype(DType::F32)?
            .max_all()?
            .to_scalar::<f32>()?;
        println!("{name} max_abs={max:.9}");
    }

    let iterations = 500;
    let baseline = elapsed_ms(&device, iterations, || {
        drop(qmat.forward(&x)?);
        drop(kmat.forward(&x)?);
        drop(vmat.forward(&x)?);
        Ok(())
    })?;
    let fused = elapsed_ms(&device, iterations, || {
        drop(candle_nn::fused_qkv::q8_0_decode(&x, &qw, &kw, &vw)?);
        Ok(())
    })?;
    println!(
        "projection baseline={baseline:.4}ms fused={fused:.4}ms speedup={:.2}x",
        baseline / fused
    );
    let baseline_post = elapsed_ms(&device, iterations, || {
        let q =
            candle_nn::ops::rms_norm(&qmat.forward(&x)?, &q_norm, 1e-6)?.to_dtype(DType::F16)?;
        let k =
            candle_nn::ops::rms_norm(&kmat.forward(&x)?, &k_norm, 1e-6)?.to_dtype(DType::F16)?;
        let v = vmat.forward(&x)?.to_dtype(DType::F16)?;
        drop((q, k, v));
        Ok(())
    })?;
    let fused_post = elapsed_ms(&device, iterations, || {
        drop(candle_nn::fused_qkv::q8_0_decode_rmsnorm_f16(
            &x, &qw, &kw, &vw, &q_norm, &k_norm, 1e-6,
        )?);
        Ok(())
    })?;
    println!(
        "projection+norm+cast baseline={baseline_post:.4}ms fused={fused_post:.4}ms speedup={:.2}x",
        baseline_post / fused_post
    );
    let rope_baseline = elapsed_ms(&device, iterations, || {
        let q = q_ref_f16
            .reshape((1, 1, Q / HEAD_DIM, HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k_ref_f16
            .reshape((1, 1, KV / HEAD_DIM, HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;
        let (qr, kr) = rope.apply(
            &q.narrow(3, 0, ROPE_DIM)?,
            &k.narrow(3, 0, ROPE_DIM)?,
            position,
        )?;
        drop(Tensor::cat(
            &[qr, q.narrow(3, ROPE_DIM, HEAD_DIM - ROPE_DIM)?],
            3,
        )?);
        drop(Tensor::cat(
            &[kr, k.narrow(3, ROPE_DIM, HEAD_DIM - ROPE_DIM)?],
            3,
        )?);
        Ok(())
    })?;
    let rope_fused = elapsed_ms(&device, iterations, || {
        drop(candle_nn::fused_qkv::partial_rope_f16(
            &q_ref_f16, &k_ref_f16, cos, sin, position, HEAD_DIM, ROPE_DIM,
        )?);
        Ok(())
    })?;
    println!(
        "partial-rope baseline={rope_baseline:.4}ms fused={rope_fused:.4}ms speedup={:.2}x",
        rope_baseline / rope_fused
    );
    Ok(())
}
