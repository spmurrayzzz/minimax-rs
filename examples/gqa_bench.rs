use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_transformers::utils::repeat_kv;
use std::time::Instant;

const QH: usize = 48;
const KVH: usize = 8;
const HD: usize = 128;

fn old(q: &Tensor, k: &Tensor, v: &Tensor) -> candle_core::Result<Tensor> {
    let k = repeat_kv(k.clone(), QH / KVH)?.contiguous()?;
    let v = repeat_kv(v.clone(), QH / KVH)?.contiguous()?;
    let scores =
        (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * (1.0 / (HD as f64).sqrt()))?;
    candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)
}

fn grouped(q: &Tensor, k: &Tensor, v: &Tensor) -> candle_core::Result<Tensor> {
    let q = q.reshape((1, KVH, QH / KVH, HD))?;
    let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * (1.0 / (HD as f64).sqrt()))?;
    candle_nn::ops::softmax_last_dim(&scores)?
        .matmul(v)?
        .reshape((1, QH, 1, HD))
}

fn old_prefill(q: &Tensor, k: &Tensor, v: &Tensor) -> candle_core::Result<Tensor> {
    let (_, _, seq_len, _) = q.dims4()?;
    let k = repeat_kv(k.clone(), QH / KVH)?.contiguous()?;
    let v = repeat_kv(v.clone(), QH / KVH)?.contiguous()?;
    let scores =
        (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * (1.0 / (HD as f64).sqrt()))?;
    candle_nn::ops::softmax_last_dim(&scores)?
        .matmul(&v)?
        .reshape((1, QH, seq_len, HD))
}

fn grouped_prefill(q: &Tensor, k: &Tensor, v: &Tensor) -> candle_core::Result<Tensor> {
    let (_, _, seq_len, _) = q.dims4()?;
    let kv_len = k.dim(2)?;
    let q = q.reshape((1, KVH, (QH / KVH) * seq_len, HD))?;
    let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * (1.0 / (HD as f64).sqrt()))?
        .reshape((1, KVH, QH / KVH, seq_len, kv_len))?;
    candle_nn::ops::softmax_last_dim(&scores)?
        .reshape((1, KVH, (QH / KVH) * seq_len, kv_len))?
        .matmul(v)?
        .reshape((1, QH, seq_len, HD))
}

fn elapsed_ms<F>(device: &Device, iterations: usize, mut f: F) -> Result<f64>
where
    F: FnMut() -> candle_core::Result<Tensor>,
{
    device.synchronize()?;
    let start = Instant::now();
    for _ in 0..iterations {
        drop(f()?);
    }
    device.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1e3 / iterations as f64)
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    unsafe { device.as_cuda_device()?.disable_event_tracking() };
    for (context, iterations) in [
        (1, 1000),
        (8, 1000),
        (32, 1000),
        (64, 1000),
        (128, 800),
        (256, 600),
        (512, 500),
        (4096, 100),
    ] {
        let q = Tensor::randn(0f32, 1f32, (1, QH, 1, HD), &device)?.to_dtype(DType::F16)?;
        let k = Tensor::randn(0f32, 1f32, (1, KVH, context, HD), &device)?.to_dtype(DType::F16)?;
        let v = Tensor::randn(0f32, 1f32, (1, KVH, context, HD), &device)?.to_dtype(DType::F16)?;
        let reference = old(&q, &k, &v)?;
        let candidate = grouped(&q, &k, &v)?;
        device.synchronize()?;
        let max_abs = (&reference - &candidate)?
            .abs()?
            .to_dtype(DType::F32)?
            .max_all()?
            .to_scalar::<f32>()?;
        let old_ms = elapsed_ms(&device, iterations, || old(&q, &k, &v))?;
        let grouped_ms = elapsed_ms(&device, iterations, || grouped(&q, &k, &v))?;
        println!(
            "decode context={context:>4} repeat={old_ms:.4}ms grouped={grouped_ms:.4}ms speedup={:.2}x max_abs={max_abs:.9}",
            old_ms / grouped_ms
        );
    }
    for (seq_len, iterations) in [(39, 100), (128, 30), (512, 5)] {
        let q = Tensor::randn(0f32, 1f32, (1, QH, seq_len, HD), &device)?.to_dtype(DType::F16)?;
        let k = Tensor::randn(0f32, 1f32, (1, KVH, seq_len, HD), &device)?.to_dtype(DType::F16)?;
        let v = Tensor::randn(0f32, 1f32, (1, KVH, seq_len, HD), &device)?.to_dtype(DType::F16)?;
        let reference = old_prefill(&q, &k, &v)?;
        let candidate = grouped_prefill(&q, &k, &v)?;
        device.synchronize()?;
        let max_abs = (&reference - &candidate)?
            .abs()?
            .to_dtype(DType::F32)?
            .max_all()?
            .to_scalar::<f32>()?;
        let old_ms = elapsed_ms(&device, iterations, || old_prefill(&q, &k, &v))?;
        let grouped_ms = elapsed_ms(&device, iterations, || grouped_prefill(&q, &k, &v))?;
        println!(
            "prefill tokens={seq_len:>3} repeat={old_ms:.4}ms grouped={grouped_ms:.4}ms speedup={:.2}x max_abs={max_abs:.9}",
            old_ms / grouped_ms
        );
    }
    Ok(())
}
