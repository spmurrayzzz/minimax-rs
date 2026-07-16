use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_transformers::{
    fused_moe::{FusedMoeGGUF, MoeCfg},
    quantized_var_builder::VarBuilder,
};
use clap::Parser;
use minimax::model_files::discover_gguf_shards;
use std::{path::PathBuf, time::Instant};

#[derive(Debug, Parser)]
struct Args {
    /// Directory containing the four split GGUF files.
    #[arg(long, env = "MINIMAX_MODEL_DIR", value_name = "DIR")]
    model: PathBuf,
}

const HIDDEN: usize = 3072;
const INTERMEDIATE: usize = 1536;
const EXPERTS: usize = 256;
const TOPK: usize = 8;

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
    let args = Args::parse();
    let shards = discover_gguf_shards(&args.model)?;

    let device = Device::new_cuda(0)?;
    unsafe { device.as_cuda_device()?.disable_event_tracking() };
    let prefixes = vec!["blk.0.".to_owned()];
    let vb = VarBuilder::from_gguf_selected(&shards, &prefixes, &device)?;
    let cfg = MoeCfg {
        moe_intermediate_size: INTERMEDIATE,
        num_experts: EXPERTS,
        norm_topk_prob: true,
        num_experts_per_tok: TOPK,
        hidden_size: HIDDEN,
        act: candle_nn::Activation::Silu,
        decoder_sparse_step: None,
    };
    let moe = FusedMoeGGUF::new(&cfg, vb.pp("blk.0"), DType::F16)?;
    println!(
        "expert dtypes: gate={:?} up={:?} down={:?}",
        moe.gate_experts.dtype(),
        moe.up_experts.dtype(),
        moe.down_experts.dtype()
    );

    for (length, iterations) in [
        (1, 50),
        (39, 10),
        (64, 8),
        (128, 6),
        (192, 5),
        (256, 4),
        (384, 3),
        (512, 3),
    ] {
        let x = Tensor::randn(0f32, 1f32, (1, length, HIDDEN), &device)?;
        // Compile/load kernels and populate the CUDA allocator before timing.
        let reference = moe.forward(&x, false)?;
        let tensor_core = moe.forward(&x, true)?;
        let tensor_core_2 = moe.forward(&x, true)?;
        device.synchronize()?;
        let wmma_max_abs = (&reference - &tensor_core)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        let wmma_nondeterminism = (&tensor_core - &tensor_core_2)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        let reference_max = reference.abs()?.max_all()?.to_scalar::<f32>()?;

        let generic_ms = elapsed_ms(&device, iterations, || moe.forward(&x, false))?;
        let wmma_ms = elapsed_ms(&device, iterations, || moe.forward(&x, true))?;
        println!(
            "tokens={length:>3} generic={generic_ms:>8.3} ms wmma={wmma_ms:>8.3} ms wmma_speedup={:>5.2}x ref_max={reference_max:.3} wmma_max_abs={wmma_max_abs:.6} wmma_nondet={wmma_nondeterminism:.6}",
            generic_ms / wmma_ms
        );
    }
    Ok(())
}
