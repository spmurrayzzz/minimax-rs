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

    // Exercise a route list larger than Candle's old shared-memory argsort
    // limit and verify that CUB grouping is stable and exact.
    let routes: Vec<u32> = (0..2048 * TOPK)
        .map(|index| ((index * 73 + index / 11) % EXPERTS) as u32)
        .collect();
    let routes_tensor = Tensor::from_vec(routes.clone(), routes.len(), &device)?;
    let (experts_sorted, route_ids_sorted) = candle_nn::moe::sort_routes(&routes_tensor, EXPERTS)?;
    let experts_sorted = experts_sorted.to_vec1::<u32>()?;
    let route_ids_sorted = route_ids_sorted.to_vec1::<u32>()?;
    let mut expected: Vec<(u32, u32)> = routes
        .into_iter()
        .enumerate()
        .map(|(route, expert)| (expert, route as u32))
        .collect();
    expected.sort_by_key(|&(expert, _)| expert);
    assert_eq!(
        experts_sorted,
        expected
            .iter()
            .map(|&(expert, _)| expert)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        route_ids_sorted,
        expected.iter().map(|&(_, route)| route).collect::<Vec<_>>()
    );
    println!("validated stable routing for {} routes", expected.len());

    for (length, iterations) in [
        (1, 50),
        (5, 30),
        (16, 20),
        (32, 12),
        (39, 10),
        (64, 8),
        (128, 6),
        (192, 5),
        (256, 4),
        (384, 3),
        (512, 3),
        (1024, 2),
        (2048, 1),
    ] {
        let x = Tensor::randn(0f32, 1f32, (1, length, HIDDEN), &device)?;
        // Compile/load kernels and populate the CUDA allocator before timing.
        let reference = moe.forward(&x, false)?;
        let optimized = moe.forward(&x, true)?;
        let optimized_2 = moe.forward(&x, true)?;
        device.synchronize()?;
        let optimized_max_abs = (&reference - &optimized)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        let optimized_nondeterminism = (&optimized - &optimized_2)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        let reference_max = reference.abs()?.max_all()?.to_scalar::<f32>()?;

        let generic_ms = elapsed_ms(&device, iterations, || moe.forward(&x, false))?;
        let optimized_ms = elapsed_ms(&device, iterations, || moe.forward(&x, true))?;
        println!(
            "tokens={length:>3} generic={generic_ms:>8.3} ms optimized={optimized_ms:>8.3} ms speedup={:>5.2}x ref_max={reference_max:.3} max_abs={optimized_max_abs:.6} nondet={optimized_nondeterminism:.6}",
            generic_ms / optimized_ms
        );
    }
    Ok(())
}
