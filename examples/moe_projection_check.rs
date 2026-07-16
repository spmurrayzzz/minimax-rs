use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_transformers::quantized_var_builder::VarBuilder;
use clap::Parser;
use minimax::model_files::discover_gguf_shards;
use std::path::PathBuf;

#[derive(Debug, Parser)]
struct Args {
    /// Directory containing the four split GGUF files.
    #[arg(long, env = "MINIMAX_MODEL_DIR", value_name = "DIR")]
    model: PathBuf,
    #[arg(long, default_value_t = 0)]
    layer: usize,
    #[arg(long, default_value = "gate")]
    kind: String,
    #[arg(long, default_value_t = 7)]
    expert: u32,
}

const EXPERTS: usize = 256;

fn stats(label: &str, got: &Tensor, reference: &Tensor) -> Result<()> {
    let got = got.to_dtype(DType::F32)?;
    let reference = reference.to_dtype(DType::F32)?;
    let diff = (&got - &reference)?;
    let max_abs = diff.abs()?.max_all()?.to_scalar::<f32>()?;
    let rms = diff.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
    let ref_rms = reference.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
    let got_rms = got.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
    let got_min = got.min_all()?.to_scalar::<f32>()?;
    let got_max = got.max_all()?.to_scalar::<f32>()?;
    println!(
        "{label}: max_abs={max_abs:.6} rms_err={rms:.6} ref_rms={ref_rms:.6} got_rms={got_rms:.6} got_range=[{got_min:.6}, {got_max:.6}]"
    );
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let layer = args.layer;
    let expert = args.expert;
    let (name, n, k) = match args.kind.as_str() {
        "gate" => ("ffn_gate_exps", 1536, 3072),
        "up" => ("ffn_up_exps", 1536, 3072),
        "down" => ("ffn_down_exps", 3072, 1536),
        other => anyhow::bail!("unknown tensor kind {other}"),
    };

    let paths = discover_gguf_shards(&args.model)?;
    let device = Device::new_cuda(0)?;
    let tensor_prefix = format!("blk.{layer}.{name}.");
    println!("loading {tensor_prefix} on CUDA:0");
    let vb = VarBuilder::from_gguf_selected(&paths, &[tensor_prefix], &device)?;
    let weights = vb
        .pp(format!("blk.{layer}.{name}"))
        .get((EXPERTS, n, k), "weight")?;
    println!(
        "weight dtype={:?} shape={:?}",
        weights.dtype(),
        weights.shape()
    );

    let m = 4usize;
    let values: Vec<f32> = (0..m * k)
        .map(|i| (((i * 17 + 11) % 257) as f32 - 128.0) / 128.0)
        .collect();
    let input = Tensor::from_vec(values, (m, k), &device)?;
    let sorted_token_ids = Tensor::from_vec((0..m as u32).collect(), m, &device)?;
    let expert_ids = Tensor::from_vec(vec![expert; m], m, &device)?;

    println!("dequantizing full tensor for reference");
    let dequant = weights.dequantize(&device)?;
    let expert_weight = dequant
        .narrow(0, expert as usize, 1)?
        .squeeze(0)?
        .contiguous()?;
    let reference_f32 = input.matmul(&expert_weight.t()?.contiguous()?)?;
    let reference_f16 = input
        .to_dtype(DType::F16)?
        .matmul(&expert_weight.to_dtype(DType::F16)?.t()?.contiguous()?)?
        .to_dtype(DType::F32)?;
    stats("reference f16-vs-f32", &reference_f16, &reference_f32)?;

    let mut previous_prefill: Option<Tensor> = None;
    let mut previous_decode: Option<Tensor> = None;
    for iteration in 0..3 {
        let prefill = candle_nn::moe::moe_gemm_gguf(
            &input,
            &weights,
            &None,
            &sorted_token_ids,
            &expert_ids,
            1,
            true,
            DType::F16,
        )?;
        device.synchronize()?;
        stats(
            &format!("prefill[{iteration}] vs f16-ref"),
            &prefill,
            &reference_f16,
        )?;
        if let Some(previous) = &previous_prefill {
            stats(&format!("prefill[{iteration}] repeat"), &prefill, previous)?;
        }
        previous_prefill = Some(prefill);

        let decode = candle_nn::moe::moe_gemm_gguf(
            &input,
            &weights,
            &None,
            &sorted_token_ids,
            &expert_ids,
            1,
            false,
            DType::F16,
        )?;
        device.synchronize()?;
        stats(
            &format!("decode[{iteration}] vs f32-ref"),
            &decode,
            &reference_f32,
        )?;
        if let Some(previous) = &previous_decode {
            stats(&format!("decode[{iteration}] repeat"), &decode, previous)?;
        }
        previous_decode = Some(decode);
    }
    Ok(())
}
