use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor, quantized::QTensor};
use candle_nn::kv_cache::KvCache;
use candle_transformers::{
    fused_moe::{FusedMoeGGUF, MoeCfg},
    models::{quantized_qwen3::RotaryEmbedding, with_tracing::QMatMul},
    quantized_nn::RmsNorm,
    quantized_var_builder::{TensorParallelShard, VarBuilder},
};
use clap::{Parser, ValueEnum};
use minimax::{
    model_files::discover_gguf_shards,
    tensor_parallel::TensorParallelGroup,
    tensor_parallel_model::{
        EXPERTS, HEAD_DIM, HIDDEN_SIZE, KV_HEADS, QUERY_HEADS, ROPE_DIM, TOPK, TensorParallelLayer,
        layer_tensor_specs, rotary_embeddings,
    },
    tp_reference::FullLayerReference,
};
use std::{path::PathBuf, sync::Arc, time::Instant};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ProfileGraph {
    Reference,
    Tensor,
    Hybrid,
}

#[derive(Debug, Parser)]
struct Args {
    /// Directory containing the four split GGUF files.
    #[arg(long, env = "MINIMAX_MODEL_DIR", value_name = "DIR")]
    model: PathBuf,
    #[arg(long, default_value_t = 0)]
    layer: usize,
    #[arg(long, default_value_t = 512)]
    context: usize,
    #[arg(long, default_value_t = 50)]
    iterations: usize,
    /// Skip parity/prefill benchmarks and delimit only this decode graph for nsys/ncu.
    #[arg(long, value_enum)]
    profile: Option<ProfileGraph>,
}

struct ReferenceAttention {
    q: QMatMul,
    k: QMatMul,
    v: QMatMul,
    q_weight: Arc<QTensor>,
    k_weight: Arc<QTensor>,
    v_weight: Arc<QTensor>,
    o: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rope: Arc<RotaryEmbedding>,
    cache: KvCache,
    fused_decode: bool,
}

impl ReferenceAttention {
    fn new(vb: &VarBuilder, rope: Arc<RotaryEmbedding>) -> candle_core::Result<Self> {
        let q_weight = vb
            .pp("attn_q")
            .get((QUERY_HEADS * HEAD_DIM, HIDDEN_SIZE), "weight")?;
        let k_weight = vb
            .pp("attn_k")
            .get((KV_HEADS * HEAD_DIM, HIDDEN_SIZE), "weight")?;
        let v_weight = vb
            .pp("attn_v")
            .get((KV_HEADS * HEAD_DIM, HIDDEN_SIZE), "weight")?;
        let fused_decode = [q_weight.dtype(), k_weight.dtype(), v_weight.dtype()]
            .iter()
            .all(|dtype| *dtype == candle_core::quantized::GgmlDType::Q8_0);
        Ok(Self {
            q: QMatMul::from_weights(q_weight.clone())?,
            k: QMatMul::from_weights(k_weight.clone())?,
            v: QMatMul::from_weights(v_weight.clone())?,
            q_weight,
            k_weight,
            v_weight,
            o: QMatMul::new(QUERY_HEADS * HEAD_DIM, HIDDEN_SIZE, vb.pp("attn_output"))?,
            q_norm: RmsNorm::new(QUERY_HEADS * HEAD_DIM, 1e-6, vb.pp("attn_q_norm"))?,
            k_norm: RmsNorm::new(KV_HEADS * HEAD_DIM, 1e-6, vb.pp("attn_k_norm"))?,
            rope,
            cache: KvCache::new(2, 4_096),
            fused_decode,
        })
    }

    fn forward(&mut self, x: &Tensor, pos: usize) -> candle_core::Result<Tensor> {
        let (_, query_len, _) = x.dims3()?;
        let (q, k, v) = if query_len == 1 && self.fused_decode {
            let (q, k, v) = candle_nn::fused_qkv::q8_0_decode_rmsnorm_f16(
                x,
                &self.q_weight,
                &self.k_weight,
                &self.v_weight,
                self.q_norm.weight(),
                self.k_norm.weight(),
                1e-6,
            )?;
            let (cos, sin) = self.rope.cos_sin_tensors();
            let (q, k) =
                candle_nn::fused_qkv::partial_rope_f16(&q, &k, cos, sin, pos, HEAD_DIM, ROPE_DIM)?;
            (
                q.reshape((1, query_len, QUERY_HEADS, HEAD_DIM))?
                    .transpose(1, 2)?
                    .contiguous()?,
                k.reshape((1, query_len, KV_HEADS, HEAD_DIM))?
                    .transpose(1, 2)?
                    .contiguous()?,
                v.reshape((1, query_len, KV_HEADS, HEAD_DIM))?
                    .transpose(1, 2)?
                    .contiguous()?,
            )
        } else {
            let q = self
                .q_norm
                .forward(&self.q.forward(x)?.contiguous()?)?
                .reshape((1, query_len, QUERY_HEADS, HEAD_DIM))?
                .transpose(1, 2)?
                .contiguous()?
                .to_dtype(DType::F16)?;
            let k = self
                .k_norm
                .forward(&self.k.forward(x)?.contiguous()?)?
                .reshape((1, query_len, KV_HEADS, HEAD_DIM))?
                .transpose(1, 2)?
                .contiguous()?
                .to_dtype(DType::F16)?;
            let v = self
                .v
                .forward(x)?
                .reshape((1, query_len, KV_HEADS, HEAD_DIM))?
                .transpose(1, 2)?
                .contiguous()?
                .to_dtype(DType::F16)?;
            let (rotated_q, rotated_k) =
                self.rope
                    .apply(&q.narrow(3, 0, ROPE_DIM)?, &k.narrow(3, 0, ROPE_DIM)?, pos)?;
            (
                Tensor::cat(&[rotated_q, q.narrow(3, ROPE_DIM, HEAD_DIM - ROPE_DIM)?], 3)?
                    .contiguous()?,
                Tensor::cat(&[rotated_k, k.narrow(3, ROPE_DIM, HEAD_DIM - ROPE_DIM)?], 3)?
                    .contiguous()?,
                v,
            )
        };
        let (k, v) = self.cache.append(&k, &v)?;
        let attended = if query_len == 1 {
            candle_nn::fused_attention::gqa_decode_f16_128(
                &q,
                &k,
                &v,
                1.0 / (HEAD_DIM as f32).sqrt(),
            )?
        } else {
            candle_flash_attn::flash_attn(
                &q.transpose(1, 2)?,
                &k.transpose(1, 2)?,
                &v.transpose(1, 2)?,
                1.0 / (HEAD_DIM as f32).sqrt(),
                true,
            )?
            .transpose(1, 2)?
        };
        self.o
            .forward(&attended.transpose(1, 2)?.contiguous()?.reshape((
                1,
                query_len,
                QUERY_HEADS * HEAD_DIM,
            ))?)
    }
}

struct HybridRank {
    attention: ReferenceAttention,
    attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
    moe: FusedMoeGGUF,
}

impl HybridRank {
    fn new(vb: &VarBuilder, rope: Arc<RotaryEmbedding>) -> candle_core::Result<Self> {
        let cfg = MoeCfg {
            moe_intermediate_size: 768,
            num_experts: EXPERTS,
            norm_topk_prob: true,
            num_experts_per_tok: TOPK,
            hidden_size: HIDDEN_SIZE,
            act: candle_nn::Activation::Silu,
            decoder_sparse_step: None,
        };
        Ok(Self {
            attention: ReferenceAttention::new(vb, rope)?,
            attention_norm: RmsNorm::new(HIDDEN_SIZE, 1e-6, vb.pp("attn_norm"))?,
            ffn_norm: RmsNorm::new(HIDDEN_SIZE, 1e-6, vb.pp("ffn_norm"))?,
            moe: FusedMoeGGUF::new(&cfg, vb.clone(), DType::F16)?,
        })
    }
}

/// Reduced-communication decision-gate prototype: attention and its KV cache
/// are replicated, while expert intermediate dimensions remain TP-sharded.
struct HybridLayer {
    ranks: [HybridRank; 2],
    weight_bytes: [usize; 2],
}

impl HybridLayer {
    fn load(
        shards: &[PathBuf],
        layer: usize,
        devices: &[Device; 2],
        ropes: &[Arc<RotaryEmbedding>; 2],
    ) -> Result<Self> {
        let mut specs = layer_tensor_specs(layer);
        for spec in &mut specs {
            let suffix = spec
                .name
                .rsplit_once('.')
                .map(|(prefix, _)| prefix)
                .unwrap_or(&spec.name);
            if [
                "attn_q",
                "attn_k",
                "attn_v",
                "attn_output",
                "attn_q_norm",
                "attn_k_norm",
            ]
            .iter()
            .any(|name| suffix.ends_with(name))
            {
                spec.shard = TensorParallelShard::Replicated;
            }
        }
        let builders = VarBuilder::from_gguf_tensor_parallel(shards, &specs, devices)?;
        let weight_bytes = [
            specs
                .iter()
                .map(|spec| {
                    builders[0]
                        .get_no_shape(&spec.name)
                        .map(|tensor| tensor.storage_size_in_bytes())
                })
                .collect::<candle_core::Result<Vec<_>>>()?
                .into_iter()
                .sum(),
            specs
                .iter()
                .map(|spec| {
                    builders[1]
                        .get_no_shape(&spec.name)
                        .map(|tensor| tensor.storage_size_in_bytes())
                })
                .collect::<candle_core::Result<Vec<_>>>()?
                .into_iter()
                .sum(),
        ];
        let prefix = format!("blk.{layer}");
        Ok(Self {
            ranks: [
                HybridRank::new(&builders[0].pp(&prefix), ropes[0].clone())?,
                HybridRank::new(&builders[1].pp(&prefix), ropes[1].clone())?,
            ],
            weight_bytes,
        })
    }

    fn forward(
        &mut self,
        inputs: [Tensor; 2],
        pos: usize,
        prefill: bool,
        collective: &mut TensorParallelGroup,
    ) -> Result<[Tensor; 2]> {
        let normalized0 = self.ranks[0].attention_norm.forward(&inputs[0])?;
        let normalized1 = self.ranks[1].attention_norm.forward(&inputs[1])?;
        let attention0 = self.ranks[0].attention.forward(&normalized0, pos)?;
        let attention1 = self.ranks[1].attention.forward(&normalized1, pos)?;
        let hidden0 = (attention0.to_dtype(DType::F32)? + inputs[0].clone())?;
        let hidden1 = (attention1.to_dtype(DType::F32)? + inputs[1].clone())?;
        let ffn0 = self.ranks[0].ffn_norm.forward(&hidden0)?;
        let ffn1 = self.ranks[1].ffn_norm.forward(&hidden1)?;
        let moe0 = self.ranks[0]
            .moe
            .forward(&ffn0, prefill)?
            .to_dtype(DType::F32)?;
        let moe1 = self.ranks[1]
            .moe
            .forward(&ffn1, prefill)?
            .to_dtype(DType::F32)?;
        collective.all_reduce_sum_with_residual([moe0, moe1], [hidden0, hidden1])
    }

    fn reset(&mut self) {
        for rank in &mut self.ranks {
            rank.attention.cache.reset();
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let shards = discover_gguf_shards(&args.model)?;
    let devices = [Device::new_cuda(0)?, Device::new_cuda(1)?];
    for device in &devices {
        unsafe { device.as_cuda_device()?.disable_event_tracking() };
    }
    let mut collective = TensorParallelGroup::new(&devices)?;
    let ropes = rotary_embeddings(&devices)?;
    let mut reference =
        FullLayerReference::load(&shards, args.layer, &devices[0], ropes[0].clone())?;
    let full_bytes = reference.weight_bytes();
    let mut tensor = TensorParallelLayer::load(&shards, args.layer, &devices, &ropes)?;
    let mut hybrid = HybridLayer::load(&shards, args.layer, &devices, &ropes)?;
    let tp_bytes = tensor.weight_bytes();
    println!(
        "loaded layer {}; backend={} NCCL={} full={:.2}MiB TP2=[{:.2}, {:.2}]MiB hybrid=[{:.2}, {:.2}]MiB",
        args.layer,
        collective.backend(),
        collective.nccl_version(),
        full_bytes as f64 / 1_048_576.0,
        tp_bytes[0] as f64 / 1_048_576.0,
        tp_bytes[1] as f64 / 1_048_576.0,
        hybrid.weight_bytes[0] as f64 / 1_048_576.0,
        hybrid.weight_bytes[1] as f64 / 1_048_576.0,
    );

    if let Some(graph) = args.profile {
        return profile_decode(
            graph,
            args.context,
            args.iterations,
            &devices,
            &mut collective,
            &mut reference,
            &mut tensor,
            &mut hybrid,
        );
    }

    for query_len in [1usize, 5, 39, 512] {
        reference.reset();
        tensor.reset();
        hybrid.reset();
        let hidden = Tensor::randn(0f32, 1f32, (1, query_len, HIDDEN_SIZE), &devices[0])?;
        let rank_inputs = collective.broadcast_from_rank0(hidden.clone())?;
        let reference_routing = reference.routing_for_input(&hidden)?;
        let tensor_routing = tensor.routing_for_inputs(&rank_inputs)?;
        let reference_ids = reference_routing.0.flatten_all()?.to_vec1::<u32>()?;
        anyhow::ensure!(
            tensor_routing[0].0.flatten_all()?.to_vec1::<u32>()? == reference_ids,
            "router IDs differ from reference at length {query_len}, rank 0"
        );
        anyhow::ensure!(
            tensor_routing[1].0.flatten_all()?.to_vec1::<u32>()? == reference_ids,
            "router IDs differ from reference at length {query_len}, rank 1"
        );
        compare(
            &format!("router weights length={query_len} rank=0"),
            &reference_routing.1,
            &tensor_routing[0].1,
        )?;
        compare(
            &format!("router weights length={query_len} rank=1"),
            &reference_routing.1,
            &tensor_routing[1].1,
        )?;
        let expected = reference.forward(&hidden, 0, query_len > 1)?;
        let actual = tensor.forward(rank_inputs, 0, query_len > 1, &mut collective)?;
        compare(
            &format!("prefill length={query_len} rank=0"),
            &expected,
            &actual[0],
        )?;
        compare(
            &format!("prefill length={query_len} rank=1"),
            &expected,
            &actual[1],
        )?;
        compare(
            &format!("prefill length={query_len} rank agreement"),
            &actual[0],
            &actual[1],
        )?;
        anyhow::ensure!(tensor.cache_lengths() == [query_len, query_len]);
        let expected_shapes = [
            Some(vec![1, 4, query_len, HEAD_DIM]),
            Some(vec![1, 4, query_len, HEAD_DIM]),
        ];
        anyhow::ensure!(tensor.cache_shapes()? == expected_shapes);
        let hybrid_inputs = collective.broadcast_from_rank0(hidden)?;
        let hybrid_actual = hybrid.forward(hybrid_inputs, 0, query_len > 1, &mut collective)?;
        compare(
            &format!("hybrid prefill length={query_len} rank=0"),
            &expected,
            &hybrid_actual[0],
        )?;
        compare(
            &format!("hybrid prefill length={query_len} rank=1"),
            &expected,
            &hybrid_actual[1],
        )?;
    }

    reference.reset();
    tensor.reset();
    hybrid.reset();
    let context_hidden = Tensor::randn(0f32, 1f32, (1, args.context, HIDDEN_SIZE), &devices[0])?;
    let context_inputs = collective.broadcast_from_rank0(context_hidden.clone())?;
    drop(reference.forward(&context_hidden, 0, true)?);
    drop(tensor.forward(context_inputs, 0, true, &mut collective)?);
    let hybrid_context_inputs = collective.broadcast_from_rank0(context_hidden.clone())?;
    drop(hybrid.forward(hybrid_context_inputs, 0, true, &mut collective)?);
    let token = Tensor::randn(0f32, 1f32, (1, 1, HIDDEN_SIZE), &devices[0])?;
    let token_inputs = collective.broadcast_from_rank0(token.clone())?;
    let expected = reference.forward(&token, args.context, false)?;
    let actual = tensor.forward(token_inputs, args.context, false, &mut collective)?;
    compare("cached decode rank=0", &expected, &actual[0])?;
    compare("cached decode rank=1", &expected, &actual[1])?;
    anyhow::ensure!(tensor.cache_lengths() == [args.context + 1; 2]);
    let hybrid_token_inputs = collective.broadcast_from_rank0(token)?;
    let hybrid_actual =
        hybrid.forward(hybrid_token_inputs, args.context, false, &mut collective)?;
    compare("hybrid cached decode rank=0", &expected, &hybrid_actual[0])?;
    compare("hybrid cached decode rank=1", &expected, &hybrid_actual[1])?;

    // Benchmark each graph from equivalent warmed caches. Cache growth is
    // intentional and mirrors autoregressive decode.
    let benchmark_token = Tensor::randn(0f32, 1f32, (1, 1, HIDDEN_SIZE), &devices[0])?;
    devices[0].synchronize()?;
    devices[1].synchronize()?;
    let broadcast_started = Instant::now();
    let benchmark_inputs = collective.broadcast_from_rank0(benchmark_token.clone())?;
    devices[0].synchronize()?;
    devices[1].synchronize()?;
    let broadcast_ms = broadcast_started.elapsed().as_secs_f64() * 1e3;

    let started = Instant::now();
    for iteration in 0..args.iterations {
        drop(reference.forward(&benchmark_token, args.context + 1 + iteration, false)?);
    }
    devices[0].synchronize()?;
    let reference_ms = started.elapsed().as_secs_f64() * 1e3 / args.iterations as f64;

    let started = Instant::now();
    for iteration in 0..args.iterations {
        let inputs = [benchmark_inputs[0].clone(), benchmark_inputs[1].clone()];
        drop(tensor.forward(inputs, args.context + 1 + iteration, false, &mut collective)?);
    }
    devices[0].synchronize()?;
    devices[1].synchronize()?;
    let tensor_ms = started.elapsed().as_secs_f64() * 1e3 / args.iterations as f64;

    let started = Instant::now();
    for iteration in 0..args.iterations {
        let inputs = [benchmark_inputs[0].clone(), benchmark_inputs[1].clone()];
        drop(hybrid.forward(inputs, args.context + 1 + iteration, false, &mut collective)?);
    }
    devices[0].synchronize()?;
    devices[1].synchronize()?;
    let hybrid_ms = started.elapsed().as_secs_f64() * 1e3 / args.iterations as f64;
    println!(
        "decode context={} broadcast/chunk={broadcast_ms:.3}ms reference={reference_ms:.3}ms TP2={tensor_ms:.3}ms ({:.3}x) hybrid={hybrid_ms:.3}ms ({:.3}x)",
        args.context,
        reference_ms / tensor_ms,
        reference_ms / hybrid_ms,
    );

    for query_len in [39usize, 512] {
        let hidden = Tensor::randn(0f32, 1f32, (1, query_len, HIDDEN_SIZE), &devices[0])?;
        devices[0].synchronize()?;
        devices[1].synchronize()?;
        let broadcast_started = Instant::now();
        let prefill_inputs = collective.broadcast_from_rank0(hidden.clone())?;
        devices[0].synchronize()?;
        devices[1].synchronize()?;
        let broadcast_ms = broadcast_started.elapsed().as_secs_f64() * 1e3;

        reference.reset();
        devices[0].synchronize()?;
        let started = Instant::now();
        drop(reference.forward(&hidden, 0, true)?);
        devices[0].synchronize()?;
        let reference_prefill = started.elapsed().as_secs_f64() * 1e3;

        tensor.reset();
        let started = Instant::now();
        let inputs = [prefill_inputs[0].clone(), prefill_inputs[1].clone()];
        drop(tensor.forward(inputs, 0, true, &mut collective)?);
        devices[0].synchronize()?;
        devices[1].synchronize()?;
        let tensor_prefill = started.elapsed().as_secs_f64() * 1e3;

        hybrid.reset();
        let started = Instant::now();
        let inputs = [prefill_inputs[0].clone(), prefill_inputs[1].clone()];
        drop(hybrid.forward(inputs, 0, true, &mut collective)?);
        devices[0].synchronize()?;
        devices[1].synchronize()?;
        let hybrid_prefill = started.elapsed().as_secs_f64() * 1e3;
        println!(
            "prefill length={query_len} broadcast/chunk={broadcast_ms:.3}ms reference={reference_prefill:.3}ms TP2={tensor_prefill:.3}ms ({:.3}x) hybrid={hybrid_prefill:.3}ms ({:.3}x)",
            reference_prefill / tensor_prefill,
            reference_prefill / hybrid_prefill,
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn profile_decode(
    graph: ProfileGraph,
    context: usize,
    iterations: usize,
    devices: &[Device; 2],
    collective: &mut TensorParallelGroup,
    reference: &mut FullLayerReference,
    tensor: &mut TensorParallelLayer,
    hybrid: &mut HybridLayer,
) -> Result<()> {
    anyhow::ensure!(iterations > 0, "--iterations must be greater than zero");
    let context_hidden = Tensor::randn(0f32, 1f32, (1, context, HIDDEN_SIZE), &devices[0])?;
    match graph {
        ProfileGraph::Reference => {
            reference.reset();
            drop(reference.forward(&context_hidden, 0, true)?);
        }
        ProfileGraph::Tensor => {
            tensor.reset();
            let inputs = collective.broadcast_from_rank0(context_hidden)?;
            drop(tensor.forward(inputs, 0, true, collective)?);
        }
        ProfileGraph::Hybrid => {
            hybrid.reset();
            let inputs = collective.broadcast_from_rank0(context_hidden)?;
            drop(hybrid.forward(inputs, 0, true, collective)?);
        }
    }
    let token = Tensor::randn(0f32, 1f32, (1, 1, HIDDEN_SIZE), &devices[0])?;
    let rank_inputs = match graph {
        ProfileGraph::Reference => None,
        ProfileGraph::Tensor | ProfileGraph::Hybrid => {
            Some(collective.broadcast_from_rank0(token.clone())?)
        }
    };
    devices[0].synchronize()?;
    devices[1].synchronize()?;
    cudarc::driver::profiler_start()?;
    let started = Instant::now();
    for iteration in 0..iterations {
        let pos = context + iteration;
        match graph {
            ProfileGraph::Reference => drop(reference.forward(&token, pos, false)?),
            ProfileGraph::Tensor => {
                let inputs = rank_inputs.as_ref().expect("rank inputs");
                drop(tensor.forward(
                    [inputs[0].clone(), inputs[1].clone()],
                    pos,
                    false,
                    collective,
                )?)
            }
            ProfileGraph::Hybrid => {
                let inputs = rank_inputs.as_ref().expect("rank inputs");
                drop(hybrid.forward(
                    [inputs[0].clone(), inputs[1].clone()],
                    pos,
                    false,
                    collective,
                )?)
            }
        }
    }
    devices[0].synchronize()?;
    devices[1].synchronize()?;
    let milliseconds = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    cudarc::driver::profiler_stop()?;
    println!(
        "profile graph={graph:?} context={context} iterations={iterations} average={milliseconds:.3}ms"
    );
    Ok(())
}

fn compare(label: &str, expected: &Tensor, actual: &Tensor) -> Result<()> {
    let actual = if actual.device().same_device(expected.device()) {
        actual.clone()
    } else {
        actual
            .contiguous()?
            .to_device(&Device::Cpu)?
            .to_device(expected.device())
            .with_context(|| format!("host-stage {label}"))?
    };
    let difference = (&actual - expected)?.abs()?.to_dtype(DType::F32)?;
    let max_abs = difference.max_all()?.to_scalar::<f32>()?;
    let mean_abs =
        (difference.sum_all()?.to_scalar::<f32>()? / difference.elem_count() as f32).abs();
    println!("{label}: max_abs={max_abs:.7} mean_abs={mean_abs:.7}");
    anyhow::ensure!(
        max_abs.is_finite() && mean_abs.is_finite(),
        "{label} produced a non-finite difference"
    );
    // Quantized row-parallel dots change only accumulation grouping. This is a
    // diagnostic ceiling, not a claim of bitwise layer equivalence.
    anyhow::ensure!(max_abs <= 4.0, "{label} max_abs={max_abs} exceeds 4.0");
    Ok(())
}
