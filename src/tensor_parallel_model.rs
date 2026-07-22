//! Rank-local MiniMax-M2.7 decoder math for fixed two-GPU tensor parallelism.

use crate::tensor_parallel::{TensorParallelGroup, TensorParallelRankGroup};
use anyhow::{Context, Result, bail};
use candle_core::{
    DType, Device, Module, Tensor,
    quantized::{GgmlDType, QTensor},
};
use candle_nn::{Embedding, kv_cache::KvCache};
use candle_transformers::{
    fused_moe::{FusedMoeGGUF, MoeCfg},
    models::{quantized_qwen3::RotaryEmbedding, with_tracing::QMatMul},
    quantized_nn::RmsNorm,
    quantized_var_builder::{TensorParallelShard, TensorParallelSpec, VarBuilder},
};
use cudarc::nccl::Id;
use std::{path::PathBuf, sync::Arc};

pub const HIDDEN_SIZE: usize = 3_072;
pub const QUERY_HEADS: usize = 48;
pub const KV_HEADS: usize = 8;
pub const HEAD_DIM: usize = 128;
pub const ROPE_DIM: usize = 64;
pub const LOCAL_QUERY_HEADS: usize = QUERY_HEADS / 2;
pub const LOCAL_KV_HEADS: usize = KV_HEADS / 2;
pub const LOCAL_ATTN_WIDTH: usize = LOCAL_QUERY_HEADS * HEAD_DIM;
pub const LOCAL_KV_WIDTH: usize = LOCAL_KV_HEADS * HEAD_DIM;
pub const EXPERTS: usize = 256;
pub const TOPK: usize = 8;
pub const LOCAL_MOE_INTERMEDIATE: usize = 768;
pub const VOCAB_SIZE: usize = 200_064;
pub const NUM_LAYERS: usize = 62;
pub const MAX_CONTEXT: usize = 196_608;
const NORM_EPS: f64 = 1e-6;

/// Exact TP placement for all tensors in one decoder layer.
pub fn layer_tensor_specs(layer: usize) -> Vec<TensorParallelSpec> {
    let prefix = format!("blk.{layer}");
    let spec = |suffix: &str, shard| TensorParallelSpec::new(format!("{prefix}.{suffix}"), shard);
    vec![
        spec("attn_q.weight", TensorParallelShard::Split { axis: 0 }),
        spec("attn_k.weight", TensorParallelShard::Split { axis: 0 }),
        spec("attn_v.weight", TensorParallelShard::Split { axis: 0 }),
        spec("attn_output.weight", TensorParallelShard::Split { axis: 1 }),
        spec("attn_q_norm.weight", TensorParallelShard::Split { axis: 0 }),
        spec("attn_k_norm.weight", TensorParallelShard::Split { axis: 0 }),
        spec("attn_norm.weight", TensorParallelShard::Replicated),
        spec("ffn_norm.weight", TensorParallelShard::Replicated),
        spec("ffn_gate_inp.weight", TensorParallelShard::Replicated),
        spec("exp_probs_b.bias", TensorParallelShard::Replicated),
        spec(
            "ffn_gate_exps.weight",
            TensorParallelShard::Split { axis: 1 },
        ),
        spec("ffn_up_exps.weight", TensorParallelShard::Split { axis: 1 }),
        spec(
            "ffn_down_exps.weight",
            TensorParallelShard::Split { axis: 2 },
        ),
    ]
}

struct ProjectedQkv {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    stats: Tensor,
}

struct RankAttention {
    q: QMatMul,
    k: QMatMul,
    v: QMatMul,
    q_weight: Arc<QTensor>,
    k_weight: Arc<QTensor>,
    v_weight: Arc<QTensor>,
    fused_decode: bool,
    o: QMatMul,
    q_norm_weight: Tensor,
    k_norm_weight: Tensor,
    rope: Arc<RotaryEmbedding>,
    cache: KvCache,
}

impl RankAttention {
    fn new(vb: &VarBuilder, rope: Arc<RotaryEmbedding>) -> candle_core::Result<Self> {
        let q_weight = vb
            .pp("attn_q")
            .get((LOCAL_ATTN_WIDTH, HIDDEN_SIZE), "weight")?;
        let k_weight = vb
            .pp("attn_k")
            .get((LOCAL_KV_WIDTH, HIDDEN_SIZE), "weight")?;
        let v_weight = vb
            .pp("attn_v")
            .get((LOCAL_KV_WIDTH, HIDDEN_SIZE), "weight")?;
        let fused_decode = [q_weight.dtype(), k_weight.dtype(), v_weight.dtype()]
            .iter()
            .all(|dtype| *dtype == GgmlDType::Q8_0);
        Ok(Self {
            q: QMatMul::from_weights(q_weight.clone())?,
            k: QMatMul::from_weights(k_weight.clone())?,
            v: QMatMul::from_weights(v_weight.clone())?,
            q_weight,
            k_weight,
            v_weight,
            fused_decode,
            o: QMatMul::new(LOCAL_ATTN_WIDTH, HIDDEN_SIZE, vb.pp("attn_output"))?,
            q_norm_weight: vb
                .pp("attn_q_norm")
                .get(LOCAL_ATTN_WIDTH, "weight")?
                .dequantize(vb.device())?
                .to_dtype(DType::F32)?,
            k_norm_weight: vb
                .pp("attn_k_norm")
                .get(LOCAL_KV_WIDTH, "weight")?
                .dequantize(vb.device())?
                .to_dtype(DType::F32)?,
            rope,
            cache: KvCache::new(2, 4_096),
        })
    }

    fn project(&self, x: &Tensor) -> candle_core::Result<ProjectedQkv> {
        if x.dim(1)? == 1 && self.fused_decode {
            let (q, k, v, stats) = candle_nn::fused_qkv::q8_0_decode_with_sums(
                x,
                &self.q_weight,
                &self.k_weight,
                &self.v_weight,
            )?;
            return Ok(ProjectedQkv { q, k, v, stats });
        }
        let q = self.q.forward(x)?.contiguous()?.to_dtype(DType::F32)?;
        let k = self.k.forward(x)?.contiguous()?.to_dtype(DType::F32)?;
        let v = self.v.forward(x)?.contiguous()?.to_dtype(DType::F32)?;
        let q_sum = q.sqr()?.sum_keepdim(candle_core::D::Minus1)?;
        let k_sum = k.sqr()?.sum_keepdim(candle_core::D::Minus1)?;
        let stats = Tensor::cat(&[q_sum, k_sum], candle_core::D::Minus1)?.contiguous()?;
        Ok(ProjectedQkv { q, k, v, stats })
    }

    fn finish(
        &mut self,
        projected: ProjectedQkv,
        global_stats: &Tensor,
        pos: usize,
    ) -> candle_core::Result<Tensor> {
        let (_, query_len, _) = projected.q.dims3()?;
        let (q, k, v) = if query_len == 1 {
            let (cos, sin) = self.rope.cos_sin_tensors();
            let (q, k, v) = candle_nn::fused_qkv::tp_rmsnorm_rope_f16(
                &projected.q,
                &projected.k,
                &projected.v,
                global_stats,
                &self.q_norm_weight,
                &self.k_norm_weight,
                cos,
                sin,
                QUERY_HEADS * HEAD_DIM,
                KV_HEADS * HEAD_DIM,
                HEAD_DIM,
                ROPE_DIM,
                pos,
                NORM_EPS as f32,
            )?;
            (
                q.reshape((1, LOCAL_QUERY_HEADS, 1, HEAD_DIM))?,
                k.reshape((1, LOCAL_KV_HEADS, 1, HEAD_DIM))?,
                v.reshape((1, LOCAL_KV_HEADS, 1, HEAD_DIM))?,
            )
        } else {
            let q_denominator =
                ((global_stats.narrow(2, 0, 1)? / (QUERY_HEADS * HEAD_DIM) as f64)? + NORM_EPS)?
                    .sqrt()?;
            let k_denominator = ((global_stats.narrow(2, 1, 1)? / (KV_HEADS * HEAD_DIM) as f64)?
                + NORM_EPS)?
                .sqrt()?;
            let q = projected
                .q
                .broadcast_div(&q_denominator)?
                .broadcast_mul(&self.q_norm_weight)?
                .to_dtype(DType::F16)?
                .reshape((1, query_len, LOCAL_QUERY_HEADS, HEAD_DIM))?
                .transpose(1, 2)?
                .contiguous()?;
            let k = projected
                .k
                .broadcast_div(&k_denominator)?
                .broadcast_mul(&self.k_norm_weight)?
                .to_dtype(DType::F16)?
                .reshape((1, query_len, LOCAL_KV_HEADS, HEAD_DIM))?
                .transpose(1, 2)?
                .contiguous()?;
            let v = projected
                .v
                .to_dtype(DType::F16)?
                .reshape((1, query_len, LOCAL_KV_HEADS, HEAD_DIM))?
                .transpose(1, 2)?
                .contiguous()?;
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
            candle_nn::fused_attention::gqa_decode_tp2_f16_128(
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
                LOCAL_ATTN_WIDTH,
            ))?)
    }
}

struct RankLayer {
    attention: RankAttention,
    attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
    moe: FusedMoeGGUF,
    device: Device,
}

impl RankLayer {
    fn new(
        vb: &VarBuilder,
        device: &Device,
        rope: Arc<RotaryEmbedding>,
    ) -> candle_core::Result<Self> {
        let cfg = MoeCfg {
            moe_intermediate_size: LOCAL_MOE_INTERMEDIATE,
            num_experts: EXPERTS,
            norm_topk_prob: true,
            num_experts_per_tok: TOPK,
            hidden_size: HIDDEN_SIZE,
            act: candle_nn::Activation::Silu,
            decoder_sparse_step: None,
        };
        Ok(Self {
            attention: RankAttention::new(vb, rope)?,
            attention_norm: RmsNorm::new(HIDDEN_SIZE, NORM_EPS, vb.pp("attn_norm"))?,
            ffn_norm: RmsNorm::new(HIDDEN_SIZE, NORM_EPS, vb.pp("ffn_norm"))?,
            moe: FusedMoeGGUF::new(&cfg, vb.clone(), DType::F16)?,
            device: device.clone(),
        })
    }

    fn project_attention(&self, input: &Tensor) -> candle_core::Result<ProjectedQkv> {
        self.attention.project(&self.attention_norm.forward(input)?)
    }

    fn forward_moe(&self, hidden: &Tensor, prefill: bool) -> candle_core::Result<Tensor> {
        self.forward_moe_normalized(&self.ffn_norm.forward(hidden)?, prefill)
    }

    fn forward_moe_normalized(
        &self,
        normalized: &Tensor,
        prefill: bool,
    ) -> candle_core::Result<Tensor> {
        self.moe.forward(normalized, prefill)?.to_dtype(DType::F32)
    }
}

/// One decoder layer represented by two rank-local weight and KV-cache shards.
pub struct TensorParallelLayer {
    layer_index: usize,
    ranks: [RankLayer; 2],
    weight_bytes: [usize; 2],
}

impl TensorParallelLayer {
    pub fn load(
        shards: &[PathBuf],
        layer_index: usize,
        devices: &[Device; 2],
        ropes: &[Arc<RotaryEmbedding>; 2],
    ) -> Result<Self> {
        let specs = layer_tensor_specs(layer_index);
        let builders = VarBuilder::from_gguf_tensor_parallel(shards, &specs, devices)
            .with_context(|| format!("load TP shards for layer {layer_index}"))?;
        let weight_bytes_for = |rank: usize| -> candle_core::Result<usize> {
            Ok(specs
                .iter()
                .map(|spec| {
                    builders[rank]
                        .get_no_shape(&spec.name)
                        .map(|tensor| tensor.storage_size_in_bytes())
                })
                .collect::<candle_core::Result<Vec<_>>>()?
                .into_iter()
                .sum())
        };
        let weight_bytes = [weight_bytes_for(0)?, weight_bytes_for(1)?];
        let prefix = format!("blk.{layer_index}");
        let rank0 = RankLayer::new(&builders[0].pp(&prefix), &devices[0], ropes[0].clone())
            .with_context(|| format!("construct layer {layer_index} rank 0"))?;
        let rank1 = RankLayer::new(&builders[1].pp(&prefix), &devices[1], ropes[1].clone())
            .with_context(|| format!("construct layer {layer_index} rank 1"))?;
        Ok(Self {
            layer_index,
            ranks: [rank0, rank1],
            weight_bytes,
        })
    }

    pub fn forward(
        &mut self,
        inputs: [Tensor; 2],
        pos: usize,
        prefill: bool,
        collective: &mut TensorParallelGroup,
    ) -> Result<[Tensor; 2]> {
        self.validate_inputs(&inputs)?;
        let query_len = inputs[0].dim(1)?;
        let initial_lengths = self.cache_lengths();
        if initial_lengths[0] != initial_lengths[1] {
            self.reset();
            bail!(
                "layer {} entered forward with divergent TP cache lengths {:?}; caches were reset",
                self.layer_index,
                initial_lengths
            )
        }
        if pos != initial_lengths[0] {
            bail!(
                "layer {} forward position {pos} does not match TP cache length {}",
                self.layer_index,
                initial_lengths[0]
            )
        }
        let expected_length = initial_lengths[0].checked_add(query_len).ok_or_else(|| {
            anyhow::anyhow!("layer {} cache length overflows usize", self.layer_index)
        })?;

        match self.forward_inner(inputs, pos, prefill, collective) {
            Ok(outputs) if self.cache_lengths() == [expected_length; 2] => Ok(outputs),
            Ok(_) => {
                let final_lengths = self.cache_lengths();
                let error = anyhow::anyhow!(
                    "layer {} completed with TP cache lengths {:?}, expected [{expected_length}, {expected_length}]",
                    self.layer_index,
                    final_lengths
                );
                if let Err(rollback) = self.rollback_caches(initial_lengths) {
                    self.reset();
                    Err(error.context(format!(
                        "cache rollback also failed ({rollback}); caches were reset"
                    )))
                } else {
                    Err(error)
                }
            }
            Err(error) => {
                if let Err(rollback) = self.rollback_caches(initial_lengths) {
                    self.reset();
                    Err(error.context(format!(
                        "cache rollback also failed ({rollback}); caches were reset"
                    )))
                } else {
                    Err(error)
                }
            }
        }
    }

    fn forward_inner(
        &mut self,
        inputs: [Tensor; 2],
        pos: usize,
        prefill: bool,
        collective: &mut TensorParallelGroup,
    ) -> Result<[Tensor; 2]> {
        let query_len = inputs[0].dim(1)?;
        let residuals = [inputs[0].clone(), inputs[1].clone()];
        let projected0 = self.ranks[0]
            .project_attention(&inputs[0])
            .with_context(|| self.context(0, "attention projection"))?;
        let projected1 = self.ranks[1]
            .project_attention(&inputs[1])
            .with_context(|| self.context(1, "attention projection"))?;

        // Collective 1: complete-width Q/K sums of squares.
        let global_stats = collective
            .all_reduce_sum([projected0.stats.clone(), projected1.stats.clone()])
            .with_context(|| format!("layer {} Q/K statistics all-reduce", self.layer_index))?;
        let attention0 = self.ranks[0]
            .attention
            .finish(projected0, &global_stats[0], pos)
            .with_context(|| self.context(0, "attention"))?
            .to_dtype(DType::F32)?;
        let attention1 = self.ranks[1]
            .attention
            .finish(projected1, &global_stats[1], pos)
            .with_context(|| self.context(1, "attention"))?
            .to_dtype(DType::F32)?;

        // Collective 2: row-parallel attention output projection.
        let (hidden, normalized) = if query_len == 1 {
            let weights = [
                self.ranks[0].ffn_norm.weight().clone(),
                self.ranks[1].ffn_norm.weight().clone(),
            ];
            let (hidden, normalized) = collective
                .all_reduce_sum_with_residual_rmsnorm(
                    [attention0, attention1],
                    residuals,
                    weights,
                    NORM_EPS as f32,
                )
                .with_context(|| {
                    format!("layer {} attention output all-reduce", self.layer_index)
                })?;
            (hidden, Some(normalized))
        } else {
            (
                collective
                    .all_reduce_sum_with_residual([attention0, attention1], residuals)
                    .with_context(|| {
                        format!("layer {} attention output all-reduce", self.layer_index)
                    })?,
                None,
            )
        };
        let (moe0, moe1) = if let Some(normalized) = normalized {
            (
                self.ranks[0]
                    .forward_moe_normalized(&normalized[0], prefill)
                    .with_context(|| self.context(0, "MoE"))?,
                self.ranks[1]
                    .forward_moe_normalized(&normalized[1], prefill)
                    .with_context(|| self.context(1, "MoE"))?,
            )
        } else {
            (
                self.ranks[0]
                    .forward_moe(&hidden[0], prefill)
                    .with_context(|| self.context(0, "MoE"))?,
                self.ranks[1]
                    .forward_moe(&hidden[1], prefill)
                    .with_context(|| self.context(1, "MoE"))?,
            )
        };

        // Collective 3: row-parallel expert down projections.
        collective
            .all_reduce_sum_with_residual([moe0, moe1], hidden)
            .with_context(|| format!("layer {} MoE output all-reduce", self.layer_index))
    }

    /// Computes rank-local router selections from matching hidden states. This
    /// does not advance attention caches and is intended for parity diagnostics.
    pub fn routing_for_inputs(&self, inputs: &[Tensor; 2]) -> Result<[(Tensor, Tensor); 2]> {
        self.validate_inputs(inputs)?;
        let normalized0 = self.ranks[0]
            .ffn_norm
            .forward(&inputs[0])
            .with_context(|| self.context(0, "diagnostic FFN norm"))?;
        let normalized1 = self.ranks[1]
            .ffn_norm
            .forward(&inputs[1])
            .with_context(|| self.context(1, "diagnostic FFN norm"))?;
        Ok([
            self.ranks[0]
                .moe
                .routing(&normalized0)
                .with_context(|| self.context(0, "diagnostic routing"))?,
            self.ranks[1]
                .moe
                .routing(&normalized1)
                .with_context(|| self.context(1, "diagnostic routing"))?,
        ])
    }

    pub fn reset(&mut self) {
        for rank in &mut self.ranks {
            rank.attention.cache.reset();
        }
    }

    fn rollback_caches(&mut self, lengths: [usize; 2]) -> Result<()> {
        for (rank, &length) in lengths.iter().enumerate() {
            self.ranks[rank]
                .attention
                .cache
                .truncate(length)
                .with_context(|| self.context(rank, "rollback KV cache"))?;
        }
        Ok(())
    }

    pub fn truncate_cache(&mut self, seq_len: usize) -> Result<()> {
        let lengths = self.cache_lengths();
        if lengths[0] != lengths[1] {
            self.reset();
            bail!(
                "layer {} cannot truncate divergent TP cache lengths {:?}; caches were reset",
                self.layer_index,
                lengths
            )
        }
        if seq_len > lengths[0] {
            bail!(
                "layer {} cannot truncate TP cache from {} to {seq_len}",
                self.layer_index,
                lengths[0]
            )
        }
        if let Err(error) = self.rollback_caches([seq_len; 2]) {
            self.reset();
            return Err(error.context("TP cache truncation failed; caches were reset"));
        }
        Ok(())
    }

    pub fn weight_bytes(&self) -> [usize; 2] {
        self.weight_bytes
    }

    pub fn cache_lengths(&self) -> [usize; 2] {
        [
            self.ranks[0].attention.cache.current_seq_len(),
            self.ranks[1].attention.cache.current_seq_len(),
        ]
    }

    pub fn cache_shapes(&self) -> Result<[Option<Vec<usize>>; 2]> {
        let shape = |rank: usize| -> Result<Option<Vec<usize>>> {
            Ok(self.ranks[rank]
                .attention
                .cache
                .k()?
                .map(|tensor| tensor.dims().to_vec()))
        };
        Ok([shape(0)?, shape(1)?])
    }

    fn validate_inputs(&self, inputs: &[Tensor; 2]) -> Result<()> {
        for (rank, input) in inputs.iter().enumerate() {
            if !input.device().same_device(&self.ranks[rank].device) {
                bail!(
                    "layer {} rank {rank} input is on the wrong device",
                    self.layer_index
                )
            }
            let (_, _, width) = input.dims3()?;
            if width != HIDDEN_SIZE {
                bail!(
                    "layer {} rank {rank} input width is {width}, expected {HIDDEN_SIZE}",
                    self.layer_index
                )
            }
            if input.dtype() != DType::F32 {
                bail!(
                    "layer {} rank {rank} input must be F32, got {:?}",
                    self.layer_index,
                    input.dtype()
                )
            }
        }
        if inputs[0].dims() != inputs[1].dims() {
            bail!(
                "layer {} rank input shape mismatch: {:?} vs {:?}",
                self.layer_index,
                inputs[0].shape(),
                inputs[1].shape()
            )
        }
        Ok(())
    }

    fn context(&self, rank: usize, operation: &str) -> String {
        format!("layer {} rank {rank} {operation}", self.layer_index)
    }
}

/// One process-owned decoder-layer shard. This keeps all CUDA state for a
/// rank inside the process that created it. Calls must remain serialized: its
/// decode output uses collective workspaces that are deliberately reused by
/// the next layer/forward after stream-ordered consumers have been enqueued.
pub struct TensorParallelRankLayer {
    layer_index: usize,
    rank: usize,
    inner: RankLayer,
    weight_bytes: usize,
}

impl TensorParallelRankLayer {
    pub fn load(
        shards: &[PathBuf],
        layer_index: usize,
        rank: usize,
        device: &Device,
        rope: Arc<RotaryEmbedding>,
    ) -> Result<Self> {
        if rank >= 2 {
            bail!("tensor-parallel rank must be 0 or 1, got {rank}")
        }
        let specs = layer_tensor_specs(layer_index);
        let builder = VarBuilder::from_gguf_tensor_parallel_rank(shards, &specs, rank, device)
            .with_context(|| format!("load TP shard for layer {layer_index} rank {rank}"))?;
        let weight_bytes = specs
            .iter()
            .map(|spec| {
                builder
                    .get_no_shape(&spec.name)
                    .map(|tensor| tensor.storage_size_in_bytes())
            })
            .collect::<candle_core::Result<Vec<_>>>()?
            .into_iter()
            .sum();
        let prefix = format!("blk.{layer_index}");
        let inner = RankLayer::new(&builder.pp(&prefix), device, rope)
            .with_context(|| format!("construct layer {layer_index} rank {rank}"))?;
        Ok(Self {
            layer_index,
            rank,
            inner,
            weight_bytes,
        })
    }

    pub fn forward(
        &mut self,
        input: Tensor,
        pos: usize,
        prefill: bool,
        collective: &mut TensorParallelRankGroup,
    ) -> Result<Tensor> {
        if collective.rank() != self.rank {
            bail!(
                "layer {} rank {} received collective rank {}",
                self.layer_index,
                self.rank,
                collective.rank()
            )
        }
        self.validate_input(&input)?;
        let query_len = input.dim(1)?;
        let initial_length = self.cache_length();
        if pos != initial_length {
            bail!(
                "layer {} rank {} forward position {pos} does not match cache length {initial_length}",
                self.layer_index,
                self.rank
            )
        }
        let expected_length = initial_length.checked_add(query_len).ok_or_else(|| {
            anyhow::anyhow!(
                "layer {} rank {} cache length overflows usize",
                self.layer_index,
                self.rank
            )
        })?;
        match self.forward_inner(input, pos, prefill, collective) {
            Ok(output) if self.cache_length() == expected_length => Ok(output),
            Ok(_) => {
                let final_length = self.cache_length();
                let error = anyhow::anyhow!(
                    "layer {} rank {} completed with cache length {final_length}, expected {expected_length}",
                    self.layer_index,
                    self.rank
                );
                self.rollback_after_error(initial_length, error)
            }
            Err(error) => self.rollback_after_error(initial_length, error),
        }
    }

    fn forward_inner(
        &mut self,
        input: Tensor,
        pos: usize,
        prefill: bool,
        collective: &mut TensorParallelRankGroup,
    ) -> Result<Tensor> {
        let query_len = input.dim(1)?;
        let residual = input.clone();
        let projected = self
            .inner
            .project_attention(&input)
            .with_context(|| self.context("attention projection"))?;
        let global_stats = collective
            .all_reduce_sum(projected.stats.clone())
            .with_context(|| self.context("Q/K statistics all-reduce"))?;
        let attention = self
            .inner
            .attention
            .finish(projected, &global_stats, pos)
            .with_context(|| self.context("attention"))?
            .to_dtype(DType::F32)?;
        let (hidden, normalized) = if query_len == 1 {
            let (hidden, normalized) = collective
                .all_reduce_sum_with_residual_rmsnorm(
                    attention,
                    residual,
                    self.inner.ffn_norm.weight().clone(),
                    NORM_EPS as f32,
                )
                .with_context(|| self.context("attention output all-reduce"))?;
            (hidden, Some(normalized))
        } else {
            let hidden = collective
                .all_reduce_sum_with_residual(attention, residual)
                .with_context(|| self.context("attention output all-reduce"))?;
            (hidden, None)
        };
        let moe = if let Some(normalized) = normalized {
            self.inner
                .forward_moe_normalized(&normalized, prefill)
                .with_context(|| self.context("MoE"))?
        } else {
            self.inner
                .forward_moe(&hidden, prefill)
                .with_context(|| self.context("MoE"))?
        };
        collective
            .all_reduce_sum_with_residual(moe, hidden)
            .with_context(|| self.context("MoE output all-reduce"))
    }

    fn rollback_after_error<T>(
        &mut self,
        initial_length: usize,
        error: anyhow::Error,
    ) -> Result<T> {
        match self.inner.attention.cache.truncate(initial_length) {
            Ok(()) => Err(error),
            Err(rollback) => {
                self.reset();
                Err(error.context(format!(
                    "rank-local cache rollback also failed ({rollback}); cache was reset"
                )))
            }
        }
    }

    fn validate_input(&self, input: &Tensor) -> Result<()> {
        if !input.device().same_device(&self.inner.device) {
            bail!(
                "layer {} rank {} input is on the wrong device",
                self.layer_index,
                self.rank
            )
        }
        let (_, _, width) = input.dims3()?;
        if width != HIDDEN_SIZE || input.dtype() != DType::F32 {
            bail!(
                "layer {} rank {} requires F32 width {HIDDEN_SIZE}, got {:?} width {width}",
                self.layer_index,
                self.rank,
                input.dtype()
            )
        }
        Ok(())
    }

    fn context(&self, operation: &str) -> String {
        format!("layer {} rank {} {operation}", self.layer_index, self.rank)
    }

    pub fn reset(&mut self) {
        self.inner.attention.cache.reset();
    }

    pub fn truncate_cache(&mut self, seq_len: usize) -> Result<()> {
        let current = self.cache_length();
        if seq_len > current {
            bail!(
                "layer {} rank {} cannot truncate cache from {current} to {seq_len}",
                self.layer_index,
                self.rank
            )
        }
        self.inner
            .attention
            .cache
            .truncate(seq_len)
            .with_context(|| self.context("truncate KV cache"))
    }

    pub fn cache_length(&self) -> usize {
        self.inner.attention.cache.current_seq_len()
    }

    pub fn cache_shape(&self) -> Result<Option<Vec<usize>>> {
        Ok(self
            .inner
            .attention
            .cache
            .k()?
            .map(|tensor| tensor.dims().to_vec()))
    }

    pub fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }
}

struct RankZeroGlobals {
    embedding: Embedding,
    norm: RmsNorm,
    output: QMatMul,
    weight_bytes: usize,
}

impl RankZeroGlobals {
    fn load(shards: &[PathBuf], device: &Device) -> Result<Self> {
        let prefixes = ["token_embd.", "output_norm.", "output."]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let builder = VarBuilder::from_gguf_selected(shards, &prefixes, device)
            .context("load tensor-parallel rank-0 global tensors")?;
        let weight_bytes = ["token_embd.weight", "output_norm.weight", "output.weight"]
            .into_iter()
            .map(|name| {
                builder
                    .get_no_shape(name)
                    .map(|tensor| tensor.storage_size_in_bytes())
            })
            .collect::<candle_core::Result<Vec<_>>>()?
            .into_iter()
            .sum();
        let embedding = Embedding::new(
            builder
                .pp("token_embd")
                .get((VOCAB_SIZE, HIDDEN_SIZE), "weight")?
                .dequantize(device)?,
            HIDDEN_SIZE,
        );
        let norm = RmsNorm::new(HIDDEN_SIZE, NORM_EPS, builder.pp("output_norm"))?;
        let output = QMatMul::new(HIDDEN_SIZE, VOCAB_SIZE, builder.pp("output"))?;
        Ok(Self {
            embedding,
            norm,
            output,
            weight_bytes,
        })
    }

    fn embed(&self, ids: &[u32], device: &Device) -> Result<Tensor> {
        let input = Tensor::from_slice(ids, (1, ids.len()), device)?;
        Ok(self.embedding.forward(&input)?.to_dtype(DType::F32)?)
    }

    fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let query_len = hidden.dim(1)?;
        let final_hidden = hidden.narrow(1, query_len - 1, 1)?;
        Ok(self
            .output
            .forward(&self.norm.forward(&final_hidden)?)?
            .to_dtype(DType::F32)?
            .squeeze(1)?
            .squeeze(0)?)
    }
}

/// Output from one process-owned TP rank. Rank 0 returns logits while rank 1
/// returns only its replicated final hidden state.
pub struct TensorParallelRankOutput {
    pub hidden: Tensor,
    pub logits: Option<Tensor>,
}

/// Complete process-owned MiniMax TP rank: all 62 rank-local decoder shards,
/// head-sharded KV caches, the process-local collective, and rank-0-only global
/// embedding/final-output tensors. The controller that drives two instances of
/// this type must own no CUDA state.
pub struct TensorParallelRankModel {
    rank: usize,
    device: Device,
    layers: Vec<TensorParallelRankLayer>,
    globals: Option<RankZeroGlobals>,
    collective: TensorParallelRankGroup,
    layer_weight_bytes: usize,
}

impl TensorParallelRankModel {
    pub fn load(shards: &[PathBuf], rank: usize, device: &Device, id: Id) -> Result<Self> {
        if shards.len() != 4 {
            bail!(
                "MiniMax tensor parallelism requires four GGUF shards, got {}",
                shards.len()
            )
        }
        if rank >= 2 {
            bail!("tensor-parallel rank must be 0 or 1, got {rank}")
        }
        if !device.is_cuda() {
            bail!("tensor-parallel rank {rank} requires a CUDA device")
        }

        // Establish the process group before expensive model loading so rank
        // initialization failures are observed while both children are alive.
        let collective = TensorParallelRankGroup::new(device, rank, id)
            .with_context(|| format!("initialize tensor-parallel rank {rank} collectives"))?;
        let rope = rotary_embedding(device)
            .with_context(|| format!("construct rank {rank} rotary embedding"))?;
        let mut layers = Vec::with_capacity(NUM_LAYERS);
        let mut layer_weight_bytes = 0usize;
        for layer_index in 0..NUM_LAYERS {
            let layer =
                TensorParallelRankLayer::load(shards, layer_index, rank, device, rope.clone())
                    .with_context(|| {
                        format!("load complete model layer {layer_index} rank {rank}")
                    })?;
            layer_weight_bytes = layer_weight_bytes
                .checked_add(layer.weight_bytes())
                .ok_or_else(|| anyhow::anyhow!("rank {rank} layer weight byte count overflow"))?;
            layers.push(layer);
        }
        let globals = if rank == 0 {
            Some(
                RankZeroGlobals::load(shards, device)
                    .context("load complete model rank-0 globals")?,
            )
        } else {
            None
        };
        Ok(Self {
            rank,
            device: device.clone(),
            layers,
            globals,
            collective,
            layer_weight_bytes,
        })
    }

    pub fn forward(&mut self, ids: &[u32], pos: usize) -> Result<TensorParallelRankOutput> {
        self.validate_ids(ids, pos)?;
        let initial_length = self.cache_length()?;
        if initial_length != pos {
            bail!(
                "rank {} model forward position {pos} does not match cache length {initial_length}",
                self.rank
            )
        }
        let expected_length = pos
            .checked_add(ids.len())
            .ok_or_else(|| anyhow::anyhow!("rank {} cache length overflows usize", self.rank))?;

        match self.forward_inner(ids, pos) {
            Ok(output) => match self.cache_length() {
                Ok(final_length) if final_length == expected_length => Ok(output),
                Ok(final_length) => {
                    let error = anyhow::anyhow!(
                        "rank {} model completed with cache length {final_length}, expected {expected_length}",
                        self.rank
                    );
                    self.rollback_after_error(initial_length, error)
                }
                Err(error) => {
                    self.reset();
                    Err(error.context(format!(
                        "rank {} model completed with divergent layer caches; all caches were reset",
                        self.rank
                    )))
                }
            },
            Err(error) => self.rollback_after_error(initial_length, error),
        }
    }

    fn forward_inner(&mut self, ids: &[u32], pos: usize) -> Result<TensorParallelRankOutput> {
        let query_len = ids.len();
        let input = match &self.globals {
            Some(globals) => globals
                .embed(ids, &self.device)
                .with_context(|| format!("rank {} input embedding", self.rank))?,
            None => Tensor::zeros((1, query_len, HIDDEN_SIZE), DType::F32, &self.device)
                .with_context(|| format!("rank {} broadcast receive allocation", self.rank))?,
        };
        let mut hidden = self
            .collective
            .broadcast_from_rank0(input)
            .with_context(|| format!("rank {} input activation broadcast", self.rank))?;
        for (layer_index, layer) in self.layers.iter_mut().enumerate() {
            hidden = layer
                .forward(hidden, pos, query_len > 1, &mut self.collective)
                .with_context(|| {
                    format!("complete model layer {layer_index} rank {}", self.rank)
                })?;
        }
        let logits = self
            .globals
            .as_ref()
            .map(|globals| globals.logits(&hidden))
            .transpose()
            .with_context(|| format!("rank {} final norm/output projection", self.rank))?;
        Ok(TensorParallelRankOutput { hidden, logits })
    }

    fn validate_ids(&self, ids: &[u32], pos: usize) -> Result<()> {
        if ids.is_empty() {
            bail!(
                "rank {} model input must contain at least one token",
                self.rank
            )
        }
        if let Some((index, id)) = ids
            .iter()
            .enumerate()
            .find(|(_, id)| **id >= VOCAB_SIZE as u32)
        {
            bail!(
                "rank {} input token {id} at position {index} is outside vocabulary size {VOCAB_SIZE}",
                self.rank
            )
        }
        let end = pos
            .checked_add(ids.len())
            .ok_or_else(|| anyhow::anyhow!("rank {} input position overflows usize", self.rank))?;
        if end > MAX_CONTEXT {
            bail!(
                "rank {} input ends at position {end}, beyond context limit {MAX_CONTEXT}",
                self.rank
            )
        }
        Ok(())
    }

    fn rollback_after_error<T>(
        &mut self,
        initial_length: usize,
        error: anyhow::Error,
    ) -> Result<T> {
        match self.truncate_cache(initial_length) {
            Ok(()) => Err(error),
            Err(rollback) => {
                self.reset();
                Err(error.context(format!(
                    "rank {} complete-model cache rollback also failed ({rollback}); all caches were reset",
                    self.rank
                )))
            }
        }
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    pub fn truncate_cache(&mut self, seq_len: usize) -> Result<()> {
        let current = self.cache_length()?;
        if seq_len > current {
            bail!(
                "rank {} cannot truncate complete-model cache from {current} to {seq_len}",
                self.rank
            )
        }
        for layer_index in 0..self.layers.len() {
            if let Err(error) = self.layers[layer_index].truncate_cache(seq_len) {
                self.reset();
                return Err(error).with_context(|| {
                    format!(
                        "truncate complete-model layer {layer_index} rank {}; all rank-local caches were reset",
                        self.rank
                    )
                });
            }
        }
        Ok(())
    }

    pub fn cache_length(&self) -> Result<usize> {
        let first = self
            .layers
            .first()
            .ok_or_else(|| anyhow::anyhow!("rank {} model has no layers", self.rank))?;
        let length = first.cache_length();
        for (layer_index, layer) in self.layers.iter().enumerate().skip(1) {
            if layer.cache_length() != length {
                bail!(
                    "rank {} layer {layer_index} cache length {} differs from layer 0 length {length}",
                    self.rank,
                    layer.cache_length()
                )
            }
        }
        Ok(length)
    }

    pub fn cache_shape(&self) -> Result<Option<Vec<usize>>> {
        self.layers
            .first()
            .ok_or_else(|| anyhow::anyhow!("rank {} model has no layers", self.rank))?
            .cache_shape()
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn nccl_version(&self) -> i32 {
        self.collective.nccl_version()
    }

    pub fn collective_backend(&self) -> &'static str {
        self.collective.backend()
    }

    pub fn layer_weight_bytes(&self) -> usize {
        self.layer_weight_bytes
    }

    pub fn global_weight_bytes(&self) -> usize {
        self.globals
            .as_ref()
            .map_or(0, |globals| globals.weight_bytes)
    }

    pub fn close_peer_mapping(&mut self) -> Result<()> {
        self.collective.close_peer_mapping()
    }

    pub fn free_local_peer_buffer(&mut self) -> Result<()> {
        self.collective.free_local_peer_buffer()
    }
}

/// Creates one process-local persistent F32 RoPE table.
pub fn rotary_embedding(device: &Device) -> Result<Arc<RotaryEmbedding>> {
    Ok(Arc::new(RotaryEmbedding::new(
        DType::F32,
        ROPE_DIM,
        196_608,
        5_000_000.,
        device,
    )?))
}

/// Creates the two persistent F32 RoPE tables shared by all TP layers.
pub fn rotary_embeddings(devices: &[Device; 2]) -> Result<[Arc<RotaryEmbedding>; 2]> {
    Ok([
        rotary_embedding(&devices[0])?,
        rotary_embedding(&devices[1])?,
    ])
}
