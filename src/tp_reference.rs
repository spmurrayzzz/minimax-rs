//! Full-width layer reference used only by tensor-parallel parity harnesses.

use crate::tensor_parallel_model::{
    EXPERTS, HEAD_DIM, HIDDEN_SIZE, KV_HEADS, QUERY_HEADS, ROPE_DIM, TOPK, layer_tensor_specs,
};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor, quantized::QTensor};
use candle_nn::kv_cache::KvCache;
use candle_transformers::{
    fused_moe::{FusedMoeGGUF, MoeCfg},
    models::{quantized_qwen3::RotaryEmbedding, with_tracing::QMatMul},
    quantized_nn::RmsNorm,
    quantized_var_builder::VarBuilder,
};
use std::{path::PathBuf, sync::Arc};

/// Full-width attention matching the existing pipeline implementation.
pub struct FullAttentionReference {
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

impl FullAttentionReference {
    pub fn new(vb: &VarBuilder, rope: Arc<RotaryEmbedding>) -> candle_core::Result<Self> {
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

    pub fn forward(&mut self, x: &Tensor, pos: usize) -> candle_core::Result<Tensor> {
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

    pub fn reset(&mut self) {
        self.cache.reset();
    }

    pub fn truncate_cache(&mut self, seq_len: usize) -> candle_core::Result<()> {
        self.cache.truncate(seq_len)
    }

    pub fn cache_length(&self) -> usize {
        self.cache.current_seq_len()
    }

    pub fn cache_shape(&self) -> candle_core::Result<Option<Vec<usize>>> {
        Ok(self.cache.k()?.map(|tensor| tensor.dims().to_vec()))
    }
}

/// Full-width layer used as the numerical reference for TP tests.
pub struct FullLayerReference {
    attention: FullAttentionReference,
    attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
    moe: FusedMoeGGUF,
    weight_bytes: usize,
}

impl FullLayerReference {
    pub fn load(
        shards: &[PathBuf],
        layer: usize,
        device: &Device,
        rope: Arc<RotaryEmbedding>,
    ) -> Result<Self> {
        let prefix = vec![format!("blk.{layer}.")];
        let builder = VarBuilder::from_gguf_selected(shards, &prefix, device)
            .with_context(|| format!("load full reference layer {layer}"))?;
        let weight_bytes = layer_tensor_specs(layer)
            .iter()
            .map(|spec| {
                builder
                    .get_no_shape(&spec.name)
                    .map(|tensor| tensor.storage_size_in_bytes())
            })
            .collect::<candle_core::Result<Vec<_>>>()?
            .into_iter()
            .sum();
        let vb = builder.pp(format!("blk.{layer}"));
        let cfg = MoeCfg {
            moe_intermediate_size: 1_536,
            num_experts: EXPERTS,
            norm_topk_prob: true,
            num_experts_per_tok: TOPK,
            hidden_size: HIDDEN_SIZE,
            act: candle_nn::Activation::Silu,
            decoder_sparse_step: None,
        };
        Ok(Self {
            attention: FullAttentionReference::new(&vb, rope)?,
            attention_norm: RmsNorm::new(HIDDEN_SIZE, 1e-6, vb.pp("attn_norm"))?,
            ffn_norm: RmsNorm::new(HIDDEN_SIZE, 1e-6, vb.pp("ffn_norm"))?,
            moe: FusedMoeGGUF::new(&cfg, vb, DType::F16)?,
            weight_bytes,
        })
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        pos: usize,
        prefill: bool,
    ) -> candle_core::Result<Tensor> {
        let residual = x.clone();
        let x = (self
            .attention
            .forward(&self.attention_norm.forward(x)?, pos)?
            .to_dtype(DType::F32)?
            + residual)?;
        let residual = x.clone();
        self.moe
            .forward(&self.ffn_norm.forward(&x)?, prefill)?
            .to_dtype(DType::F32)?
            + residual
    }

    pub fn routing_for_input(&self, input: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        self.moe.routing(&self.ffn_norm.forward(input)?)
    }

    pub fn reset(&mut self) {
        self.attention.reset();
    }

    pub fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }

    pub fn truncate_cache(&mut self, seq_len: usize) -> candle_core::Result<()> {
        self.attention.truncate_cache(seq_len)
    }

    pub fn cache_length(&self) -> usize {
        self.attention.cache_length()
    }

    pub fn cache_shape(&self) -> candle_core::Result<Option<Vec<usize>>> {
        self.attention.cache_shape()
    }
}
