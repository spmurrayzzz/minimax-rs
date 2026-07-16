//! MiniMax-M2.7 execution graph.  This follows llama.cpp's model definition:
//! pre-norm GQA, independently-normalised Q/K, partial (64 dimension) RoPE, and
//! sigmoid-gated top-8-of-256 SwiGLU experts.
use anyhow::Result;
use candle_core::{
    DType, Device, Module, Tensor,
    quantized::{GgmlDType, QTensor},
};
use candle_nn::Embedding;
use candle_nn::kv_cache::KvCache;
use candle_transformers::{
    fused_moe::{FusedMoeGGUF, MoeCfg},
    models::{quantized_qwen3::RotaryEmbedding, with_tracing::QMatMul},
    quantized_nn::RmsNorm,
    quantized_var_builder::VarBuilder,
};
use std::sync::Arc;

const H: usize = 3072;
const QH: usize = 48;
const KVH: usize = 8;
const HD: usize = 128;
const ROT: usize = 64;
const VOCAB: usize = 200064;
const LAYERS: usize = 62;
const EXPERTS: usize = 256;
const TOPK: usize = 8;

struct Attention {
    q: QMatMul,
    k: QMatMul,
    v: QMatMul,
    qw: Arc<QTensor>,
    kw: Arc<QTensor>,
    vw: Arc<QTensor>,
    o: QMatMul,
    qn: RmsNorm,
    kn: RmsNorm,
    rope: Arc<RotaryEmbedding>,
    cache: KvCache,
    fused_decode: bool,
}
impl Attention {
    fn new(v: &VarBuilder, rope: Arc<RotaryEmbedding>) -> candle_core::Result<Self> {
        let qw = v.pp("attn_q").get((QH * HD, H), "weight")?;
        let kw = v.pp("attn_k").get((KVH * HD, H), "weight")?;
        let vw = v.pp("attn_v").get((KVH * HD, H), "weight")?;
        let fused_decode = [qw.dtype(), kw.dtype(), vw.dtype()]
            .iter()
            .all(|dtype| *dtype == GgmlDType::Q8_0);
        Ok(Self {
            q: QMatMul::from_weights(qw.clone())?,
            k: QMatMul::from_weights(kw.clone())?,
            v: QMatMul::from_weights(vw.clone())?,
            qw,
            kw,
            vw,
            o: QMatMul::new(QH * HD, H, v.pp("attn_output"))?,
            qn: RmsNorm::new(QH * HD, 1e-6, v.pp("attn_q_norm"))?,
            kn: RmsNorm::new(KVH * HD, 1e-6, v.pp("attn_k_norm"))?,
            rope,
            cache: KvCache::new(2, 4096),
            fused_decode,
        })
    }
    fn reset(&mut self) {
        self.cache.reset();
    }
    fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        pos: usize,
    ) -> candle_core::Result<Tensor> {
        let (_, l, _) = x.dims3()?;
        let (q, k, v) = if l == 1 && self.fused_decode {
            // Decode projects Q/K/V from the same activation. Sharing its Q8_1
            // quantization and fusing Q/K RMSNorm plus F16 conversion removes
            // eight tiny launches per layer.
            let (q, k, v) = candle_nn::fused_qkv::q8_0_decode_rmsnorm_f16(
                x,
                &self.qw,
                &self.kw,
                &self.vw,
                self.qn.weight(),
                self.kn.weight(),
                1e-6,
            )?;
            // MiniMax rotates only the first 64 dimensions of each 128-d head.
            // Keep the F32 tables resident and fuse both Q/K rotations with the
            // partial-dimension copy instead of casting/catting per layer.
            let (cos, sin) = self.rope.cos_sin_tensors();
            let (q, k) = candle_nn::fused_qkv::partial_rope_f16(&q, &k, cos, sin, pos, HD, ROT)?;
            (
                q.reshape((1, l, QH, HD))?.transpose(1, 2)?.contiguous()?,
                k.reshape((1, l, KVH, HD))?.transpose(1, 2)?.contiguous()?,
                v.reshape((1, l, KVH, HD))?.transpose(1, 2)?.contiguous()?,
            )
        } else {
            let q = self
                .qn
                .forward(&self.q.forward(x)?.contiguous()?)?
                .reshape((1, l, QH, HD))?
                .transpose(1, 2)?
                .contiguous()?
                .to_dtype(DType::F16)?;
            let k = self
                .kn
                .forward(&self.k.forward(x)?.contiguous()?)?
                .reshape((1, l, KVH, HD))?
                .transpose(1, 2)?
                .contiguous()?
                .to_dtype(DType::F16)?;
            let v = self
                .v
                .forward(x)?
                .reshape((1, l, KVH, HD))?
                .transpose(1, 2)?
                .contiguous()?
                .to_dtype(DType::F16)?;
            let (qr, kr) = self
                .rope
                .apply(&q.narrow(3, 0, ROT)?, &k.narrow(3, 0, ROT)?, pos)?;
            (
                Tensor::cat(&[qr, q.narrow(3, ROT, HD - ROT)?], 3)?.contiguous()?,
                Tensor::cat(&[kr, k.narrow(3, ROT, HD - ROT)?], 3)?.contiguous()?,
                v,
            )
        };
        let (k, v) = self.cache.append(&k, &v)?;

        // Group Q heads by their shared KV head. Flattening repetition and
        // sequence into GEMM's M dimension performs true GQA without copying K
        // and V six times; it is also much faster for long-context decode.
        let kv_len = k.dim(2)?;
        let q = q.reshape((1, KVH, (QH / KVH) * l, HD))?;
        let mut scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)?
            * (1.0 / (HD as f64).sqrt()))?
        .reshape((1, KVH, QH / KVH, l, kv_len))?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(&m.to_dtype(scores.dtype())?)?;
        }
        let y = candle_nn::ops::softmax_last_dim(&scores)?
            .reshape((1, KVH, (QH / KVH) * l, kv_len))?
            .matmul(&v)?
            .reshape((1, QH, l, HD))?;
        self.o
            .forward(&y.transpose(1, 2)?.contiguous()?.reshape((1, l, QH * HD))?)
    }
}
struct Layer {
    attn: Attention,
    an: RmsNorm,
    fnorm: RmsNorm,
    moe: FusedMoeGGUF,
    device: Device,
}
impl Layer {
    fn new(
        v: &VarBuilder,
        device: &Device,
        rope: Arc<RotaryEmbedding>,
    ) -> candle_core::Result<Self> {
        let cfg = MoeCfg {
            moe_intermediate_size: 1536,
            num_experts: EXPERTS,
            norm_topk_prob: true,
            num_experts_per_tok: TOPK,
            hidden_size: H,
            act: candle_nn::Activation::Silu,
            decoder_sparse_step: None,
        };
        Ok(Self {
            attn: Attention::new(v, rope)?,
            an: RmsNorm::new(H, 1e-6, v.pp("attn_norm"))?,
            fnorm: RmsNorm::new(H, 1e-6, v.pp("ffn_norm"))?,
            moe: FusedMoeGGUF::new(&cfg, v.clone(), DType::F16)?,
            device: device.clone(),
        })
    }
    fn reset(&mut self) {
        self.attn.reset();
    }
    fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        pos: usize,
        prefill: bool,
    ) -> anyhow::Result<Tensor> {
        let x = x.to_device(&self.device)?.to_dtype(DType::F32)?;
        let r = x.clone();
        let n = self
            .an
            .forward(&x)
            .map_err(|e| anyhow::anyhow!("attn norm: {e}"))?;
        let a = self
            .attn
            .forward(&n, mask, pos)
            .map_err(|e| anyhow::anyhow!("attention: {e}"))?
            .to_dtype(DType::F32)?;
        let x = (a + r).map_err(|e| anyhow::anyhow!("attn residual: {e}"))?;
        let r = x.clone();
        let n = self
            .fnorm
            .forward(&x)
            .map_err(|e| anyhow::anyhow!("ffn norm: {e}"))?;
        let y = self
            .moe
            .forward(&n, prefill)
            .map_err(|e| anyhow::anyhow!("moe: {e}"))?
            .to_dtype(DType::F32)?;
        (y + r).map_err(|e| anyhow::anyhow!("ffn residual: {e}"))
    }
}
pub struct Model {
    embed: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    out: QMatMul,
    devices: Vec<Device>,
}
impl Model {
    pub fn load(shards: &[std::path::PathBuf], devices: &[Device]) -> Result<Self> {
        if shards.len() != 4 || devices.len() != 2 {
            anyhow::bail!("MiniMax requires four GGUF shards and two CUDA devices")
        }
        let global_prefixes = vec![
            "token_embd.".to_owned(),
            "output_norm.".to_owned(),
            "output.".to_owned(),
        ];
        let global = VarBuilder::from_gguf_selected(shards, &global_prefixes, &devices[0])?;
        let embed = Embedding::new(
            global
                .pp("token_embd")
                .get((VOCAB, H), "weight")?
                .dequantize(&devices[0])?,
            H,
        );
        // Build positions/frequencies in F32. An F16 position table loses
        // integer precision after 2048 and overflows before this model's 196K
        // context limit; apply() casts the final sin/cos slice to Q/K dtype.
        let rope0 = Arc::new(RotaryEmbedding::new(
            DType::F32,
            ROT,
            196608,
            5_000_000.,
            &devices[0],
        )?);
        let rope1 = Arc::new(RotaryEmbedding::new(
            DType::F32,
            ROT,
            196608,
            5_000_000.,
            &devices[1],
        )?);
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let (d, r) = if i < 31 {
                (&devices[0], rope0.clone())
            } else {
                (&devices[1], rope1.clone())
            };
            let prefixes = vec![format!("blk.{i}.")];
            let v = VarBuilder::from_gguf_selected(shards, &prefixes, d)?;
            layers.push(Layer::new(&v.pp(format!("blk.{i}")), d, r)?);
        }
        let norm = RmsNorm::new(H, 1e-6, global.pp("output_norm"))?;
        let out = QMatMul::new(H, VOCAB, global.pp("output"))?;
        Ok(Self {
            embed,
            layers,
            norm,
            out,
            devices: devices.to_vec(),
        })
    }
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }
    pub fn forward(&mut self, ids: &[u32], pos: usize) -> Result<Tensor> {
        if let Some((index, id)) = ids.iter().enumerate().find(|(_, id)| **id >= VOCAB as u32) {
            anyhow::bail!(
                "input token {id} at position {index} is outside vocabulary size {VOCAB}"
            );
        }
        let input = Tensor::from_slice(ids, (1, ids.len()), &self.devices[0])?;
        let mut x = self.embed.forward(&input)?;
        let l = ids.len();
        // Materialize the small causal mask directly on each device. Copying it
        // peer-to-peer for every layer creates unnecessary cross-device stream
        // dependencies and exposed a CUDA race at the layer-30/31 boundary.
        let masks = if l > 1 {
            let values: Vec<f32> = (0..l)
                .flat_map(|i| {
                    (0..(l + pos)).map(move |j| if j <= i + pos { 0.0 } else { f32::NEG_INFINITY })
                })
                .collect();
            Some(
                self.devices
                    .iter()
                    .map(|device| {
                        Tensor::from_vec(values.clone(), (1, 1, l, l + pos), device)?
                            .to_dtype(DType::F16)
                    })
                    .collect::<candle_core::Result<Vec<_>>>()?,
            )
        } else {
            None
        };
        for (layer_index, layer) in self.layers.iter_mut().enumerate() {
            if !x.device().same_device(&layer.device) {
                // Candle/cudarc's peer-copy stream is not reliably ordered on
                // this two-Blackwell setup: the destination occasionally sees
                // a previously freed activation. Stage the one inter-GPU
                // activation through host memory. This is only
                // one small transfer at the 30/31 layer boundary per chunk.
                let host = x.contiguous()?.to_device(&Device::Cpu)?;
                x = host.to_device(&layer.device)?;
                layer.device.synchronize()?;
            }
            let device_index = usize::from(layer_index >= 31);
            let mask = masks.as_ref().map(|masks| &masks[device_index]);
            x = layer
                .forward(&x, mask, pos, l > 1)
                .map_err(|e| anyhow::anyhow!("layer {layer_index}: {e}"))?;
        }
        let x = x.narrow(1, l - 1, 1)?;
        let x = if x.device().same_device(&self.devices[0]) {
            x
        } else {
            let host = x.contiguous()?.to_device(&Device::Cpu)?;
            let x = host.to_device(&self.devices[0])?;
            self.devices[0].synchronize()?;
            x
        };
        let normalized = self.norm.forward(&x)?;
        let logits = self
            .out
            .forward(&normalized)?
            .to_dtype(DType::F32)?
            .squeeze(1)?
            .squeeze(0)?;
        Ok(logits)
    }
}
