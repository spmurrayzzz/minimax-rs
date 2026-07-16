use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor, quantized::QTensor};
use candle_nn::kv_cache::KvCache;
use candle_transformers::{
    fused_moe::{FusedMoeGGUF, MoeCfg},
    models::{quantized_qwen3::RotaryEmbedding, with_tracing::QMatMul},
    quantized_nn::RmsNorm,
    quantized_var_builder::VarBuilder,
    utils::repeat_kv,
};
use std::{path::PathBuf, sync::Arc, time::Instant};

const H: usize = 3072;
const QH: usize = 48;
const KVH: usize = 8;
const HD: usize = 128;
const ROT: usize = 64;

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
    fused_post: bool,
    fused_rope: bool,
    grouped_gqa: bool,
}

impl Attention {
    fn new(v: &VarBuilder, rope: Arc<RotaryEmbedding>) -> candle_core::Result<Self> {
        let qw = v.pp("attn_q").get((QH * HD, H), "weight")?;
        let kw = v.pp("attn_k").get((KVH * HD, H), "weight")?;
        let vw = v.pp("attn_v").get((KVH * HD, H), "weight")?;
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
            cache: KvCache::new(2, 512),
            fused_post: std::env::var_os("PLAIN_QKV").is_none(),
            fused_rope: std::env::var_os("PLAIN_ROPE").is_none(),
            grouped_gqa: std::env::var_os("REPEAT_KV").is_none(),
        })
    }

    fn forward(&mut self, x: &Tensor, pos: usize) -> candle_core::Result<Tensor> {
        let (_, l, _) = x.dims3()?;
        let (q, k, v) = if l == 1 && self.fused_post {
            candle_nn::fused_qkv::q8_0_decode_rmsnorm_f16(
                x,
                &self.qw,
                &self.kw,
                &self.vw,
                self.qn.weight(),
                self.kn.weight(),
                1e-6,
            )?
        } else if l == 1 {
            let (q, k, v) = candle_nn::fused_qkv::q8_0_decode(x, &self.qw, &self.kw, &self.vw)?;
            (
                self.qn.forward(&q)?.to_dtype(DType::F16)?,
                self.kn.forward(&k)?.to_dtype(DType::F16)?,
                v.to_dtype(DType::F16)?,
            )
        } else {
            (
                self.qn
                    .forward(&self.q.forward(x)?.contiguous()?)?
                    .to_dtype(DType::F16)?,
                self.kn
                    .forward(&self.k.forward(x)?.contiguous()?)?
                    .to_dtype(DType::F16)?,
                self.v.forward(x)?.to_dtype(DType::F16)?,
            )
        };
        let (q, k) = if l == 1 && self.fused_rope {
            let (cos, sin) = self.rope.cos_sin_tensors();
            let (q, k) = candle_nn::fused_qkv::partial_rope_f16(&q, &k, cos, sin, pos, HD, ROT)?;
            (
                q.reshape((1, l, QH, HD))?.transpose(1, 2)?.contiguous()?,
                k.reshape((1, l, KVH, HD))?.transpose(1, 2)?.contiguous()?,
            )
        } else {
            let q = q.reshape((1, l, QH, HD))?.transpose(1, 2)?.contiguous()?;
            let k = k.reshape((1, l, KVH, HD))?.transpose(1, 2)?.contiguous()?;
            let (qr, kr) = self
                .rope
                .apply(&q.narrow(3, 0, ROT)?, &k.narrow(3, 0, ROT)?, pos)?;
            (
                Tensor::cat(&[qr, q.narrow(3, ROT, HD - ROT)?], 3)?.contiguous()?,
                Tensor::cat(&[kr, k.narrow(3, ROT, HD - ROT)?], 3)?.contiguous()?,
            )
        };
        let v = v.reshape((1, l, KVH, HD))?.transpose(1, 2)?.contiguous()?;
        let (k, v) = self.cache.append(&k, &v)?;
        let y = if self.grouped_gqa {
            let kv_len = k.dim(2)?;
            let q = q.reshape((1, KVH, (QH / KVH) * l, HD))?;
            let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)?
                * (1.0 / (HD as f64).sqrt()))?
            .reshape((1, KVH, QH / KVH, l, kv_len))?;
            candle_nn::ops::softmax_last_dim(&scores)?
                .reshape((1, KVH, (QH / KVH) * l, kv_len))?
                .matmul(&v)?
                .reshape((1, QH, l, HD))?
        } else {
            let k = repeat_kv(k, QH / KVH)?.contiguous()?;
            let v = repeat_kv(v, QH / KVH)?.contiguous()?;
            let scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)?
                * (1.0 / (HD as f64).sqrt()))?;
            candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?
        };
        self.o
            .forward(&y.transpose(1, 2)?.contiguous()?.reshape((1, l, QH * HD))?)
    }
}

struct Layer {
    attn: Attention,
    an: RmsNorm,
    fnorm: RmsNorm,
    moe: FusedMoeGGUF,
}

impl Layer {
    fn new(v: &VarBuilder, _device: &Device, rope: Arc<RotaryEmbedding>) -> Result<Self> {
        let cfg = MoeCfg {
            moe_intermediate_size: 1536,
            num_experts: 256,
            norm_topk_prob: true,
            num_experts_per_tok: 8,
            hidden_size: H,
            act: candle_nn::Activation::Silu,
            decoder_sparse_step: None,
        };
        Ok(Self {
            attn: Attention::new(v, rope)?,
            an: RmsNorm::new(H, 1e-6, v.pp("attn_norm"))?,
            fnorm: RmsNorm::new(H, 1e-6, v.pp("ffn_norm"))?,
            moe: FusedMoeGGUF::new(&cfg, v.clone(), DType::F16)?,
        })
    }

    fn forward(&mut self, x: &Tensor, pos: usize) -> Result<Tensor> {
        let n = self.an.forward(x)?;
        let x = (self.attn.forward(&n, pos)?.to_dtype(DType::F32)? + x)?;
        let n = self.fnorm.forward(&x)?;
        Ok((self.moe.forward(&n, false)? + x)?)
    }

    fn profile(&mut self, x: &Tensor, pos: usize, device: &Device) -> Result<()> {
        let start = Instant::now();
        let n = self.an.forward(x)?;
        device.synchronize()?;
        let norm1 = start.elapsed();
        let a = self.attn.forward(&n, pos)?.to_dtype(DType::F32)?;
        device.synchronize()?;
        let attention = start.elapsed() - norm1;
        let x = (a + x)?;
        let n = self.fnorm.forward(&x)?;
        device.synchronize()?;
        let norm2_residual = start.elapsed() - norm1 - attention;
        let y = self.moe.forward(&n, false)?;
        device.synchronize()?;
        let moe = start.elapsed() - norm1 - attention - norm2_residual;
        drop((y + x)?);
        device.synchronize()?;
        let residual = start.elapsed() - norm1 - attention - norm2_residual - moe;
        println!(
            "stages norm1={:.3}ms attention={:.3}ms norm2+res={:.3}ms moe={:.3}ms residual={:.3}ms total={:.3}ms",
            norm1.as_secs_f64() * 1e3,
            attention.as_secs_f64() * 1e3,
            norm2_residual.as_secs_f64() * 1e3,
            moe.as_secs_f64() * 1e3,
            residual.as_secs_f64() * 1e3,
            start.elapsed().as_secs_f64() * 1e3,
        );
        Ok(())
    }
}

fn main() -> Result<()> {
    let context = std::env::args()
        .nth(1)
        .as_deref()
        .unwrap_or("512")
        .parse::<usize>()?;
    let iterations = std::env::args()
        .nth(2)
        .as_deref()
        .unwrap_or("100")
        .parse::<usize>()?;
    let model_dir = PathBuf::from("/storage/models/minimax-m2.7-gguf/UD-Q4_K_XL");
    let mut shards = std::fs::read_dir(model_dir)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "gguf"))
        .collect::<Vec<_>>();
    shards.sort();
    let device = Device::new_cuda(0)?;
    // This benchmark intentionally uses one CUDA stream.
    unsafe { device.as_cuda_device()?.disable_event_tracking() };
    let prefix = vec!["blk.0.".to_owned()];
    let vb = VarBuilder::from_gguf_selected(&shards, &prefix, &device)?;
    let rope = Arc::new(RotaryEmbedding::new(
        DType::F32,
        ROT,
        196_608,
        5_000_000.,
        &device,
    )?);
    let mut layer = Layer::new(&vb.pp("blk.0"), &device, rope)?;

    let prefill = Tensor::randn(0f32, 1f32, (1, context, H), &device)?;
    drop(layer.attn.forward(&prefill, 0)?);
    let token = Tensor::randn(0f32, 1f32, (1, 1, H), &device)?;
    drop(layer.forward(&token, context)?);
    device.synchronize()?;

    layer.profile(&token, context + 1, &device)?;
    let start = Instant::now();
    for i in 0..iterations {
        drop(layer.forward(&token, context + 2 + i)?);
    }
    device.synchronize()?;
    let ms = start.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    println!(
        "decode context={context} average={ms:.3}ms/layer projected={:.2} tok/s",
        1e3 / (ms * 62.)
    );
    Ok(())
}
