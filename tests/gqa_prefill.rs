use anyhow::Result;
use candle_core::{DType, Device, Tensor};

const QUERY_HEADS: usize = 48;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const MAX_CONTEXT: usize = 196_608;

fn grouped_causal_reference(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    past_len: usize,
) -> candle_core::Result<Tensor> {
    let (_, _, query_len, _) = q.dims4()?;
    let kv_len = k.dim(2)?;
    let grouped_q = q.reshape((1, KV_HEADS, (QUERY_HEADS / KV_HEADS) * query_len, HEAD_DIM))?;
    let scores = (grouped_q.matmul(&k.transpose(2, 3)?.contiguous()?)?
        * (1.0 / (HEAD_DIM as f64).sqrt()))?
    .reshape((1, KV_HEADS, QUERY_HEADS / KV_HEADS, query_len, kv_len))?;
    let mask = (0..query_len)
        .flat_map(|query| {
            (0..kv_len).map(move |key| {
                if key <= past_len + query {
                    0.0f32
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect::<Vec<_>>();
    let mask = Tensor::from_vec(mask, (1, 1, 1, query_len, kv_len), q.device())?
        .to_dtype(scores.dtype())?;
    candle_nn::ops::softmax_last_dim(&scores.broadcast_add(&mask)?)?
        .reshape((1, KV_HEADS, (QUERY_HEADS / KV_HEADS) * query_len, kv_len))?
        .matmul(v)?
        .reshape((1, QUERY_HEADS, query_len, HEAD_DIM))
}

fn random_cache(
    device: &Device,
    query_len: usize,
    past_len: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let kv_len = past_len + query_len;
    let capacity = kv_len.div_ceil(4096) * 4096;
    let q = Tensor::randn(0f32, 1f32, (1, QUERY_HEADS, query_len, HEAD_DIM), device)?
        .to_dtype(DType::F16)?;
    let k_backing = Tensor::randn(0f32, 1f32, (1, KV_HEADS, capacity, HEAD_DIM), device)?
        .to_dtype(DType::F16)?;
    let v_backing = Tensor::randn(0f32, 1f32, (1, KV_HEADS, capacity, HEAD_DIM), device)?
        .to_dtype(DType::F16)?;
    Ok((
        q,
        k_backing.narrow(2, 0, kv_len)?,
        v_backing.narrow(2, 0, kv_len)?,
    ))
}

#[test]
#[ignore = "requires an sm_120 CUDA device"]
fn fused_prefill_matches_grouped_causal_attention() -> Result<()> {
    let device = Device::new_cuda(0)?;
    unsafe { device.as_cuda_device()?.disable_event_tracking() };

    for (query_len, past_len) in [(2, 0), (5, 7), (39, 0), (512, 0), (17, 512)] {
        let (q, k, v) = random_cache(&device, query_len, past_len)?;
        let reference = grouped_causal_reference(&q, &k, &v, past_len)?;
        let candidate = candle_nn::fused_attention::gqa_prefill_f16_128(
            &q,
            &k,
            &v,
            past_len,
            1.0 / (HEAD_DIM as f32).sqrt(),
        )?;
        let max_abs = (&reference - &candidate)?
            .abs()?
            .to_dtype(DType::F32)?
            .max_all()?
            .to_scalar::<f32>()?;
        eprintln!("prefill query_len={query_len} past_len={past_len} max_abs={max_abs}");
        assert!(
            max_abs <= 0.003_906_25,
            "query_len={query_len} past_len={past_len} max_abs={max_abs}"
        );
    }

    for context in [1, 8, 513] {
        let (q, k, v) = random_cache(&device, 1, context - 1)?;
        let reference = grouped_causal_reference(&q, &k, &v, context - 1)?;
        let candidate = candle_nn::fused_attention::gqa_decode_f16_128(
            &q,
            &k,
            &v,
            1.0 / (HEAD_DIM as f32).sqrt(),
        )?;
        let max_abs = (&reference - &candidate)?
            .abs()?
            .to_dtype(DType::F32)?
            .max_all()?
            .to_scalar::<f32>()?;
        eprintln!("decode context={context} max_abs={max_abs}");
        assert!(
            max_abs <= 0.003_906_25,
            "decode context={context} max_abs={max_abs}"
        );
    }
    Ok(())
}

#[test]
fn near_limit_prefill_workspace_is_bounded() -> Result<()> {
    let workspace = candle_nn::fused_attention::gqa_prefill_workspace_bytes(512, MAX_CONTEXT)?;
    assert_eq!(workspace, 159_645_696);
    assert!(workspace <= 192 * 1024 * 1024);
    Ok(())
}

#[test]
#[ignore = "allocates a near-limit synthetic KV cache on CUDA"]
fn fused_prefill_reaches_the_context_limit() -> Result<()> {
    let device = Device::new_cuda(0)?;
    unsafe { device.as_cuda_device()?.disable_event_tracking() };
    let q = Tensor::zeros((1, QUERY_HEADS, 512, HEAD_DIM), DType::F16, &device)?;
    let k = Tensor::zeros((1, KV_HEADS, MAX_CONTEXT, HEAD_DIM), DType::F16, &device)?;
    let v = Tensor::zeros((1, KV_HEADS, MAX_CONTEXT, HEAD_DIM), DType::F16, &device)?;
    let output = candle_nn::fused_attention::gqa_prefill_f16_128(
        &q,
        &k,
        &v,
        MAX_CONTEXT - 512,
        1.0 / (HEAD_DIM as f32).sqrt(),
    )?;
    assert_eq!(output.dims4()?, (1, QUERY_HEADS, 512, HEAD_DIM));
    assert_eq!(
        output
            .abs()?
            .to_dtype(DType::F32)?
            .max_all()?
            .to_scalar::<f32>()?,
        0.0
    );
    Ok(())
}
