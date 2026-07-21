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

fn flash_prefill(q: &Tensor, k: &Tensor, v: &Tensor) -> candle_core::Result<Tensor> {
    candle_flash_attn::flash_attn(
        &q.transpose(1, 2)?,
        &k.transpose(1, 2)?,
        &v.transpose(1, 2)?,
        1.0 / (HEAD_DIM as f32).sqrt(),
        true,
    )?
    .transpose(1, 2)
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
fn flash_prefill_matches_grouped_causal_attention() -> Result<()> {
    let device = Device::new_cuda(0)?;
    unsafe { device.as_cuda_device()?.disable_event_tracking() };

    for (query_len, past_len) in [(2, 0), (5, 7), (39, 0), (512, 0), (17, 512)] {
        let (q, k, v) = random_cache(&device, query_len, past_len)?;
        let reference = grouped_causal_reference(&q, &k, &v, past_len)?;
        let candidate = flash_prefill(&q, &k, &v)?;
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

    for context in [1, 8, 513, 8_192, 8_193, 12_288] {
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

    // The production cache has padded head strides, but the public kernel must
    // also zero-stage its final MMA tail correctly for exact allocations.
    let context = 8_193;
    let q =
        Tensor::randn(0f32, 1f32, (1, QUERY_HEADS, 1, HEAD_DIM), &device)?.to_dtype(DType::F16)?;
    let k = Tensor::randn(0f32, 1f32, (1, KV_HEADS, context, HEAD_DIM), &device)?
        .to_dtype(DType::F16)?;
    let v = Tensor::randn(0f32, 1f32, (1, KV_HEADS, context, HEAD_DIM), &device)?
        .to_dtype(DType::F16)?;
    let reference = grouped_causal_reference(&q, &k, &v, context - 1)?;
    let candidate =
        candle_nn::fused_attention::gqa_decode_f16_128(&q, &k, &v, 1.0 / (HEAD_DIM as f32).sqrt())?;
    let max_abs = (&reference - &candidate)?
        .abs()?
        .to_dtype(DType::F32)?
        .max_all()?
        .to_scalar::<f32>()?;
    eprintln!("exact decode context={context} max_abs={max_abs}");
    assert!(
        max_abs <= 0.003_906_25,
        "context={context} max_abs={max_abs}"
    );

    Ok(())
}

#[test]
fn legacy_split_k_near_limit_workspace_is_bounded() -> Result<()> {
    let workspace = candle_nn::fused_attention::gqa_prefill_workspace_bytes(512, MAX_CONTEXT)?;
    assert_eq!(workspace, 159_645_696);
    assert!(workspace <= 192 * 1024 * 1024);
    Ok(())
}

#[test]
#[ignore = "allocates a near-limit synthetic KV cache on CUDA"]
fn flash_prefill_reaches_the_context_limit() -> Result<()> {
    let device = Device::new_cuda(0)?;
    unsafe { device.as_cuda_device()?.disable_event_tracking() };
    let q = Tensor::zeros((1, QUERY_HEADS, 512, HEAD_DIM), DType::F16, &device)?;
    let k = Tensor::zeros((1, KV_HEADS, MAX_CONTEXT, HEAD_DIM), DType::F16, &device)?;
    let v = Tensor::zeros((1, KV_HEADS, MAX_CONTEXT, HEAD_DIM), DType::F16, &device)?;
    let output = flash_prefill(&q, &k, &v)?;
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
