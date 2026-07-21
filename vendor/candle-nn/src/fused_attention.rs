//! CUDA attention kernels used by quantized autoregressive models.

use candle::{DType, Result, Tensor};

const QUERY_HEADS: usize = 48;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const MAX_QUERY_LEN: usize = 512;
const MAX_SPLITS: usize = 128;
const PARTIAL_STRIDE: usize = HEAD_DIM + 2;
const PREFILL_KEYS_PER_SPLIT: usize = 16_384;
const TARGET_SPLIT_BLOCKS: usize = 256;

/// Fused single-token F16 GQA for MiniMax's 48 query heads, 8 KV heads, and
/// 128-element heads.
///
/// The split-K kernel performs online softmax and consumes the strided
/// preallocated KV cache directly. It does not transpose K or materialize the
/// `(query_heads, context)` score tensor.
pub fn gqa_decode_f16_128(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    // Keep enough independent blocks in flight to fill the 188-SM target GPU,
    // without making each split so short that launch/combine overhead wins.
    // These crossover points come from examples/gqa_bench.rs on sm_120.
    let seq_len = k.dim(2)?;
    let num_splits = if seq_len >= 8_192 {
        // The MMA kernel partitions 128-key tiles. Long-running blocks need
        // fewer splits than the scalar path. The 64-split middle band is the
        // favorable launch wave on the 188-SM target; 16 wins on either side.
        if (20_480..49_152).contains(&seq_len) {
            64
        } else {
            16
        }
    } else {
        match seq_len {
            0..2_048 => 32,
            _ => 64,
        }
    };
    gqa_decode_f16_128_with_splits(q, k, v, scale, num_splits)
}

/// Fused causal multi-token F16 GQA for MiniMax prefill.
///
/// Sequence splits are reduced with online softmax, so workspace is bounded by
/// `(query_len, num_splits, query_heads, head_dim)` rather than by the complete
/// `(query_len, kv_len)` attention-score matrix.
pub fn gqa_prefill_f16_128(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    past_len: usize,
    scale: f32,
) -> Result<Tensor> {
    let query_len = q.dim(2)?;
    let kv_len = k.dim(2)?;
    let num_splits = prefill_num_splits(query_len, kv_len)?;
    gqa_f16_128_with_splits(q, k, v, past_len, scale, num_splits)
}

/// Total temporary/output allocation made by [`gqa_prefill_f16_128`] for the
/// supplied dimensions.
#[doc(hidden)]
pub fn gqa_prefill_workspace_bytes(query_len: usize, kv_len: usize) -> Result<usize> {
    let num_splits = prefill_num_splits(query_len, kv_len)?;
    allocation_sizes(query_len, num_splits).map(|(_, _, bytes)| bytes)
}

fn prefill_num_splits(query_len: usize, kv_len: usize) -> Result<usize> {
    if query_len == 0 || query_len > MAX_QUERY_LEN {
        candle::bail!(
            "fused GQA prefill query length must be in 1..={MAX_QUERY_LEN}, got {query_len}"
        )
    }
    if kv_len < query_len || kv_len > i32::MAX as usize {
        candle::bail!(
            "fused GQA prefill KV length must be in {query_len}..={}, got {kv_len}",
            i32::MAX
        )
    }
    let query_blocks = query_len
        .checked_mul(KV_HEADS)
        .ok_or_else(|| candle::Error::Msg("fused GQA prefill dimensions overflow".into()))?;
    let occupancy_splits = TARGET_SPLIT_BLOCKS.div_ceil(query_blocks);
    let context_splits = kv_len.div_ceil(PREFILL_KEYS_PER_SPLIT);
    Ok(occupancy_splits.max(context_splits).clamp(1, MAX_SPLITS))
}

fn allocation_sizes(query_len: usize, num_splits: usize) -> Result<(usize, usize, usize)> {
    let partial_count = query_len
        .checked_mul(num_splits)
        .and_then(|n| n.checked_mul(QUERY_HEADS))
        .and_then(|n| n.checked_mul(PARTIAL_STRIDE))
        .ok_or_else(|| candle::Error::Msg("fused GQA partial workspace overflows usize".into()))?;
    let output_count = query_len
        .checked_mul(QUERY_HEADS)
        .and_then(|n| n.checked_mul(HEAD_DIM))
        .ok_or_else(|| candle::Error::Msg("fused GQA output size overflows usize".into()))?;
    let bytes = partial_count
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|n| {
            output_count
                .checked_mul(std::mem::size_of::<half::f16>())
                .and_then(|output_bytes| n.checked_add(output_bytes))
        })
        .ok_or_else(|| candle::Error::Msg("fused GQA allocation size overflows usize".into()))?;
    Ok((partial_count, output_count, bytes))
}

/// Variant exposed for kernel tuning and benchmarks.
#[doc(hidden)]
pub fn gqa_decode_f16_128_with_splits(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    num_splits: usize,
) -> Result<Tensor> {
    let seq_len = k.dim(2)?;
    gqa_f16_128_with_splits(q, k, v, seq_len.saturating_sub(1), scale, num_splits)
}

/// Variant exposed for kernel validation and tuning.
#[doc(hidden)]
pub fn gqa_prefill_f16_128_with_splits(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    past_len: usize,
    scale: f32,
    num_splits: usize,
) -> Result<Tensor> {
    gqa_f16_128_with_splits(q, k, v, past_len, scale, num_splits)
}

#[cfg(feature = "cuda")]
fn gqa_f16_128_with_splits(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    past_len: usize,
    scale: f32,
    num_splits: usize,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::cuda_backend::kernels::ffi;
    use candle::op::BackpropOp;

    if q.dtype() != DType::F16 || k.dtype() != DType::F16 || v.dtype() != DType::F16 {
        candle::bail!("fused GQA attention requires F16 Q/K/V")
    }
    let (q_batch, query_heads, query_len, q_head_dim) = q.dims4()?;
    if (q_batch, query_heads, q_head_dim) != (1, QUERY_HEADS, HEAD_DIM) {
        candle::bail!(
            "fused GQA attention expected Q shape (1, {QUERY_HEADS}, query_len, {HEAD_DIM}), got {:?}",
            q.shape()
        )
    }
    if query_len == 0 || query_len > MAX_QUERY_LEN {
        candle::bail!(
            "fused GQA attention query length must be in 1..={MAX_QUERY_LEN}, got {query_len}"
        )
    }
    let (batch, kv_heads, seq_len, head_dim) = k.dims4()?;
    if (batch, kv_heads, head_dim) != (1, KV_HEADS, HEAD_DIM) || v.dims4()? != k.dims4()? {
        candle::bail!(
            "fused GQA attention expected matching K/V shape (1, {KV_HEADS}, seq, {HEAD_DIM}), got {:?} and {:?}",
            k.shape(),
            v.shape()
        )
    }
    let expected_seq_len = past_len
        .checked_add(query_len)
        .ok_or_else(|| candle::Error::Msg("fused GQA context length overflows usize".into()))?;
    if seq_len != expected_seq_len || seq_len > i32::MAX as usize {
        candle::bail!(
            "fused GQA attention expected KV length past_len + query_len = {expected_seq_len}, got {seq_len}"
        )
    }
    if !(1..=MAX_SPLITS).contains(&num_splits) {
        candle::bail!("fused GQA split count must be in 1..={MAX_SPLITS}, got {num_splits}")
    }
    for tensor in [k, v] {
        if !tensor.device().same_device(q.device()) {
            candle::bail!("fused GQA Q/K/V must share a CUDA device")
        }
    }

    let (q_storage, q_layout) = q.storage_and_layout();
    let q_slice = match &*q_storage {
        candle::Storage::Cuda(storage) => storage.as_cuda_slice::<half::f16>()?,
        _ => candle::bail!("fused GQA attention requires CUDA tensors"),
    };
    let q_stride = q_layout.stride();
    if q_stride[3] != 1 {
        candle::bail!("unsupported fused GQA Q layout {q_layout:?}")
    }
    let q_slice = q_slice.slice(q_layout.start_offset()..);

    let (k_storage, k_layout) = k.storage_and_layout();
    let k_slice = match &*k_storage {
        candle::Storage::Cuda(storage) => storage.as_cuda_slice::<half::f16>()?,
        _ => candle::bail!("fused GQA attention requires CUDA tensors"),
    };
    let k_stride = k_layout.stride();
    if k_stride[3] != 1 || k_stride[2] != HEAD_DIM {
        candle::bail!("unsupported fused GQA K layout {k_layout:?}")
    }
    let k_slice = k_slice.slice(k_layout.start_offset()..);

    let (v_storage, v_layout) = v.storage_and_layout();
    let v_slice = match &*v_storage {
        candle::Storage::Cuda(storage) => storage.as_cuda_slice::<half::f16>()?,
        _ => candle::bail!("fused GQA attention requires CUDA tensors"),
    };
    let v_stride = v_layout.stride();
    if v_stride[3] != 1 || v_stride[2] != HEAD_DIM {
        candle::bail!("unsupported fused GQA V layout {v_layout:?}")
    }
    let v_slice = v_slice.slice(v_layout.start_offset()..);

    let to_i32 = |name: &str, value: usize| -> Result<i32> {
        i32::try_from(value)
            .map_err(|_| candle::Error::Msg(format!("fused GQA {name} stride is too large")))
    };
    let q_head_stride = to_i32("Q head", q_stride[1])?;
    let q_seq_stride = to_i32("Q sequence", q_stride[2])?;
    let k_head_stride = to_i32("K head", k_stride[1])?;
    let v_head_stride = to_i32("V head", v_stride[1])?;

    let dev = q.device().as_cuda_device()?;
    let (partial_count, output_count, _) = allocation_sizes(query_len, num_splits)?;
    let partials = unsafe { dev.alloc::<f32>(partial_count) }?;
    let output = unsafe { dev.alloc::<half::f16>(output_count) }?;
    let stream = dev.cuda_stream();
    unsafe {
        let q_ptr = q_slice.device_ptr(&stream).0 as *const std::ffi::c_void;
        let k_ptr = k_slice.device_ptr(&stream).0 as *const std::ffi::c_void;
        let v_ptr = v_slice.device_ptr(&stream).0 as *const std::ffi::c_void;
        let partials_ptr =
            partials.device_ptr(partials.stream()).0 as *mut std::ffi::c_void;
        let output_ptr = output.device_ptr(output.stream()).0 as *mut std::ffi::c_void;
        let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
        if query_len == 1 && seq_len >= 8_192 {
            ffi::launch_gqa_decode_mma_f16_128(
                q_ptr,
                k_ptr,
                v_ptr,
                partials_ptr,
                output_ptr,
                seq_len as i32,
                q_head_stride,
                k_head_stride,
                v_head_stride,
                num_splits as i32,
                scale,
                stream.context().ordinal() as i32,
                stream_ptr,
            );
        } else {
            ffi::launch_gqa_f16_128(
                q_ptr,
                k_ptr,
                v_ptr,
                partials_ptr,
                output_ptr,
                query_len as i32,
                past_len as i32,
                q_head_stride,
                q_seq_stride,
                k_head_stride,
                v_head_stride,
                num_splits as i32,
                scale,
                stream_ptr,
            );
        }
    }

    let output = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
    Ok(Tensor::from_storage(
        candle::Storage::Cuda(output),
        (1, QUERY_HEADS, query_len, HEAD_DIM),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(not(feature = "cuda"))]
fn gqa_f16_128_with_splits(
    _: &Tensor,
    _: &Tensor,
    _: &Tensor,
    _: usize,
    _: f32,
    _: usize,
) -> Result<Tensor> {
    candle::bail!("fused GQA attention is only implemented for CUDA")
}
