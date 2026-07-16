//! CUDA attention kernels used by quantized autoregressive models.

use candle::{DType, Result, Tensor};

const QUERY_HEADS: usize = 48;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const MAX_SPLITS: usize = 128;
const PARTIAL_STRIDE: usize = HEAD_DIM + 2;

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
    let num_splits = match k.dim(2)? {
        0..2_048 => 32,
        2_048..24_576 => 64,
        24_576..49_152 => 80,
        _ => 96,
    };
    gqa_decode_f16_128_with_splits(q, k, v, scale, num_splits)
}

/// Variant exposed for kernel tuning and benchmarks.
#[doc(hidden)]
#[cfg(feature = "cuda")]
pub fn gqa_decode_f16_128_with_splits(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    num_splits: usize,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::cuda_backend::kernels::ffi;
    use candle::op::BackpropOp;

    if q.dtype() != DType::F16 || k.dtype() != DType::F16 || v.dtype() != DType::F16 {
        candle::bail!("fused GQA decode requires F16 Q/K/V")
    }
    if q.dims4()? != (1, QUERY_HEADS, 1, HEAD_DIM) {
        candle::bail!(
            "fused GQA decode expected Q shape (1, {QUERY_HEADS}, 1, {HEAD_DIM}), got {:?}",
            q.shape()
        )
    }
    let (batch, kv_heads, seq_len, head_dim) = k.dims4()?;
    if (batch, kv_heads, head_dim) != (1, KV_HEADS, HEAD_DIM) || v.dims4()? != k.dims4()? {
        candle::bail!(
            "fused GQA decode expected matching K/V shape (1, {KV_HEADS}, seq, {HEAD_DIM}), got {:?} and {:?}",
            k.shape(),
            v.shape()
        )
    }
    if seq_len == 0 || seq_len > i32::MAX as usize {
        candle::bail!("invalid fused GQA context length {seq_len}")
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
        _ => candle::bail!("fused GQA requires CUDA tensors"),
    };
    let q_stride = q_layout.stride();
    // The sequence dimension is a singleton and may retain an arbitrary stride
    // after transpose; the kernel only addresses Q by head and feature.
    if q_stride[3] != 1 || q_stride[1] != HEAD_DIM {
        candle::bail!("unsupported fused GQA Q layout {q_layout:?}")
    }
    let q_slice = q_slice.slice(q_layout.start_offset()..);

    let (k_storage, k_layout) = k.storage_and_layout();
    let k_slice = match &*k_storage {
        candle::Storage::Cuda(storage) => storage.as_cuda_slice::<half::f16>()?,
        _ => candle::bail!("fused GQA requires CUDA tensors"),
    };
    let k_stride = k_layout.stride();
    if k_stride[3] != 1 || k_stride[2] != HEAD_DIM {
        candle::bail!("unsupported fused GQA K layout {k_layout:?}")
    }
    let k_slice = k_slice.slice(k_layout.start_offset()..);

    let (v_storage, v_layout) = v.storage_and_layout();
    let v_slice = match &*v_storage {
        candle::Storage::Cuda(storage) => storage.as_cuda_slice::<half::f16>()?,
        _ => candle::bail!("fused GQA requires CUDA tensors"),
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
    let k_head_stride = to_i32("K head", k_stride[1])?;
    let v_head_stride = to_i32("V head", v_stride[1])?;

    let dev = q.device().as_cuda_device()?;
    let partial_count = num_splits * QUERY_HEADS * PARTIAL_STRIDE;
    let partials = unsafe { dev.alloc::<f32>(partial_count) }?;
    let output = unsafe { dev.alloc::<half::f16>(QUERY_HEADS * HEAD_DIM) }?;
    let stream = dev.cuda_stream();
    unsafe {
        ffi::launch_gqa_decode_f16_128(
            q_slice.device_ptr(&stream).0 as *const std::ffi::c_void,
            k_slice.device_ptr(&stream).0 as *const std::ffi::c_void,
            v_slice.device_ptr(&stream).0 as *const std::ffi::c_void,
            partials.device_ptr(partials.stream()).0 as *mut std::ffi::c_void,
            output.device_ptr(output.stream()).0 as *mut std::ffi::c_void,
            seq_len as i32,
            q_head_stride,
            k_head_stride,
            v_head_stride,
            num_splits as i32,
            scale,
            stream.cu_stream() as *mut std::ffi::c_void,
        );
    }

    let output = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
    Ok(Tensor::from_storage(
        candle::Storage::Cuda(output),
        (1, QUERY_HEADS, 1, HEAD_DIM),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(not(feature = "cuda"))]
#[doc(hidden)]
pub fn gqa_decode_f16_128_with_splits(
    _: &Tensor,
    _: &Tensor,
    _: &Tensor,
    _: f32,
    _: usize,
) -> Result<Tensor> {
    candle::bail!("fused GQA decode is only implemented for CUDA")
}
