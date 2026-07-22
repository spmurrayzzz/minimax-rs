//! Fused single-token Q/K/V projection for CUDA GGUF models.

use candle::quantized::QTensor;
use candle::{DType, Result, Tensor};

/// Projects one F32 activation through three Q8_0 matrices while sharing the
/// activation quantization and CUDA launch. The returned tensors have shapes
/// `[1, 1, q_rows]`, `[1, 1, k_rows]`, and `[1, 1, v_rows]`.
#[cfg(feature = "cuda")]
pub fn q8_0_decode(
    input: &Tensor,
    q_weight: &QTensor,
    k_weight: &QTensor,
    v_weight: &QTensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::cuda_backend::kernels::ffi;
    use candle::op::BackpropOp;
    use candle::quantized::GgmlDType;

    if input.dtype() != DType::F32 || input.dims().len() != 3 || input.dims()[..2] != [1, 1] {
        candle::bail!(
            "fused qkv decode expects a contiguous f32 [1, 1, hidden] tensor, got {:?} {:?}",
            input.shape(),
            input.dtype()
        )
    }
    let hidden = input.dim(2)?;
    let (q_rows, q_hidden) = q_weight.shape().dims2()?;
    let (k_rows, k_hidden) = k_weight.shape().dims2()?;
    let (v_rows, v_hidden) = v_weight.shape().dims2()?;
    if q_hidden != hidden || k_hidden != hidden || v_hidden != hidden {
        candle::bail!(
            "fused qkv weight/input mismatch: input {hidden}, q {:?}, k {:?}, v {:?}",
            q_weight.shape(),
            k_weight.shape(),
            v_weight.shape()
        )
    }
    if q_weight.dtype() != GgmlDType::Q8_0
        || k_weight.dtype() != GgmlDType::Q8_0
        || v_weight.dtype() != GgmlDType::Q8_0
    {
        candle::bail!("fused qkv decode requires Q8_0 weights")
    }
    let dev = input.device().as_cuda_device()?;
    for weight in [q_weight, k_weight, v_weight] {
        if !weight.device().same_device(input.device()) {
            candle::bail!("fused qkv weights and input must be on the same CUDA device")
        }
    }

    let input = input.contiguous()?;
    let (input_storage, input_layout) = input.storage_and_layout();
    let input_slice = match &*input_storage {
        candle::Storage::Cuda(storage) => storage.as_cuda_slice::<f32>()?,
        _ => candle::bail!("fused qkv input must be a CUDA tensor"),
    };
    let (start, end) = input_layout
        .contiguous_offsets()
        .ok_or_else(|| candle::Error::Msg("fused qkv input must be contiguous".into()))?;
    let input_slice = input_slice.slice(start..end);

    // The dense MMVQ path pads activation rows to 512 and stores one 36-byte
    // Q8_1 block per 32 values.
    let hidden_padded = hidden.div_ceil(512) * 512;
    let scratch_bytes = hidden_padded / 32 * 36;
    let scratch = unsafe { dev.alloc::<u8>(scratch_bytes) }?;
    let output_len = q_rows + k_rows + v_rows;
    let output = unsafe { dev.alloc::<f32>(output_len) }?;
    let stream = dev.cuda_stream();
    unsafe {
        ffi::launch_mmvq_gguf_qkv_q8_0_f32(
            input_slice.device_ptr(&stream).0 as *const std::ffi::c_void,
            scratch.device_ptr(scratch.stream()).0 as *mut std::ffi::c_void,
            q_weight.device_ptr()? as *const std::ffi::c_void,
            k_weight.device_ptr()? as *const std::ffi::c_void,
            v_weight.device_ptr()? as *const std::ffi::c_void,
            output.device_ptr(output.stream()).0 as *mut std::ffi::c_void,
            hidden as i32,
            hidden_padded as i32,
            q_rows as i32,
            k_rows as i32,
            v_rows as i32,
            stream.cu_stream() as *mut std::ffi::c_void,
        );
    }

    let output = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
    let output = Tensor::from_storage(
        candle::Storage::Cuda(output),
        (1, 1, output_len),
        BackpropOp::none(),
        false,
    );
    let q = output.narrow(2, 0, q_rows)?;
    let k = output.narrow(2, q_rows, k_rows)?;
    let v = output.narrow(2, q_rows + k_rows, v_rows)?;
    Ok((q, k, v))
}

/// Projects TP-local Q/K/V and emits the two local Q/K sums of squares used by
/// global tensor-parallel RMSNorm. The activation is quantized only once.
#[cfg(feature = "cuda")]
pub fn q8_0_decode_with_sums(
    input: &Tensor,
    q_weight: &QTensor,
    k_weight: &QTensor,
    v_weight: &QTensor,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::cuda_backend::kernels::ffi;
    use candle::op::BackpropOp;

    let (q, k, v) = q8_0_decode(input, q_weight, k_weight, v_weight)?;
    let q_rows = q.elem_count();
    let k_rows = k.elem_count();
    let dev = input.device().as_cuda_device()?;
    let stream = dev.cuda_stream();
    let (q_storage, q_layout) = q.storage_and_layout();
    let q_slice = match &*q_storage {
        candle::Storage::Cuda(storage) => storage.as_cuda_slice::<f32>()?,
        _ => candle::bail!("fused TP Q must be a CUDA tensor"),
    };
    let (q_start, q_end) = q_layout
        .contiguous_offsets()
        .ok_or_else(|| candle::Error::Msg("fused TP Q must be contiguous".into()))?;
    let q_slice = q_slice.slice(q_start..q_end);
    let (k_storage, k_layout) = k.storage_and_layout();
    let k_slice = match &*k_storage {
        candle::Storage::Cuda(storage) => storage.as_cuda_slice::<f32>()?,
        _ => candle::bail!("fused TP K must be a CUDA tensor"),
    };
    let (k_start, k_end) = k_layout
        .contiguous_offsets()
        .ok_or_else(|| candle::Error::Msg("fused TP K must be contiguous".into()))?;
    let k_slice = k_slice.slice(k_start..k_end);
    let stats = unsafe { dev.alloc::<f32>(2) }?;
    unsafe {
        ffi::launch_qkv_tp_sumsq_f32(
            q_slice.device_ptr(&stream).0 as *const std::ffi::c_void,
            k_slice.device_ptr(&stream).0 as *const std::ffi::c_void,
            stats.device_ptr(stats.stream()).0 as *mut std::ffi::c_void,
            q_rows as i32,
            k_rows as i32,
            stream.cu_stream() as *mut std::ffi::c_void,
        );
    }
    let stats = Tensor::from_storage(
        candle::Storage::Cuda(candle::CudaStorage::wrap_cuda_slice(stats, dev.clone())),
        (1, 1, 2),
        BackpropOp::none(),
        false,
    );
    Ok((q.clone(), k.clone(), v, stats))
}

/// As [`q8_0_decode`], but also applies Q/K RMSNorm and converts all three
/// projections to F16 in one post-projection kernel.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn q8_0_decode_rmsnorm_f16(
    input: &Tensor,
    q_weight: &QTensor,
    k_weight: &QTensor,
    v_weight: &QTensor,
    q_norm: &Tensor,
    k_norm: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::cuda_backend::kernels::ffi;
    use candle::op::BackpropOp;
    use candle::quantized::GgmlDType;

    if input.dtype() != DType::F32 || input.dims().len() != 3 || input.dims()[..2] != [1, 1] {
        candle::bail!(
            "fused qkv decode expects a contiguous f32 [1, 1, hidden] tensor, got {:?} {:?}",
            input.shape(),
            input.dtype()
        )
    }
    let hidden = input.dim(2)?;
    let (q_rows, q_hidden) = q_weight.shape().dims2()?;
    let (k_rows, k_hidden) = k_weight.shape().dims2()?;
    let (v_rows, v_hidden) = v_weight.shape().dims2()?;
    if q_hidden != hidden || k_hidden != hidden || v_hidden != hidden {
        candle::bail!("fused qkv weight/input shape mismatch")
    }
    if q_weight.dtype() != GgmlDType::Q8_0
        || k_weight.dtype() != GgmlDType::Q8_0
        || v_weight.dtype() != GgmlDType::Q8_0
    {
        candle::bail!("fused qkv decode requires Q8_0 weights")
    }
    if q_norm.dtype() != DType::F32
        || k_norm.dtype() != DType::F32
        || q_norm.dims1()? != q_rows
        || k_norm.dims1()? != k_rows
    {
        candle::bail!("fused qkv RMSNorm weights must be matching F32 vectors")
    }
    let dev = input.device().as_cuda_device()?;
    for device in [
        q_weight.device(),
        k_weight.device(),
        v_weight.device(),
        q_norm.device().clone(),
        k_norm.device().clone(),
    ] {
        if !device.same_device(input.device()) {
            candle::bail!("fused qkv inputs must be on the same CUDA device")
        }
    }

    let input = input.contiguous()?;
    let (input_storage, input_layout) = input.storage_and_layout();
    let input_slice = match &*input_storage {
        candle::Storage::Cuda(storage) => storage.as_cuda_slice::<f32>()?,
        _ => candle::bail!("fused qkv input must be a CUDA tensor"),
    };
    let (start, end) = input_layout
        .contiguous_offsets()
        .ok_or_else(|| candle::Error::Msg("fused qkv input must be contiguous".into()))?;
    let input_slice = input_slice.slice(start..end);

    let (q_norm_storage, q_norm_layout) = q_norm.storage_and_layout();
    let q_norm_slice = match &*q_norm_storage {
        candle::Storage::Cuda(storage) => storage.as_cuda_slice::<f32>()?,
        _ => candle::bail!("fused qkv norm must be a CUDA tensor"),
    };
    let (start, end) = q_norm_layout
        .contiguous_offsets()
        .ok_or_else(|| candle::Error::Msg("fused qkv norm must be contiguous".into()))?;
    let q_norm_slice = q_norm_slice.slice(start..end);
    let (k_norm_storage, k_norm_layout) = k_norm.storage_and_layout();
    let k_norm_slice = match &*k_norm_storage {
        candle::Storage::Cuda(storage) => storage.as_cuda_slice::<f32>()?,
        _ => candle::bail!("fused qkv norm must be a CUDA tensor"),
    };
    let (start, end) = k_norm_layout
        .contiguous_offsets()
        .ok_or_else(|| candle::Error::Msg("fused qkv norm must be contiguous".into()))?;
    let k_norm_slice = k_norm_slice.slice(start..end);

    let hidden_padded = hidden.div_ceil(512) * 512;
    let scratch = unsafe { dev.alloc::<u8>(hidden_padded / 32 * 36) }?;
    let output_len = q_rows + k_rows + v_rows;
    let projection_output = unsafe { dev.alloc::<f32>(output_len) }?;
    let output = unsafe { dev.alloc::<half::f16>(output_len) }?;
    let stream = dev.cuda_stream();
    unsafe {
        ffi::launch_mmvq_gguf_qkv_q8_0_f32_rmsnorm_f16(
            input_slice.device_ptr(&stream).0 as *const std::ffi::c_void,
            scratch.device_ptr(scratch.stream()).0 as *mut std::ffi::c_void,
            q_weight.device_ptr()? as *const std::ffi::c_void,
            k_weight.device_ptr()? as *const std::ffi::c_void,
            v_weight.device_ptr()? as *const std::ffi::c_void,
            projection_output
                .device_ptr(projection_output.stream())
                .0 as *mut std::ffi::c_void,
            q_norm_slice.device_ptr(&stream).0 as *const std::ffi::c_void,
            k_norm_slice.device_ptr(&stream).0 as *const std::ffi::c_void,
            output.device_ptr(output.stream()).0 as *mut std::ffi::c_void,
            hidden as i32,
            hidden_padded as i32,
            q_rows as i32,
            k_rows as i32,
            v_rows as i32,
            eps,
            stream.cu_stream() as *mut std::ffi::c_void,
        );
    }

    let output = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
    let output = Tensor::from_storage(
        candle::Storage::Cuda(output),
        (1, 1, output_len),
        BackpropOp::none(),
        false,
    );
    Ok((
        output.narrow(2, 0, q_rows)?,
        output.narrow(2, q_rows, k_rows)?,
        output.narrow(2, q_rows + k_rows, v_rows)?,
    ))
}

/// Finishes one-token TP Q/K/V after the rank-local statistics have been
/// globally reduced. RMSNorm, F16 conversion, partial RoPE, and output packing
/// share one CUDA launch and one allocation.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn tp_rmsnorm_rope_f16(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    global_stats: &Tensor,
    q_norm: &Tensor,
    k_norm: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    q_full_rows: usize,
    k_full_rows: usize,
    head_dim: usize,
    rope_dim: usize,
    position: usize,
    eps: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::cuda_backend::kernels::ffi;
    use candle::op::BackpropOp;

    let q_rows = q.elem_count();
    let k_rows = k.elem_count();
    let v_rows = v.elem_count();
    if [q, k, v, global_stats, q_norm, k_norm, cos, sin]
        .iter()
        .any(|tensor| tensor.dtype() != DType::F32)
        || global_stats.elem_count() != 2
        || q_norm.elem_count() != q_rows
        || k_norm.elem_count() != k_rows
        || q_full_rows < q_rows
        || k_full_rows < k_rows
        || !q_rows.is_multiple_of(head_dim)
        || !k_rows.is_multiple_of(head_dim)
        || !v_rows.is_multiple_of(head_dim)
        || rope_dim == 0
        || rope_dim > head_dim
        || !rope_dim.is_multiple_of(2)
    {
        candle::bail!("invalid tensors or dimensions for fused TP QKV finish")
    }
    let table_dims = cos.dims2()?;
    if sin.dims2()? != table_dims || position >= table_dims.0 || table_dims.1 != rope_dim / 2 {
        candle::bail!("invalid RoPE tables or position for fused TP QKV finish")
    }
    for tensor in [k, v, global_stats, q_norm, k_norm, cos, sin] {
        if !tensor.device().same_device(q.device()) {
            candle::bail!("fused TP QKV tensors must share a CUDA device")
        }
    }
    let dev = q.device().as_cuda_device()?;
    let stream = dev.cuda_stream();

    let f32_ptr = |tensor: &Tensor, name: &str| -> Result<*const std::ffi::c_void> {
        let (storage, layout) = tensor.storage_and_layout();
        let slice = match &*storage {
            candle::Storage::Cuda(storage) => storage.as_cuda_slice::<f32>()?,
            _ => candle::bail!("{name} must be a CUDA tensor"),
        };
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle::Error::Msg(format!("{name} must be contiguous")))?;
        let ptr = slice.slice(start..end).device_ptr(&stream).0 as *const std::ffi::c_void;
        Ok(ptr)
    };
    let q_ptr = f32_ptr(q, "TP Q")?;
    let k_ptr = f32_ptr(k, "TP K")?;
    let v_ptr = f32_ptr(v, "TP V")?;
    let stats_ptr = f32_ptr(global_stats, "TP Q/K statistics")?;
    let qn_ptr = f32_ptr(q_norm, "TP Q norm")?;
    let kn_ptr = f32_ptr(k_norm, "TP K norm")?;
    let cos_ptr = f32_ptr(cos, "TP RoPE cosine table")?;
    let sin_ptr = f32_ptr(sin, "TP RoPE sine table")?;

    let output_len = q_rows + k_rows + v_rows;
    let output = unsafe { dev.alloc::<half::f16>(output_len) }?;
    unsafe {
        ffi::launch_qkv_tp_rmsnorm_rope_f16(
            q_ptr,
            k_ptr,
            v_ptr,
            stats_ptr,
            qn_ptr,
            kn_ptr,
            cos_ptr,
            sin_ptr,
            output.device_ptr(output.stream()).0 as *mut std::ffi::c_void,
            q_rows as i32,
            k_rows as i32,
            v_rows as i32,
            q_full_rows as i32,
            k_full_rows as i32,
            head_dim as i32,
            rope_dim as i32,
            position as i32,
            eps,
            stream.cu_stream() as *mut std::ffi::c_void,
        );
    }
    let output = Tensor::from_storage(
        candle::Storage::Cuda(candle::CudaStorage::wrap_cuda_slice(output, dev.clone())),
        (1, 1, output_len),
        BackpropOp::none(),
        false,
    );
    Ok((
        output.narrow(2, 0, q_rows)?,
        output.narrow(2, q_rows, k_rows)?,
        output.narrow(2, q_rows + k_rows, v_rows)?,
    ))
}

/// Applies half-split partial RoPE to one-token Q/K tensors and packs the
/// results into one allocation. Cosine and sine tables remain F32 so no table
/// cast/allocation is needed on each decode step.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn partial_rope_f16(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    position: usize,
    head_dim: usize,
    rope_dim: usize,
) -> Result<(Tensor, Tensor)> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::cuda_backend::kernels::ffi;
    use candle::op::BackpropOp;

    let q_rows = q.dim(candle::D::Minus1)?;
    let k_rows = k.dim(candle::D::Minus1)?;
    if q.dtype() != DType::F16
        || k.dtype() != DType::F16
        || cos.dtype() != DType::F32
        || sin.dtype() != DType::F32
        || q.elem_count() != q_rows
        || k.elem_count() != k_rows
        || rope_dim == 0
        || rope_dim > head_dim
        || !rope_dim.is_multiple_of(2)
        || !q_rows.is_multiple_of(head_dim)
        || !k_rows.is_multiple_of(head_dim)
    {
        candle::bail!("invalid tensors or dimensions for fused partial RoPE")
    }
    let table_dims = cos.dims2()?;
    if sin.dims2()? != table_dims || position >= table_dims.0 || table_dims.1 != rope_dim / 2 {
        candle::bail!("invalid RoPE tables or position")
    }
    for tensor in [k, cos, sin] {
        if !tensor.device().same_device(q.device()) {
            candle::bail!("fused partial RoPE tensors must share a CUDA device")
        }
    }
    let dev = q.device().as_cuda_device()?;
    let stream = dev.cuda_stream();

    let (q_storage, q_layout) = q.storage_and_layout();
    let q_slice = match &*q_storage {
        candle::Storage::Cuda(storage) => storage.as_cuda_slice::<half::f16>()?,
        _ => candle::bail!("fused partial RoPE requires CUDA tensors"),
    };
    let (start, end) = q_layout
        .contiguous_offsets()
        .ok_or_else(|| candle::Error::Msg("Q must be contiguous".into()))?;
    let q_slice = q_slice.slice(start..end);
    let (k_storage, k_layout) = k.storage_and_layout();
    let k_slice = match &*k_storage {
        candle::Storage::Cuda(storage) => storage.as_cuda_slice::<half::f16>()?,
        _ => candle::bail!("fused partial RoPE requires CUDA tensors"),
    };
    let (start, end) = k_layout
        .contiguous_offsets()
        .ok_or_else(|| candle::Error::Msg("K must be contiguous".into()))?;
    let k_slice = k_slice.slice(start..end);
    let (cos_storage, cos_layout) = cos.storage_and_layout();
    let cos_slice = match &*cos_storage {
        candle::Storage::Cuda(storage) => storage.as_cuda_slice::<f32>()?,
        _ => candle::bail!("fused partial RoPE requires CUDA tensors"),
    };
    let (start, end) = cos_layout
        .contiguous_offsets()
        .ok_or_else(|| candle::Error::Msg("cos table must be contiguous".into()))?;
    let cos_slice = cos_slice.slice(start..end);
    let (sin_storage, sin_layout) = sin.storage_and_layout();
    let sin_slice = match &*sin_storage {
        candle::Storage::Cuda(storage) => storage.as_cuda_slice::<f32>()?,
        _ => candle::bail!("fused partial RoPE requires CUDA tensors"),
    };
    let (start, end) = sin_layout
        .contiguous_offsets()
        .ok_or_else(|| candle::Error::Msg("sin table must be contiguous".into()))?;
    let sin_slice = sin_slice.slice(start..end);

    let output = unsafe { dev.alloc::<half::f16>(q_rows + k_rows) }?;
    unsafe {
        ffi::launch_partial_rope_f16(
            q_slice.device_ptr(&stream).0 as *const std::ffi::c_void,
            k_slice.device_ptr(&stream).0 as *const std::ffi::c_void,
            cos_slice.device_ptr(&stream).0 as *const std::ffi::c_void,
            sin_slice.device_ptr(&stream).0 as *const std::ffi::c_void,
            output.device_ptr(output.stream()).0 as *mut std::ffi::c_void,
            q_rows as i32,
            k_rows as i32,
            head_dim as i32,
            rope_dim as i32,
            position as i32,
            stream.cu_stream() as *mut std::ffi::c_void,
        );
    }
    let output = Tensor::from_storage(
        candle::Storage::Cuda(candle::CudaStorage::wrap_cuda_slice(output, dev.clone())),
        (1, 1, q_rows + k_rows),
        BackpropOp::none(),
        false,
    );
    Ok((
        output.narrow(2, 0, q_rows)?,
        output.narrow(2, q_rows, k_rows)?,
    ))
}

#[cfg(not(feature = "cuda"))]
pub fn q8_0_decode(
    _: &Tensor,
    _: &QTensor,
    _: &QTensor,
    _: &QTensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    candle::bail!("fused qkv decode is only implemented for CUDA")
}

#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn q8_0_decode_rmsnorm_f16(
    _: &Tensor,
    _: &QTensor,
    _: &QTensor,
    _: &QTensor,
    _: &Tensor,
    _: &Tensor,
    _: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    candle::bail!("fused qkv decode is only implemented for CUDA")
}

#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn partial_rope_f16(
    _: &Tensor,
    _: &Tensor,
    _: &Tensor,
    _: &Tensor,
    _: usize,
    _: usize,
    _: usize,
) -> Result<(Tensor, Tensor)> {
    candle::bail!("fused partial RoPE is only implemented for CUDA")
}
