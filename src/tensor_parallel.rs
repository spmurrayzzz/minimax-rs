//! Fixed two-rank NCCL collectives for the MiniMax tensor-parallel graph.
//!
//! Both ranks use the CUDA streams already owned by their Candle devices. A
//! single host controller uses persistent peer-visible signals for small
//! latency-sensitive operations and grouped NCCL for larger tensors.

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Shape, Storage, Tensor, op::BackpropOp};
use cudarc::{
    driver::{CudaStream, DevicePtr, result as driver_result, sys as driver_sys},
    nccl::{Comm, MultiCommGroup, ReduceOp, result},
    runtime::{result as runtime_result, sys as runtime_sys},
};
use std::{ffi::c_void, sync::Arc};

/// The only tensor-parallel world size supported by MiniMax-M2.7.
pub const TP_WORLD_SIZE: usize = 2;
const PEER_ALL_REDUCE_MAX_ELEMENTS: usize = 8_192;
const PEER_SIGNAL_BYTES: usize = 128;
const PROCESS_PEER_DATA_BYTES: usize = PEER_ALL_REDUCE_MAX_ELEMENTS * std::mem::size_of::<f32>();
const PROCESS_PEER_BUFFER_BYTES: usize = PEER_SIGNAL_BYTES + PROCESS_PEER_DATA_BYTES;
const CUDA_IPC_HANDLE_BYTES: usize = 64;

#[derive(Debug)]
struct PeerBuffer {
    base: usize,
    stream: Arc<CudaStream>,
}

impl PeerBuffer {
    fn new(stream: Arc<CudaStream>) -> Result<Self> {
        stream.context().bind_to_thread()?;
        let bytes = PEER_SIGNAL_BYTES;
        let base = unsafe { runtime_result::malloc_sync(bytes) }
            .context("allocate persistent peer all-reduce buffer")?;
        if let Err(error) = unsafe { runtime_result::memset_d8_sync(base, 0, bytes) } {
            let _ = unsafe { runtime_result::free_sync(base) };
            return Err(error).context("zero persistent peer all-reduce buffer");
        }
        Ok(Self {
            base: base as usize,
            stream,
        })
    }
}

impl Drop for PeerBuffer {
    fn drop(&mut self) {
        let _ = self.stream.context().bind_to_thread();
        let _ = self.stream.synchronize();
        let _ = unsafe { runtime_result::free_sync(self.base as *mut c_void) };
    }
}

#[derive(Debug)]
struct PeerAllReduce {
    buffers: [PeerBuffer; TP_WORLD_SIZE],
    epoch: u32,
}

impl PeerAllReduce {
    fn new(streams: &[Arc<CudaStream>; TP_WORLD_SIZE]) -> Result<Self> {
        for rank in 0..TP_WORLD_SIZE {
            let peer = 1 - rank;
            streams[rank].context().bind_to_thread()?;
            let status =
                unsafe { driver_sys::cuCtxEnablePeerAccess(streams[peer].context().cu_ctx(), 0) };
            if status != driver_sys::CUresult::CUDA_SUCCESS
                && status != driver_sys::CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED
            {
                bail!("enable CUDA peer access for TP rank {rank}: {status:?}")
            }
            let mut pool = std::ptr::null_mut();
            let status =
                unsafe { runtime_sys::cudaDeviceGetDefaultMemPool(&mut pool, rank as i32) };
            if status != runtime_sys::cudaError::cudaSuccess {
                bail!("query CUDA default memory pool for TP rank {rank}: {status:?}")
            }
            let location = runtime_sys::cudaMemLocation {
                type_: runtime_sys::cudaMemLocationType::cudaMemLocationTypeDevice,
                __bindgen_anon_1: runtime_sys::cudaMemLocation__bindgen_ty_1 { id: peer as i32 },
            };
            let access = runtime_sys::cudaMemAccessDesc {
                location,
                flags: runtime_sys::cudaMemAccessFlags::cudaMemAccessFlagsProtReadWrite,
            };
            let status = unsafe { runtime_sys::cudaMemPoolSetAccess(pool, &access, 1) };
            if status != runtime_sys::cudaError::cudaSuccess {
                bail!("enable peer access to TP rank {rank} memory pool: {status:?}")
            }
        }
        Ok(Self {
            buffers: [
                PeerBuffer::new(streams[0].clone())?,
                PeerBuffer::new(streams[1].clone())?,
            ],
            epoch: 0,
        })
    }

    fn next_epoch(&mut self) -> u32 {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.epoch = 1;
        }
        self.epoch
    }
}

#[derive(Debug)]
struct ProcessPeerAllReduce {
    local: Option<usize>,
    peer: Option<usize>,
    stream: Arc<CudaStream>,
    stats_workspace: Tensor,
    output_workspaces: [Tensor; 2],
    normalized_workspace: Tensor,
    broadcast_workspace: Tensor,
    next_output: usize,
}

impl ProcessPeerAllReduce {
    fn new(stream: Arc<CudaStream>, comm: &Comm, rank: usize, device: &Device) -> Result<Self> {
        stream.context().bind_to_thread()?;
        let local = unsafe { runtime_result::malloc_sync(PROCESS_PEER_BUFFER_BYTES) }
            .with_context(|| format!("allocate rank-{rank} CUDA IPC collective buffer"))?;
        if let Err(error) =
            unsafe { runtime_result::memset_d8_sync(local, 0, PROCESS_PEER_BUFFER_BYTES) }
        {
            let _ = unsafe { runtime_result::free_sync(local) };
            return Err(error).with_context(|| format!("zero rank-{rank} CUDA IPC buffer"));
        }

        let setup = (|| -> Result<usize> {
            let mut local_handle =
                std::mem::MaybeUninit::<runtime_sys::cudaIpcMemHandle_t>::uninit();
            let status =
                unsafe { runtime_sys::cudaIpcGetMemHandle(local_handle.as_mut_ptr(), local) };
            if status != runtime_sys::cudaError::cudaSuccess {
                bail!("export rank-{rank} CUDA IPC buffer: {status:?}")
            }
            let local_handle = unsafe { local_handle.assume_init() };
            let local_bytes = local_handle
                .reserved
                .iter()
                .map(|&byte| byte as u8)
                .collect::<Vec<_>>();
            let send = stream
                .clone_htod(&local_bytes)
                .with_context(|| format!("upload rank-{rank} CUDA IPC handle"))?;
            let mut gathered = unsafe { stream.alloc::<u8>(CUDA_IPC_HANDLE_BYTES * TP_WORLD_SIZE) }
                .with_context(|| format!("allocate rank-{rank} IPC handle exchange"))?;
            comm.all_gather(&send, &mut gathered)
                .with_context(|| format!("exchange CUDA IPC handles on rank {rank}"))?;
            let gathered = stream
                .clone_dtoh(&gathered)
                .with_context(|| format!("download CUDA IPC handles on rank {rank}"))?;
            let peer_rank = 1 - rank;
            let start = peer_rank * CUDA_IPC_HANDLE_BYTES;
            let mut peer_handle = runtime_sys::cudaIpcMemHandle_t {
                reserved: [0; CUDA_IPC_HANDLE_BYTES],
            };
            for (target, &source) in peer_handle
                .reserved
                .iter_mut()
                .zip(&gathered[start..start + CUDA_IPC_HANDLE_BYTES])
            {
                *target = source as std::ffi::c_char;
            }
            let mut peer = std::ptr::null_mut();
            let status = unsafe {
                runtime_sys::cudaIpcOpenMemHandle(
                    &mut peer,
                    peer_handle,
                    runtime_sys::cudaIpcMemLazyEnablePeerAccess,
                )
            };
            if status != runtime_sys::cudaError::cudaSuccess {
                bail!("open rank-{peer_rank} CUDA IPC buffer on rank {rank}: {status:?}")
            }
            Ok(peer as usize)
        })();
        match setup {
            Ok(peer) => {
                let workspaces = (|| -> candle_core::Result<[Tensor; 5]> {
                    Ok([
                        Tensor::zeros(PEER_ALL_REDUCE_MAX_ELEMENTS, DType::F32, device)?,
                        Tensor::zeros(PEER_ALL_REDUCE_MAX_ELEMENTS, DType::F32, device)?,
                        Tensor::zeros(PEER_ALL_REDUCE_MAX_ELEMENTS, DType::F32, device)?,
                        Tensor::zeros(PEER_ALL_REDUCE_MAX_ELEMENTS, DType::F32, device)?,
                        Tensor::zeros(PEER_ALL_REDUCE_MAX_ELEMENTS, DType::F32, device)?,
                    ])
                })();
                match workspaces {
                    Ok([stats, output0, output1, normalized, broadcast]) => Ok(Self {
                        local: Some(local as usize),
                        peer: Some(peer),
                        stream,
                        stats_workspace: stats,
                        output_workspaces: [output0, output1],
                        normalized_workspace: normalized,
                        broadcast_workspace: broadcast,
                        next_output: 0,
                    }),
                    Err(error) => {
                        let _ = unsafe { runtime_sys::cudaIpcCloseMemHandle(peer as *mut c_void) };
                        // The peer may already have imported `local`. Do not
                        // cudaFree it without a cross-process close barrier;
                        // process teardown will reclaim this initialization-only leak.
                        Err(error.into())
                    }
                }
            }
            Err(error) => {
                // Once handle exchange starts, the other process may have
                // imported `local` even if this rank reports an error. Avoid
                // cudaFree here because CUDA defines that race as undefined.
                Err(error)
            }
        }
    }

    fn local_base(&self) -> usize {
        self.local
            .expect("process peer local buffer was already released")
    }

    fn peer_base(&self) -> usize {
        self.peer.expect("process peer mapping was already closed")
    }

    fn local_data(&self) -> usize {
        self.local_base() + PEER_SIGNAL_BYTES
    }

    fn peer_data(&self) -> usize {
        self.peer_base() + PEER_SIGNAL_BYTES
    }

    fn close_peer_mapping(&mut self) -> Result<()> {
        self.stream.context().bind_to_thread()?;
        self.stream.synchronize()?;
        let Some(peer) = self.peer.take() else {
            return Ok(());
        };
        let status = unsafe { runtime_sys::cudaIpcCloseMemHandle(peer as *mut c_void) };
        if status != runtime_sys::cudaError::cudaSuccess {
            self.peer = Some(peer);
            bail!("close imported CUDA IPC collective buffer: {status:?}")
        }
        Ok(())
    }

    fn free_local_buffer(&mut self) -> Result<()> {
        if self.peer.is_some() {
            bail!("cannot free exported CUDA IPC buffer before closing the peer mapping")
        }
        self.stream.context().bind_to_thread()?;
        self.stream.synchronize()?;
        let Some(local) = self.local.take() else {
            return Ok(());
        };
        if let Err(error) = unsafe { runtime_result::free_sync(local as *mut c_void) } {
            self.local = Some(local);
            return Err(error).context("free exported CUDA IPC collective buffer");
        }
        Ok(())
    }

    fn output_workspace(&mut self, shape: &Shape) -> candle_core::Result<Tensor> {
        let count = shape.elem_count();
        let is_statistics = shape.dims().last() == Some(&2);
        let workspace = if is_statistics {
            &self.stats_workspace
        } else {
            let index = self.next_output;
            self.next_output = (self.next_output + 1) % self.output_workspaces.len();
            &self.output_workspaces[index]
        };
        workspace.narrow(0, 0, count)?.reshape(shape)
    }

    fn normalized_workspace(&self, shape: &Shape) -> candle_core::Result<Tensor> {
        self.normalized_workspace
            .narrow(0, 0, shape.elem_count())?
            .reshape(shape)
    }

    fn broadcast_workspace(&self, shape: &Shape) -> candle_core::Result<Tensor> {
        self.broadcast_workspace
            .narrow(0, 0, shape.elem_count())?
            .reshape(shape)
    }
}

impl Drop for ProcessPeerAllReduce {
    fn drop(&mut self) {
        let _ = self.stream.context().bind_to_thread();
        let _ = self.stream.synchronize();
        if let Some(peer) = self.peer.take() {
            let _ = unsafe { runtime_sys::cudaIpcCloseMemHandle(peer as *mut c_void) };
        }
        // Freeing an exported allocation while another process still has it
        // imported is undefined. Normal shutdown calls `free_local_buffer`
        // only after both ranks acknowledge closing their peer mappings. On
        // an uncoordinated error path, deliberately leave this tiny allocation
        // for CUDA process teardown rather than racing the importer.
    }
}

/// Two NCCL communicators bound to the exact streams owned by two Candle CUDA
/// devices.
#[derive(Debug)]
pub struct TensorParallelGroup {
    comms: [Comm; TP_WORLD_SIZE],
    devices: [Device; TP_WORLD_SIZE],
    nccl_version: i32,
    peer_all_reduce: PeerAllReduce,
}

impl TensorParallelGroup {
    /// Creates the fixed two-rank communicator group.
    ///
    /// Initialization errors are returned to the caller; tensor mode must not
    /// silently fall back to a host-staged collective.
    pub fn new(devices: &[Device]) -> Result<Self> {
        if devices.len() != TP_WORLD_SIZE {
            bail!(
                "tensor parallelism requires exactly {TP_WORLD_SIZE} CUDA devices, got {}",
                devices.len()
            )
        }
        for (rank, device) in devices.iter().enumerate() {
            if !device.is_cuda() {
                bail!("tensor-parallel rank {rank} is not a CUDA device")
            }
        }
        if devices[0].same_device(&devices[1]) {
            bail!("tensor-parallel ranks must use two distinct CUDA devices")
        }

        let streams: [Arc<CudaStream>; TP_WORLD_SIZE] = devices
            .iter()
            .map(|device| Ok(device.as_cuda_device()?.cuda_stream()))
            .collect::<candle_core::Result<Vec<_>>>()?
            .try_into()
            .map_err(|streams: Vec<_>| {
                anyhow::anyhow!(
                    "found {} CUDA streams, expected {TP_WORLD_SIZE}",
                    streams.len()
                )
            })?;
        let peer_all_reduce =
            PeerAllReduce::new(&streams).context("failed to initialize peer all-reduce buffers")?;
        let comms: [Comm; TP_WORLD_SIZE] = Comm::from_devices(streams.to_vec())
            .context("failed to initialize the two-rank NCCL communicator")?
            .try_into()
            .map_err(|comms: Vec<Comm>| {
                anyhow::anyhow!(
                    "NCCL initialized {} communicators, expected {TP_WORLD_SIZE}",
                    comms.len()
                )
            })?;
        for (rank, comm) in comms.iter().enumerate() {
            if comm.rank() != rank || comm.world_size() != TP_WORLD_SIZE {
                bail!(
                    "invalid NCCL communicator at index {rank}: rank={}, world_size={}",
                    comm.rank(),
                    comm.world_size()
                )
            }
        }
        let nccl_version = result::get_nccl_version().context("failed to query NCCL version")?;

        Ok(Self {
            comms,
            devices: [devices[0].clone(), devices[1].clone()],
            nccl_version,
            peer_all_reduce,
        })
    }

    pub fn backend(&self) -> &'static str {
        "peer+nccl"
    }

    /// NCCL's integer version (`major * 10000 + minor * 100 + patch`).
    pub fn nccl_version(&self) -> i32 {
        self.nccl_version
    }

    pub fn devices(&self) -> &[Device; TP_WORLD_SIZE] {
        &self.devices
    }

    /// Sums matching contiguous F32 tensors and returns one out-of-place result
    /// on each rank.
    pub fn all_reduce_sum(&mut self, inputs: [Tensor; TP_WORLD_SIZE]) -> Result<[Tensor; 2]> {
        let shape = validate_pair(&inputs, &self.devices, "all-reduce")?;
        let count = shape.elem_count();
        if count <= PEER_ALL_REDUCE_MAX_ELEMENTS {
            return self
                .peer_all_reduce_sum(inputs, &shape, None, None)
                .map(|result| result.0);
        }

        let (storage0, layout0) = inputs[0].storage_and_layout();
        let input0 = cuda_f32_slice(&storage0, 0, "all-reduce input")?;
        let input0 = input0.slice(layout0.start_offset()..layout0.start_offset() + count);
        let (storage1, layout1) = inputs[1].storage_and_layout();
        let input1 = cuda_f32_slice(&storage1, 1, "all-reduce input")?;
        let input1 = input1.slice(layout1.start_offset()..layout1.start_offset() + count);

        let cuda0 = self.devices[0].as_cuda_device()?;
        let cuda1 = self.devices[1].as_cuda_device()?;
        let mut output0 = unsafe { cuda0.alloc::<f32>(count) }
            .context("failed to allocate rank-0 all-reduce output")?;
        let mut output1 = unsafe { cuda1.alloc::<f32>(count) }
            .context("failed to allocate rank-1 all-reduce output")?;
        let output_view0 = output0.slice_mut(..);
        let output_view1 = output1.slice_mut(..);

        let mut group = MultiCommGroup::new(&self.comms).context("failed to start NCCL group")?;
        group
            .all_reduce(0, input0, output_view0, &ReduceOp::Sum)
            .context("rank 0 failed to enqueue NCCL all-reduce")?;
        group
            .all_reduce(1, input1, output_view1, &ReduceOp::Sum)
            .context("rank 1 failed to enqueue NCCL all-reduce")?;
        group
            .finish()
            .context("failed to finish NCCL all-reduce group")?;

        tracing::trace!(
            operation = "all_reduce_sum",
            count,
            dtype = "f32",
            "enqueued tensor-parallel collective"
        );
        Ok([
            tensor_from_f32_slice(output0, cuda0, &shape),
            tensor_from_f32_slice(output1, cuda1, &shape),
        ])
    }

    /// Sums matching rank tensors and adds each rank's local residual in the
    /// same peer kernel when the latency path is eligible.
    pub fn all_reduce_sum_with_residual(
        &mut self,
        inputs: [Tensor; TP_WORLD_SIZE],
        residuals: [Tensor; TP_WORLD_SIZE],
    ) -> Result<[Tensor; TP_WORLD_SIZE]> {
        let shape = validate_pair(&inputs, &self.devices, "residual all-reduce")?;
        let residual_shape = validate_pair(&residuals, &self.devices, "all-reduce residual")?;
        if shape != residual_shape {
            bail!(
                "residual all-reduce shape mismatch: inputs {:?}, residuals {:?}",
                shape,
                residual_shape
            )
        }
        if shape.elem_count() <= PEER_ALL_REDUCE_MAX_ELEMENTS {
            return self
                .peer_all_reduce_sum(inputs, &shape, Some(&residuals), None)
                .map(|result| result.0);
        }
        let reduced = self.all_reduce_sum(inputs)?;
        Ok([
            (&reduced[0] + &residuals[0])?,
            (&reduced[1] + &residuals[1])?,
        ])
    }

    /// Decode-only peer reduction that also emits RMS-normalized hidden states.
    pub fn all_reduce_sum_with_residual_rmsnorm(
        &mut self,
        inputs: [Tensor; TP_WORLD_SIZE],
        residuals: [Tensor; TP_WORLD_SIZE],
        norm_weights: [Tensor; TP_WORLD_SIZE],
        eps: f32,
    ) -> Result<([Tensor; TP_WORLD_SIZE], [Tensor; TP_WORLD_SIZE])> {
        let shape = validate_pair(&inputs, &self.devices, "RMSNorm all-reduce")?;
        let residual_shape = validate_pair(&residuals, &self.devices, "RMSNorm residual")?;
        let weight_shape = validate_pair(&norm_weights, &self.devices, "RMSNorm weight")?;
        if shape != residual_shape || weight_shape != Shape::from(shape.elem_count()) {
            bail!(
                "RMSNorm all-reduce shape mismatch: inputs {:?}, residuals {:?}, weights {:?}",
                shape,
                residual_shape,
                weight_shape
            )
        }
        if shape.elem_count() > PEER_ALL_REDUCE_MAX_ELEMENTS {
            bail!(
                "fused RMSNorm all-reduce supports at most {PEER_ALL_REDUCE_MAX_ELEMENTS} elements"
            )
        }
        let (hidden, normalized) =
            self.peer_all_reduce_sum(inputs, &shape, Some(&residuals), Some((&norm_weights, eps)))?;
        Ok((
            hidden,
            normalized.expect("RMSNorm outputs requested from peer reduction"),
        ))
    }

    fn peer_all_reduce_sum(
        &mut self,
        inputs: [Tensor; TP_WORLD_SIZE],
        shape: &Shape,
        residuals: Option<&[Tensor; TP_WORLD_SIZE]>,
        rmsnorm: Option<(&[Tensor; TP_WORLD_SIZE], f32)>,
    ) -> Result<([Tensor; TP_WORLD_SIZE], Option<[Tensor; TP_WORLD_SIZE]>)> {
        use candle_core::cuda_backend::kernels::ffi;

        let count = shape.elem_count();
        let streams = [
            self.devices[0].as_cuda_device()?.cuda_stream(),
            self.devices[1].as_cuda_device()?.cuda_stream(),
        ];
        let (storage0, layout0) = inputs[0].storage_and_layout();
        let input0 = cuda_f32_slice(&storage0, 0, "peer all-reduce input")?;
        let input0 = input0.slice(layout0.start_offset()..layout0.start_offset() + count);
        let (storage1, layout1) = inputs[1].storage_and_layout();
        let input1 = cuda_f32_slice(&storage1, 1, "peer all-reduce input")?;
        let input1 = input1.slice(layout1.start_offset()..layout1.start_offset() + count);
        let input_ptrs = [
            input0.device_ptr(&streams[0]).0,
            input1.device_ptr(&streams[1]).0,
        ];

        let cuda0 = self.devices[0].as_cuda_device()?;
        let cuda1 = self.devices[1].as_cuda_device()?;
        let output0 = unsafe { cuda0.alloc::<f32>(count) }
            .context("failed to allocate rank-0 peer all-reduce output")?;
        let output1 = unsafe { cuda1.alloc::<f32>(count) }
            .context("failed to allocate rank-1 peer all-reduce output")?;
        let normalized = if rmsnorm.is_some() {
            Some([
                unsafe { cuda0.alloc::<f32>(count) }
                    .context("failed to allocate rank-0 fused RMSNorm output")?,
                unsafe { cuda1.alloc::<f32>(count) }
                    .context("failed to allocate rank-1 fused RMSNorm output")?,
            ])
        } else {
            None
        };
        let output_ptrs = [
            output0.device_ptr(&streams[0]).0,
            output1.device_ptr(&streams[1]).0,
        ];
        let normalized_ptrs = if let Some(normalized) = normalized.as_ref() {
            [
                normalized[0].device_ptr(&streams[0]).0,
                normalized[1].device_ptr(&streams[1]).0,
            ]
        } else {
            [0; TP_WORLD_SIZE]
        };
        let residual_ptrs = if let Some(residuals) = residuals {
            [
                cuda_f32_ptr(&residuals[0], &streams[0], 0, "all-reduce residual")?,
                cuda_f32_ptr(&residuals[1], &streams[1], 1, "all-reduce residual")?,
            ]
        } else {
            [0; TP_WORLD_SIZE]
        };
        let (norm_ptrs, norm_eps) = if let Some((weights, eps)) = rmsnorm {
            (
                [
                    cuda_f32_ptr(&weights[0], &streams[0], 0, "RMSNorm weight")?,
                    cuda_f32_ptr(&weights[1], &streams[1], 1, "RMSNorm weight")?,
                ],
                eps,
            )
        } else {
            ([0; TP_WORLD_SIZE], 0.0)
        };
        let epoch = self.peer_all_reduce.next_epoch();
        for rank in 0..TP_WORLD_SIZE {
            let peer = 1 - rank;
            streams[rank].context().bind_to_thread()?;
            unsafe {
                ffi::launch_tp2_peer_all_reduce_f32(
                    input_ptrs[rank] as *const c_void,
                    input_ptrs[peer] as *const c_void,
                    residual_ptrs[rank] as *const c_void,
                    norm_ptrs[rank] as *const c_void,
                    output_ptrs[rank] as *mut c_void,
                    normalized_ptrs[rank] as *mut c_void,
                    self.peer_all_reduce.buffers[rank].base as *mut c_void,
                    self.peer_all_reduce.buffers[peer].base as *mut c_void,
                    epoch,
                    count as i32,
                    norm_eps,
                    streams[rank].cu_stream() as *mut c_void,
                );
            }
        }

        tracing::trace!(
            operation = "peer_all_reduce_sum",
            count,
            dtype = "f32",
            "enqueued tensor-parallel peer collective"
        );
        let normalized = normalized.map(|[rank0, rank1]| {
            [
                tensor_from_f32_slice(rank0, cuda0, shape),
                tensor_from_f32_slice(rank1, cuda1, shape),
            ]
        });
        Ok((
            [
                tensor_from_f32_slice(output0, cuda0, shape),
                tensor_from_f32_slice(output1, cuda1, shape),
            ],
            normalized,
        ))
    }

    fn peer_broadcast_from_rank0(
        &mut self,
        input: Tensor,
        shape: &Shape,
    ) -> Result<[Tensor; TP_WORLD_SIZE]> {
        use candle_core::cuda_backend::kernels::ffi;

        let count = shape.elem_count();
        let streams = [
            self.devices[0].as_cuda_device()?.cuda_stream(),
            self.devices[1].as_cuda_device()?.cuda_stream(),
        ];
        let source = cuda_f32_ptr(&input, &streams[0], 0, "peer broadcast input")?;
        let cuda1 = self.devices[1].as_cuda_device()?;
        let output1 = unsafe { cuda1.alloc::<f32>(count) }
            .context("failed to allocate rank-1 peer broadcast output")?;
        let output1_ptr = output1.device_ptr(&streams[1]).0;
        let epoch = self.peer_all_reduce.next_epoch();
        for rank in 0..TP_WORLD_SIZE {
            let peer = 1 - rank;
            streams[rank].context().bind_to_thread()?;
            unsafe {
                ffi::launch_tp2_peer_broadcast_f32(
                    source as *const c_void,
                    if rank == 0 {
                        std::ptr::null_mut()
                    } else {
                        output1_ptr as *mut c_void
                    },
                    self.peer_all_reduce.buffers[rank].base as *mut c_void,
                    self.peer_all_reduce.buffers[peer].base as *mut c_void,
                    epoch,
                    rank as i32,
                    count as i32,
                    streams[rank].cu_stream() as *mut c_void,
                );
            }
        }
        Ok([input, tensor_from_f32_slice(output1, cuda1, shape)])
    }

    /// Broadcasts a contiguous rank-0 F32 tensor into out-of-place buffers on
    /// both ranks.
    pub fn broadcast_from_rank0(&mut self, input: Tensor) -> Result<[Tensor; 2]> {
        validate_rank_tensor(&input, &self.devices[0], 0, "broadcast input")?;
        let shape = input.shape().clone();
        let count = shape.elem_count();
        if count == 0 {
            bail!("broadcast input must not be empty")
        }
        if count <= PEER_ALL_REDUCE_MAX_ELEMENTS {
            return self.peer_broadcast_from_rank0(input, &shape);
        }

        let (storage, layout) = input.storage_and_layout();
        let input = cuda_f32_slice(&storage, 0, "broadcast input")?;
        let input = input.slice(layout.start_offset()..layout.start_offset() + count);

        let cuda0 = self.devices[0].as_cuda_device()?;
        let cuda1 = self.devices[1].as_cuda_device()?;
        let mut output0 = unsafe { cuda0.alloc::<f32>(count) }
            .context("failed to allocate rank-0 broadcast output")?;
        let mut output1 = unsafe { cuda1.alloc::<f32>(count) }
            .context("failed to allocate rank-1 broadcast output")?;
        let output_view0 = output0.slice_mut(..);
        let output_view1 = output1.slice_mut(..);

        let mut group = MultiCommGroup::new(&self.comms).context("failed to start NCCL group")?;
        group
            .broadcast(0, Some(input), output_view0, 0)
            .context("rank 0 failed to enqueue NCCL broadcast")?;
        group
            .broadcast::<f32>(1, None, output_view1, 0)
            .context("rank 1 failed to enqueue NCCL broadcast")?;
        group
            .finish()
            .context("failed to finish NCCL broadcast group")?;

        tracing::trace!(
            operation = "broadcast",
            count,
            dtype = "f32",
            root = 0,
            "enqueued tensor-parallel collective"
        );
        Ok([
            tensor_from_f32_slice(output0, cuda0, &shape),
            tensor_from_f32_slice(output1, cuda1, &shape),
        ])
    }
}

/// One process-local collective rank. Unlike [`TensorParallelGroup`], this
/// type creates only its local CUDA device state, then maps one small
/// peer-owned CUDA IPC staging buffer for latency-sensitive collectives.
/// Larger tensors continue to use its process-local NCCL communicator.
#[derive(Debug)]
pub struct TensorParallelRankGroup {
    comm: Comm,
    peer_all_reduce: ProcessPeerAllReduce,
    device: Device,
    rank: usize,
    nccl_version: i32,
}

impl TensorParallelRankGroup {
    pub fn new(device: &Device, rank: usize, id: cudarc::nccl::Id) -> Result<Self> {
        if rank >= TP_WORLD_SIZE {
            bail!("tensor-parallel rank must be 0 or 1, got {rank}")
        }
        if !device.is_cuda() {
            bail!("tensor-parallel rank {rank} is not a CUDA device")
        }
        let stream = device.as_cuda_device()?.cuda_stream();
        let comm = Comm::from_rank(stream.clone(), rank, TP_WORLD_SIZE, id)
            .with_context(|| format!("initialize process-local NCCL rank {rank}"))?;
        let peer_all_reduce = ProcessPeerAllReduce::new(stream, &comm, rank, device)
            .with_context(|| format!("initialize process-local peer buffers on rank {rank}"))?;
        let nccl_version = result::get_nccl_version().context("failed to query NCCL version")?;
        Ok(Self {
            comm,
            peer_all_reduce,
            device: device.clone(),
            rank,
            nccl_version,
        })
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn backend(&self) -> &'static str {
        "ipc-peer+nccl"
    }

    pub fn nccl_version(&self) -> i32 {
        self.nccl_version
    }

    /// First half of coordinated process shutdown. Both ranks must complete
    /// this step before either rank frees its exported buffer.
    pub fn close_peer_mapping(&mut self) -> Result<()> {
        self.peer_all_reduce.close_peer_mapping()
    }

    /// Second half of coordinated process shutdown, called only after the
    /// controller has received `close_peer_mapping` acknowledgements from all
    /// ranks.
    pub fn free_local_peer_buffer(&mut self) -> Result<()> {
        self.peer_all_reduce.free_local_buffer()
    }

    /// The small-tensor result aliases a persistent workspace and remains
    /// stable only until this serialized rank begins its next forward pass.
    pub fn all_reduce_sum(&mut self, input: Tensor) -> Result<Tensor> {
        validate_rank_tensor(&input, &self.device, self.rank, "rank all-reduce")?;
        if input.elem_count() <= PEER_ALL_REDUCE_MAX_ELEMENTS {
            return self
                .peer_all_reduce_sum(input, None, None)
                .map(|value| value.0);
        }
        self.nccl_all_reduce_sum(input)
    }

    pub fn all_reduce_sum_with_residual(
        &mut self,
        input: Tensor,
        residual: Tensor,
    ) -> Result<Tensor> {
        validate_rank_tensor(&input, &self.device, self.rank, "rank residual all-reduce")?;
        validate_rank_tensor(
            &residual,
            &self.device,
            self.rank,
            "rank all-reduce residual",
        )?;
        if input.shape() != residual.shape() {
            bail!(
                "rank {} residual all-reduce shape mismatch: {:?} vs {:?}",
                self.rank,
                input.shape(),
                residual.shape()
            )
        }
        if input.elem_count() <= PEER_ALL_REDUCE_MAX_ELEMENTS {
            return self
                .peer_all_reduce_sum(input, Some(&residual), None)
                .map(|value| value.0);
        }
        Ok((self.nccl_all_reduce_sum(input)? + residual)?)
    }

    pub fn all_reduce_sum_with_residual_rmsnorm(
        &mut self,
        input: Tensor,
        residual: Tensor,
        norm_weight: Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor)> {
        validate_rank_tensor(&input, &self.device, self.rank, "rank RMSNorm all-reduce")?;
        validate_rank_tensor(&residual, &self.device, self.rank, "rank RMSNorm residual")?;
        validate_rank_tensor(&norm_weight, &self.device, self.rank, "rank RMSNorm weight")?;
        if input.shape() != residual.shape()
            || norm_weight.shape() != &Shape::from(input.elem_count())
        {
            bail!(
                "rank {} RMSNorm all-reduce shape mismatch: input {:?}, residual {:?}, weight {:?}",
                self.rank,
                input.shape(),
                residual.shape(),
                norm_weight.shape()
            )
        }
        if input.elem_count() > PEER_ALL_REDUCE_MAX_ELEMENTS {
            bail!(
                "rank-local fused RMSNorm all-reduce supports at most {PEER_ALL_REDUCE_MAX_ELEMENTS} elements"
            )
        }
        let (hidden, normalized) =
            self.peer_all_reduce_sum(input, Some(&residual), Some((&norm_weight, eps)))?;
        Ok((
            hidden,
            normalized.expect("RMSNorm output requested from process peer reduction"),
        ))
    }

    fn nccl_all_reduce_sum(&self, input: Tensor) -> Result<Tensor> {
        let shape = input.shape().clone();
        let count = shape.elem_count();
        if count == 0 {
            bail!("rank all-reduce input must not be empty")
        }
        let (storage, layout) = input.storage_and_layout();
        let input = cuda_f32_slice(&storage, self.rank, "rank all-reduce input")?;
        let input = input.slice(layout.start_offset()..layout.start_offset() + count);
        let cuda = self.device.as_cuda_device()?;
        let mut output = unsafe { cuda.alloc::<f32>(count) }
            .with_context(|| format!("allocate rank-{} all-reduce output", self.rank))?;
        {
            let mut output_view = output.slice_mut(..);
            self.comm
                .all_reduce(&input, &mut output_view, &ReduceOp::Sum)
                .with_context(|| format!("enqueue NCCL all-reduce on rank {}", self.rank))?;
        }
        Ok(tensor_from_f32_slice(output, cuda, &shape))
    }

    fn peer_all_reduce_sum(
        &mut self,
        input: Tensor,
        residual: Option<&Tensor>,
        rmsnorm: Option<(&Tensor, f32)>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        use candle_core::cuda_backend::kernels::ffi;

        let shape = input.shape().clone();
        let count = shape.elem_count();
        let cuda = self.device.as_cuda_device()?;
        let stream = cuda.cuda_stream();
        let input_ptr = cuda_f32_ptr(&input, &stream, self.rank, "rank peer input")?;
        unsafe {
            driver_result::memcpy_dtod_async(
                self.peer_all_reduce.local_data() as u64,
                input_ptr,
                count * std::mem::size_of::<f32>(),
                stream.cu_stream(),
            )
        }
        .with_context(|| format!("stage rank-{} peer all-reduce input", self.rank))?;
        // These aliases point into persistent process-local workspaces. The
        // rank stream serializes every use, so a later collective cannot
        // overwrite a value before its consumer has run.
        let output = self.peer_all_reduce.output_workspace(&shape)?;
        let normalized = if rmsnorm.is_some() {
            Some(self.peer_all_reduce.normalized_workspace(&shape)?)
        } else {
            None
        };
        let output_ptr = cuda_f32_ptr(&output, &stream, self.rank, "rank peer output")?;
        let normalized_ptr = normalized
            .as_ref()
            .map(|output| cuda_f32_ptr(output, &stream, self.rank, "rank normalized output"))
            .transpose()?
            .unwrap_or(0);
        let residual_ptr = residual
            .map(|tensor| cuda_f32_ptr(tensor, &stream, self.rank, "rank residual"))
            .transpose()?
            .unwrap_or(0);
        let (norm_ptr, norm_eps) = rmsnorm
            .map(|(weight, eps)| {
                Ok::<_, anyhow::Error>((
                    cuda_f32_ptr(weight, &stream, self.rank, "rank RMSNorm weight")?,
                    eps,
                ))
            })
            .transpose()?
            .unwrap_or((0, 0.0));
        unsafe {
            ffi::launch_tp2_peer_all_reduce_f32(
                self.peer_all_reduce.local_data() as *const c_void,
                self.peer_all_reduce.peer_data() as *const c_void,
                residual_ptr as *const c_void,
                norm_ptr as *const c_void,
                output_ptr as *mut c_void,
                normalized_ptr as *mut c_void,
                self.peer_all_reduce.local_base() as *mut c_void,
                self.peer_all_reduce.peer_base() as *mut c_void,
                0,
                count as i32,
                norm_eps,
                stream.cu_stream() as *mut c_void,
            );
        }
        Ok((output, normalized))
    }

    /// Broadcasts `input` from rank 0. The rank-1 input has the same shape and
    /// dtype but its values are ignored.
    pub fn broadcast_from_rank0(&mut self, input: Tensor) -> Result<Tensor> {
        use candle_core::cuda_backend::kernels::ffi;

        validate_rank_tensor(&input, &self.device, self.rank, "rank broadcast")?;
        let shape = input.shape().clone();
        let count = shape.elem_count();
        if count == 0 {
            bail!("rank broadcast input must not be empty")
        }
        if count > PEER_ALL_REDUCE_MAX_ELEMENTS {
            let (storage, layout) = input.storage_and_layout();
            let input = cuda_f32_slice(&storage, self.rank, "rank broadcast input")?;
            let input = input.slice(layout.start_offset()..layout.start_offset() + count);
            let cuda = self.device.as_cuda_device()?;
            let mut output = unsafe { cuda.alloc::<f32>(count) }
                .with_context(|| format!("allocate rank-{} broadcast output", self.rank))?;
            {
                let mut output_view = output.slice_mut(..);
                self.comm
                    .broadcast(Some(&input), &mut output_view, 0)
                    .with_context(|| format!("enqueue NCCL broadcast on rank {}", self.rank))?;
            }
            return Ok(tensor_from_f32_slice(output, cuda, &shape));
        }

        let cuda = self.device.as_cuda_device()?;
        let stream = cuda.cuda_stream();
        if self.rank == 0 {
            let input_ptr = cuda_f32_ptr(&input, &stream, self.rank, "rank broadcast input")?;
            unsafe {
                driver_result::memcpy_dtod_async(
                    self.peer_all_reduce.local_data() as u64,
                    input_ptr,
                    count * std::mem::size_of::<f32>(),
                    stream.cu_stream(),
                )
            }
            .context("stage rank-0 peer broadcast input")?;
        }
        let output = if self.rank == 1 {
            Some(self.peer_all_reduce.broadcast_workspace(&shape)?)
        } else {
            None
        };
        let output_ptr = output
            .as_ref()
            .map(|output| cuda_f32_ptr(output, &stream, self.rank, "rank broadcast output"))
            .transpose()?
            .unwrap_or(0);
        unsafe {
            ffi::launch_tp2_peer_broadcast_f32(
                if self.rank == 0 {
                    self.peer_all_reduce.local_data() as *const c_void
                } else {
                    self.peer_all_reduce.peer_data() as *const c_void
                },
                output_ptr as *mut c_void,
                self.peer_all_reduce.local_base() as *mut c_void,
                self.peer_all_reduce.peer_base() as *mut c_void,
                0,
                self.rank as i32,
                count as i32,
                stream.cu_stream() as *mut c_void,
            );
        }
        if let Some(output) = output {
            Ok(output)
        } else {
            Ok(input)
        }
    }
}

fn validate_pair(
    inputs: &[Tensor; TP_WORLD_SIZE],
    devices: &[Device; TP_WORLD_SIZE],
    operation: &str,
) -> Result<Shape> {
    for rank in 0..TP_WORLD_SIZE {
        validate_rank_tensor(&inputs[rank], &devices[rank], rank, operation)?;
    }
    if inputs[0].shape() != inputs[1].shape() {
        bail!(
            "{operation} rank shape mismatch: rank 0 has {:?}, rank 1 has {:?}",
            inputs[0].shape(),
            inputs[1].shape()
        )
    }
    if inputs[0].elem_count() == 0 {
        bail!("{operation} inputs must not be empty")
    }
    Ok(inputs[0].shape().clone())
}

fn validate_rank_tensor(
    tensor: &Tensor,
    device: &Device,
    rank: usize,
    operation: &str,
) -> Result<()> {
    if tensor.dtype() != DType::F32 {
        bail!(
            "{operation} rank {rank} requires F32, got {:?}",
            tensor.dtype()
        )
    }
    if !tensor.device().same_device(device) {
        bail!("{operation} rank {rank} tensor is on the wrong CUDA device")
    }
    if !tensor.layout().is_contiguous() {
        bail!("{operation} rank {rank} tensor must be contiguous")
    }
    Ok(())
}

fn cuda_f32_slice<'a>(
    storage: &'a Storage,
    rank: usize,
    description: &str,
) -> Result<&'a cudarc::driver::CudaSlice<f32>> {
    match storage {
        Storage::Cuda(storage) => storage
            .as_cuda_slice::<f32>()
            .with_context(|| format!("rank {rank} {description} is not F32 CUDA storage")),
        _ => bail!("rank {rank} {description} is not CUDA storage"),
    }
}

fn cuda_f32_ptr(
    tensor: &Tensor,
    stream: &Arc<CudaStream>,
    rank: usize,
    description: &str,
) -> Result<u64> {
    let (storage, layout) = tensor.storage_and_layout();
    let slice = cuda_f32_slice(&storage, rank, description)?;
    let (start, end) = layout
        .contiguous_offsets()
        .ok_or_else(|| anyhow::anyhow!("rank {rank} {description} is not contiguous"))?;
    let ptr = slice.slice(start..end).device_ptr(stream).0;
    Ok(ptr)
}

fn tensor_from_f32_slice(
    slice: cudarc::driver::CudaSlice<f32>,
    device: &candle_core::CudaDevice,
    shape: &Shape,
) -> Tensor {
    let storage = candle_core::CudaStorage::wrap_cuda_slice(slice, device.clone());
    Tensor::from_storage(
        Storage::Cuda(storage),
        shape.clone(),
        BackpropOp::none(),
        false,
    )
}
