//! Varbuilder for Loading gguf files
//!
//! VarBuilder is a utility to store quantized tensors from a [GGUF model file](https://huggingface.co/docs/hub/gguf).
//! These tensors can be loaded from disk using `from_gguf` or from an in-memory
//! buffer using `from_gguf_buffer`.

use candle::quantized::{GgmlDType, QTensor};
use candle::{Device, Result, Shape};
use std::{
    collections::{HashMap, HashSet},
    io::{Read, Seek},
    path::PathBuf,
    sync::Arc,
};

/// Placement of one GGUF tensor in a fixed two-rank tensor-parallel graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorParallelShard {
    /// Upload the complete raw tensor to both ranks.
    Replicated,
    /// Split one logical axis into two equal, raw-block-preserving shards.
    Split { axis: usize },
    /// Upload only to rank 0 (embedding/final-output tensors, for example).
    Rank0Only,
}

/// Exact tensor name and its two-rank placement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorParallelSpec {
    pub name: String,
    pub shard: TensorParallelShard,
}

impl TensorParallelSpec {
    pub fn new(name: impl Into<String>, shard: TensorParallelShard) -> Self {
        Self {
            name: name.into(),
            shard,
        }
    }
}

#[derive(Debug)]
struct LocatedTensor {
    path: PathBuf,
    shape: Shape,
    dtype: GgmlDType,
    raw_offset: u64,
    byte_len: usize,
}

fn locate_tensor_parallel_tensors<P: AsRef<std::path::Path>>(
    paths: &[P],
    specs: &[TensorParallelSpec],
) -> Result<HashMap<String, LocatedTensor>> {
    if specs.is_empty() {
        candle::bail!("tensor-parallel GGUF specification must not be empty")
    }

    let mut requested = HashSet::with_capacity(specs.len());
    for spec in specs {
        if !requested.insert(spec.name.as_str()) {
            candle::bail!("duplicate tensor-parallel specification for {}", spec.name)
        }
    }

    // Locate and validate every selected tensor before allocating on a CUDA
    // device. Shards may split a layer at arbitrary file boundaries.
    let mut locations = HashMap::with_capacity(specs.len());
    for path in paths {
        let path = path.as_ref();
        let mut file = std::fs::File::open(path)?;
        let declared_tensors = gguf_declared_tensor_count(&mut file)?;
        file.seek(std::io::SeekFrom::Start(0))?;
        let content = candle::quantized::gguf_file::Content::read(&mut file)?;
        if content.tensor_infos.len() != declared_tensors {
            candle::bail!(
                "GGUF shard {} declares {declared_tensors} tensors but contains only {} unique tensor names; duplicate tensor definitions are not supported",
                path.display(),
                content.tensor_infos.len()
            )
        }
        let file_len = file.metadata()?.len();
        for (name, info) in &content.tensor_infos {
            if !requested.contains(name.as_str()) {
                continue;
            }
            if locations.contains_key(name) {
                candle::bail!(
                    "duplicate GGUF tensor {name} found while scanning {}",
                    path.display()
                )
            }
            let elem_count = checked_elem_count(&info.shape, name)?;
            let block_size = info.ggml_dtype.block_size();
            if !elem_count.is_multiple_of(block_size) {
                candle::bail!(
                    "GGUF tensor {name} has {elem_count} elements, not divisible by {:?} block size {block_size}",
                    info.ggml_dtype
                )
            }
            let byte_len = (elem_count / block_size)
                .checked_mul(info.ggml_dtype.type_size())
                .ok_or_else(|| {
                    candle::Error::Msg(format!("GGUF tensor {name} byte size overflows usize"))
                })?;
            let raw_offset = content
                .tensor_data_offset
                .checked_add(info.offset)
                .ok_or_else(|| {
                    candle::Error::Msg(format!("GGUF tensor {name} file offset overflows u64"))
                })?;
            let raw_end = raw_offset.checked_add(byte_len as u64).ok_or_else(|| {
                candle::Error::Msg(format!("GGUF tensor {name} file range overflows u64"))
            })?;
            if raw_end > file_len {
                candle::bail!(
                    "GGUF tensor {name} needs bytes {raw_offset}..{raw_end}, but {} is only {file_len} bytes",
                    path.display()
                )
            }
            locations.insert(
                name.clone(),
                LocatedTensor {
                    path: path.to_owned(),
                    shape: info.shape.clone(),
                    dtype: info.ggml_dtype,
                    raw_offset,
                    byte_len,
                },
            );
        }
    }

    for spec in specs {
        let location = locations
            .get(&spec.name)
            .ok_or_else(|| candle::Error::Msg(format!("missing GGUF tensor {}", spec.name)))?;
        if let TensorParallelShard::Split { axis } = spec.shard {
            validate_split(&spec.name, location.dtype, &location.shape, axis)?;
        }
    }
    Ok(locations)
}

// VarBuilder specialized for QTensors
#[derive(Clone)]
pub struct VarBuilder {
    data: Arc<std::collections::HashMap<String, Arc<QTensor>>>,
    path: Vec<String>,
    device: Device,
}

impl VarBuilder {
    pub fn from_gguf<P: AsRef<std::path::Path>>(p: P, device: &Device) -> Result<Self> {
        let mut file = std::fs::File::open(p)?;
        let content = candle::quantized::gguf_file::Content::read(&mut file)?;
        let mut data = std::collections::HashMap::new();
        for tensor_name in content.tensor_infos.keys() {
            let tensor = content.tensor(&mut file, tensor_name, device)?;
            data.insert(tensor_name.to_string(), Arc::new(tensor));
        }
        Ok(Self {
            data: Arc::new(data),
            path: Vec::new(),
            device: device.clone(),
        })
    }

    /// Load only tensors whose names start with one of `prefixes`. This is
    /// important for split GGUF models: shard boundaries may bisect a layer,
    /// and loading an entire shard onto one GPU would waste VRAM.
    pub fn from_gguf_selected<P: AsRef<std::path::Path>>(
        paths: &[P],
        prefixes: &[String],
        device: &Device,
    ) -> Result<Self> {
        let mut data = std::collections::HashMap::new();
        for path in paths {
            let mut file = std::fs::File::open(path)?;
            let content = candle::quantized::gguf_file::Content::read(&mut file)?;
            for tensor_name in content.tensor_infos.keys() {
                if prefixes.iter().any(|p| tensor_name.starts_with(p)) {
                    let tensor = content.tensor(&mut file, tensor_name, device)?;
                    data.insert(tensor_name.to_string(), Arc::new(tensor));
                }
            }
        }
        Ok(Self {
            data: Arc::new(data),
            path: Vec::new(),
            device: device.clone(),
        })
    }

    /// Loads an exact set of tensors into a fixed pair of rank-local builders.
    ///
    /// All files and shard specifications are validated before the first
    /// tensor is uploaded. Each selected tensor is then read once, split in
    /// host memory on GGML block boundaries, and uploaded without any
    /// dequantize/re-quantize round trip.
    pub fn from_gguf_tensor_parallel<P: AsRef<std::path::Path>>(
        paths: &[P],
        specs: &[TensorParallelSpec],
        devices: &[Device; 2],
    ) -> Result<[Self; 2]> {
        let locations = locate_tensor_parallel_tensors(paths, specs)?;

        let mut rank_data = [HashMap::new(), HashMap::new()];
        for spec in specs {
            let location = locations
                .get(&spec.name)
                .expect("all tensor locations were validated");
            let raw = read_raw_tensor(location)?;
            match spec.shard {
                TensorParallelShard::Replicated => {
                    let rank0 = candle::quantized::ggml_file::qtensor_from_ggml(
                        location.dtype,
                        &raw,
                        location.shape.dims().to_vec(),
                        &devices[0],
                    )?;
                    let rank1 = candle::quantized::ggml_file::qtensor_from_ggml(
                        location.dtype,
                        &raw,
                        location.shape.dims().to_vec(),
                        &devices[1],
                    )?;
                    rank_data[0].insert(spec.name.clone(), Arc::new(rank0));
                    rank_data[1].insert(spec.name.clone(), Arc::new(rank1));
                }
                TensorParallelShard::Split { axis } => {
                    let (raw_shards, shard_shapes) = split_quantized_bytes(
                        &spec.name,
                        location.dtype,
                        &location.shape,
                        axis,
                        &raw,
                    )?;
                    drop(raw);
                    for rank in 0..2 {
                        let tensor = candle::quantized::ggml_file::qtensor_from_ggml(
                            location.dtype,
                            &raw_shards[rank],
                            shard_shapes[rank].dims().to_vec(),
                            &devices[rank],
                        )?;
                        rank_data[rank].insert(spec.name.clone(), Arc::new(tensor));
                    }
                }
                TensorParallelShard::Rank0Only => {
                    let tensor = candle::quantized::ggml_file::qtensor_from_ggml(
                        location.dtype,
                        &raw,
                        location.shape.dims().to_vec(),
                        &devices[0],
                    )?;
                    rank_data[0].insert(spec.name.clone(), Arc::new(tensor));
                }
            }
        }

        let [rank0, rank1] = rank_data;
        Ok([
            Self {
                data: Arc::new(rank0),
                path: Vec::new(),
                device: devices[0].clone(),
            },
            Self {
                data: Arc::new(rank1),
                path: Vec::new(),
                device: devices[1].clone(),
            },
        ])
    }

    /// Loads one rank of a fixed two-rank tensor-parallel specification.
    ///
    /// This is the process-isolated counterpart to
    /// [`Self::from_gguf_tensor_parallel`]: the caller creates exactly one
    /// CUDA device and never materializes the other rank's tensors there.
    pub fn from_gguf_tensor_parallel_rank<P: AsRef<std::path::Path>>(
        paths: &[P],
        specs: &[TensorParallelSpec],
        rank: usize,
        device: &Device,
    ) -> Result<Self> {
        if rank >= 2 {
            candle::bail!("tensor-parallel GGUF rank must be 0 or 1, got {rank}")
        }
        let locations = locate_tensor_parallel_tensors(paths, specs)?;
        let mut data = HashMap::with_capacity(specs.len());
        for spec in specs {
            if rank == 1 && spec.shard == TensorParallelShard::Rank0Only {
                continue;
            }
            let location = locations
                .get(&spec.name)
                .expect("all tensor locations were validated");
            let raw = read_raw_tensor(location)?;
            let tensor = match spec.shard {
                TensorParallelShard::Replicated | TensorParallelShard::Rank0Only => {
                    candle::quantized::ggml_file::qtensor_from_ggml(
                        location.dtype,
                        &raw,
                        location.shape.dims().to_vec(),
                        device,
                    )?
                }
                TensorParallelShard::Split { axis } => {
                    let (raw_shards, shard_shapes) = split_quantized_bytes(
                        &spec.name,
                        location.dtype,
                        &location.shape,
                        axis,
                        &raw,
                    )?;
                    candle::quantized::ggml_file::qtensor_from_ggml(
                        location.dtype,
                        &raw_shards[rank],
                        shard_shapes[rank].dims().to_vec(),
                        device,
                    )?
                }
            };
            data.insert(spec.name.clone(), Arc::new(tensor));
        }
        Ok(Self {
            data: Arc::new(data),
            path: Vec::new(),
            device: device.clone(),
        })
    }

    pub fn from_gguf_buffer(buffer: &[u8], device: &Device) -> Result<Self> {
        let mut cursor = std::io::Cursor::new(buffer);
        let content = candle::quantized::gguf_file::Content::read(&mut cursor)?;
        let mut data = std::collections::HashMap::new();
        for tensor_name in content.tensor_infos.keys() {
            let tensor = content.tensor(&mut cursor, tensor_name, device)?;
            data.insert(tensor_name.to_string(), Arc::new(tensor));
        }
        Ok(Self {
            data: Arc::new(data),
            path: Vec::new(),
            device: device.clone(),
        })
    }

    pub fn pp<S: ToString>(&self, s: S) -> Self {
        let mut path = self.path.clone();
        path.push(s.to_string());
        Self {
            data: self.data.clone(),
            path,
            device: self.device.clone(),
        }
    }

    fn path(&self, tensor_name: &str) -> String {
        if self.path.is_empty() {
            tensor_name.to_string()
        } else {
            [&self.path.join("."), tensor_name].join(".")
        }
    }

    pub fn get<S: Into<Shape>>(&self, s: S, name: &str) -> Result<Arc<QTensor>> {
        let path = self.path(name);
        match self.data.get(&path) {
            None => {
                candle::bail!("cannot find tensor {path}")
            }
            Some(qtensor) => {
                let shape = s.into();
                if qtensor.shape() != &shape {
                    candle::bail!(
                        "shape mismatch for {name}, got {:?}, expected {shape:?}",
                        qtensor.shape()
                    )
                }
                Ok(qtensor.clone())
            }
        }
    }

    pub fn get_no_shape(&self, name: &str) -> Result<Arc<QTensor>> {
        let path = self.path(name);
        match self.data.get(&path) {
            None => {
                candle::bail!("cannot find tensor {name}")
            }
            Some(qtensor) => Ok(qtensor.clone()),
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }
}

fn gguf_declared_tensor_count<R: Read + Seek>(reader: &mut R) -> Result<usize> {
    reader.seek(std::io::SeekFrom::Start(0))?;
    let mut prefix = [0u8; 8];
    reader.read_exact(&mut prefix)?;
    let magic = u32::from_le_bytes(prefix[..4].try_into().expect("four-byte GGUF magic"));
    if !matches!(magic, 0x4655_4747 | 0x4747_5546) {
        candle::bail!("unknown GGUF magic 0x{magic:08x}")
    }
    let version = u32::from_le_bytes(prefix[4..].try_into().expect("four-byte GGUF version"));
    let declared = match version {
        1 => {
            let mut count = [0u8; 4];
            reader.read_exact(&mut count)?;
            u32::from_le_bytes(count) as u64
        }
        2 | 3 => {
            let mut count = [0u8; 8];
            reader.read_exact(&mut count)?;
            u64::from_le_bytes(count)
        }
        _ => candle::bail!("unsupported GGUF version {version}"),
    };
    usize::try_from(declared).map_err(|_| {
        candle::Error::Msg(format!(
            "GGUF tensor count {declared} does not fit in usize"
        ))
    })
}

fn checked_elem_count(shape: &Shape, name: &str) -> Result<usize> {
    shape.dims().iter().try_fold(1usize, |count, &dim| {
        count.checked_mul(dim).ok_or_else(|| {
            candle::Error::Msg(format!("GGUF tensor {name} element count overflows usize"))
        })
    })
}

fn read_raw_tensor(location: &LocatedTensor) -> Result<Vec<u8>> {
    let mut file = std::fs::File::open(&location.path)?;
    file.seek(std::io::SeekFrom::Start(location.raw_offset))?;
    let mut raw = vec![0u8; location.byte_len];
    file.read_exact(&mut raw)?;
    Ok(raw)
}

fn validate_split(name: &str, dtype: GgmlDType, shape: &Shape, axis: usize) -> Result<()> {
    let dims = shape.dims();
    if axis >= dims.len() {
        candle::bail!(
            "cannot split GGUF tensor {name} with dtype {dtype:?} and shape {shape:?} on axis {axis}"
        )
    }
    let dim = dims[axis];
    if !dim.is_multiple_of(2) {
        candle::bail!(
            "cannot split GGUF tensor {name} with dtype {dtype:?} on axis {axis}: dimension {dim} is odd"
        )
    }
    let boundary = dim / 2;
    if axis + 1 == dims.len() && !boundary.is_multiple_of(dtype.block_size()) {
        candle::bail!(
            "unaligned GGUF split for tensor {name}: dtype {dtype:?}, axis {axis}, boundary {boundary}, block size {}",
            dtype.block_size()
        )
    }
    Ok(())
}

/// Splits the row-major GGML block lattice without changing any quantized
/// block bytes. The last logical dimension is represented as a dimension of
/// whole GGML blocks; all preceding dimensions are unchanged.
fn split_quantized_bytes(
    name: &str,
    dtype: GgmlDType,
    shape: &Shape,
    axis: usize,
    raw: &[u8],
) -> Result<([Vec<u8>; 2], [Shape; 2])> {
    validate_split(name, dtype, shape, axis)?;
    let mut physical_dims = shape.dims().to_vec();
    let last = physical_dims.len() - 1;
    physical_dims[last] /= dtype.block_size();

    let physical_axis = physical_dims[axis];
    if !physical_axis.is_multiple_of(2) {
        // This can only differ from the logical evenness check on the final
        // axis, where it provides a clearer defensive error.
        candle::bail!(
            "unaligned GGUF split for tensor {name}: dtype {dtype:?}, axis {axis}, physical block dimension {physical_axis}"
        )
    }
    let outer = physical_dims[..axis]
        .iter()
        .try_fold(1usize, |value, &dim| value.checked_mul(dim))
        .ok_or_else(|| {
            candle::Error::Msg(format!(
                "GGUF tensor {name} outer split size overflows usize"
            ))
        })?;
    let inner = physical_dims[axis + 1..]
        .iter()
        .try_fold(1usize, |value, &dim| value.checked_mul(dim))
        .ok_or_else(|| {
            candle::Error::Msg(format!(
                "GGUF tensor {name} inner split size overflows usize"
            ))
        })?;
    let type_size = dtype.type_size();
    let expected = outer
        .checked_mul(physical_axis)
        .and_then(|value| value.checked_mul(inner))
        .and_then(|value| value.checked_mul(type_size))
        .ok_or_else(|| {
            candle::Error::Msg(format!("GGUF tensor {name} raw split size overflows usize"))
        })?;
    if raw.len() != expected {
        candle::bail!(
            "GGUF tensor {name} raw byte length mismatch: got {}, expected {expected} for dtype {dtype:?} and shape {shape:?}",
            raw.len()
        )
    }

    let rank_axis = physical_axis / 2;
    let rank_chunk_bytes = rank_axis
        .checked_mul(inner)
        .and_then(|value| value.checked_mul(type_size))
        .ok_or_else(|| {
            candle::Error::Msg(format!(
                "GGUF tensor {name} shard byte size overflows usize"
            ))
        })?;
    let outer_stride_bytes = rank_chunk_bytes.checked_mul(2).ok_or_else(|| {
        candle::Error::Msg(format!("GGUF tensor {name} split stride overflows usize"))
    })?;
    let mut shards = [
        Vec::with_capacity(raw.len() / 2),
        Vec::with_capacity(raw.len() / 2),
    ];
    for outer_index in 0..outer {
        let start = outer_index * outer_stride_bytes;
        shards[0].extend_from_slice(&raw[start..start + rank_chunk_bytes]);
        shards[1].extend_from_slice(&raw[start + rank_chunk_bytes..start + outer_stride_bytes]);
    }

    let mut shard_dims = shape.dims().to_vec();
    shard_dims[axis] /= 2;
    let shard_shape = Shape::from(shard_dims);
    Ok((shards, [shard_shape.clone(), shard_shape]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const USED_DTYPES: &[GgmlDType] = &[
        GgmlDType::F32,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
        GgmlDType::Q8_0,
    ];

    fn synthetic_bytes(dtype: GgmlDType, shape: &[usize]) -> Vec<u8> {
        let elements = shape.iter().product::<usize>();
        assert!(elements.is_multiple_of(dtype.block_size()));
        let len = elements / dtype.block_size() * dtype.type_size();
        (0..len)
            .map(|index| ((index * 131 + 17) % 251) as u8)
            .collect()
    }

    fn reassemble(
        dtype: GgmlDType,
        original_shape: &[usize],
        axis: usize,
        shards: &[Vec<u8>; 2],
    ) -> Vec<u8> {
        let mut physical = original_shape.to_vec();
        let last = physical.len() - 1;
        physical[last] /= dtype.block_size();
        let outer = physical[..axis].iter().product::<usize>();
        let inner = physical[axis + 1..].iter().product::<usize>();
        let chunk = physical[axis] / 2 * inner * dtype.type_size();
        let mut output = Vec::with_capacity(shards[0].len() + shards[1].len());
        for outer_index in 0..outer {
            let start = outer_index * chunk;
            output.extend_from_slice(&shards[0][start..start + chunk]);
            output.extend_from_slice(&shards[1][start..start + chunk]);
        }
        output
    }

    #[test]
    fn raw_splits_reassemble_byte_for_byte_on_first_middle_and_last_axes() {
        for &dtype in USED_DTYPES {
            let block = dtype.block_size();
            for (shape, axis) in [
                (vec![4, block * 4], 0),
                (vec![2, 4, block * 2], 1),
                (vec![2, 3, block * 4], 2),
            ] {
                let raw = synthetic_bytes(dtype, &shape);
                let (shards, shard_shapes) = split_quantized_bytes(
                    "synthetic.weight",
                    dtype,
                    &Shape::from(shape.clone()),
                    axis,
                    &raw,
                )
                .unwrap();
                let mut expected_shape = shape.clone();
                expected_shape[axis] /= 2;
                assert_eq!(shard_shapes[0].dims(), expected_shape);
                assert_eq!(shard_shapes[1].dims(), expected_shape);
                assert_eq!(reassemble(dtype, &shape, axis, &shards), raw);
            }
        }
    }

    #[test]
    fn minimax_tp2_weight_shapes_match_the_execution_graph() {
        let cases = [
            (GgmlDType::Q8_0, vec![6_144, 3_072], 0, vec![3_072, 3_072]),
            (GgmlDType::Q8_0, vec![1_024, 3_072], 0, vec![512, 3_072]),
            (GgmlDType::F32, vec![6_144], 0, vec![3_072]),
            (GgmlDType::F32, vec![1_024], 0, vec![512]),
            (GgmlDType::Q8_0, vec![3_072, 6_144], 1, vec![3_072, 3_072]),
            (
                GgmlDType::Q4K,
                vec![256, 1_536, 3_072],
                1,
                vec![256, 768, 3_072],
            ),
            (
                GgmlDType::Q5K,
                vec![256, 3_072, 1_536],
                2,
                vec![256, 3_072, 768],
            ),
        ];
        for (dtype, shape, axis, expected) in cases {
            let raw = synthetic_bytes(dtype, &shape);
            let (_, shard_shapes) = split_quantized_bytes(
                "blk.0.synthetic.weight",
                dtype,
                &Shape::from(shape),
                axis,
                &raw,
            )
            .unwrap();
            assert_eq!(shard_shapes[0].dims(), expected);
            assert_eq!(shard_shapes[1].dims(), expected);
        }
    }

    #[test]
    fn invalid_split_boundaries_are_rejected_with_tensor_context() {
        let odd = split_quantized_bytes(
            "odd.weight",
            GgmlDType::Q4K,
            &Shape::from(vec![3, 256]),
            0,
            &synthetic_bytes(GgmlDType::Q4K, &[3, 256]),
        )
        .unwrap_err();
        assert!(odd.to_string().contains("odd.weight"));
        assert!(odd.to_string().contains("odd"));

        let unaligned = split_quantized_bytes(
            "unaligned.weight",
            GgmlDType::Q4K,
            &Shape::from(vec![2, 256]),
            1,
            &synthetic_bytes(GgmlDType::Q4K, &[2, 256]),
        )
        .unwrap_err();
        let message = unaligned.to_string();
        assert!(message.contains("unaligned.weight"));
        assert!(message.contains("Q4K"));
        assert!(message.contains("axis 1"));
        assert!(message.contains("boundary 128"));

        let bad_axis =
            validate_split("axis.weight", GgmlDType::F32, &Shape::from(vec![2, 4]), 2).unwrap_err();
        assert!(bad_axis.to_string().contains("axis 2"));
    }

    fn temp_file(label: &str) -> PathBuf {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        std::env::temp_dir().join(format!(
            "candle-tp-loader-{label}-{}-{}.gguf",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn write_tensor(path: &std::path::Path, name: &str, dtype: GgmlDType, shape: &[usize]) {
        let raw = synthetic_bytes(dtype, shape);
        let tensor = candle::quantized::ggml_file::qtensor_from_ggml(
            dtype,
            &raw,
            shape.to_vec(),
            &Device::Cpu,
        )
        .unwrap();
        let mut file = std::fs::File::create(path).unwrap();
        candle::quantized::gguf_file::write(&mut file, &[], &[(name, &tensor)]).unwrap();
    }

    #[test]
    fn paired_loader_preserves_raw_shards_and_rejects_missing_or_duplicate_tensors() {
        let path = temp_file("source");
        let duplicate_path = temp_file("duplicate");
        let name = "blk.0.ffn_gate_exps.weight";
        let shape = [2, 4, 512];
        write_tensor(&path, name, GgmlDType::Q4K, &shape);
        write_tensor(&duplicate_path, name, GgmlDType::Q4K, &shape);
        let devices = [Device::Cpu, Device::Cpu];
        let spec = TensorParallelSpec::new(name, TensorParallelShard::Split { axis: 1 });

        let builders = VarBuilder::from_gguf_tensor_parallel(
            std::slice::from_ref(&path),
            std::slice::from_ref(&spec),
            &devices,
        )
        .unwrap();
        let original = synthetic_bytes(GgmlDType::Q4K, &shape);
        let (expected, _) = split_quantized_bytes(
            name,
            GgmlDType::Q4K,
            &Shape::from(shape.to_vec()),
            1,
            &original,
        )
        .unwrap();
        for rank in 0..2 {
            let tensor = builders[rank].get((2, 2, 512), name).unwrap();
            assert_eq!(tensor.data().unwrap().as_ref(), expected[rank]);
        }

        let missing = VarBuilder::from_gguf_tensor_parallel(
            std::slice::from_ref(&path),
            &[TensorParallelSpec::new(
                "missing.weight",
                TensorParallelShard::Replicated,
            )],
            &devices,
        )
        .err()
        .expect("missing tensor must fail");
        assert!(missing.to_string().contains("missing.weight"));

        let duplicate = VarBuilder::from_gguf_tensor_parallel(
            &[path.clone(), duplicate_path.clone()],
            std::slice::from_ref(&spec),
            &devices,
        )
        .err()
        .expect("duplicate GGUF tensor must fail");
        assert!(duplicate.to_string().contains("duplicate GGUF tensor"));

        let duplicate_spec = VarBuilder::from_gguf_tensor_parallel(
            std::slice::from_ref(&path),
            &[spec.clone(), spec],
            &devices,
        )
        .err()
        .expect("duplicate specification must fail");
        assert!(duplicate_spec
            .to_string()
            .contains("duplicate tensor-parallel specification"));

        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(duplicate_path);
    }
}
