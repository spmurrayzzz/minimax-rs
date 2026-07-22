use candle_core::{Device, Shape, quantized::GgmlDType};
use candle_transformers::quantized_var_builder::{
    TensorParallelShard, TensorParallelSpec, VarBuilder,
};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

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

fn temp_file(label: &str) -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    std::env::temp_dir().join(format!(
        "minimax-tp-loader-{label}-{}-{}.gguf",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

fn qtensor(dtype: GgmlDType, shape: &[usize], raw: &[u8]) -> candle_core::quantized::QTensor {
    candle_core::quantized::ggml_file::qtensor_from_ggml(dtype, raw, shape.to_vec(), &Device::Cpu)
        .expect("synthetic QTensor")
}

fn write_tensor(path: &std::path::Path, name: &str, dtype: GgmlDType, shape: &[usize]) -> Vec<u8> {
    let raw = synthetic_bytes(dtype, shape);
    let tensor = qtensor(dtype, shape, &raw);
    let mut file = std::fs::File::create(path).expect("create synthetic GGUF");
    candle_core::quantized::gguf_file::write(&mut file, &[], &[(name, &tensor)])
        .expect("write synthetic GGUF");
    raw
}

fn write_duplicate_tensor(path: &std::path::Path, name: &str) {
    let shape = [2, 256];
    let raw0 = synthetic_bytes(GgmlDType::Q4K, &shape);
    let mut raw1 = raw0.clone();
    raw1[0] ^= 0xff;
    let tensor0 = qtensor(GgmlDType::Q4K, &shape, &raw0);
    let tensor1 = qtensor(GgmlDType::Q4K, &shape, &raw1);
    let mut file = std::fs::File::create(path).expect("create duplicate GGUF");
    candle_core::quantized::gguf_file::write(&mut file, &[], &[(name, &tensor0), (name, &tensor1)])
        .expect("write duplicate GGUF");
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
fn raw_tp_loader_splits_every_model_dtype_on_each_valid_axis() -> anyhow::Result<()> {
    let devices = [Device::Cpu, Device::Cpu];
    for &dtype in USED_DTYPES {
        let block = dtype.block_size();
        for (case, shape, axis) in [
            ("first", vec![4, block * 4], 0),
            ("middle", vec![2, 4, block * 2], 1),
            ("last", vec![2, 3, block * 4], 2),
        ] {
            let path = temp_file(case);
            let name = "synthetic.weight";
            let raw = write_tensor(&path, name, dtype, &shape);
            let builders = VarBuilder::from_gguf_tensor_parallel(
                std::slice::from_ref(&path),
                &[TensorParallelSpec::new(
                    name,
                    TensorParallelShard::Split { axis },
                )],
                &devices,
            )?;
            let tensors = [
                builders[0].get_no_shape(name)?,
                builders[1].get_no_shape(name)?,
            ];
            let raw_shards = [
                tensors[0].data()?.into_owned(),
                tensors[1].data()?.into_owned(),
            ];
            let mut expected_shape = shape.clone();
            expected_shape[axis] /= 2;
            assert_eq!(tensors[0].shape(), &Shape::from(expected_shape.clone()));
            assert_eq!(tensors[1].shape(), &Shape::from(expected_shape.clone()));
            assert_eq!(reassemble(dtype, &shape, axis, &raw_shards), raw);
            for rank in 0..2 {
                let builder = VarBuilder::from_gguf_tensor_parallel_rank(
                    std::slice::from_ref(&path),
                    &[TensorParallelSpec::new(
                        name,
                        TensorParallelShard::Split { axis },
                    )],
                    rank,
                    &Device::Cpu,
                )?;
                let tensor = builder.get_no_shape(name)?;
                assert_eq!(tensor.shape(), &Shape::from(expected_shape.clone()));
                assert_eq!(tensor.data()?.as_ref(), raw_shards[rank]);
            }
            let _ = std::fs::remove_file(path);
        }
    }
    Ok(())
}

#[test]
fn loader_rejects_invalid_missing_and_duplicate_definitions_before_loading() {
    let devices = [Device::Cpu, Device::Cpu];
    let source = temp_file("validation");
    write_tensor(&source, "odd.weight", GgmlDType::Q4K, &[3, 256]);

    let odd = VarBuilder::from_gguf_tensor_parallel(
        std::slice::from_ref(&source),
        &[TensorParallelSpec::new(
            "odd.weight",
            TensorParallelShard::Split { axis: 0 },
        )],
        &devices,
    )
    .err()
    .expect("odd split must fail");
    assert!(odd.to_string().contains("odd"));

    let unaligned_source = temp_file("unaligned");
    write_tensor(
        &unaligned_source,
        "unaligned.weight",
        GgmlDType::Q4K,
        &[2, 256],
    );
    let unaligned = VarBuilder::from_gguf_tensor_parallel(
        std::slice::from_ref(&unaligned_source),
        &[TensorParallelSpec::new(
            "unaligned.weight",
            TensorParallelShard::Split { axis: 1 },
        )],
        &devices,
    )
    .err()
    .expect("unaligned split must fail");
    assert!(unaligned.to_string().contains("boundary 128"));

    let missing = VarBuilder::from_gguf_tensor_parallel(
        std::slice::from_ref(&source),
        &[TensorParallelSpec::new(
            "missing.weight",
            TensorParallelShard::Replicated,
        )],
        &devices,
    )
    .err()
    .expect("missing tensor must fail");
    assert!(missing.to_string().contains("missing.weight"));

    let cross_duplicate = temp_file("cross-duplicate");
    write_tensor(&cross_duplicate, "odd.weight", GgmlDType::Q4K, &[3, 256]);
    let duplicate = VarBuilder::from_gguf_tensor_parallel(
        &[source.clone(), cross_duplicate.clone()],
        &[TensorParallelSpec::new(
            "odd.weight",
            TensorParallelShard::Replicated,
        )],
        &devices,
    )
    .err()
    .expect("cross-shard duplicate must fail");
    assert!(duplicate.to_string().contains("duplicate GGUF tensor"));

    let same_file_duplicate = temp_file("same-file-duplicate");
    write_duplicate_tensor(&same_file_duplicate, "duplicate.weight");
    let duplicate = VarBuilder::from_gguf_tensor_parallel(
        std::slice::from_ref(&same_file_duplicate),
        &[TensorParallelSpec::new(
            "duplicate.weight",
            TensorParallelShard::Replicated,
        )],
        &devices,
    )
    .err()
    .expect("same-shard duplicate must fail");
    assert!(
        duplicate
            .to_string()
            .contains("duplicate tensor definitions")
    );

    let duplicate_spec = TensorParallelSpec::new("odd.weight", TensorParallelShard::Replicated);
    let duplicate = VarBuilder::from_gguf_tensor_parallel(
        std::slice::from_ref(&source),
        &[duplicate_spec.clone(), duplicate_spec],
        &devices,
    )
    .err()
    .expect("duplicate spec must fail");
    assert!(
        duplicate
            .to_string()
            .contains("duplicate tensor-parallel specification")
    );

    for path in [
        source,
        unaligned_source,
        cross_duplicate,
        same_file_duplicate,
    ] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn replicated_and_rank_zero_only_placements_have_expected_visibility() -> anyhow::Result<()> {
    let path = temp_file("placements");
    let replicated_raw = synthetic_bytes(GgmlDType::F32, &[2, 4]);
    let rank_zero_raw = synthetic_bytes(GgmlDType::F32, &[4, 2]);
    let replicated = qtensor(GgmlDType::F32, &[2, 4], &replicated_raw);
    let rank_zero = qtensor(GgmlDType::F32, &[4, 2], &rank_zero_raw);
    let mut file = std::fs::File::create(&path)?;
    candle_core::quantized::gguf_file::write(
        &mut file,
        &[],
        &[
            ("replicated.weight", &replicated),
            ("rank_zero.weight", &rank_zero),
        ],
    )?;
    let builders = VarBuilder::from_gguf_tensor_parallel(
        std::slice::from_ref(&path),
        &[
            TensorParallelSpec::new("replicated.weight", TensorParallelShard::Replicated),
            TensorParallelSpec::new("rank_zero.weight", TensorParallelShard::Rank0Only),
        ],
        &[Device::Cpu, Device::Cpu],
    )?;
    assert_eq!(
        builders[0]
            .get_no_shape("replicated.weight")?
            .data()?
            .as_ref(),
        replicated_raw
    );
    assert_eq!(
        builders[1]
            .get_no_shape("replicated.weight")?
            .data()?
            .as_ref(),
        replicated_raw
    );
    assert_eq!(
        builders[0]
            .get_no_shape("rank_zero.weight")?
            .data()?
            .as_ref(),
        rank_zero_raw
    );
    assert!(!builders[1].contains_key("rank_zero.weight"));
    for rank in 0..2 {
        let builder = VarBuilder::from_gguf_tensor_parallel_rank(
            std::slice::from_ref(&path),
            &[
                TensorParallelSpec::new("replicated.weight", TensorParallelShard::Replicated),
                TensorParallelSpec::new("rank_zero.weight", TensorParallelShard::Rank0Only),
            ],
            rank,
            &Device::Cpu,
        )?;
        assert_eq!(
            builder.get_no_shape("replicated.weight")?.data()?.as_ref(),
            replicated_raw
        );
        assert_eq!(builder.contains_key("rank_zero.weight"), rank == 0);
    }
    assert!(
        VarBuilder::from_gguf_tensor_parallel_rank(
            std::slice::from_ref(&path),
            &[TensorParallelSpec::new(
                "replicated.weight",
                TensorParallelShard::Replicated,
            )],
            2,
            &Device::Cpu,
        )
        .is_err()
    );
    let _ = std::fs::remove_file(path);
    Ok(())
}
