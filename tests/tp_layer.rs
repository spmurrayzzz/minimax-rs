use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use minimax::{
    model_files::discover_gguf_shards,
    tensor_parallel::TensorParallelGroup,
    tensor_parallel_model::{HEAD_DIM, HIDDEN_SIZE, TensorParallelLayer, rotary_embeddings},
    tp_reference::FullLayerReference,
};

fn model_dir() -> Result<std::path::PathBuf> {
    std::env::var_os("MINIMAX_MODEL_DIR")
        .map(Into::into)
        .context("MINIMAX_MODEL_DIR must point to the four MiniMax GGUF shards")
}

#[test]
#[ignore = "requires MiniMax weights, two CUDA devices, and NCCL"]
fn layer_zero_keeps_rank_outputs_and_kv_caches_consistent() -> Result<()> {
    let shards = discover_gguf_shards(&model_dir()?)?;
    let devices = [Device::new_cuda(0)?, Device::new_cuda(1)?];
    for device in &devices {
        unsafe { device.as_cuda_device()?.disable_event_tracking() };
    }
    let mut collective = TensorParallelGroup::new(&devices)?;
    let ropes = rotary_embeddings(&devices)?;
    let mut layer = TensorParallelLayer::load(&shards, 0, &devices, &ropes)?;
    let mut reference = FullLayerReference::load(&shards, 0, &devices[0], ropes[0].clone())?;

    for query_len in [1usize, 5, 39, 512] {
        layer.reset();
        reference.reset();
        let source = Tensor::randn(0f32, 1f32, (1, query_len, HIDDEN_SIZE), &devices[0])?;
        let inputs = collective.broadcast_from_rank0(source.clone())?;
        let reference_routing = reference.routing_for_input(&source)?;
        let routing = layer.routing_for_inputs(&inputs)?;
        assert_eq!(
            reference_routing.0.flatten_all()?.to_vec1::<u32>()?,
            routing[0].0.flatten_all()?.to_vec1::<u32>()?,
            "router IDs differ from full reference for length {query_len}"
        );
        assert_close(
            &reference_routing.1,
            &routing[0].1,
            0.0,
            0.0,
            &format!("router weights length {query_len}"),
        )?;
        assert_eq!(
            routing[0].0.flatten_all()?.to_vec1::<u32>()?,
            routing[1].0.flatten_all()?.to_vec1::<u32>()?,
            "router ID mismatch for length {query_len}"
        );
        assert_eq!(
            routing[0].1.flatten_all()?.to_vec1::<f32>()?,
            routing[1].1.flatten_all()?.to_vec1::<f32>()?,
            "router weight mismatch for length {query_len}"
        );
        let expected = reference.forward(&source, 0, query_len > 1)?;
        let outputs = layer.forward(inputs, 0, query_len > 1, &mut collective)?;
        assert_close(
            &expected,
            &outputs[0],
            2.0,
            0.02,
            &format!("full layer prefill length {query_len}"),
        )?;
        let rank0 = outputs[0].flatten_all()?.to_vec1::<f32>()?;
        let rank1 = outputs[1].flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(rank0, rank1, "rank output mismatch for length {query_len}");
        assert!(rank0.iter().all(|value| value.is_finite()));
        assert_eq!(layer.cache_lengths(), [query_len, query_len]);
        assert_eq!(
            layer.cache_shapes()?,
            [
                Some(vec![1, 4, query_len, HEAD_DIM]),
                Some(vec![1, 4, query_len, HEAD_DIM]),
            ]
        );
    }

    layer.truncate_cache(5)?;
    assert_eq!(layer.cache_lengths(), [5, 5]);
    let invalid_truncate = layer
        .truncate_cache(6)
        .expect_err("cache growth through truncate must fail");
    assert!(invalid_truncate.to_string().contains("cannot truncate"));
    assert_eq!(layer.cache_lengths(), [5, 5]);
    let source = Tensor::zeros((1, 1, HIDDEN_SIZE), DType::F32, &devices[0])?;
    let inputs = collective.broadcast_from_rank0(source)?;
    drop(layer.forward(inputs, 5, false, &mut collective)?);
    assert_eq!(layer.cache_lengths(), [6, 6]);
    let source = Tensor::zeros((1, 1, HIDDEN_SIZE), DType::F32, &devices[0])?;
    let inputs = collective.broadcast_from_rank0(source)?;
    let wrong_position = layer
        .forward(inputs, 5, false, &mut collective)
        .expect_err("position/cache mismatch must fail");
    assert!(
        wrong_position
            .to_string()
            .contains("does not match TP cache")
    );
    assert_eq!(layer.cache_lengths(), [6, 6]);
    layer.reset();
    assert_eq!(layer.cache_lengths(), [0, 0]);

    reference.reset();
    let context = Tensor::randn(0f32, 1f32, (1, 39, HIDDEN_SIZE), &devices[0])?;
    let context_inputs = collective.broadcast_from_rank0(context.clone())?;
    drop(reference.forward(&context, 0, true)?);
    drop(layer.forward(context_inputs, 0, true, &mut collective)?);
    let token = Tensor::randn(0f32, 1f32, (1, 1, HIDDEN_SIZE), &devices[0])?;
    let token_inputs = collective.broadcast_from_rank0(token.clone())?;
    let expected = reference.forward(&token, 39, false)?;
    let actual = layer.forward(token_inputs, 39, false, &mut collective)?;
    assert_close(&expected, &actual[0], 0.05, 0.01, "cached decode")?;
    Ok(())
}

fn assert_close(
    expected: &Tensor,
    actual: &Tensor,
    max_tolerance: f32,
    mean_tolerance: f32,
    label: &str,
) -> Result<()> {
    let difference = (actual - expected)?.abs()?.to_dtype(DType::F32)?;
    let max_abs = difference.max_all()?.to_scalar::<f32>()?;
    let mean_abs = difference.sum_all()?.to_scalar::<f32>()? / difference.elem_count() as f32;
    anyhow::ensure!(
        max_abs <= max_tolerance && mean_abs <= mean_tolerance,
        "{label}: max_abs={max_abs}, mean_abs={mean_abs}, tolerances={max_tolerance}/{mean_tolerance}"
    );
    Ok(())
}
