use candle_core::{Device, Tensor};
use minimax::tensor_parallel::TensorParallelGroup;

fn devices() -> candle_core::Result<[Device; 2]> {
    let devices = [Device::new_cuda(0)?, Device::new_cuda(1)?];
    for device in &devices {
        unsafe { device.as_cuda_device()?.disable_event_tracking() };
    }
    Ok(devices)
}

#[test]
#[ignore = "requires two CUDA devices and NCCL"]
fn f32_pair_collectives_match_cpu_references_and_survive_reallocation_stress() -> anyhow::Result<()>
{
    let devices = devices()?;
    let mut group = TensorParallelGroup::new(&devices)?;
    for dims in [
        vec![1, 2],
        vec![1, 1, 3_072],
        vec![1, 39, 3_072],
        vec![1, 512, 3_072],
    ] {
        let count = dims.iter().product::<usize>();
        let left_values = (0..count)
            .map(|index| index as f32 * 0.25)
            .collect::<Vec<_>>();
        let right_values = (0..count)
            .map(|index| 7.0 - index as f32 * 0.125)
            .collect::<Vec<_>>();
        let expected = left_values
            .iter()
            .zip(&right_values)
            .map(|(left, right)| left + right)
            .collect::<Vec<_>>();
        let outputs = group.all_reduce_sum([
            Tensor::from_vec(left_values, dims.clone(), &devices[0])?,
            Tensor::from_vec(right_values, dims.clone(), &devices[1])?,
        ])?;
        for (rank, output) in outputs.iter().enumerate() {
            let actual = output.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(
                actual, expected,
                "all-reduce mismatch on rank {rank} for {dims:?}"
            );
        }
    }

    for count in [3_072, 39 * 3_072] {
        let source_values = (0..count).map(|index| index as f32).collect::<Vec<_>>();
        let outputs = group.broadcast_from_rank0(Tensor::from_vec(
            source_values.clone(),
            count,
            &devices[0],
        )?)?;
        for (rank, output) in outputs.iter().enumerate() {
            assert_eq!(
                output.flatten_all()?.to_vec1::<f32>()?,
                source_values,
                "broadcast mismatch on rank {rank} for {count} elements"
            );
        }
    }
    let broadcast_sizes = [1usize, 17, 3_072, 8_191];
    for iteration in 0..500usize {
        let count = broadcast_sizes[iteration % broadcast_sizes.len()];
        let value = iteration as f32;
        let outputs = group.broadcast_from_rank0(Tensor::from_vec(
            vec![value; count],
            count,
            &devices[0],
        )?)?;
        if iteration % 37 == 0 {
            for (rank, output) in outputs.iter().enumerate() {
                assert!(
                    output
                        .flatten_all()?
                        .to_vec1::<f32>()?
                        .iter()
                        .all(|&actual| actual == value),
                    "broadcast stress mismatch on rank {rank}, iteration {iteration}"
                );
            }
        }
    }

    let sizes = [1usize, 2, 17, 3_072, 39 * 3_072, 512 * 3_072, 8_191];
    for iteration in 0..2_000usize {
        let count = sizes[iteration % sizes.len()];
        let outputs = group.all_reduce_sum([
            Tensor::from_vec(vec![iteration as f32; count], count, &devices[0])?,
            Tensor::from_vec(vec![0.5f32; count], count, &devices[1])?,
        ])?;
        if iteration % 199 == 0 {
            let expected = iteration as f32 + 0.5;
            for (rank, output) in outputs.iter().enumerate() {
                let actual = output.flatten_all()?.to_vec1::<f32>()?;
                assert!(
                    actual.iter().all(|&value| value == expected),
                    "stress mismatch on rank {rank}, iteration {iteration}"
                );
            }
        }
    }
    devices[0].synchronize()?;
    devices[1].synchronize()?;
    Ok(())
}
