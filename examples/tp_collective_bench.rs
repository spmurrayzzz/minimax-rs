use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use clap::Parser;
use minimax::tensor_parallel::TensorParallelGroup;
use std::time::Instant;

#[derive(Debug, Parser)]
struct Args {
    /// Timed all-reduces for each representative shape.
    #[arg(long, default_value_t = 100)]
    iterations: usize,
    /// Varying-size allocation/free stress iterations.
    #[arg(long, default_value_t = 2_000)]
    stress_iterations: usize,
}

const SHAPES: &[&[usize]] = &[&[1, 2], &[1, 1, 3_072], &[1, 39, 3_072], &[1, 512, 3_072]];

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let args = Args::parse();
    let devices = [Device::new_cuda(0)?, Device::new_cuda(1)?];
    for device in &devices {
        // Match inference, including the allocation/free behavior this
        // benchmark is intended to stress.
        unsafe { device.as_cuda_device()?.disable_event_tracking() };
    }
    let mut group = TensorParallelGroup::new(&devices)?;
    println!(
        "collective backend={} NCCL version={}",
        group.backend(),
        group.nccl_version()
    );

    for &dims in SHAPES {
        let count = dims.iter().product::<usize>();
        let rank0 = Tensor::from_vec(vec![1f32; count], dims, &devices[0])?;
        let rank1 = Tensor::from_vec(vec![2f32; count], dims, &devices[1])?;
        let outputs = group.all_reduce_sum([rank0.clone(), rank1.clone()])?;
        verify_constant(&outputs[0], 3.0)?;
        verify_constant(&outputs[1], 3.0)?;

        for _ in 0..10 {
            drop(group.all_reduce_sum([rank0.clone(), rank1.clone()])?);
        }
        devices[0].synchronize()?;
        devices[1].synchronize()?;
        let start = Instant::now();
        for _ in 0..args.iterations {
            drop(group.all_reduce_sum([rank0.clone(), rank1.clone()])?);
        }
        devices[0].synchronize()?;
        devices[1].synchronize()?;
        let elapsed = start.elapsed();
        println!(
            "all_reduce shape={dims:?} elements={count} dtype=f32 average={:.3}us",
            elapsed.as_secs_f64() * 1e6 / args.iterations as f64
        );
    }

    let source = Tensor::from_vec(
        (0..3_072).map(|value| value as f32).collect(),
        (1, 1, 3_072),
        &devices[0],
    )?;
    let broadcast = group.broadcast_from_rank0(source.clone())?;
    for (rank, output) in broadcast.iter().enumerate() {
        let values = output.flatten_all()?.to_vec1::<f32>()?;
        for (index, value) in values.into_iter().enumerate() {
            anyhow::ensure!(
                value == index as f32,
                "broadcast mismatch on rank {rank} at {index}: {value}"
            );
        }
    }
    for _ in 0..10 {
        drop(group.broadcast_from_rank0(source.clone())?);
    }
    devices[0].synchronize()?;
    devices[1].synchronize()?;
    let start = Instant::now();
    for _ in 0..args.iterations {
        drop(group.broadcast_from_rank0(source.clone())?);
    }
    devices[0].synchronize()?;
    devices[1].synchronize()?;
    println!(
        "broadcast shape=[1, 1, 3072] average={:.3}us",
        start.elapsed().as_secs_f64() * 1e6 / args.iterations as f64
    );

    let stress_sizes = [2usize, 3_072, 39 * 3_072, 512 * 3_072, 17, 8_191];
    for iteration in 0..args.stress_iterations {
        let count = stress_sizes[iteration % stress_sizes.len()];
        let rank0 = Tensor::from_vec(vec![iteration as f32; count], count, &devices[0])?;
        let rank1 = Tensor::from_vec(vec![1f32; count], count, &devices[1])?;
        let outputs = group.all_reduce_sum([rank0, rank1])?;
        if iteration % 251 == 0 {
            let expected = iteration as f32 + 1.0;
            verify_constant(&outputs[0], expected)
                .with_context(|| format!("rank 0 stress iteration {iteration}"))?;
            verify_constant(&outputs[1], expected)
                .with_context(|| format!("rank 1 stress iteration {iteration}"))?;
        }
    }
    devices[0].synchronize()?;
    devices[1].synchronize()?;
    println!(
        "completed {} varying-size stress iterations",
        args.stress_iterations
    );
    Ok(())
}

fn verify_constant(tensor: &Tensor, expected: f32) -> Result<()> {
    for (index, value) in tensor
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .enumerate()
    {
        anyhow::ensure!(
            value == expected,
            "element {index}: got {value}, expected {expected}"
        );
    }
    Ok(())
}
