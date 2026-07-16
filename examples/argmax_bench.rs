use anyhow::Result;
use candle_core::{D, Device, Tensor};
use std::time::Instant;

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let logits = Tensor::randn(0f32, 1f32, 200_064, &device)?;
    device.synchronize()?;
    let iterations = 100;

    let start = Instant::now();
    let mut cpu_id = 0;
    for _ in 0..iterations {
        let values = logits.to_vec1::<f32>()?;
        cpu_id = values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
    }
    let cpu_ms = start.elapsed().as_secs_f64() * 1e3 / iterations as f64;

    let start = Instant::now();
    let mut gpu_id = 0u32;
    for _ in 0..iterations {
        gpu_id = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
    }
    let gpu_ms = start.elapsed().as_secs_f64() * 1e3 / iterations as f64;

    println!("cpu={cpu_ms:.4} ms gpu={gpu_ms:.4} ms ids={cpu_id}/{gpu_id}");
    Ok(())
}
