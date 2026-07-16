use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::kv_cache::{ConcatKvCache, KvCache};
use std::time::Instant;

const LAYERS: usize = 62;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const PREFILL: usize = 512;
const DECODE: usize = 256;

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let prefill = Tensor::zeros((1, KV_HEADS, PREFILL, HEAD_DIM), DType::F16, &device)?;
    let token = Tensor::zeros((1, KV_HEADS, 1, HEAD_DIM), DType::F16, &device)?;

    let mut concat = (0..LAYERS)
        .map(|_| ConcatKvCache::new(2))
        .collect::<Vec<_>>();
    for cache in &mut concat {
        cache.append(&prefill, &prefill)?;
    }
    device.synchronize()?;
    let start = Instant::now();
    for _ in 0..DECODE {
        for cache in &mut concat {
            cache.append(&token, &token)?;
        }
    }
    device.synchronize()?;
    println!(
        "concat: {:?} ({:.3} ms/token)",
        start.elapsed(),
        start.elapsed().as_secs_f64() * 1e3 / DECODE as f64
    );
    drop(concat);
    device.synchronize()?;

    let mut chunked = (0..LAYERS)
        .map(|_| KvCache::new(2, PREFILL))
        .collect::<Vec<_>>();
    for cache in &mut chunked {
        cache.append(&prefill, &prefill)?;
    }
    device.synchronize()?;
    let start = Instant::now();
    for _ in 0..DECODE {
        for cache in &mut chunked {
            cache.append(&token, &token)?;
        }
    }
    device.synchronize()?;
    println!(
        "chunked: {:?} ({:.3} ms/token)",
        start.elapsed(),
        start.elapsed().as_secs_f64() * 1e3 / DECODE as f64
    );

    Ok(())
}
