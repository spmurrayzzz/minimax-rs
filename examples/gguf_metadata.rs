use anyhow::Result;
use candle_core::quantized::gguf_file;
use clap::Parser;
use minimax::model_files::discover_gguf_shards;
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Debug, Parser)]
struct Args {
    /// Directory containing the four split GGUF files.
    #[arg(long, env = "MINIMAX_MODEL_DIR", value_name = "DIR")]
    model: PathBuf,
    /// Print every tensor instead of only dtype counts and layer-0 tensors.
    #[arg(long)]
    all: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let shards = discover_gguf_shards(&args.model)?;
    let mut counts = BTreeMap::<String, usize>::new();
    let mut selected = Vec::new();
    for (shard_index, path) in shards.iter().enumerate() {
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file)?;
        for (name, info) in content.tensor_infos {
            *counts.entry(format!("{:?}", info.ggml_dtype)).or_default() += 1;
            if args.all || name.starts_with("blk.0.") {
                selected.push((shard_index + 1, name, info.shape, info.ggml_dtype));
            }
        }
    }
    println!("GGUF dtype counts:");
    for (dtype, count) in counts {
        println!("  {dtype}: {count}");
    }
    selected.sort_by(|left, right| left.1.cmp(&right.1));
    println!("{} tensors:", if args.all { "all" } else { "layer 0" });
    for (shard, name, shape, dtype) in selected {
        println!("  shard={shard} {name}: {:?} {dtype:?}", shape.dims());
    }
    Ok(())
}
