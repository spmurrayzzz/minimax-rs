use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

pub const MODEL_DIR_ENV: &str = "MINIMAX_MODEL_DIR";
pub const EXPECTED_SHARD_COUNT: usize = 4;

/// Find the split GGUF shards for a MiniMax model in deterministic order.
pub fn discover_gguf_shards(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let entries = std::fs::read_dir(model_dir)
        .with_context(|| format!("cannot read model directory {}", model_dir.display()))?;
    let mut shards = Vec::new();
    for entry in entries {
        let path = entry
            .with_context(|| format!("cannot inspect model directory {}", model_dir.display()))?
            .path();
        if path
            .extension()
            .is_some_and(|extension| extension == "gguf")
        {
            shards.push(path);
        }
    }
    shards.sort();
    if shards.len() != EXPECTED_SHARD_COUNT {
        bail!(
            "expected {EXPECTED_SHARD_COUNT} split GGUF files in {}, found {}",
            model_dir.display(),
            shards.len()
        );
    }
    Ok(shards)
}
