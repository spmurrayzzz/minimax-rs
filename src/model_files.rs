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
        if path.is_file()
            && path
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        sync::atomic::{AtomicUsize, Ordering},
    };

    static NEXT_TEST_DIR: AtomicUsize = AtomicUsize::new(0);

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            let sequence = NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "minimax-model-files-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir(&path).expect("create test model directory");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }

        fn create_file(&self, name: &str) -> PathBuf {
            let path = self.0.join(name);
            fs::write(&path, []).expect("create test shard");
            path
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn discovers_regular_gguf_files_in_deterministic_order() {
        let dir = TestDir::new();
        let shard_3 = dir.create_file("model-00003-of-00004.gguf");
        let shard_1 = dir.create_file("model-00001-of-00004.gguf");
        let shard_4 = dir.create_file("model-00004-of-00004.gguf");
        let shard_2 = dir.create_file("model-00002-of-00004.gguf");
        dir.create_file("notes.txt");
        fs::create_dir(dir.path().join("not-a-file.gguf")).expect("create decoy directory");

        assert_eq!(
            discover_gguf_shards(dir.path()).expect("discover shards"),
            vec![shard_1, shard_2, shard_3, shard_4]
        );
    }

    #[test]
    fn rejects_incomplete_or_extra_shard_sets() {
        for count in [3, 5] {
            let dir = TestDir::new();
            for index in 0..count {
                dir.create_file(&format!("shard-{index}.gguf"));
            }

            let error = discover_gguf_shards(dir.path()).expect_err("invalid shard count");
            let message = error.to_string();
            assert!(message.contains("expected 4 split GGUF files"));
            assert!(message.contains(&format!("found {count}")));
        }
    }

    #[test]
    fn reports_when_the_model_directory_cannot_be_read() {
        let dir = TestDir::new();
        let missing = dir.path().join("missing");

        let error = discover_gguf_shards(&missing).expect_err("missing directory");
        assert!(error.to_string().contains("cannot read model directory"));
        assert!(error.to_string().contains(&missing.display().to_string()));
    }
}
