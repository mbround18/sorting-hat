//! A content-addressed digest cache.
//!
//! Scanning 400 PDFs through a local model is minutes of GPU time, so a digest
//! is keyed by file fingerprint and reused across runs. Renaming or moving a
//! source file does not invalidate its entry.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use crate::types::Digest;

pub struct Cache {
    path: PathBuf,
    entries: HashMap<String, Digest>,
}

impl Cache {
    /// Load the digest log, tolerating a truncated final line from a killed run.
    pub fn open(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let path = dir.join("digests.jsonl");

        let mut entries = HashMap::new();
        if path.exists() {
            let file = File::open(&path)?;
            for (n, line) in BufReader::new(file).lines().enumerate() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<Digest>(&line) {
                    // A later entry for the same hash supersedes an earlier one.
                    Ok(digest) => {
                        entries.insert(digest.hash.clone(), digest);
                    }
                    Err(err) => tracing::warn!(line = n + 1, %err, "discarding bad cache entry"),
                }
            }
        }

        Ok(Self { path, entries })
    }

    pub fn get(&self, hash: &str) -> Option<&Digest> {
        self.entries.get(hash)
    }

    /// Append a digest and make it durable before returning.
    pub fn put(&mut self, digest: Digest) -> Result<()> {
        let line = serde_json::to_string(&digest)?;
        let mut file = OpenOptions::new().create(true).append(true).open(&self.path)?;
        writeln!(file, "{line}")?;
        file.flush()?;
        self.entries.insert(digest.hash.clone(), digest);
        Ok(())
    }

    /// Drop entries the backend was unsure of, so they can be read again — by a
    /// better backend, typically the vision pass over a document that yielded no
    /// text the first time. Returns how many were dropped.
    ///
    /// The log is append-only with last-entry-wins, so removing an entry means
    /// rewriting the file.
    pub fn drop_below(&mut self, confidence: f32) -> Result<usize> {
        let before = self.entries.len();
        self.entries.retain(|_, d| d.confidence >= confidence);
        let dropped = before - self.entries.len();
        if dropped > 0 {
            self.rewrite()?;
        }
        Ok(dropped)
    }

    /// Drop entries the backend could not name, so a higher-resolution render
    /// or a different backend can try again. Returns how many were dropped.
    pub fn drop_unnamed(&mut self) -> Result<usize> {
        let before = self.entries.len();
        self.entries.retain(|_, d| !crate::naming::is_unknown(&d.title));
        let dropped = before - self.entries.len();
        if dropped > 0 {
            self.rewrite()?;
        }
        Ok(dropped)
    }

    /// Write the live entries back out, replacing the log.
    fn rewrite(&self) -> Result<()> {
        let tmp = self.path.with_extension("jsonl.tmp");
        {
            let mut file = File::create(&tmp)
                .with_context(|| format!("creating {}", tmp.display()))?;
            for digest in self.entries.values() {
                writeln!(file, "{}", serde_json::to_string(digest)?)?;
            }
            file.flush()?;
        }
        // Rename over the original so an interrupted rewrite cannot truncate it.
        std::fs::rename(&tmp, &self.path)
            .with_context(|| format!("replacing {}", self.path.display()))?;
        Ok(())
    }

    /// Drop every entry. Used by `--rescan`.
    pub fn clear(&mut self) -> Result<()> {
        self.entries.clear();
        if self.path.exists() {
            std::fs::remove_file(&self.path)?;
        }
        Ok(())
    }
}
