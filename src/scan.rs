//! Walk the source tree and identify the PDFs worth looking at.

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::types::{Duplicate, SourceDoc};

/// Directory names we never descend into: VCS metadata, and the tool's own output.
const SKIP_DIRS: &[&str] = &[".git", ".svn", "target", "node_modules", ".venv", "__pycache__"];

/// Find every PDF under `root`, skipping anything inside `exclude`.
pub fn find_pdfs(root: &Path, exclude: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let exclude: Vec<PathBuf> = exclude.iter().filter_map(|p| p.canonicalize().ok()).collect();
    let mut out = Vec::new();

    for entry in WalkDir::new(root).follow_links(false).into_iter().filter_entry(|e| {
        if !e.file_type().is_dir() {
            return true;
        }
        let name = e.file_name().to_string_lossy();
        if name.starts_with('.') && e.depth() > 0 {
            return false;
        }
        if SKIP_DIRS.contains(&name.as_ref()) {
            return false;
        }
        // Never walk into our own library output, even if it lives under the source.
        !e.path().canonicalize().map(|p| exclude.iter().any(|x| p.starts_with(x))).unwrap_or(false)
    }) {
        let entry = entry.context("walking source tree")?;
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("pdf")) {
            out.push(path.to_path_buf());
        }
    }
    out.sort();
    Ok(out)
}

/// Hash each file so the cache survives renames and so we can spot duplicates.
///
/// Hashing 14 GB fully would dominate the run, so this fingerprints the head,
/// tail and length of each file instead — enough to key a cache and to flag
/// copies of the same download, without reading every byte.
pub fn fingerprint(paths: &[PathBuf]) -> Vec<SourceDoc> {
    paths
        .par_iter()
        .filter_map(|path| match fingerprint_one(path) {
            Ok(doc) => Some(doc),
            Err(err) => {
                tracing::warn!(path = %path.display(), %err, "skipping unreadable file");
                None
            }
        })
        .collect()
}

const SAMPLE: usize = 256 * 1024;

fn fingerprint_one(path: &Path) -> Result<SourceDoc> {
    let mut file = std::fs::File::open(path)?;
    let bytes = file.metadata()?.len();

    let mut hasher = blake3::Hasher::new();
    hasher.update(&bytes.to_le_bytes());

    let mut head = vec![0u8; SAMPLE.min(bytes as usize)];
    file.read_exact(&mut head)?;
    hasher.update(&head);

    if bytes > SAMPLE as u64 * 2 {
        use std::io::Seek;
        file.seek(std::io::SeekFrom::End(-(SAMPLE as i64)))?;
        let mut tail = vec![0u8; SAMPLE];
        file.read_exact(&mut tail)?;
        hasher.update(&tail);
    }

    Ok(SourceDoc { path: path.to_path_buf(), hash: hasher.finalize().to_hex().to_string(), bytes })
}

/// Split the corpus into one representative per hash plus the duplicate groups.
pub fn dedupe(docs: Vec<SourceDoc>) -> (Vec<SourceDoc>, Vec<Duplicate>) {
    let mut groups: HashMap<String, Vec<SourceDoc>> = HashMap::new();
    for doc in docs {
        groups.entry(doc.hash.clone()).or_default().push(doc);
    }

    let mut unique = Vec::new();
    let mut duplicates = Vec::new();
    for (hash, mut group) in groups {
        // Prefer the shallowest path as the keeper: it is usually the original.
        group.sort_by_key(|d| (d.path.components().count(), d.path.clone()));
        let kept = group.remove(0);
        if !group.is_empty() {
            duplicates.push(Duplicate {
                hash,
                kept: kept.path.clone(),
                others: group.into_iter().map(|d| d.path).collect(),
            });
        }
        unique.push(kept);
    }

    unique.sort_by(|a, b| a.path.cmp(&b.path));
    duplicates.sort_by(|a, b| a.kept.cmp(&b.kept));
    (unique, duplicates)
}
