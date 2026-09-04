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

/// Hash every file in full, so identity is exact rather than probable.
///
/// SHA-256 over the whole file, not a sample of it: hashing this corpus takes
/// about eight seconds with hardware SHA and several cores, which is far too
/// cheap to justify the risk that two documents differing only in the middle
/// are declared the same. It is also the hash everyone already has a tool for,
/// so anything this program reports can be checked with `sha256sum`.
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

fn fingerprint_one(path: &Path) -> Result<SourceDoc> {
    use sha2::{Digest, Sha256};

    let mut file = std::fs::File::open(path)?;
    let bytes = file.metadata()?.len();

    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    Ok(SourceDoc { path: path.to_path_buf(), hash: format!("{:x}", hasher.finalize()), bytes })
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the hash to the published SHA-256 of "abc", so a future change to
    /// how files are read cannot silently produce a different identity — and so
    /// anything this tool prints stays checkable with `sha256sum`.
    #[test]
    fn hashes_match_the_sha256_standard() {
        let dir = std::env::temp_dir().join(format!("sorting-hat-hash-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("abc.bin");
        std::fs::write(&file, b"abc").unwrap();

        let doc = fingerprint_one(&file).unwrap();
        assert_eq!(
            doc.hash,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(doc.bytes, 3);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The buffered read must give the same answer as a single-shot hash for a
    /// file larger than the read buffer.
    #[test]
    fn hashes_files_larger_than_the_read_buffer() {
        use sha2::{Digest, Sha256};

        let dir = std::env::temp_dir().join(format!("sorting-hat-big-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("big.bin");
        let data: Vec<u8> = (0..(3 << 20)).map(|i| (i % 251) as u8).collect();
        std::fs::write(&file, &data).unwrap();

        let doc = fingerprint_one(&file).unwrap();
        assert_eq!(doc.hash, format!("{:x}", Sha256::digest(&data)));

        std::fs::remove_dir_all(&dir).ok();
    }
}
