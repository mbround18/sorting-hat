//! Core data types shared across the pipeline stages.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// A PDF found in the source tree, before any content is read.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceDoc {
    pub path: PathBuf,
    /// blake3 of the file contents; the cache key for everything downstream.
    pub hash: String,
    pub bytes: u64,
}

/// Raw material pulled out of a PDF: embedded metadata plus a text probe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Probe {
    pub file_name: String,
    pub pdf_title: Option<String>,
    pub pdf_author: Option<String>,
    pub pdf_subject: Option<String>,
    pub pdf_creator: Option<String>,
    pub page_count: Option<usize>,
    /// Text from the leading pages, whitespace-collapsed and truncated.
    pub text: String,
    /// True when the PDF yielded essentially no extractable text (image scan).
    pub scanned: bool,
}

/// What the model concluded about a single document. Pass one output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Digest {
    pub hash: String,
    pub path: PathBuf,
    /// Human-facing title, cleaned up. Drives the destination file name.
    pub title: String,
    pub game_system: String,
    pub doc_type: String,
    pub setting: String,
    pub level_range: String,
    pub publisher: String,
    pub topics: Vec<String>,
    pub summary: String,
    /// 0.0-1.0. Low confidence documents are parked instead of filed.
    pub confidence: f32,
    /// Which backend produced this digest, for provenance in the plan.
    pub source: String,
}

/// The folder tree the model designs from the whole corpus. Pass two, part one.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Taxonomy {
    /// Leaf folder paths, e.g. "D&D 5e/Adventures/Tier 1". Slash separated.
    pub leaves: Vec<String>,
    /// Optional one-line rationale per leaf, shown in the plan review.
    #[serde(default)]
    pub notes: Vec<String>,
}

impl Taxonomy {
    pub fn contains(&self, leaf: &str) -> bool {
        self.leaves.iter().any(|l| l == leaf)
    }
}

/// One filing decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assignment {
    pub hash: String,
    pub source: PathBuf,
    /// Destination relative to the library root, including file name.
    pub dest: PathBuf,
    pub leaf: String,
    pub title: String,
    pub confidence: f32,
    pub reason: String,
}

/// The full reviewable plan written to disk before anything moves.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    pub created: chrono::DateTime<chrono::Utc>,
    pub source_root: PathBuf,
    pub library_root: PathBuf,
    pub mode: LinkMode,
    pub taxonomy: Taxonomy,
    pub assignments: Vec<Assignment>,
    /// Documents the pipeline refused to file, with the reason why.
    pub unfiled: Vec<Unfiled>,
    /// Groups of identical files (same hash); only the first is filed.
    pub duplicates: Vec<Duplicate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Unfiled {
    pub path: PathBuf,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Duplicate {
    pub hash: String,
    pub kept: PathBuf,
    pub others: Vec<PathBuf>,
}

/// How a filed document gets into the library.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
#[clap(rename_all = "kebab-case")]
pub enum LinkMode {
    /// Hard link. No extra disk, originals untouched. Same filesystem only.
    Hardlink,
    /// Symbolic link. Works across filesystems, originals untouched.
    Symlink,
    /// Full copy. Safest, doubles disk usage.
    Copy,
    /// Move the original. Destructive to the source tree.
    Move,
}

/// Written on every apply so a run can be reversed exactly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UndoManifest {
    pub created: chrono::DateTime<chrono::Utc>,
    pub mode: LinkMode,
    pub library_root: PathBuf,
    pub actions: Vec<UndoAction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UndoAction {
    /// File that was created in the library.
    pub created: PathBuf,
    /// Where it came from. Only meaningful for `Move`.
    pub original: PathBuf,
}
