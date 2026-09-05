//! Writing what we know back into the filed PDFs.
//!
//! Metadata and a chapter index are separate ideas and get separate commands,
//! but they cannot be separate *rewrites*: lopdf frequently cannot re-read a
//! file it has just written — 91 of 247 here — so a second pass over its own
//! output would silently skip a third of the library.
//!
//! Every enrichment therefore starts again from the pristine source: the filed
//! copy is replaced, then everything known about the document is applied in one
//! load and one save. That makes the commands idempotent and order-independent,
//! at the cost of copying the file again. Derived outlines are cached so the
//! expensive half — reading headings, and the model judging them — is not
//! repeated.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::metadata;
use crate::outline;
use crate::types::{Digest, Plan};

/// Outlines derived for each source document, cached between runs.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct OutlineCache {
    /// Keyed by the source document's SHA-256, so a moved or renamed file keeps
    /// its outline and a changed one does not.
    pub by_hash: HashMap<String, Vec<CachedEntry>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedEntry {
    pub level: usize,
    pub title: String,
    pub page: usize,
}

impl From<&outline::Entry> for CachedEntry {
    fn from(e: &outline::Entry) -> Self {
        Self { level: e.level, title: e.title.clone(), page: e.page }
    }
}

impl From<&CachedEntry> for outline::Entry {
    fn from(e: &CachedEntry) -> Self {
        Self { level: e.level, title: e.title.clone(), page: e.page }
    }
}

impl OutlineCache {
    pub fn open(dir: &Path) -> Result<Self> {
        let path = dir.join("outlines.json");
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        Ok(serde_json::from_str(&raw).unwrap_or_default())
    }

    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join("outlines.json");
        std::fs::write(&path, serde_json::to_vec_pretty(self)?)
            .with_context(|| format!("writing {}", path.display()))
    }

    pub fn get(&self, hash: &str) -> Option<Vec<outline::Entry>> {
        self.by_hash.get(hash).map(|v| v.iter().map(Into::into).collect())
    }

    pub fn put(&mut self, hash: &str, entries: &[outline::Entry]) {
        self.by_hash.insert(hash.to_string(), entries.iter().map(Into::into).collect());
    }
}

/// What one document should end up carrying.
pub struct Wanted<'a> {
    pub metadata: Option<&'a Digest>,
    pub outline: Option<Vec<outline::Entry>>,
}

/// What actually happened to it.
#[derive(Debug, Default)]
pub struct Outcome {
    pub refreshed: usize,
    pub stamped: usize,
    pub indexed: usize,
    pub bookmarks: usize,
    pub skipped: Vec<(PathBuf, String)>,
}

/// Above this confidence, our metadata replaces what the PDF already carries.
const OVERWRITE_CONFIDENCE: f32 = 0.6;

/// Replace each filed copy from its source and write everything wanted into it.
pub fn refresh(
    plan: &Plan,
    cache: &OutlineCache,
    max_growth: f32,
    dry_run: bool,
) -> Result<Outcome> {
    let mut outcome = Outcome::default();

    let bar = indicatif::ProgressBar::new(plan.assignments.len() as u64);
    bar.set_style(
        indicatif::ProgressStyle::with_template("enriching   [{bar:32}] {pos}/{len} {eta_precise}")
            .expect("static template")
            .progress_chars("=> "),
    );

    for assignment in &plan.assignments {
        bar.inc(1);
        let dest = plan.library_root.join(&assignment.dest);

        let wanted = Wanted {
            metadata: assignment.digest.as_ref(),
            outline: cache.get(&assignment.hash),
        };
        if wanted.metadata.is_none() && wanted.outline.is_none() {
            continue;
        }
        if dry_run {
            outcome.refreshed += 1;
            if wanted.metadata.is_some() {
                outcome.stamped += 1;
            }
            if let Some(entries) = &wanted.outline {
                outcome.indexed += 1;
                outcome.bookmarks += entries.len();
            }
            continue;
        }

        match one(&assignment.source, &dest, &wanted, max_growth) {
            Ok(done) => {
                outcome.refreshed += 1;
                if done.stamped {
                    outcome.stamped += 1;
                }
                if done.bookmarks > 0 {
                    outcome.indexed += 1;
                    outcome.bookmarks += done.bookmarks;
                }
                if let Some(why) = done.skipped {
                    outcome.skipped.push((dest, why));
                }
            }
            Err(err) => {
                tracing::warn!(path = %dest.display(), %err, "could not enrich");
                outcome.skipped.push((dest, format!("{err:#}")));
            }
        }
    }
    bar.finish_and_clear();
    Ok(outcome)
}

struct Done {
    stamped: bool,
    bookmarks: usize,
    skipped: Option<String>,
}

/// Restore one document from its source, then write everything into it at once.
fn one(source: &Path, dest: &Path, wanted: &Wanted<'_>, max_growth: f32) -> Result<Done> {
    if !source.exists() {
        anyhow::bail!("source has gone missing since the plan was made");
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Unlink before copying. Copying onto a hard link writes straight through
    // to the original, so a library placed by link would have its sources
    // rewritten — removing the entry first breaks that connection and leaves a
    // genuinely separate file.
    if dest.exists() {
        std::fs::remove_file(dest)
            .with_context(|| format!("clearing {}", dest.display()))?;
    }
    std::fs::copy(source, dest)
        .with_context(|| format!("restoring {} from its source", dest.display()))?;

    let mut doc = lopdf::Document::load(dest)
        .with_context(|| format!("reading {}", dest.display()))?;
    if doc.is_encrypted() {
        anyhow::bail!("{}", outline::Skip::Encrypted);
    }

    let mut done = Done { stamped: false, bookmarks: 0, skipped: None };
    let mut changed = false;

    if let Some(digest) = wanted.metadata {
        let overwrite = digest.confidence >= OVERWRITE_CONFIDENCE;
        if !metadata::apply_to_doc(&mut doc, digest, overwrite).is_empty() {
            done.stamped = true;
            changed = true;
        }
    }

    if let Some(entries) = &wanted.outline {
        if outline::has_outline(&doc) {
            done.skipped = Some(outline::Skip::AlreadyIndexed.to_string());
        } else {
            match outline::apply_to_doc(&mut doc, entries) {
                Ok(n) => {
                    done.bookmarks = n;
                    changed = true;
                }
                Err(skip) => done.skipped = Some(skip.to_string()),
            }
        }
    }

    if changed {
        metadata::save_atomically(&mut doc, dest, max_growth)?;
    }
    Ok(done)
}
