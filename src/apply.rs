//! Executing an approved plan, and reversing one.

use anyhow::{anyhow, bail, Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};

use crate::metadata;
use crate::outline;
use crate::types::{LinkMode, Plan, UndoAction, UndoManifest};

/// Above this confidence, our metadata replaces what the PDF already carries.
const OVERWRITE_CONFIDENCE: f32 = 0.6;

pub struct Report {
    pub filed: usize,
    pub skipped: usize,
    pub failed: Vec<(PathBuf, String)>,
    pub manifest: Option<PathBuf>,
    pub stamped: usize,
    pub stamp_failed: Vec<(PathBuf, String)>,
    pub indexed: usize,
    pub bookmarks: usize,
    pub index_skipped: Vec<(PathBuf, String)>,
}

/// What each filed document should be enriched with.
///
/// Whether metadata is written is passed separately, because it is also the
/// flag that gates the link-mode check; this carries the work itself.
#[derive(Default)]
pub struct Enrichment {
    /// Bookmark entries per source path, prepared before anything is written.
    pub outlines: std::collections::HashMap<PathBuf, Vec<outline::Entry>>,
    pub max_growth: f32,
}

/// Carry out a plan. Every created file is recorded first, so an interrupted
/// run is still fully reversible.
pub fn apply(
    plan: &Plan,
    work_dir: &Path,
    write_metadata: bool,
    enrich: &Enrichment,
) -> Result<Report> {
    // Checked before anything is created, so a refused combination costs nothing.
    if write_metadata {
        metadata::may_write(plan.mode)?;
        if plan.assignments.iter().all(|a| a.digest.is_none()) {
            anyhow::bail!(
                "this plan was made before metadata was recorded; re-run `sorting-hat plan` \
to produce one that can be stamped"
            );
        }
    }

    let root = &plan.library_root;
    std::fs::create_dir_all(root).with_context(|| format!("creating {}", root.display()))?;

    let bar = ProgressBar::new(plan.assignments.len() as u64);
    bar.set_style(
        ProgressStyle::with_template("filing      [{bar:32}] {pos}/{len}")
            .expect("static template")
            .progress_chars("=> "),
    );

    let mut manifest = UndoManifest {
        created: chrono::Utc::now(),
        mode: plan.mode,
        library_root: root.clone(),
        actions: Vec::new(),
    };
    let mut report = Report {
        filed: 0,
        skipped: 0,
        failed: Vec::new(),
        manifest: None,
        stamped: 0,
        stamp_failed: Vec::new(),
        indexed: 0,
        bookmarks: 0,
        index_skipped: Vec::new(),
    };

    for assignment in &plan.assignments {
        bar.inc(1);
        let dest = root.join(&assignment.dest);

        match place(&assignment.source, &dest, plan.mode) {
            Ok(Placed::Created) => {
                // Record the file before stamping it: if the stamp fails, undo
                // must still know the file is there.
                manifest.actions.push(UndoAction {
                    created: dest.clone(),
                    original: assignment.source.clone(),
                });
                report.filed += 1;

                // One load, one save, every change at once: lopdf cannot
                // reliably re-read what it has written, so a document gets
                // exactly one rewrite or it gets none.
                let wants_outline = enrich.outlines.get(&assignment.source);
                if write_metadata || wants_outline.is_some() {
                    match rewrite(&dest, assignment, write_metadata, wants_outline, enrich.max_growth)
                    {
                        Ok(done) => {
                            if done.stamped {
                                report.stamped += 1;
                            }
                            if done.bookmarks > 0 {
                                report.indexed += 1;
                                report.bookmarks += done.bookmarks;
                            }
                            if let Some(why) = done.outline_skipped {
                                report.index_skipped.push((dest.clone(), why));
                            }
                        }
                        Err(err) => {
                            tracing::warn!(path = %dest.display(), %err, "could not enrich");
                            report.stamp_failed.push((dest.clone(), format!("{err:#}")));
                        }
                    }
                }
            }
            Ok(Placed::AlreadyThere) => report.skipped += 1,
            Err(err) => {
                tracing::warn!(source = %assignment.source.display(), %err, "could not file");
                report.failed.push((assignment.source.clone(), format!("{err:#}")));
            }
        }
    }
    bar.finish_and_clear();

    if !manifest.actions.is_empty() {
        std::fs::create_dir_all(work_dir)?;
        let path = work_dir.join(format!(
            "undo-{}.json",
            manifest.created.format("%Y%m%dT%H%M%SZ")
        ));
        std::fs::write(&path, serde_json::to_vec_pretty(&manifest)?)
            .with_context(|| format!("writing undo manifest {}", path.display()))?;
        report.manifest = Some(path);
    }

    Ok(report)
}

struct Enriched {
    stamped: bool,
    bookmarks: usize,
    outline_skipped: Option<String>,
}

/// Load a filed document once, make every requested change, save it once.
fn rewrite(
    dest: &Path,
    assignment: &crate::types::Assignment,
    write_metadata: bool,
    entries: Option<&Vec<outline::Entry>>,
    _max_growth: f32,
) -> Result<Enriched> {
    let mut doc = lopdf::Document::load(dest)
        .with_context(|| format!("reading {} to enrich it", dest.display()))?;
    if doc.is_encrypted() {
        anyhow::bail!("PDF is encrypted; leaving it alone");
    }

    let mut changed = false;
    let mut done = Enriched { stamped: false, bookmarks: 0, outline_skipped: None };

    if write_metadata {
        if let Some(digest) = &assignment.digest {
            let overwrite = digest.confidence >= OVERWRITE_CONFIDENCE;
            if !metadata::apply_to_doc(&mut doc, digest, overwrite).is_empty() {
                done.stamped = true;
                changed = true;
            }
        }
    }

    if let Some(entries) = entries {
        if outline::has_outline(&doc) {
            done.outline_skipped = Some(outline::Skip::AlreadyIndexed.to_string());
        } else {
            match outline::apply_to_doc(&mut doc, entries) {
                Ok(n) => {
                    done.bookmarks = n;
                    changed = true;
                }
                Err(skip) => done.outline_skipped = Some(skip.to_string()),
            }
        }
    }

    if changed {
        // The same plausibility guard the metadata path uses: a rewrite that
        // balloons the file is discarded and the good copy kept.
        metadata::save_atomically(&mut doc, dest)?;
    }
    Ok(done)
}

enum Placed {
    Created,
    AlreadyThere,
}

fn place(source: &Path, dest: &Path, mode: LinkMode) -> Result<Placed> {
    if !source.exists() {
        bail!("source has gone missing since the plan was made");
    }
    if dest.exists() {
        // Re-running a plan must not clobber or duplicate work already done.
        return Ok(Placed::AlreadyThere);
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }

    match mode {
        LinkMode::Hardlink => match std::fs::hard_link(source, dest) {
            Ok(()) => Ok(Placed::Created),
            // Hard links cannot cross filesystems; a copy still satisfies intent.
            Err(err) if err.raw_os_error() == Some(18) => {
                std::fs::copy(source, dest).context("hard link crossed a filesystem; copy failed")?;
                Ok(Placed::Created)
            }
            Err(err) => Err(anyhow!(err).context("hard linking")),
        },
        LinkMode::Symlink => {
            let target = source
                .canonicalize()
                .with_context(|| format!("resolving {}", source.display()))?;
            std::os::unix::fs::symlink(&target, dest).context("symlinking")?;
            Ok(Placed::Created)
        }
        LinkMode::Copy => {
            std::fs::copy(source, dest).context("copying")?;
            Ok(Placed::Created)
        }
        LinkMode::Move => {
            match std::fs::rename(source, dest) {
                Ok(()) => Ok(Placed::Created),
                Err(err) if err.raw_os_error() == Some(18) => {
                    std::fs::copy(source, dest).context("cross-filesystem move: copy failed")?;
                    std::fs::remove_file(source)
                        .context("cross-filesystem move: removing the original failed")?;
                    Ok(Placed::Created)
                }
                Err(err) => Err(anyhow!(err).context("moving")),
            }
        }
    }
}

/// Reverse an applied run using its manifest.
pub fn undo(manifest_path: &Path) -> Result<(usize, Vec<String>)> {
    let raw = std::fs::read_to_string(manifest_path)
        .with_context(|| format!("reading {}", manifest_path.display()))?;
    let manifest: UndoManifest = serde_json::from_str(&raw)
        .with_context(|| format!("parsing {}", manifest_path.display()))?;

    let mut reversed = 0usize;
    let mut problems = Vec::new();

    for action in &manifest.actions {
        if !action.created.exists() {
            continue;
        }
        let result = if manifest.mode == LinkMode::Move {
            // Put the original back where it came from before removing anything.
            if action.original.exists() {
                Err(anyhow!(
                    "{} already exists; leaving {} in place",
                    action.original.display(),
                    action.created.display()
                ))
            } else {
                action
                    .original
                    .parent()
                    .map(std::fs::create_dir_all)
                    .transpose()
                    .and_then(|_| std::fs::rename(&action.created, &action.original))
                    .map_err(Into::into)
            }
        } else {
            std::fs::remove_file(&action.created).map_err(Into::into)
        };

        match result {
            Ok(()) => reversed += 1,
            Err(err) => problems.push(format!("{}: {err:#}", action.created.display())),
        }
    }

    prune_empty_dirs(&manifest.library_root);
    Ok((reversed, problems))
}

/// Remove directories the undo emptied, deepest first, leaving the root itself.
fn prune_empty_dirs(root: &Path) {
    let Ok(entries) = walkdir::WalkDir::new(root)
        .contents_first(true)
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
    else {
        return;
    };
    for entry in entries {
        if entry.file_type().is_dir() && entry.path() != root {
            let _ = std::fs::remove_dir(entry.path());
        }
    }
}
