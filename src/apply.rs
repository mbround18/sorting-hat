//! Executing an approved plan, and reversing one.

use anyhow::{anyhow, bail, Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};

use crate::metadata;
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
}

/// Carry out a plan. Every created file is recorded first, so an interrupted
/// run is still fully reversible.
pub fn apply(plan: &Plan, work_dir: &Path, write_metadata: bool) -> Result<Report> {
    // Checked before anything is created, so a refused combination costs nothing.
    if write_metadata {
        metadata::may_write(plan.mode)?;
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

                if write_metadata {
                    // Replace what the PDF already says only when the backend
                    // was confident. Publishers' own metadata is often better
                    // than a hesitant guess, but a confident reading should
                    // displace an authoring-tool default.
                    let overwrite = assignment.digest.confidence >= OVERWRITE_CONFIDENCE;
                    match metadata::stamp(&dest, &assignment.digest, overwrite) {
                        Ok(fields) if !fields.is_empty() => report.stamped += 1,
                        Ok(_) => {}
                        Err(err) => {
                            tracing::warn!(path = %dest.display(), %err, "could not stamp metadata");
                            report.stamp_failed.push((dest, format!("{err:#}")));
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
