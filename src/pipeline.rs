//! Stage orchestration: source tree in, reviewable plan out. Nothing here writes
//! to the library — that is `apply`'s job, and only after you have read the plan.

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use crate::brain::Brain;
use crate::cache::Cache;
use crate::config::Config;
use crate::naming;
use crate::types::{Assignment, Digest, Duplicate, LinkMode, Plan, Probe, SourceDoc, Taxonomy, Unfiled};
use crate::{extract, scan};

/// Where documents go when nothing in the taxonomy fits well enough.
pub const UNSORTED: &str = "_Unsorted";

/// Describes a page image when a PDF yields no text of its own.
pub trait Eyes {
    fn name(&self) -> &str;
    fn describe(&mut self, image: &std::path::Path, probe: &Probe) -> Result<crate::brain::DigestFields>;
}

pub struct Options {
    pub mode: LinkMode,
    /// Ignore cached digests and re-read every document.
    pub rescan: bool,
    /// Re-read documents whose cached digest scored below this confidence.
    pub redigest_below: Option<f32>,
    /// Re-read documents the backend could not name.
    pub redigest_unknown: bool,
    /// Stop after this many documents. For trying the pipeline out cheaply.
    pub limit: Option<usize>,
}

fn bar(len: u64, what: &str) -> ProgressBar {
    let bar = ProgressBar::new(len);
    bar.set_style(
        ProgressStyle::with_template("{msg:<12} [{bar:32}] {pos}/{len} {eta_precise}")
            .expect("static template")
            .progress_chars("=> "),
    );
    bar.set_message(what.to_string());
    bar
}

pub fn build_plan(
    cfg: &Config,
    brain: &mut dyn Brain,
    eyes: Option<&mut (dyn Eyes + '_)>,
    opts: &Options,
) -> Result<Plan> {
    let exclude = vec![cfg.library.clone(), cfg.work_dir.clone()];
    let paths = scan::find_pdfs(&cfg.source, &exclude)?;
    if paths.is_empty() {
        anyhow::bail!("no PDFs found under {}", cfg.source.display());
    }
    tracing::info!(found = paths.len(), "PDFs discovered");

    let docs = scan::fingerprint(&paths);
    let (mut docs, duplicates) = scan::dedupe(docs);
    if let Some(limit) = opts.limit {
        docs.truncate(limit);
    }
    tracing::info!(unique = docs.len(), duplicate_groups = duplicates.len(), "fingerprinted");

    let mut cache = Cache::open(&cfg.cache_dir())?;
    match cache.rekey(&docs) {
        Ok(moved) if moved > 0 => {
            tracing::info!(moved, "carried cached digests onto the current hashing scheme")
        }
        Ok(_) => {}
        Err(err) => tracing::warn!(%err, "could not re-key the digest cache"),
    }
    if opts.rescan {
        cache.clear()?;
    } else {
        if let Some(threshold) = opts.redigest_below {
            let dropped = cache.drop_below(threshold)?;
            tracing::info!(dropped, threshold, "dropped low-confidence digests for a second look");
        }
        if opts.redigest_unknown {
            let dropped = cache.drop_unnamed()?;
            tracing::info!(dropped, "dropped unnamed digests for a second look");
        }
    }

    let (digests, unfiled) = digest_all(cfg, brain, eyes, &mut cache, &docs)?;
    if digests.is_empty() {
        anyhow::bail!("every document failed to produce a digest; nothing to plan");
    }

    let (digests, near) = if cfg.dedupe.near_duplicates {
        fold_near_duplicates(cfg, digests)
    } else {
        (digests, Vec::new())
    };
    let mut duplicates = duplicates;
    if !near.is_empty() {
        let folded: usize = near.iter().map(|d| d.others.len()).sum();
        tracing::info!(titles = near.len(), files = folded, "folded near-duplicate re-saves");
        duplicates.extend(near);
    }

    tracing::info!(docs = digests.len(), "designing taxonomy");
    let taxonomy = brain.design_taxonomy(&digests, &cfg.taxonomy)?;
    tracing::info!(leaves = taxonomy.leaves.len(), "taxonomy designed");

    let filings = file_all(cfg, brain, &taxonomy, &digests);
    let (taxonomy, filings) = fold_thin_leaves(taxonomy, filings, cfg.taxonomy.min_docs_per_leaf);
    let assignments = to_assignments(&digests, &filings);

    Ok(Plan {
        created: chrono::Utc::now(),
        source_root: cfg.source.clone(),
        library_root: cfg.library.clone(),
        mode: opts.mode,
        taxonomy,
        assignments,
        unfiled,
        duplicates,
    })
}

/// Read and describe every document, reusing cached digests where possible.
fn digest_all(
    cfg: &Config,
    brain: &mut dyn Brain,
    mut eyes: Option<&mut (dyn Eyes + '_)>,
    cache: &mut Cache,
    docs: &[SourceDoc],
) -> Result<(Vec<Digest>, Vec<Unfiled>)> {
    let mut digests = Vec::with_capacity(docs.len());
    let mut unfiled = Vec::new();

    let pending: Vec<&SourceDoc> = docs
        .iter()
        .filter(|doc| match cache.get(&doc.hash) {
            Some(cached) => {
                // Reuse the description, but track where the file lives now.
                let mut cached = cached.clone();
                cached.path = doc.path.clone();
                digests.push(cached);
                false
            }
            None => true,
        })
        .collect();

    tracing::info!(cached = digests.len(), to_read = pending.len(), "digest cache");
    if pending.is_empty() {
        return Ok((digests, unfiled));
    }

    // Text extraction is IO- and CPU-bound and embarrassingly parallel; the
    // model call afterwards is not, so the two stages are kept separate.
    let probe_bar = bar(pending.len() as u64, "extracting");
    let probes: Vec<(&SourceDoc, Result<Probe>)> = pending
        .par_iter()
        .map(|doc| {
            let probe = extract::probe(&doc.path, &cfg.extract)
                .with_context(|| format!("extracting {}", doc.path.display()));
            probe_bar.inc(1);
            (*doc, probe)
        })
        .collect();
    probe_bar.finish_and_clear();

    let digest_bar = bar(probes.len() as u64, "reading");
    for (doc, probe) in probes {
        digest_bar.inc(1);
        let probe = match probe {
            Ok(probe) => probe,
            Err(err) => {
                unfiled.push(Unfiled { path: doc.path.clone(), reason: format!("{err:#}") });
                continue;
            }
        };

        // A PDF with no extractable text is a picture. Look at it instead of
        // guessing from the file name — for a product code like `PZO31005E.pdf`
        // the file name carries nothing at all.
        let seen = match (probe.scanned, eyes.as_deref_mut()) {
            (true, Some(eyes)) => match look(cfg, eyes, doc, &probe) {
                Ok(fields) => Some((fields, eyes.name().to_string())),
                Err(err) => {
                    tracing::warn!(path = %doc.path.display(), %err, "vision pass failed");
                    None
                }
            },
            _ => None,
        };

        let outcome = match seen {
            Some((fields, source)) => Ok((fields, source)),
            None => brain.digest(&probe).map(|f| (f, brain.name().to_string())),
        };

        match outcome {
            Ok((mut fields, source)) => {
                // A backend that could not read a title must not name the file
                // "Unknown.pdf" — the original name, however ugly, carries more.
                if naming::is_unknown(&fields.title) {
                    fields.title = fallback_title(&doc.path);
                }
                let digest = Digest {
                    hash: doc.hash.clone(),
                    path: doc.path.clone(),
                    title: fields.title,
                    game_system: fields.game_system,
                    doc_type: fields.doc_type,
                    setting: fields.setting,
                    level_range: fields.level_range,
                    publisher: fields.publisher,
                    topics: fields.topics,
                    summary: fields.summary,
                    confidence: fields.confidence.clamp(0.0, 1.0),
                    source,
                };
                cache.put(digest.clone())?;
                digests.push(digest);
            }
            Err(err) => {
                tracing::warn!(path = %doc.path.display(), %err, "digest failed");
                unfiled.push(Unfiled {
                    path: doc.path.clone(),
                    reason: format!("could not be described: {err:#}"),
                });
            }
        }
    }
    digest_bar.finish_and_clear();

    Ok((digests, unfiled))
}

/// The source file's own name, unchanged, for when nothing better is known.
///
/// Deliberately not tidied: if no backend could read a title, the name the file
/// arrived with is the only real information left, and altering it would only
/// make the document harder to trace back to its source.
fn fallback_title(path: &std::path::Path) -> String {
    let stem = path.file_stem().unwrap_or_default().to_string_lossy().into_owned();
    if stem.is_empty() {
        "Untitled".to_string()
    } else {
        stem
    }
}

/// Fold together documents that are the same work saved twice.
///
/// Exact copies are caught by fingerprint before anything is read. This catches
/// the re-download: same title, same page count, a few kilobytes apart because
/// the metadata or compression differs. The largest copy is kept — it is the
/// least likely to have been through a lossy re-save — and the rest are listed
/// as duplicates rather than filed alongside as "Title (2)".
fn fold_near_duplicates(cfg: &Config, digests: Vec<Digest>) -> (Vec<Digest>, Vec<Duplicate>) {
    let mut groups: HashMap<String, Vec<Digest>> = HashMap::new();
    for digest in digests {
        groups.entry(title_key(&digest.title)).or_default().push(digest);
    }

    // Page counts are only needed for titles that actually collide.
    let contested: Vec<PathBuf> = groups
        .values()
        .filter(|g| g.len() > 1)
        .flat_map(|g| g.iter().map(|d| d.path.clone()))
        .collect();
    let pages: HashMap<PathBuf, Option<usize>> = contested
        .par_iter()
        .map(|path| (path.clone(), extract::page_count(path, cfg.extract.timeout_secs)))
        .collect();

    let size = |path: &PathBuf| std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    let mut kept = Vec::new();
    let mut duplicates = Vec::new();

    for (_, mut group) in groups {
        if group.len() == 1 {
            kept.append(&mut group);
            continue;
        }
        // Largest first: that copy becomes the candidate every other is judged against.
        group.sort_by_key(|d| std::cmp::Reverse(size(&d.path)));

        while !group.is_empty() {
            let winner = group.remove(0);
            let winner_size = size(&winner.path);
            let winner_pages = pages.get(&winner.path).copied().flatten();

            let mut same = Vec::new();
            group.retain(|other| {
                let other_pages = pages.get(&other.path).copied().flatten();
                // A different page count means a different edition or printing,
                // not a re-save; keep both and let the user see them.
                if winner_pages != other_pages {
                    return true;
                }
                let other_size = size(&other.path);
                let spread = winner_size.abs_diff(other_size) as f32 / winner_size.max(1) as f32;
                if spread <= cfg.dedupe.size_tolerance {
                    same.push(other.path.clone());
                    false
                } else {
                    true
                }
            });

            if !same.is_empty() {
                duplicates.push(Duplicate {
                    hash: winner.hash.clone(),
                    kept: winner.path.clone(),
                    others: same,
                });
            }
            kept.push(winner);
        }
    }

    kept.sort_by(|a, b| a.path.cmp(&b.path));
    duplicates.sort_by(|a, b| a.kept.cmp(&b.kept));
    (kept, duplicates)
}

/// A title reduced to what identifies the work, for grouping re-saves.
fn title_key(title: &str) -> String {
    title
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Render the first page and have the vision backend read it.
fn look(
    cfg: &Config,
    eyes: &mut (dyn Eyes + '_),
    doc: &SourceDoc,
    probe: &Probe,
) -> Result<crate::brain::DigestFields> {
    let image = extract::render_first_page(
        &doc.path,
        &cfg.render_dir(),
        cfg.vision.max_pixels,
        cfg.extract.timeout_secs,
    )?;
    let fields = eyes.describe(&image, probe);
    // The render is a scratch file; keep the work directory from filling up.
    let _ = std::fs::remove_file(&image);
    fields
}

/// Ask the brain where each document belongs. Failures land in `_Unsorted`
/// rather than dropping the document.
fn file_all(
    cfg: &Config,
    brain: &mut dyn Brain,
    taxonomy: &Taxonomy,
    digests: &[Digest],
) -> Vec<(String, f32, String)> {
    let bar = bar(digests.len() as u64, "filing");
    let filings = digests
        .iter()
        .map(|digest| {
            bar.inc(1);
            match brain.file(digest, taxonomy) {
                Ok(filing) if filing.confidence >= cfg.taxonomy.min_confidence => {
                    (filing.folder, filing.confidence, filing.reason)
                }
                Ok(filing) => (
                    UNSORTED.to_string(),
                    filing.confidence,
                    format!("low confidence for {:?}: {}", filing.folder, filing.reason),
                ),
                Err(err) => {
                    tracing::warn!(title = %digest.title, %err, "filing failed");
                    (UNSORTED.to_string(), 0.0, format!("filing failed: {err:#}"))
                }
            }
        })
        .collect();
    bar.finish_and_clear();
    filings
}

/// Collapse leaves that ended up with too few documents into their parent, so
/// the library does not sprout single-file folders.
fn fold_thin_leaves(
    taxonomy: Taxonomy,
    filings: Vec<(String, f32, String)>,
    min_docs: usize,
) -> (Taxonomy, Vec<(String, f32, String)>) {
    if min_docs <= 1 {
        return (taxonomy, filings);
    }

    let mut counts: HashMap<&str, usize> = HashMap::new();
    for (folder, _, _) in &filings {
        *counts.entry(folder.as_str()).or_default() += 1;
    }

    // A leaf folds to its parent, unless the parent is the root or the leaf is
    // the unsorted bucket, which always stays put.
    let remap: HashMap<String, String> = counts
        .iter()
        .filter(|(folder, n)| **n < min_docs && **folder != UNSORTED)
        .filter_map(|(folder, _)| {
            let parent = folder.rsplit_once('/')?.0.to_string();
            Some((folder.to_string(), parent))
        })
        .collect();

    if remap.is_empty() {
        return (taxonomy, filings);
    }

    let filings: Vec<_> = filings
        .into_iter()
        .map(|(folder, conf, reason)| match remap.get(&folder) {
            Some(parent) => (
                parent.clone(),
                conf,
                format!("{reason}; folded up from thin folder {folder:?}"),
            ),
            None => (folder, conf, reason),
        })
        .collect();

    let mut leaves: Vec<String> = filings.iter().map(|(f, _, _)| f.clone()).collect();
    leaves.sort();
    leaves.dedup();
    (Taxonomy { leaves, notes: taxonomy.notes }, filings)
}

/// Turn filings into concrete, collision-free destination paths.
fn to_assignments(digests: &[Digest], filings: &[(String, f32, String)]) -> Vec<Assignment> {
    let mut taken: HashSet<PathBuf> = HashSet::new();
    let mut assignments = Vec::with_capacity(digests.len());

    for (digest, (folder, confidence, reason)) in digests.iter().zip(filings) {
        let dir = naming::sanitize_folder(folder);
        let name = naming::file_name(&digest.title, &digest.path);
        let dest = naming::unique(&mut taken, dir.join(name));

        assignments.push(Assignment {
            hash: digest.hash.clone(),
            source: digest.path.clone(),
            dest,
            leaf: folder.clone(),
            title: digest.title.clone(),
            confidence: *confidence,
            reason: reason.clone(),
            digest: Some(digest.clone()),
        });
    }

    assignments.sort_by(|a, b| a.dest.cmp(&b.dest));
    assignments
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filing(folder: &str) -> (String, f32, String) {
        (folder.to_string(), 0.9, String::new())
    }

    #[test]
    fn thin_leaves_fold_into_their_parent() {
        let tax = Taxonomy {
            leaves: vec!["A/B/C".into(), "A/B/D".into()],
            notes: vec![],
        };
        let filings = vec![filing("A/B/C"), filing("A/B/D"), filing("A/B/D")];
        let (tax, filings) = fold_thin_leaves(tax, filings, 2);
        assert_eq!(filings[0].0, "A/B", "the single-document leaf folds up");
        assert_eq!(filings[1].0, "A/B/D", "the leaf that met the threshold stays");
        assert!(tax.leaves.contains(&"A/B".to_string()));
    }

    #[test]
    fn unsorted_never_folds_away() {
        let tax = Taxonomy { leaves: vec![UNSORTED.into()], notes: vec![] };
        let (_, filings) = fold_thin_leaves(tax, vec![filing(UNSORTED)], 5);
        assert_eq!(filings[0].0, UNSORTED);
    }

    #[test]
    fn top_level_leaves_have_no_parent_to_fold_into() {
        let tax = Taxonomy { leaves: vec!["Solo".into()], notes: vec![] };
        let (_, filings) = fold_thin_leaves(tax, vec![filing("Solo")], 5);
        assert_eq!(filings[0].0, "Solo");
    }

    #[test]
    fn unknown_titles_fall_back_to_the_file_name() {
        // The original name is preserved verbatim, punctuation and all.
        assert_eq!(fallback_title(std::path::Path::new("/a/PZO30102E.pdf")), "PZO30102E");
        assert_eq!(fallback_title(std::path::Path::new("/a/some_map.name.pdf")), "some_map.name");
    }

    #[test]
    fn title_key_ignores_punctuation_and_case() {
        assert_eq!(title_key("Ghosts of Saltmarsh"), title_key("ghosts  of saltmarsh"));
        assert_eq!(title_key("Skull & Shackles: Part 6"), "skull shackles part 6");
        assert_ne!(title_key("Volo's Guide"), title_key("Xanathar's Guide"));
    }

    #[test]
    fn identical_titles_get_distinct_destinations() {
        let make = |path: &str| Digest {
            hash: path.into(),
            path: path.into(),
            title: "Same Title".into(),
            game_system: "x".into(),
            doc_type: "x".into(),
            setting: "x".into(),
            level_range: "x".into(),
            publisher: "x".into(),
            topics: vec![],
            summary: String::new(),
            confidence: 0.9,
            source: "test".into(),
        };
        let digests = vec![make("a.pdf"), make("b.pdf")];
        let filings = vec![filing("A"), filing("A")];
        let out = to_assignments(&digests, &filings);
        assert_ne!(out[0].dest, out[1].dest);
    }
}
