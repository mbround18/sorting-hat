//! Stamping what the model worked out back into the PDF itself.
//!
//! A tidy folder tree lives only in this tool's head: copy a file elsewhere and
//! the knowledge is gone. Writing the title, author and subject into the PDF's
//! own Info dictionary makes it travel with the document — every reader, every
//! file manager and every future run of this tool can see it.
//!
//! This rewrites the file, so it is only ever safe on a private copy. See
//! [`may_write`].

use anyhow::{anyhow, Context, Result};
use lopdf::{Dictionary, Document, Object};
use std::path::Path;

use crate::naming::is_unknown;
use crate::types::{Digest, LinkMode};

/// Whether stamping metadata is allowed in this link mode.
///
/// Never through a link. A hard link is the same inode as the original and a
/// symlink points straight at it, so stamping either rewrites the source PDF in
/// place — silently editing a library the user asked us only to read.
///
/// `copy` and `move` both yield a file that is the user's to change: a copy is
/// private to the library, and a move has taken the original out of the source
/// tree on purpose.
pub fn may_write(mode: LinkMode) -> Result<()> {
    match mode {
        LinkMode::Copy | LinkMode::Move => Ok(()),
        LinkMode::Hardlink => Err(anyhow!(
            "--write-metadata cannot be used with --mode hardlink.\n\
A hard link is the same file as the original, so stamping it would rewrite the \
source PDFs in place. Re-plan with --mode copy, or --mode move if the originals \
are meant to become the library."
        )),
        LinkMode::Symlink => Err(anyhow!(
            "--write-metadata cannot be used with --mode symlink.\n\
A symlink points at the original, so stamping it would rewrite the source PDF. \
Re-plan with --mode copy, or --mode move if the originals are meant to become \
the library."
        )),
    }
}

/// What gets written into the PDF's Info dictionary.
fn entries(digest: &Digest) -> Vec<(&'static str, String)> {
    let mut out = Vec::new();

    if !is_unknown(&digest.title) {
        out.push(("Title", digest.title.clone()));
    }
    if !is_unknown(&digest.publisher) {
        out.push(("Author", digest.publisher.clone()));
    }
    if !digest.summary.trim().is_empty() {
        out.push(("Subject", digest.summary.clone()));
    }

    // Keywords carry the facts a search would want but no standard field holds:
    // system, type, setting and level range, alongside the model's own topics.
    let mut keywords: Vec<String> = Vec::new();
    for value in [&digest.game_system, &digest.doc_type, &digest.setting, &digest.level_range] {
        if !is_unknown(value) {
            keywords.push(value.clone());
        }
    }
    keywords.extend(digest.topics.iter().filter(|t| !is_unknown(t)).cloned());
    keywords.dedup();
    if !keywords.is_empty() {
        out.push(("Keywords", keywords.join(", ")));
    }

    out
}

/// Set the Info dictionary on an already-loaded document.
///
/// Separate from [`stamp`] so that metadata and a bookmark tree can be written
/// in one load and one save. lopdf cannot always re-read what it just wrote —
/// 91 of 247 files here — so a second pass over its own output is not an
/// option, and every change a document needs must happen together.
pub fn apply_to_doc(doc: &mut Document, digest: &Digest, overwrite: bool) -> Vec<&'static str> {
    let fields = entries(digest);
    if fields.is_empty() {
        return Vec::new();
    }

    let mut info = current_info(doc);
    let mut written: Vec<&'static str> = Vec::new();
    for (key, value) in fields {
        let occupied = info
            .get(key.as_bytes())
            .ok()
            .and_then(|o| o.as_str().ok())
            .is_some_and(|b| !b.is_empty());
        if occupied && !overwrite {
            continue;
        }
        info.set(key, Object::string_literal(value));
        written.push(key);
    }
    if written.is_empty() {
        return written;
    }

    // Record who did this, so a later run can tell a stamped file from an
    // untouched one without guessing.
    info.set("Producer", Object::string_literal("sorting-hat"));

    let id = doc.add_object(Object::Dictionary(info));
    doc.trailer.set("Info", Object::Reference(id));
    written
}

fn current_info(doc: &Document) -> Dictionary {
    doc.trailer
        .get(b"Info")
        .ok()
        .and_then(|o| o.as_reference().ok())
        .and_then(|id| doc.get_object(id).ok())
        .and_then(|o| o.as_dict().ok())
        .cloned()
        .unwrap_or_default()
}

/// Save beside the target then rename over it, so an interrupted or failed
/// write can never leave a half-rewritten PDF where a good one was.
///
/// The rewritten file is also checked for plausibility. lopdf rebuilds the
/// whole document, and a PDF that used object streams comes back expanded —
/// one 2.8 MB rulebook became 8.2 MB in testing. `compress` puts the streams
/// back; if the result is still wildly different from the original, the stamp
/// is abandoned and the good file left alone.
pub fn save_atomically(doc: &mut Document, path: &Path, max_growth: f32) -> Result<()> {
    let before = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    doc.compress();

    let tmp = path.with_extension("pdf.sorting-hat-tmp");
    doc.save(&tmp).with_context(|| format!("writing {}", tmp.display()))?;
    let after = std::fs::metadata(&tmp).map(|m| m.len()).unwrap_or(0);

    if let Err(err) = plausible(before, after, max_growth) {
        let _ = std::fs::remove_file(&tmp);
        return Err(err);
    }

    std::fs::rename(&tmp, path).with_context(|| format!("replacing {}", path.display()))?;
    Ok(())
}

/// Whether a rewritten PDF's size is close enough to the original to trust.
///
/// Metadata is a few hundred bytes, so a small change either way is expected;
/// anything dramatic means the rewrite did something other than what we asked.
fn plausible(before: u64, after: u64, max_growth: f32) -> Result<()> {
    if before == 0 {
        return Ok(());
    }
    if after < before / 2 {
        return Err(anyhow!(
            "rewriting shrank the PDF from {before} to {after} bytes; discarding the result"
        ));
    }
    // A megabyte of slack, so small files are not judged by ratio alone.
    let ceiling = before + ((before as f32 * max_growth) as u64).max(1 << 20);
    if after > ceiling {
        return Err(anyhow!(
            "rewriting grew the PDF from {before} to {after} bytes; discarding the result rather \
than bloating the library"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest() -> Digest {
        Digest {
            hash: "h".into(),
            path: "x.pdf".into(),
            title: "Town Square".into(),
            game_system: "Pathfinder 1e".into(),
            doc_type: "maps".into(),
            setting: "unknown".into(),
            level_range: "unknown".into(),
            publisher: "Paizo".into(),
            topics: vec!["battle map".into(), "town".into()],
            summary: "A gridded town square for encounters.".into(),
            confidence: 0.9,
            source: "vision".into(),
        }
    }

    #[test]
    fn never_stamps_through_a_link() {
        assert!(may_write(LinkMode::Hardlink).is_err());
        assert!(may_write(LinkMode::Symlink).is_err());
        assert!(may_write(LinkMode::Copy).is_ok());
        assert!(may_write(LinkMode::Move).is_ok());
    }

    #[test]
    fn builds_the_expected_fields() {
        let fields = entries(&digest());
        let map: std::collections::HashMap<_, _> = fields.into_iter().collect();
        assert_eq!(map["Title"], "Town Square");
        assert_eq!(map["Author"], "Paizo");
        assert!(map["Subject"].starts_with("A gridded town square"));
        // "unknown" setting and level range must not reach the keywords.
        assert_eq!(map["Keywords"], "Pathfinder 1e, maps, battle map, town");
    }

    #[test]
    fn rejects_implausible_rewrites() {
        let mb = 1u64 << 20;
        assert!(plausible(10 * mb, 10 * mb + 500, 0.25).is_ok(), "a few hundred bytes is normal");
        assert!(plausible(10 * mb, 4 * mb, 0.25).is_err(), "losing half the file is corruption");
        assert!(plausible(10 * mb, 30 * mb, 0.25).is_err(), "tripling the file is bloat");
        assert!(plausible(1000, 1500, 0.25).is_ok(), "small files get absolute slack, not a ratio");
        assert!(plausible(0, 5000, 0.25).is_ok(), "an unknown original cannot be judged");
    }

    #[test]
    fn says_nothing_when_it_knows_nothing() {
        let mut d = digest();
        d.title = "unknown".into();
        d.publisher = "unknown".into();
        d.summary = String::new();
        d.game_system = "unknown".into();
        d.doc_type = "unknown".into();
        d.topics.clear();
        assert!(entries(&d).is_empty(), "nothing known should mean nothing written");
    }
}
