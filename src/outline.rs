//! Building a chapter index — a PDF bookmark tree — for documents that have none.
//!
//! The expensive way to find chapters is to read every page. This does not do
//! that. Headings are set in larger type than body text, and `pdftohtml -xml`
//! reports the size of every text run, so a heading is recognisable without
//! understanding a word of it: about half a second for a hundred-page book, and
//! a fiftyfold reduction in what anything smarter has to look at.
//!
//! Type size also gives the hierarchy for free. In a typical rulebook the parts
//! are set at 45pt, chapters at 36, sections at 30 and subsections at 23; those
//! four sizes become four levels of the tree without anyone naming them.

use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

/// One line of text that might be a heading.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// 1-based page number, as the PDF counts them.
    pub page: usize,
    /// Type size in points; larger means higher in the hierarchy.
    pub size: f32,
    /// Distance from the top of the page, used to recover reading order.
    pub top: f32,
    pub text: String,
}

/// A finished bookmark, ready to be written into the document.
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    /// Nesting depth, 0 being top level.
    pub level: usize,
    pub title: String,
    pub page: usize,
}

/// Pull heading candidates out of a PDF by type size.
///
/// `min_ratio` is how much larger than body text a run must be to be considered.
pub fn candidates(path: &Path, min_ratio: f32, timeout_secs: u64) -> Result<Vec<Candidate>> {
    let xml = run_pdftohtml(path, timeout_secs)?;
    Ok(parse(&xml, min_ratio))
}

fn run_pdftohtml(path: &Path, timeout_secs: u64) -> Result<String> {
    let secs = timeout_secs.to_string();
    let output = Command::new("timeout")
        .args([
            secs.as_str(),
            "pdftohtml",
            "-xml",
            "-i", // skip images: only the text and its metrics matter
            "-q",
            "-stdout",
            &path.to_string_lossy(),
        ])
        .output()
        .with_context(|| format!("running pdftohtml on {}", path.display()))?;
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Parse pdftohtml's XML into heading candidates.
///
/// The body text size is taken to be whichever size covers the most characters
/// in the document, which is far more robust than the most *frequent* size:
/// a book with many short headings can have more heading runs than body runs.
fn parse(xml: &str, min_ratio: f32) -> Vec<Candidate> {
    let fonts = font_sizes(xml);
    let runs = text_runs(xml, &fonts);

    let mut weight: BTreeMap<u32, usize> = BTreeMap::new();
    for run in &runs {
        *weight.entry(run.size.to_bits()).or_default() += run.text.chars().count();
    }
    let Some(body) = weight.iter().max_by_key(|(_, chars)| **chars).map(|(bits, _)| f32::from_bits(*bits))
    else {
        return Vec::new();
    };
    if body <= 0.0 {
        return Vec::new();
    }

    let mut out: Vec<Candidate> = runs
        .into_iter()
        .filter(|run| run.size >= body * min_ratio && plausible_heading(&run.text))
        .collect();

    // pdftohtml emits runs in layout order, not reading order, so a sidebar can
    // arrive before the chapter title above it. An outline listed out of order
    // is worse than no outline, so reading order is restored here: down the
    // page, page by page.
    out.sort_by(|a, b| {
        a.page
            .cmp(&b.page)
            .then(a.top.partial_cmp(&b.top).unwrap_or(std::cmp::Ordering::Equal))
    });

    // Same heading on consecutive pages is a running header, not a chapter.
    out.dedup_by(|a, b| a.text == b.text && a.page.abs_diff(b.page) <= 1);
    out
}

/// Whether a line of large text reads like a heading rather than decoration.
fn plausible_heading(text: &str) -> bool {
    let t = text.trim();
    // A drop-cap is one big letter; a page number is just digits.
    if t.chars().count() < 3 || t.chars().count() > 80 {
        return false;
    }
    if !t.chars().any(|c| c.is_alphabetic()) {
        return false;
    }
    // Needs at least one run of letters, so "1 2 3" and "• • •" are rejected.
    t.split_whitespace().any(|w| w.chars().filter(|c| c.is_alphabetic()).count() >= 2)
}

fn font_sizes(xml: &str) -> BTreeMap<String, f32> {
    let mut sizes = BTreeMap::new();
    for chunk in xml.split("<fontspec ").skip(1) {
        let Some(id) = attribute(chunk, "id") else { continue };
        let Some(size) = attribute(chunk, "size").and_then(|s| s.parse::<f32>().ok()) else {
            continue;
        };
        sizes.insert(id, size);
    }
    sizes
}

fn text_runs(xml: &str, fonts: &BTreeMap<String, f32>) -> Vec<Candidate> {
    let mut out = Vec::new();
    let mut page = 0usize;
    let mut rest = xml;

    // Scanned as whole elements rather than by splitting on '<': a heading may
    // contain nested markup, and "Chapter 2: <b>Endings</b>" must not be read
    // as the two fragments "Chapter 2: " and "Endings".
    loop {
        let Some(open) = rest.find('<') else { break };
        let tail = &rest[open..];

        if let Some(after) = tail.strip_prefix("<page ") {
            if let Some(n) = attribute(after, "number").and_then(|s| s.parse().ok()) {
                page = n;
            }
            rest = after;
            continue;
        }

        let Some(after) = tail.strip_prefix("<text ") else {
            rest = &tail[1..];
            continue;
        };
        let Some(head_end) = after.find('>') else { break };
        let (attrs, body_and_rest) = after.split_at(head_end);
        let body_and_rest = &body_and_rest[1..];

        let Some(close) = body_and_rest.find("</text>") else {
            rest = body_and_rest;
            continue;
        };
        let body = &body_and_rest[..close];
        rest = &body_and_rest[close + "</text>".len()..];

        let Some(font) = attribute(attrs, "font") else { continue };
        let Some(size) = fonts.get(&font).copied() else { continue };

        let top = attribute(attrs, "top").and_then(|v| v.parse().ok()).unwrap_or(0.0);
        let text = strip_markup(body);
        if !text.is_empty() {
            out.push(Candidate { page, size, top, text });
        }
    }
    out
}

/// Read `name="value"` out of a tag body.
fn attribute(chunk: &str, name: &str) -> Option<String> {
    let key = format!("{name}=\"");
    let start = chunk.find(&key)? + key.len();
    let end = chunk[start..].find('"')? + start;
    Some(chunk[start..end].to_string())
}

/// Drop nested tags (`<b>`, `<i>`) and decode the few entities pdftohtml emits.
fn strip_markup(body: &str) -> String {
    let mut out = String::new();
    let mut depth = 0usize;
    for ch in body.chars() {
        match ch {
            '<' => depth += 1,
            '>' => depth = depth.saturating_sub(1),
            _ if depth == 0 => out.push(ch),
            _ => {}
        }
    }
    let out = out
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&#160;", " ");
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Turn candidates into a nested outline using their type sizes as levels.
///
/// Sizes are ranked largest first, so the biggest type becomes level 0 whatever
/// its actual point size; a document set entirely in 14pt with 18pt headings
/// nests exactly like one using 45pt parts.
pub fn nest(candidates: &[Candidate], max_depth: usize) -> Vec<Entry> {
    let mut sizes: Vec<f32> = candidates.iter().map(|c| c.size).collect();
    sizes.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    sizes.dedup();

    candidates
        .iter()
        .map(|c| {
            let level = sizes.iter().position(|s| *s == c.size).unwrap_or(0);
            Entry {
                level: level.min(max_depth.saturating_sub(1)),
                title: c.text.clone(),
                page: c.page,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const XML: &str = r##"<pdf2xml>
<page number="1" height="800" width="600">
<fontspec id="0" size="14" family="Serif" color="#000000"/>
<fontspec id="1" size="36" family="Serif" color="#000000"/>
<text top="10" left="10" width="100" height="20" font="1">Chapter 1: Beginnings</text>
<text top="40" left="10" width="100" height="10" font="0">Body text that goes on and on and on for a while.</text>
<text top="60" left="10" width="100" height="10" font="0">More body text, comfortably the bulk of the page.</text>
</page>
<page number="2" height="800" width="600">
<text top="10" left="10" width="100" height="20" font="1">Chapter 2: <b>Endings</b></text>
<text top="40" left="10" width="100" height="10" font="0">Yet more ordinary body text down here.</text>
</page>
</pdf2xml>"##;

    #[test]
    fn finds_headings_by_type_size() {
        let found = parse(XML, 1.6);
        assert_eq!(found.len(), 2, "{found:?}");
        assert_eq!(found[0].text, "Chapter 1: Beginnings");
        assert_eq!(found[0].page, 1);
        assert_eq!(found[1].page, 2);
    }

    #[test]
    fn strips_nested_markup_from_a_heading() {
        let found = parse(XML, 1.6);
        assert_eq!(found[1].text, "Chapter 2: Endings");
    }

    #[test]
    fn body_size_is_decided_by_characters_not_run_count() {
        // Three tiny headings, one long paragraph: the paragraph is the body.
        let xml = r##"<pdf2xml><page number="1">
<fontspec id="0" size="10"/><fontspec id="1" size="30"/>
<text font="1">Aaa</text><text font="1">Bbb</text><text font="1">Ccc</text>
<text font="0">This single paragraph carries far more characters than the three short headings above it do.</text>
</page></pdf2xml>"##;
        let found = parse(xml, 1.6);
        assert_eq!(found.len(), 3, "headings, not the paragraph: {found:?}");
    }

    #[test]
    fn rejects_drop_caps_and_page_numbers() {
        assert!(!plausible_heading("A"));
        assert!(!plausible_heading("42"));
        assert!(!plausible_heading("1 2 3"));
        assert!(plausible_heading("Chapter 1: Beginnings"));
    }

    #[test]
    fn drops_running_headers_repeated_across_facing_pages() {
        let repeated = vec![
            Candidate { page: 4, size: 30.0, top: 0.0, text: "Player's Handbook".into() },
            Candidate { page: 5, size: 30.0, top: 0.0, text: "Player's Handbook".into() },
            Candidate { page: 6, size: 30.0, top: 0.0, text: "Combat".into() },
        ];
        let mut v = repeated.clone();
        v.dedup_by(|a, b| a.text == b.text && a.page.abs_diff(b.page) <= 1);
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn restores_reading_order_within_a_page() {
        // pdftohtml emitted the sidebar first; the chapter title sits above it.
        let xml = r##"<pdf2xml><page number="1">
<fontspec id="0" size="10"/><fontspec id="1" size="30"/>
<text top="400" font="1">A Sidebar Aside</text>
<text top="50" font="1">Chapter One Begins</text>
<text font="0">Body text long enough to be the most common size on this page by far.</text>
</page></pdf2xml>"##;
        let found = parse(xml, 1.6);
        assert_eq!(found[0].text, "Chapter One Begins", "higher on the page comes first");
        assert_eq!(found[1].text, "A Sidebar Aside");
    }

    #[test]
    fn nesting_ranks_sizes_rather_than_trusting_points() {
        let c = vec![
            Candidate { page: 1, size: 45.0, top: 0.0, text: "Part One".into() },
            Candidate { page: 2, size: 36.0, top: 0.0, text: "Chapter One".into() },
            Candidate { page: 3, size: 36.0, top: 0.0, text: "Chapter Two".into() },
            Candidate { page: 4, size: 23.0, top: 0.0, text: "A Section".into() },
        ];
        let nested = nest(&c, 4);
        assert_eq!(nested[0].level, 0);
        assert_eq!(nested[1].level, 1);
        assert_eq!(nested[2].level, 1);
        assert_eq!(nested[3].level, 2);
    }

    #[test]
    fn nesting_respects_the_depth_limit() {
        let c: Vec<Candidate> = (0..6)
            .map(|i| Candidate { page: 1, size: 40.0 - i as f32, top: i as f32, text: format!("H{i}") })
            .collect();
        let nested = nest(&c, 3);
        assert!(nested.iter().all(|e| e.level < 3), "{nested:?}");
    }
}

/// Debug helper: print what the font pass finds for one document.
pub fn dump(path: &Path, min_ratio: f32, max_depth: usize, timeout_secs: u64) -> Result<()> {
    let found = candidates(path, min_ratio, timeout_secs)?;
    let nested = nest(&found, max_depth);
    println!("{} candidates", nested.len());
    for e in &nested {
        println!("{}{}  ..... p{}", "    ".repeat(e.level), e.title, e.page);
    }
    Ok(())
}
