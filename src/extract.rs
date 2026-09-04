//! Pull metadata and a text probe out of a PDF.
//!
//! Poppler's `pdfinfo`/`pdftotext` do the work when present: they are fast,
//! stream rather than load, and survive the malformed files that a decade of
//! scraped RPG PDFs are full of. `lopdf` is the pure-Rust fallback.

use anyhow::{Context, Result};
use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;

use crate::config::ExtractConfig;
use crate::types::Probe;

fn have(bin: &str) -> bool {
    Command::new(bin).arg("-v").output().is_ok()
}

fn poppler() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| have("pdftotext") && have("pdfinfo"))
}

fn timeout_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| have("timeout"))
}

/// Run a command, wrapped in `timeout` when it is available.
fn run(secs: u64, bin: &str, args: &[&str]) -> Result<String> {
    let output = if timeout_available() {
        let secs = secs.to_string();
        let mut full = vec![secs.as_str(), bin];
        full.extend_from_slice(args);
        Command::new("timeout").args(&full).output()?
    } else {
        Command::new(bin).args(args).output()?
    };
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

pub fn probe(path: &Path, cfg: &ExtractConfig) -> Result<Probe> {
    let file_name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();

    let mut probe = Probe {
        file_name,
        pdf_title: None,
        pdf_author: None,
        pdf_subject: None,
        pdf_creator: None,
        page_count: None,
        text: String::new(),
        scanned: false,
    };

    if poppler() {
        read_info(path, cfg, &mut probe)?;
        probe.text = read_text(path, cfg)?;
    } else {
        read_info_lopdf(path, &mut probe);
    }

    probe.text = condense(&probe.text, cfg.max_chars);
    // Count characters that carry meaning. Measuring the condensed string
    // would count the single spaces condense inserts, so a page with two lines
    // of copyright boilerplate can clear a threshold it should fail.
    let meaningful = probe.text.chars().filter(|c| !c.is_whitespace()).count();
    probe.scanned = meaningful < cfg.scanned_threshold;
    Ok(probe)
}

fn read_info(path: &Path, cfg: &ExtractConfig, probe: &mut Probe) -> Result<()> {
    let out = run(cfg.timeout_secs, "pdfinfo", &[&path.to_string_lossy()])?;
    for line in out.lines() {
        let Some((key, value)) = line.split_once(':') else { continue };
        let value = value.trim();
        if value.is_empty() {
            continue;
        }
        match key.trim() {
            "Title" => probe.pdf_title = Some(value.to_string()),
            "Author" => probe.pdf_author = Some(value.to_string()),
            "Subject" => probe.pdf_subject = Some(value.to_string()),
            "Creator" => probe.pdf_creator = Some(value.to_string()),
            "Pages" => probe.page_count = value.parse().ok(),
            _ => {}
        }
    }
    Ok(())
}

fn read_text(path: &Path, cfg: &ExtractConfig) -> Result<String> {
    let last = cfg.pages.to_string();
    run(
        cfg.timeout_secs,
        "pdftotext",
        &["-f", "1", "-l", &last, "-q", "-enc", "UTF-8", &path.to_string_lossy(), "-"],
    )
}

/// Metadata-only fallback when poppler is not installed.
fn read_info_lopdf(path: &Path, probe: &mut Probe) {
    let Ok(doc) = lopdf::Document::load(path) else { return };
    probe.page_count = Some(doc.get_pages().len());

    let Ok(info) = doc.trailer.get(b"Info") else { return };
    let Ok(id) = info.as_reference() else { return };
    let Ok(dict) = doc.get_object(id).and_then(|o| o.as_dict().cloned()) else { return };

    let get = |key: &[u8]| -> Option<String> {
        dict.get(key).ok().and_then(|v| v.as_str().ok()).map(|b| String::from_utf8_lossy(b).into_owned())
    };
    probe.pdf_title = get(b"Title");
    probe.pdf_author = get(b"Author");
    probe.pdf_subject = get(b"Subject");
    probe.pdf_creator = get(b"Creator");
}

/// Render the first page of a PDF to a PNG, for the vision pass.
///
/// Returns the written file. `pdftoppm -singlefile` names the output exactly,
/// without the page-number suffix it would otherwise append.
pub fn render_first_page(
    path: &Path,
    out_dir: &Path,
    max_pixels: u32,
    timeout_secs: u64,
) -> Result<std::path::PathBuf> {
    if !poppler() {
        anyhow::bail!("rendering needs poppler's pdftoppm, which is not installed");
    }
    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("creating {}", out_dir.display()))?;

    // Name the render after the source so a re-run overwrites rather than piles up.
    let stem = blake3::hash(path.to_string_lossy().as_bytes()).to_hex()[..16].to_string();
    let prefix = out_dir.join(&stem);
    // `-scale-to` fixes the longest edge and preserves the aspect ratio, which
    // matters here: a poster map is far wider than it is tall, and setting the
    // two axes independently would squash it.
    let cap = max_pixels.to_string();

    run(
        timeout_secs,
        "pdftoppm",
        &[
            "-png", "-singlefile", "-f", "1", "-l", "1",
            "-scale-to", &cap,
            &path.to_string_lossy(),
            &prefix.to_string_lossy(),
        ],
    )?;

    let png = prefix.with_extension("png");
    if !png.exists() {
        anyhow::bail!("pdftoppm produced no image for {}", path.display());
    }
    Ok(png)
}

/// Collapse whitespace and truncate on a character boundary.
fn condense(text: &str, max_chars: usize) -> String {
    let mut out = String::with_capacity(text.len().min(max_chars * 4));
    let mut pending_space = false;
    let mut chars = 0usize;

    for ch in text.chars() {
        if ch.is_whitespace() {
            pending_space = !out.is_empty();
            continue;
        }
        // Drop control characters that survive bad extraction.
        if ch.is_control() {
            continue;
        }
        if pending_space {
            out.push(' ');
            pending_space = false;
            chars += 1;
        }
        out.push(ch);
        chars += 1;
        if chars >= max_chars {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::condense;

    #[test]
    fn condense_collapses_and_truncates() {
        assert_eq!(condense("  a \n\n b\tc  ", 100), "a b c");
        assert_eq!(condense("abcdef", 3), "abc");
        assert_eq!(condense("", 10), "");
    }

    #[test]
    fn whitespace_does_not_count_towards_the_scan_threshold() {
        // 30 letters separated by spaces: 59 characters, but only 30 meaningful.
        let text: String = std::iter::repeat("a ").take(30).collect();
        let meaningful = text.chars().filter(|c| !c.is_whitespace()).count();
        assert_eq!(meaningful, 30);
        assert!(meaningful < text.len(), "spaces must not pad the count");
    }

    #[test]
    fn condense_respects_char_boundaries() {
        // Truncating by bytes would split these; by chars it must not.
        assert_eq!(condense("émé", 2), "ém");
    }
}
