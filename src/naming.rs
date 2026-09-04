//! Turning model output into file system paths that survive contact with reality.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Characters no destination component may contain, on any platform we care about.
const FORBIDDEN: &[char] = &['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\0'];

/// Names Windows reserves. Harmless on Linux, but the library may end up synced.
const RESERVED: &[&str] = &[
    "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
    "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
];

/// Longest single path component. ext4 allows 255 bytes; leave room for suffixes.
const MAX_COMPONENT: usize = 180;

/// Whether a backend effectively declined to name something.
pub fn is_unknown(title: &str) -> bool {
    let t = title.trim();
    t.is_empty() || t.eq_ignore_ascii_case("unknown") || t.eq_ignore_ascii_case("untitled")
}

/// Make one path component safe: no separators, no control characters, no
/// leading/trailing dots or spaces, never empty.
pub fn sanitize_component(raw: &str) -> String {
    let mut out: String = raw
        .chars()
        .map(|c| if FORBIDDEN.contains(&c) || c.is_control() { ' ' } else { c })
        .collect();

    out = out.split_whitespace().collect::<Vec<_>>().join(" ");
    out = out.trim_matches(|c: char| c == '.' || c.is_whitespace()).to_string();

    truncate_bytes(&mut out, MAX_COMPONENT);

    if out.is_empty() {
        return "Untitled".to_string();
    }
    if RESERVED.contains(&out.to_uppercase().as_str()) {
        out.push('_');
    }
    out
}

/// Sanitize a slash-separated folder path, dropping empty and traversal segments.
pub fn sanitize_folder(raw: &str) -> PathBuf {
    let mut path = PathBuf::new();
    for segment in raw.split('/') {
        let segment = segment.trim();
        if segment.is_empty() || segment == "." || segment == ".." {
            continue;
        }
        path.push(sanitize_component(segment));
    }
    if path.as_os_str().is_empty() {
        path.push("Unsorted");
    }
    path
}

/// Truncate to a byte budget without splitting a character.
fn truncate_bytes(s: &mut String, max: usize) {
    if s.len() <= max {
        return;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s.truncate(end);
    let trimmed = s.trim_end().to_string();
    *s = trimmed;
}

/// Build a destination file name from a title, preserving the source extension.
pub fn file_name(title: &str, source: &Path) -> String {
    let ext = source
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_else(|| "pdf".to_string());

    let mut stem = sanitize_component(title);
    // Titles sometimes arrive with the extension already glued on.
    if let Some(without) = stem.to_lowercase().strip_suffix(&format!(".{ext}")) {
        stem = stem[..without.len()].trim().to_string();
        if stem.is_empty() {
            stem = "Untitled".into();
        }
    }
    truncate_bytes(&mut stem, MAX_COMPONENT - ext.len() - 8);
    format!("{stem}.{ext}")
}

/// Claim a path, appending ` (2)`, ` (3)`… until it is unique within `taken`.
pub fn unique(taken: &mut HashSet<PathBuf>, candidate: PathBuf) -> PathBuf {
    if taken.insert(candidate.clone()) {
        return candidate;
    }
    let parent = candidate.parent().map(Path::to_path_buf).unwrap_or_default();
    let stem = candidate.file_stem().unwrap_or_default().to_string_lossy().into_owned();
    let ext = candidate.extension().map(|e| e.to_string_lossy().into_owned());

    for n in 2..10_000 {
        let name = match &ext {
            Some(ext) => format!("{stem} ({n}).{ext}"),
            None => format!("{stem} ({n})"),
        };
        let next = parent.join(name);
        if taken.insert(next.clone()) {
            return next;
        }
    }
    unreachable!("ten thousand documents cannot share one title")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_separators_and_control_characters() {
        assert_eq!(sanitize_component("a/b:c\td"), "a b c d");
    }

    #[test]
    fn never_yields_an_empty_component() {
        assert_eq!(sanitize_component("..."), "Untitled");
        assert_eq!(sanitize_component("   "), "Untitled");
    }

    #[test]
    fn folder_paths_cannot_escape_the_library() {
        assert_eq!(sanitize_folder("../../etc"), PathBuf::from("etc"));
        assert_eq!(sanitize_folder("/a//b/"), PathBuf::from("a/b"));
        assert_eq!(sanitize_folder(""), PathBuf::from("Unsorted"));
    }

    #[test]
    fn keeps_the_source_extension_once() {
        assert_eq!(file_name("Ghosts of Saltmarsh", Path::new("x.PDF")), "Ghosts of Saltmarsh.pdf");
        assert_eq!(file_name("Ghosts.pdf", Path::new("x.pdf")), "Ghosts.pdf");
    }

    #[test]
    fn long_titles_stay_within_the_component_limit() {
        let name = file_name(&"ä".repeat(500), Path::new("x.pdf"));
        assert!(name.len() <= 255, "{}", name.len());
    }

    #[test]
    fn recognises_a_declined_title() {
        assert!(is_unknown("unknown"));
        assert!(is_unknown("  Unknown "));
        assert!(is_unknown(""));
        assert!(!is_unknown("Mytheos Dungeon"));
    }

    #[test]
    fn collisions_get_numbered() {
        let mut taken = HashSet::new();
        let a = unique(&mut taken, PathBuf::from("d/x.pdf"));
        let b = unique(&mut taken, PathBuf::from("d/x.pdf"));
        let c = unique(&mut taken, PathBuf::from("d/x.pdf"));
        assert_eq!(a, PathBuf::from("d/x.pdf"));
        assert_eq!(b, PathBuf::from("d/x (2).pdf"));
        assert_eq!(c, PathBuf::from("d/x (3).pdf"));
    }
}
