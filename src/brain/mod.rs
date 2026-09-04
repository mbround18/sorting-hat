//! The pluggable "understanding" layer.
//!
//! Two backends implement the same three jobs: describe a document, design a
//! taxonomy for the corpus, and file a document into it. The llama backend does
//! it with a local GGUF model on the GPU; the heuristic backend does it with
//! rules, so the rest of the pipeline can be exercised without a model.

pub mod heuristic;
#[cfg(feature = "llama")]
pub mod engine;
#[cfg(feature = "llama")]
pub mod llama;
#[cfg(feature = "llama")]
pub mod vision;
pub mod prompt;

use anyhow::Result;
use serde::Deserialize;

use crate::config::TaxonomyConfig;
use crate::types::{Digest, Probe, Taxonomy};

/// A filing decision for one document.
#[derive(Debug, Clone, Deserialize)]
pub struct Filing {
    pub folder: String,
    pub confidence: f32,
    #[serde(default)]
    pub reason: String,
}

/// What a backend extracts from a single document, before path/hash are attached.
#[derive(Debug, Clone, Deserialize)]
pub struct DigestFields {
    pub title: String,
    pub game_system: String,
    pub doc_type: String,
    #[serde(default = "unknown")]
    pub setting: String,
    #[serde(default = "unknown")]
    pub level_range: String,
    #[serde(default = "unknown")]
    pub publisher: String,
    #[serde(default)]
    pub topics: Vec<String>,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub confidence: f32,
}

fn unknown() -> String {
    "unknown".to_string()
}

/// Inference is inherently sequential here — one model, one context — so this
/// deliberately does not require `Send + Sync`. Parallelism in the pipeline
/// belongs to text extraction, which never touches a backend.
pub trait Brain {
    /// Backend identifier, recorded on every digest for provenance.
    fn name(&self) -> &str;

    fn digest(&mut self, probe: &Probe) -> Result<DigestFields>;

    fn design_taxonomy(&mut self, corpus: &[Digest], cfg: &TaxonomyConfig) -> Result<Taxonomy>;

    fn file(&mut self, digest: &Digest, taxonomy: &Taxonomy) -> Result<Filing>;
}

/// Extract the first balanced JSON value from a model response.
///
/// Grammar-constrained output should already be clean, but a backend running
/// without grammar support can still wrap it in prose or fences.
pub fn extract_json(text: &str) -> Option<&str> {
    let bytes = text.as_bytes();
    let start = bytes.iter().position(|&b| b == b'{' || b == b'[')?;
    let open = bytes[start];
    let close = if open == b'{' { b'}' } else { b']' };

    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            _ if b == open => depth += 1,
            _ if b == close => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[start..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::extract_json;

    #[test]
    fn finds_object_in_prose() {
        assert_eq!(extract_json("sure! {\"a\":1} done"), Some("{\"a\":1}"));
    }

    #[test]
    fn ignores_braces_inside_strings() {
        assert_eq!(extract_json(r#"{"a":"}"}"#), Some(r#"{"a":"}"}"#));
    }

    #[test]
    fn handles_arrays_and_nesting() {
        assert_eq!(extract_json("x [1,[2],3] y"), Some("[1,[2],3]"));
    }

    #[test]
    fn none_when_unbalanced() {
        assert_eq!(extract_json("{\"a\":1"), None);
    }
}
