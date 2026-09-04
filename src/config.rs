//! Runtime configuration, loaded from TOML and overridable on the command line.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub source: PathBuf,
    pub work_dir: PathBuf,
    pub library: PathBuf,
    pub extract: ExtractConfig,
    pub model: ModelConfig,
    pub taxonomy: TaxonomyConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ExtractConfig {
    /// How many leading pages to pull text from.
    pub pages: usize,
    /// Characters of probe text handed to the model.
    pub max_chars: usize,
    /// Below this many extracted characters the PDF is treated as a scan.
    pub scanned_threshold: usize,
    /// Seconds before an external extractor call is abandoned.
    pub timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    /// Path to a GGUF model file.
    pub path: PathBuf,
    /// Layers pushed onto the GPU. 999 means "all of them".
    pub gpu_layers: u32,
    pub context: u32,
    pub max_tokens: i32,
    pub seed: u32,
    /// Chat template wrapper. "chatml" suits most instruct GGUFs.
    pub template: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TaxonomyConfig {
    /// Upper bound on leaf folders the model may invent.
    pub max_leaves: usize,
    /// Maximum path depth, e.g. 3 = "System/Type/Subtype".
    pub max_depth: usize,
    /// A leaf with fewer documents than this gets folded into its parent.
    pub min_docs_per_leaf: usize,
    /// Assignments below this confidence land in `_Unsorted` instead.
    pub min_confidence: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            source: PathBuf::from("."),
            work_dir: PathBuf::from("tmp"),
            library: PathBuf::from("tmp/library"),
            extract: ExtractConfig::default(),
            model: ModelConfig::default(),
            taxonomy: TaxonomyConfig::default(),
        }
    }
}

impl Default for ExtractConfig {
    fn default() -> Self {
        Self { pages: 12, max_chars: 6000, scanned_threshold: 200, timeout_secs: 60 }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("models/model.gguf"),
            gpu_layers: 999,
            context: 8192,
            max_tokens: 512,
            seed: 1337,
            template: "chatml".into(),
        }
    }
}

impl Default for TaxonomyConfig {
    fn default() -> Self {
        Self { max_leaves: 40, max_depth: 3, min_docs_per_leaf: 3, min_confidence: 0.45 }
    }
}

impl Config {
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let Some(path) = path else { return Ok(Self::default()) };
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        toml::from_str(&raw).with_context(|| format!("parsing config {}", path.display()))
    }

    pub fn cache_dir(&self) -> PathBuf {
        self.work_dir.join("cache")
    }

    pub fn plan_path(&self) -> PathBuf {
        self.work_dir.join("plan.json")
    }
}
