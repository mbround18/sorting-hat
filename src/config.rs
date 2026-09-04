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
    pub vision: VisionConfig,
    pub taxonomy: TaxonomyConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ExtractConfig {
    /// How many leading pages to pull text from.
    pub pages: usize,
    /// Characters of probe text handed to the model.
    pub max_chars: usize,
    /// Below this many non-whitespace characters across `pages`, the PDF is
    /// treated as an image scan and handed to the vision pass.
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

/// The vision model, used only for PDFs that yield no extractable text.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VisionConfig {
    /// Turn the vision pass off entirely; text-poor PDFs then fall back to
    /// judging by file name alone.
    pub enabled: bool,
    /// A vision-language GGUF.
    pub path: PathBuf,
    /// Its multimodal projector. A VL model needs both files.
    pub mmproj: PathBuf,
    pub gpu_layers: u32,
    pub context: u32,
    pub max_tokens: i32,
    pub template: String,
    /// Longest edge of the rendered page, in pixels; the aspect ratio is kept.
    /// Higher reads finer print but costs image tokens quadratically.
    pub max_pixels: u32,
    /// Upper bound on the tokens the projector may spend on one image. -1 uses
    /// the model's own default. This is what actually bounds the vision
    /// encoder's buffers, so cap it here rather than by starving the render.
    ///
    /// Measured against Qwen2.5-VL-7B: 2048 is safe, 3072 overruns the encoder
    /// and aborts llama.cpp from C, which no Rust error handling can catch.
    /// Raise it only after testing on a handful of documents.
    pub image_max_tokens: i32,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            path: PathBuf::from("models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"),
            mmproj: PathBuf::from("models/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf"),
            gpu_layers: 999,
            context: 8192,
            max_tokens: 512,
            template: "chatml".into(),
            max_pixels: 2100,
            image_max_tokens: 2048,
        }
    }
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
            vision: VisionConfig::default(),
            taxonomy: TaxonomyConfig::default(),
        }
    }
}

impl Default for ExtractConfig {
    fn default() -> Self {
        Self { pages: 12, max_chars: 6000, scanned_threshold: 800, timeout_secs: 60 }
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

    /// Where rendered pages are written for the vision pass.
    pub fn render_dir(&self) -> PathBuf {
        self.work_dir.join("renders")
    }
}
