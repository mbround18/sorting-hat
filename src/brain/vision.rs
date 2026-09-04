//! Vision backend for PDFs that carry no extractable text.
//!
//! Map products, poster sheets and image-scanned books are pure pictures: the
//! text model gets nothing but a file name, which for something like
//! `PZO31005E.pdf` is nothing at all. This renders the first page and asks a
//! vision model what it is looking at, so those documents get a real title and
//! a real category instead of being parked.

use anyhow::{anyhow, Context, Result};
use std::path::Path;

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText};

use super::llama::{label, load_model, open_context};
use super::{engine, extract_json, prompt, DigestFields};
use crate::config::VisionConfig;
use crate::types::Probe;

pub struct VisionBrain {
    // Declaration order is drop order: contexts before the model they borrow.
    mtmd: MtmdContext,
    ctx: LlamaContext<'static>,
    model: &'static LlamaModel,
    cfg: VisionConfig,
    label: String,
}

impl VisionBrain {
    pub fn load(cfg: &VisionConfig) -> Result<Self> {
        if !cfg.mmproj.exists() {
            return Err(anyhow!(
                "multimodal projector not found at {}\nA vision model needs both its GGUF and its mmproj file.",
                cfg.mmproj.display()
            ));
        }
        let model = load_model(&cfg.path, cfg.gpu_layers)?;
        let ctx = open_context(model, cfg.context)?;

        let params = MtmdContextParams {
            use_gpu: true,
            print_timings: false,
            n_threads: num_threads(),
            media_marker: std::ffi::CString::new(llama_cpp_2::mtmd::mtmd_default_marker())?,
            // -1 leaves the visual token budget to the model's own default.
            image_min_tokens: -1,
            image_max_tokens: -1,
        };
        let mtmd = MtmdContext::init_from_file(&cfg.mmproj.to_string_lossy(), model, &params)
            .with_context(|| format!("loading projector {}", cfg.mmproj.display()))?;

        if !mtmd.support_vision() {
            return Err(anyhow!(
                "{} is not a vision projector",
                cfg.mmproj.display()
            ));
        }

        Ok(Self { mtmd, ctx, model, cfg: cfg.clone(), label: label("vision", &cfg.path) })
    }

    pub fn name(&self) -> &str {
        &self.label
    }

    /// Look at a rendered page and catalogue the document it came from.
    pub fn describe(&mut self, image: &Path, probe: &Probe) -> Result<DigestFields> {
        let marker = llama_cpp_2::mtmd::mtmd_default_marker();
        let user = format!("{marker}\n{}", prompt::vision_digest_user(probe));
        let text = engine::wrap(&self.cfg.template, &prompt::vision_digest_system(), &user);

        let bitmap = MtmdBitmap::from_file(&self.mtmd, &image.to_string_lossy(), false)
            .with_context(|| format!("reading rendered page {}", image.display()))?;

        let chunks = self
            .mtmd
            .tokenize(
                MtmdInputText { text, add_special: true, parse_special: true },
                &[&bitmap],
            )
            .context("tokenising image prompt")?;

        let budget = self.cfg.context as usize;
        if chunks.total_tokens() + self.cfg.max_tokens as usize >= budget {
            return Err(anyhow!(
                "the rendered page needs {} tokens, which does not fit a {budget}-token context; lower vision.max_pixels",
                chunks.total_tokens()
            ));
        }

        self.ctx.clear_kv_cache();
        // Runs the projector over the image and decodes text and image chunks in
        // order, leaving logits on the final token.
        let pos = chunks
            .eval_chunks(&self.mtmd, &self.ctx, 0, 0, self.cfg.context as i32, true)
            .map_err(|err| anyhow!("evaluating image prompt: {err}"))?;

        let raw = engine::generate(
            self.model,
            &mut self.ctx,
            pos,
            &prompt::digest_grammar(),
            self.cfg.max_tokens,
        )?;

        let json =
            extract_json(&raw).ok_or_else(|| anyhow!("model produced no JSON value: {raw:?}"))?;
        serde_json::from_str(json).with_context(|| format!("parsing model output {json:?}"))
    }
}

fn num_threads() -> i32 {
    std::thread::available_parallelism().map(|n| n.get() as i32).unwrap_or(4)
}

impl crate::pipeline::Eyes for VisionBrain {
    fn name(&self) -> &str {
        VisionBrain::name(self)
    }

    fn describe(&mut self, image: &Path, probe: &Probe) -> Result<DigestFields> {
        VisionBrain::describe(self, image, probe)
    }
}
