//! Embedded llama.cpp backend. Runs a local GGUF model, offloaded to the GPU.
//!
//! All three jobs are grammar-constrained, so the model physically cannot emit
//! malformed JSON or name a folder that is not in the taxonomy.

use anyhow::{anyhow, Context, Result};
use std::num::NonZeroU32;
use std::path::Path;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;

use super::{engine, extract_json, prompt, Brain, DigestFields, Filing};
use crate::config::{ModelConfig, TaxonomyConfig};
use crate::types::{Digest, Probe, Taxonomy};

pub struct LlamaBrain {
    // Declaration order is drop order: the context must go before the model.
    ctx: LlamaContext<'static>,
    model: &'static LlamaModel,
    cfg: ModelConfig,
    label: String,
}

/// Load a GGUF model and open one context against it.
///
/// The model is deliberately leaked: `LlamaContext` borrows it for its lifetime,
/// and one long-lived model per process is the only shape this program needs.
pub(super) fn load_model(path: &Path, gpu_layers: u32) -> Result<&'static LlamaModel> {
    if !path.exists() {
        return Err(anyhow!(
            "model not found at {}\nDownload a GGUF and point the config at it.",
            path.display()
        ));
    }
    let backend = engine::backend()?;
    let params = LlamaModelParams::default().with_n_gpu_layers(gpu_layers);
    Ok(Box::leak(Box::new(
        LlamaModel::load_from_file(backend, path, &params)
            .with_context(|| format!("loading model {}", path.display()))?,
    )))
}

pub(super) fn open_context(model: &'static LlamaModel, context: u32) -> Result<LlamaContext<'static>> {
    let params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(context))
        // A whole prompt is submitted as one batch, so n_batch must be able to
        // hold the largest prompt the context allows.
        .with_n_batch(context);
    model.new_context(engine::backend()?, params).context("creating llama context")
}

pub(super) fn label(prefix: &str, path: &Path) -> String {
    path.file_stem()
        .map(|s| format!("{prefix}:{}", s.to_string_lossy()))
        .unwrap_or_else(|| prefix.to_string())
}

impl LlamaBrain {
    pub fn load(cfg: &ModelConfig) -> Result<Self> {
        let model = load_model(&cfg.path, cfg.gpu_layers)?;
        let ctx = open_context(model, cfg.context)?;
        Ok(Self { ctx, model, cfg: cfg.clone(), label: label("llama", &cfg.path) })
    }

    /// Exactly how many tokens a prompt costs, according to the real tokeniser.
    fn prompt_tokens(&self, system: &str, user: &str) -> usize {
        let prompt = engine::wrap(&self.cfg.template, system, user);
        self.model
            .str_to_token(&prompt, llama_cpp_2::model::AddBos::Always)
            .map(|t| t.len())
            .unwrap_or(usize::MAX)
    }

    /// The largest leading slice of `lines` whose prompt and reply both fit.
    ///
    /// Measured with the model's own tokeniser rather than estimated. Every
    /// estimate of this has been wrong — seventeen tokens per candidate, then
    /// twelve for the reply, against a real cost near twenty-two — and being
    /// wrong is silent: the call is refused and the document quietly keeps its
    /// unrefined headings.
    fn fitting_batch(&self, lines: &[(usize, String, usize)]) -> usize {
        let budget = (self.cfg.context as usize).saturating_sub(SAFETY_MARGIN);
        let system = prompt::headings_system();
        let mut take = lines.len().min(MAX_HEADING_BATCH);

        while take > 1 {
            let user = prompt::headings_user(&lines[..take]);
            let cost = self
                .prompt_tokens(&system, &user)
                .saturating_add(heading_reply_budget(take) as usize);
            if cost <= budget {
                return take;
            }
            // Aim at the overshoot rather than halving blindly, so even a very
            // long document settles in two or three measurements.
            let over = cost as f64 / budget as f64;
            take = (((take as f64 / over) * 0.9) as usize).clamp(1, take - 1);
        }
        1
    }

    /// One grammar-constrained completion. Deterministic: greedy sampling.
    fn complete(
        &mut self,
        system: &str,
        user: &str,
        grammar: &str,
        max_tokens: i32,
    ) -> Result<String> {
        let prompt = engine::wrap(&self.cfg.template, system, user);
        self.ctx.clear_kv_cache();
        let pos = engine::prefill(self.model, &mut self.ctx, &prompt)?;

        let budget = self.cfg.context as i32;
        if pos + max_tokens >= budget {
            return Err(anyhow!(
                "prompt is {pos} tokens, which leaves no room for {max_tokens} of output in a {budget}-token context"
            ));
        }

        engine::generate(self.model, &mut self.ctx, pos, grammar, max_tokens)
    }

    fn complete_json<T: serde::de::DeserializeOwned>(
        &mut self,
        system: &str,
        user: &str,
        grammar: &str,
        max_tokens: i32,
    ) -> Result<T> {
        let raw = self.complete(system, user, grammar, max_tokens)?;
        let json =
            extract_json(&raw).ok_or_else(|| anyhow!("model produced no JSON value: {raw:?}"))?;
        serde_json::from_str(json).with_context(|| format!("parsing model output {json:?}"))
    }
}

impl Brain for LlamaBrain {
    fn name(&self) -> &str {
        &self.label
    }

    fn digest(&mut self, probe: &Probe) -> Result<DigestFields> {
        self.complete_json(
            &prompt::digest_system(),
            &prompt::digest_user(probe),
            &prompt::digest_grammar(),
            self.cfg.max_tokens,
        )
    }

    fn design_taxonomy(&mut self, corpus: &[Digest], cfg: &TaxonomyConfig) -> Result<Taxonomy> {
        // The catalogue can outgrow the context window. Fold it down by
        // designing on a batch, then re-designing over the previous leaves plus
        // the next batch, so late documents can still bend the tree.
        let batch = catalogue_batch_size(self.cfg.context);
        let mut leaves: Vec<String> = Vec::new();

        for chunk in corpus.chunks(batch) {
            let mut user = prompt::taxonomy_user(chunk);
            if !leaves.is_empty() {
                user.push_str(&format!(
                    "\nFolders already designed for the rest of the collection — keep the ones \
that still fit and merge rather than duplicate:\n{}\n",
                    leaves.join("\n")
                ));
            }
            let designed: Vec<String> = self.complete_json(
                &prompt::taxonomy_system(cfg),
                &user,
                &prompt::taxonomy_grammar(cfg),
                (cfg.max_leaves * 24).min(4096) as i32,
            )?;
            leaves = designed;
        }

        leaves.retain(|l| !l.trim().is_empty());
        leaves.sort();
        leaves.dedup();
        if leaves.is_empty() {
            return Err(anyhow!("model designed an empty taxonomy"));
        }
        Ok(Taxonomy { leaves, notes: Vec::new() })
    }

    fn refine_headings(
        &mut self,
        lines: &[(usize, String, usize)],
        max_level: usize,
    ) -> Result<Option<Vec<(usize, usize)>>> {
        if lines.is_empty() {
            return Ok(None);
        }
        // A long book offers more candidates than the window holds, so they are
        // judged in batches. Indices stay absolute, so order is preserved
        // however the batches fall.
        let mut kept: Vec<(usize, usize)> = Vec::new();
        let highest = lines.iter().map(|(i, _, _)| *i).max().unwrap_or(0);
        let mut rest = lines;

        while !rest.is_empty() {
            let take = self.fitting_batch(rest);
            let (chunk, remainder) = rest.split_at(take);
            rest = remainder;

            let picks: Vec<super::Selection> = self.complete_json(
                &prompt::headings_system(),
                &prompt::headings_user(chunk),
                &prompt::headings_grammar(highest, max_level),
                heading_reply_budget(chunk.len()),
            )?;
            // The grammar bounds the shape, not the meaning: an index outside
            // this batch is a mistake, and dropping it costs one heading rather
            // than corrupting the outline.
            let valid: std::collections::HashSet<usize> =
                chunk.iter().map(|(i, _, _)| *i).collect();
            kept.extend(
                picks
                    .into_iter()
                    .filter(|p| valid.contains(&p.i))
                    .map(|p| (p.i, p.l.min(max_level.saturating_sub(1)))),
            );
        }

        kept.sort_by_key(|(i, _)| *i);
        kept.dedup_by_key(|(i, _)| *i);
        Ok(Some(kept))
    }

    fn file(&mut self, digest: &Digest, taxonomy: &Taxonomy) -> Result<Filing> {
        let filing: Filing = self.complete_json(
            &prompt::assign_system(),
            &prompt::assign_user(digest, taxonomy),
            &prompt::assign_grammar(taxonomy),
            192,
        )?;
        // The grammar guarantees this, but a taxonomy leaf containing a quote
        // could in principle escape it. Verify rather than trust.
        if !taxonomy.contains(&filing.folder) {
            return Err(anyhow!(
                "model chose folder {:?}, which is not in the taxonomy",
                filing.folder
            ));
        }
        Ok(filing)
    }
}

/// Roughly how many catalogue lines fit alongside the instructions.
/// Tokens one kept candidate costs in the reply.
///
/// `{"i":123,"l":1},` is about eleven; the allowance is deliberately generous
/// because running out mid-array yields unparseable JSON and the entire answer
/// is discarded, not merely shortened.
const HEADING_REPLY_TOKENS: usize = 16;
/// Fixed part of the reply budget, on top of the per-candidate cost.
const REPLY_SLACK: usize = 64;
/// Room left over the measured prompt, for the chat template and rounding.
const SAFETY_MARGIN: usize = 256;
/// Never ask about more than this many candidates at once, however large the
/// window: a reply of thousands of entries is slow and hard to check.
const MAX_HEADING_BATCH: usize = 200;

/// Output budget for a batch of `n` candidates, in the worst case where the
/// model keeps every one of them.
fn heading_reply_budget(n: usize) -> i32 {
    (n * HEADING_REPLY_TOKENS + REPLY_SLACK) as i32
}


fn catalogue_batch_size(context: u32) -> usize {
    // ~30 tokens per catalogue line, leaving half the window for output and slack.
    ((context as usize / 2) / 30).clamp(20, 400)
}

#[cfg(test)]
mod budget {
    use super::*;

    /// The worst case is the model keeping every candidate, which happens on
    /// books whose large type really is all chapter headings.
    #[test]
    fn a_reply_budget_covers_every_candidate_being_kept() {
        for n in [1usize, 50, 200] {
            assert!(heading_reply_budget(n) as usize >= n * 11 + 8, "{n} candidates");
        }
    }

    #[test]
    fn a_catalogue_batch_leaves_room_to_answer() {
        for context in [4096u32, 8192, 32768] {
            let n = catalogue_batch_size(context);
            assert!(n * 30 < context as usize, "context {context}: {n} lines is too many");
        }
    }
}
