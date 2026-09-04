//! Embedded llama.cpp backend. Runs a local GGUF model, offloaded to the GPU.
//!
//! All three jobs are grammar-constrained, so the model physically cannot emit
//! malformed JSON or name a folder that is not in the taxonomy.

use anyhow::{anyhow, Context, Result};
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Mutex;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

use super::{extract_json, prompt, Brain, DigestFields, Filing};
use crate::config::{ModelConfig, TaxonomyConfig};
use crate::types::{Digest, Probe, Taxonomy};

pub struct LlamaBrain {
    // Declaration order is the drop order: the context must go before the
    // model, and the model before the backend.
    ctx: Mutex<LlamaContext<'static>>,
    model: &'static LlamaModel,
    _backend: &'static LlamaBackend,
    cfg: ModelConfig,
    label: String,
}

impl LlamaBrain {
    /// Load a GGUF model and open one context against it.
    ///
    /// The backend and model are deliberately leaked: `LlamaContext` borrows the
    /// model for its lifetime, and a single long-lived engine per process is the
    /// only shape this program needs.
    pub fn load(cfg: &ModelConfig) -> Result<Self> {
        if !Path::new(&cfg.path).exists() {
            return Err(anyhow!(
                "model not found at {}\nDownload a GGUF and point model.path at it.",
                cfg.path.display()
            ));
        }

        // llama.cpp writes a great deal to stderr by default — model tensors,
        // CUDA graph churn. Route it through tracing and leave it off unless the
        // user raises the log level.
        llama_cpp_2::send_logs_to_tracing(
            llama_cpp_2::LogOptions::default().with_logs_enabled(false),
        );

        let backend: &'static LlamaBackend =
            Box::leak(Box::new(LlamaBackend::init().context("initialising llama.cpp")?));

        let params = LlamaModelParams::default().with_n_gpu_layers(cfg.gpu_layers);
        let model: &'static LlamaModel = Box::leak(Box::new(
            LlamaModel::load_from_file(backend, &cfg.path, &params)
                .with_context(|| format!("loading model {}", cfg.path.display()))?,
        ));

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(cfg.context))
            // A whole prompt is submitted as one batch, so n_batch must be able
            // to hold the largest prompt the context allows.
            .with_n_batch(cfg.context);
        let ctx = model.new_context(backend, ctx_params).context("creating llama context")?;

        let label = cfg
            .path
            .file_stem()
            .map(|s| format!("llama:{}", s.to_string_lossy()))
            .unwrap_or_else(|| "llama".to_string());

        Ok(Self { ctx: Mutex::new(ctx), model, _backend: backend, cfg: cfg.clone(), label })
    }

    /// One grammar-constrained completion. Deterministic: greedy sampling.
    fn complete(&self, system: &str, user: &str, grammar: &str, max_tokens: i32) -> Result<String> {
        let prompt = wrap(&self.cfg.template, system, user);
        let tokens = self
            .model
            .str_to_token(&prompt, AddBos::Always)
            .context("tokenising prompt")?;

        let budget = self.cfg.context as usize;
        if tokens.len() + max_tokens as usize >= budget {
            return Err(anyhow!(
                "prompt is {} tokens, which leaves no room for {max_tokens} of output in a {budget}-token context",
                tokens.len()
            ));
        }

        let mut ctx = self.ctx.lock().map_err(|_| anyhow!("llama context poisoned"))?;
        ctx.clear_kv_cache();

        let mut batch = LlamaBatch::new(tokens.len().max(512), 1);
        let last = tokens.len() - 1;
        for (i, token) in tokens.iter().enumerate() {
            batch.add(*token, i as i32, &[0], i == last)?;
        }
        ctx.decode(&mut batch).context("decoding prompt")?;

        let grammar = LlamaSampler::grammar(self.model, grammar, "root")
            .map_err(|err| anyhow!("invalid grammar: {err}"))?;
        let mut sampler = LlamaSampler::chain_simple([grammar, LlamaSampler::greedy()]);

        // Accumulate bytes rather than strings: a multi-byte character can be
        // split across two tokens, and decoding each piece alone would mangle it.
        let mut out: Vec<u8> = Vec::new();
        let mut pos = batch.n_tokens();
        for _ in 0..max_tokens {
            // `sample` accepts the token into the sampler chain itself; calling
            // `accept` as well advances the grammar twice and aborts llama.cpp.
            let token = sampler.sample(&ctx, -1);
            if self.model.is_eog_token(token) {
                break;
            }
            out.extend_from_slice(&self.model.token_to_piece_bytes(token, 32, false, None)?);

            batch.clear();
            batch.add(token, pos, &[0], true)?;
            pos += 1;
            ctx.decode(&mut batch).context("decoding generated token")?;
        }

        Ok(String::from_utf8_lossy(&out).into_owned())
    }

    fn complete_json<T: serde::de::DeserializeOwned>(
        &self,
        system: &str,
        user: &str,
        grammar: &str,
        max_tokens: i32,
    ) -> Result<T> {
        let raw = self.complete(system, user, grammar, max_tokens)?;
        let json = extract_json(&raw)
            .ok_or_else(|| anyhow!("model produced no JSON value: {raw:?}"))?;
        serde_json::from_str(json).with_context(|| format!("parsing model output {json:?}"))
    }
}

/// Wrap system and user turns in the model's chat format.
fn wrap(template: &str, system: &str, user: &str) -> String {
    match template {
        "llama3" => format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        "mistral" => format!("<s>[INST] {system}\n\n{user} [/INST]"),
        "plain" => format!("{system}\n\n{user}\n\n"),
        // chatml, and the default for anything unrecognised.
        _ => format!(
            "<|im_start|>system\n{system}<|im_end|>\n\
             <|im_start|>user\n{user}<|im_end|>\n\
             <|im_start|>assistant\n"
        ),
    }
}

impl Brain for LlamaBrain {
    fn name(&self) -> &str {
        &self.label
    }

    fn digest(&self, probe: &Probe) -> Result<DigestFields> {
        self.complete_json(
            &prompt::digest_system(),
            &prompt::digest_user(probe),
            &prompt::digest_grammar(),
            self.cfg.max_tokens,
        )
    }

    fn design_taxonomy(&self, corpus: &[Digest], cfg: &TaxonomyConfig) -> Result<Taxonomy> {
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

    fn file(&self, digest: &Digest, taxonomy: &Taxonomy) -> Result<Filing> {
        let filing: Filing = self.complete_json(
            &prompt::assign_system(),
            &prompt::assign_user(digest, taxonomy),
            &prompt::assign_grammar(taxonomy),
            192,
        )?;
        // The grammar guarantees this, but a taxonomy leaf containing a quote
        // could in principle escape it. Verify rather than trust.
        if !taxonomy.contains(&filing.folder) {
            return Err(anyhow!("model chose folder {:?}, which is not in the taxonomy", filing.folder));
        }
        Ok(filing)
    }
}

/// Roughly how many catalogue lines fit alongside the instructions.
fn catalogue_batch_size(context: u32) -> usize {
    // ~30 tokens per catalogue line, leaving half the window for output and slack.
    ((context as usize / 2) / 30).clamp(20, 400)
}
