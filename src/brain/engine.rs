//! Shared llama.cpp plumbing: the process-wide backend, prompt formatting, and
//! the grammar-constrained generation loop that both backends run.

use anyhow::{anyhow, Context, Result};
use std::sync::OnceLock;

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;

/// The llama.cpp backend, initialised once and shared by every model.
///
/// `LlamaBackend::init` may only be called once per process, and both the text
/// and vision models need it, so it is created on first use and leaked.
pub fn backend() -> Result<&'static LlamaBackend> {
    static BACKEND: OnceLock<Result<&'static LlamaBackend, String>> = OnceLock::new();

    BACKEND
        .get_or_init(|| {
            // llama.cpp is noisy on stderr — model tensors, CUDA graph churn.
            // Route it through tracing and leave it off unless asked for.
            llama_cpp_2::send_logs_to_tracing(
                llama_cpp_2::LogOptions::default().with_logs_enabled(false),
            );
            LlamaBackend::init()
                .map(|backend| &*Box::leak(Box::new(backend)))
                .map_err(|err| err.to_string())
        })
        .as_ref()
        .copied()
        .map_err(|err| anyhow!("initialising llama.cpp: {err}"))
}

/// Wrap system and user turns in the model's chat format.
pub fn wrap(template: &str, system: &str, user: &str) -> String {
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

/// Generate under a GBNF grammar, starting from an already-decoded prompt.
///
/// `start_pos` is the position the prompt ended at: the caller has decoded it
/// with logits requested on the final token.
pub fn generate(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    start_pos: i32,
    grammar: &str,
    max_tokens: i32,
) -> Result<String> {
    let grammar = LlamaSampler::grammar(model, grammar, "root")
        .map_err(|err| anyhow!("invalid grammar: {err}"))?;
    let mut sampler = LlamaSampler::chain_simple([grammar, LlamaSampler::greedy()]);

    // Accumulate bytes rather than strings: a multi-byte character can be split
    // across two tokens, and decoding each piece alone would mangle it.
    let mut out: Vec<u8> = Vec::new();
    let mut batch = LlamaBatch::new(1, 1);
    let mut pos = start_pos;

    for _ in 0..max_tokens {
        // `sample` accepts the token into the sampler chain itself; calling
        // `accept` as well advances the grammar twice and aborts llama.cpp.
        let token = sampler.sample(ctx, -1);
        if model.is_eog_token(token) {
            break;
        }
        out.extend_from_slice(&model.token_to_piece_bytes(token, 32, false, None)?);

        batch.clear();
        batch.add(token, pos, &[0], true)?;
        pos += 1;
        ctx.decode(&mut batch).context("decoding generated token")?;
    }

    Ok(String::from_utf8_lossy(&out).into_owned())
}

/// Decode a plain text prompt, leaving logits on its final token.
pub fn prefill(model: &LlamaModel, ctx: &mut LlamaContext, prompt: &str) -> Result<i32> {
    use llama_cpp_2::model::AddBos;

    let tokens = model.str_to_token(prompt, AddBos::Always).context("tokenising prompt")?;
    let last = tokens
        .len()
        .checked_sub(1)
        .ok_or_else(|| anyhow!("prompt tokenised to nothing"))?;

    let mut batch = LlamaBatch::new(tokens.len(), 1);
    for (i, token) in tokens.iter().enumerate() {
        batch.add(*token, i as i32, &[0], i == last)?;
    }
    ctx.decode(&mut batch).context("decoding prompt")?;
    Ok(batch.n_tokens())
}
