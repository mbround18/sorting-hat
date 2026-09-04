# sorting-hat

Takes a directory of PDFs that were fire-hosed in with no structure, reads what
each one actually is using a local model on the GPU, designs a folder tree that
fits *that* collection, and files everything into it.

Built for a 406-document, 14 GB pile of tabletop RPG PDFs, but nothing in it is
D&D-specific except a keyword table in the fallback backend.

## How it works

Three passes, because the interesting part is the second one.

1. **Read.** Every PDF is fingerprinted, deduplicated, and probed — embedded
   metadata plus the text of its opening pages. The model turns that into a
   catalogue entry: title, game system, document type, setting, level range,
   topics, and how sure it is.
2. **Design.** The whole catalogue goes back to the model, which designs a
   folder tree *for this collection*: no empty branches, no categories for
   material that isn't there. Then each document is filed into that tree.
3. **Apply.** Nothing has moved yet. You read the plan; if you like it, you
   apply it, and every created file is recorded so the run can be reversed.

Every model call is constrained by a GBNF grammar. The digest grammar makes
malformed JSON unrepresentable, and the filing grammar is generated from the
taxonomy itself — so the model *cannot* file a document into a folder that
doesn't exist. That removes the failure mode where an LLM sorter quietly invents
a 400th category.

## Use

    # read everything and propose a library — changes nothing on disk
    sorting-hat plan --config sorting-hat.toml

    # look at what it wants to do, in detail
    sorting-hat show --full

    # do it
    sorting-hat apply

    # change your mind
    sorting-hat undo

Default mode is `--mode hardlink`: the library is built out of hard links, so it
costs no extra disk and the originals are never touched. `--mode symlink`,
`copy` and `move` are also available; `move` is the only destructive one.

Digests are cached by content fingerprint, so a second run only reads documents
that are new — and renaming a source file doesn't invalidate its entry.

## Backends

- `--backend llama` — a local GGUF model, all layers on the GPU. This is the one
  that produces a good library.
- `--backend heuristic` — keyword rules, no model, no GPU. It exists so the
  pipeline can be exercised and tested without a 10 GB download, and as a
  fallback if the model fails to load. Its output is noticeably worse.
- `--backend auto` (default) — the model if it's available, otherwise rules.

## Building

    cargo build --release --features cuda      # GPU
    cargo build --release --features llama     # CPU-only model
    cargo build --release                      # heuristic only, no native deps

See `build-notes.md` for the toolchain pins this machine needs.

## Layout

    src/scan.rs      find PDFs, fingerprint, deduplicate
    src/extract.rs   metadata + text probe (poppler, with a pure-Rust fallback)
    src/brain/       the understanding layer: prompts, grammars, two backends
    src/pipeline.rs  stage orchestration, taxonomy folding, path assignment
    src/naming.rs    turning model output into paths that survive a file system
    src/apply.rs     execution and undo
