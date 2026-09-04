# sorting-hat

Takes a directory of PDFs that were fire-hosed in with no structure, reads what
each one actually is using a local model on the GPU, designs a folder tree that
fits *that* collection, and files everything into it.

Built for a 406-document, 14 GB pile of tabletop RPG PDFs, but nothing in it is
D&D-specific except a keyword table in the fallback backend.

## How it works

Three passes, because the interesting part is the second one.

1. **Read.** Every PDF is hashed, deduplicated, and probed — embedded
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

## Duplicates

Two different problems, and each needs its own answer.

**Identical files** are found by SHA-256 over the whole file. Hashing 14 GB
takes about eight seconds on a machine with hardware SHA, so there is no reason
to hash anything less than all of it. The hashes are the ordinary ones — check
any of them with `sha256sum`.

**The same work saved twice** is the harder case, and no hash will ever find it.
A re-download differs from the original by a few kilobytes of metadata or
compression, so its hash is completely different. In the test corpus that was
109 titles — 218 files, 2.70 GB — against a single exact duplicate. They are
matched instead on title, page count, and file size within a tolerance: a
different page count means a different edition, so both are kept. The largest
copy is filed and the rest are listed under `duplicates` in the plan. Nothing is
ever deleted.

Turn the second one off with `[dedupe] near_duplicates = false`.

## Writing the naming back into the PDF

    sorting-hat plan --mode copy
    sorting-hat apply --write-metadata

A folder tree lives only in this tool's head — copy a file out of the library
and the knowledge is gone. `--write-metadata` stamps what the model worked out
into the PDF's own Info dictionary: title, author (the publisher), subject (the
one-line summary) and keywords (system, type, setting, level range, topics). It
travels with the file, into every reader and file manager.

**It refuses to run in `hardlink` or `symlink` mode.** A hard link *is* the
original file — same inode — and a symlink points straight at it, so stamping
either would rewrite your source PDFs in place. Only `copy` and `move` produce a
file that is yours to change. The check happens before anything is created.

Existing metadata is only replaced when the backend was confident (0.6 and
above). Below that, whatever the publisher set wins. This is what lets a
confident reading displace an authoring-tool default like `Diapositiva 1` while
a hesitant guess leaves a real title alone.

Every rewrite is written beside the target and renamed over it, and is checked
for plausibility first: a file that comes back at less than half its size, or
appreciably larger, is discarded and the good PDF left untouched.

### Known limitation

Around 18% of a typical collection (74 of 406 in the test corpus) uses
cross-reference and object streams — `pdfinfo` reports `Optimized: yes`. `lopdf`
cannot write those back, so it rebuilds the file expanded: one 2.8 MB rulebook
came back at 8.2 MB. The size guard catches this and skips the stamp, so those
documents are still filed and renamed correctly — they just keep their original
metadata. Running the library through `qpdf --object-streams=generate` afterwards
would recover the compression if it matters.

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
