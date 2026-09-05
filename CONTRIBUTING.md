# Contributing

Thanks for looking. This is a small tool with strong opinions, most of which
were beaten into it by real files rather than reasoned out in advance. That
history is worth knowing before changing things.

## Getting set up

    sorting-hat doctor

That reports the external tools, the models, the GPU and the paths, and says
what each missing piece costs you. Start there — most setup problems this
project has hit were invisible until something failed halfway through a long
run.

You need [poppler-utils](https://poppler.freedesktop.org/) (`pdftotext`,
`pdfinfo`, `pdftoppm`, `pdftohtml`). Everything else is optional:

    cargo build --release                    # no model, heuristic backend only
    cargo build --release --features llama   # local model, CPU
    cargo build --release --features cuda    # local model, GPU

`build-notes.md` records the toolchain pins that the GPU build needs. If your
build fails on `libclang`, `cudart_static` or the CUDA host compiler, the answer
is probably already in there.

## Running the tests

    cargo test --features cuda

The tests do not need a GPU or a model — the feature flag only has to match how
you are building. They run in well under a second, so there is no reason to skip
them.

## What the tests are for

Most of them exist because something went wrong on a real document, and the
failure was **silent**. That is the pattern worth understanding: this program
talks to a language model, a PDF parser and four external binaries, and all four
have failure modes that look like success.

A few worth reading before you change the code they cover:

- `brain::prompt::grammar_shape` — a GBNF grammar must keep one rule per line and
  must keep the escaped quotes in its JSON keys. Get either wrong and llama.cpp
  refuses the grammar at load time, the call falls back, and the only evidence is
  a warning in a log nobody is reading.
- `brain::llama::budget` — a reply that runs out of tokens mid-array is
  unparseable, so the *entire* answer is discarded rather than shortened. Batch
  sizes are now measured with the real tokeniser, not estimated. Three separate
  estimates of the per-candidate cost were wrong before that.
- `extract::a_document_with_too_little_text_declines_to_identify_itself` — an
  image-only PDF yields no text, so every one of them hashes identically. Without
  this guard, thirteen unrelated maps in the test corpus collapsed into one
  document.
- `outline::rejoins_small_caps_headings_split_by_the_extractor` — a heading set in
  small caps arrives as two runs, so `CLASS FEATURES` becomes `LASS EATURES`.
  Grouping by top edge fails, and sorting by baseline fails too; lines have to be
  found first and read left to right afterwards.
- `naming::folder_paths_cannot_escape_the_library` — folder names come from a
  language model, and are used as paths.

If you fix a bug that a real file taught you, please add the test *and* say in
its doc comment what the file did. A test named after its symptom is worth far
more here than one named after its function.

## Things to know before changing the internals

**lopdf cannot reliably re-read what it writes.** Loading 247 source PDFs
succeeded 246 times; loading the copies lopdf produced from them succeeded 156
times. The content is fine and every reader accepts them, but a second lopdf pass
over its own output silently skips about a third of a library. This is why every
enrichment starts again from the pristine source and applies everything in one
load and one save. Do not add a second rewrite pass.

**Writing through a link rewrites the original.** A hard link is the same inode
and a symlink points straight at it, so copying or stamping through one edits the
source. `metadata` and `bookmarks` refuse link modes, and enrichment unlinks a
destination before restoring it. Both guards matter; keep both.

**The model must not be able to invent an identifier.** Filing is constrained by a
grammar generated from the taxonomy, so a folder that does not exist cannot be
named. Heading selection returns indices into a list it was shown, never text.
When adding a model call, prefer a grammar that makes the wrong answer
unrepresentable over validation that catches it afterwards — though validate too.

**Nothing is deleted, ever.** Duplicates are listed, not removed. Documents that
cannot be enriched are reported and left alone. `apply` writes an undo manifest.
Keep it that way.

## Style

Match what is there. Comments explain *why*, especially when the reason is a
specific piece of misbehaviour in a PDF or a model — that is knowledge that
cannot be recovered by reading the code. Commit messages are prose, and say what
the real files taught rather than which functions moved.

## Submitting

Fork, branch, and open a pull request. Please run `cargo test --features cuda`
and `cargo clippy` first, and describe what you actually observed rather than
what you expected. If it involves a specific PDF behaving strangely, say which
kind — that detail is usually the whole story.

By contributing you agree that your work is licensed under both the MIT licence
and the Apache Licence 2.0, as the project is.
