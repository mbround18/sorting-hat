## What this changes

<!-- What it does, and why. If a real document taught you this, say what the
     document did — that is the part that cannot be recovered from the diff. -->

## How you know it works

<!-- What you ran and what you observed. "Tests pass" is necessary but rarely
     sufficient here: most of this program's failures were silent, and looked
     exactly like success until someone checked the output. -->

## Checklist

- [ ] `cargo test --features cuda` passes
- [ ] `cargo clippy` is clean
- [ ] A bug fix comes with a test, and the test's doc comment says what went wrong
- [ ] Nothing new deletes or overwrites data without being asked
- [ ] Nothing new writes into the source tree, or through a hard link or symlink
- [ ] Any new model call is constrained by a grammar, and its output is still
      validated afterwards
- [ ] Only one lopdf rewrite per document — it cannot reliably re-read its own
      output (see CONTRIBUTING.md)

## Anything you are unsure about

<!-- Perfectly fine to open a PR with open questions. -->
