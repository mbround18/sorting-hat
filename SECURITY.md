# Security Policy

## Supported versions

This is a small tool with no releases yet. Fixes land on `main`; please test
against `main` before reporting.

## Reporting a vulnerability

Use GitHub's private reporting: go to the repository's **Security** tab and
choose **Report a vulnerability**. That opens a private advisory visible only to
the maintainer. Please do not open a public issue for anything you believe is
exploitable.

Include what you did, what happened, and which PDF or model triggered it if
that matters. A file that reproduces the problem is worth more than a
description of it — but see the note on samples below.

Expect an acknowledgement within a week. This is a spare-time project, so
please be patient beyond that.

## What is in scope

This program reads untrusted files and runs a language model over their
contents, then writes to your file system. The interesting risks follow from
that:

- **Path traversal.** Folder names come from a language model and become
  directories. `naming::sanitize_folder` strips separators and `..`, and there
  are tests for it. A way past that is a real vulnerability.
- **Writing outside the library.** Nothing should ever be written outside the
  configured library directory, and nothing should ever be written to the source
  tree. A hard link and a symlink are the same file as their target, so writing
  through one reaches the original; `metadata` and `bookmarks` refuse link modes
  and enrichment unlinks before restoring. A route around either guard is in
  scope.
- **Destroying data.** The program is designed never to delete: duplicates are
  listed rather than removed, unwritable documents are left alone, and every
  `apply` records an undo manifest. Anything that loses a file is in scope, and
  a PDF rewrite that corrupts the file it was meant to enrich especially so.
- **Prompt injection.** Document text goes into a model prompt. Model output is
  constrained by grammars that make an invented folder or heading
  unrepresentable rather than merely invalid. If you can make a crafted PDF
  produce a filing decision outside those grammars, that is in scope.
- **Crashes on hostile input.** A malformed PDF should be reported and skipped,
  not abort the run.

## What is out of scope

- Vulnerabilities in dependencies — report those upstream, to
  [poppler](https://poppler.freedesktop.org/),
  [lopdf](https://github.com/J-F-Liu/lopdf) or
  [llama.cpp](https://github.com/ggml-org/llama.cpp). Tell us too if this
  project can work around one.
- The quality of the model's judgement. A document filed in the wrong folder is
  a bug, not a vulnerability.
- Anything requiring you to already be able to run code as the user.

## Sending sample files

Do not attach copyrighted material. If a specific document is needed to
reproduce a problem, describe its structure — encrypted, image-only, object
streams, small-caps headings — or construct a minimal PDF that shows the same
shape. That is usually enough.
