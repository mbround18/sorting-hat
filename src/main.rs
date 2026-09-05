//! sorting-hat — read a firehose of PDFs, work out what they are, and propose a
//! library worth living in. Nothing moves until you have read the plan.

mod apply;
mod brain;
mod cache;
mod config;
mod extract;
mod metadata;
mod naming;
mod outline;
mod pipeline;
mod report;
mod scan;
mod systems;
mod types;

use anyhow::{Context, Result};
#[cfg(not(feature = "llama"))]
use anyhow::bail;
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::io::Write;
use std::path::PathBuf;

use brain::Brain;
use types::{LinkMode, Plan};

#[derive(Parser)]
#[command(
    name = "sorting-hat",
    version,
    about,
    long_about = "Reads a directory of PDFs, works out what each one is using a local model on \
the GPU, and proposes a library worth living in. Nothing moves until you have read the plan.",
    after_help = "\
EXAMPLES
  sorting-hat sort -s ~/PDFs -l ~/Library     read, plan and file in one go
  sorting-hat plan -s ~/PDFs                  propose a library, change nothing
  sorting-hat show --full                     read the plan in detail
  sorting-hat apply                           carry it out
  sorting-hat apply --copy --metadata         copy, and write the naming into each PDF
  sorting-hat undo                            put everything back
  sorting-hat doctor                          check this machine is set up

Start with `plan`. It only reads.",
    subcommand_help_heading = "Commands",
)]
struct Cli {
    #[command(subcommand)]
    command: Command,

    #[command(flatten)]
    common: Common,
}

#[derive(Args, Clone)]
struct Common {
    /// TOML config file. Command line flags win over it.
    #[arg(long, short, global = true, help_heading = "Paths", value_name = "FILE")]
    config: Option<PathBuf>,

    /// Directory to read PDFs from, recursively.
    #[arg(long, short, global = true, help_heading = "Paths", value_name = "DIR")]
    source: Option<PathBuf>,

    /// Where the sorted library is built.
    #[arg(long, short, global = true, help_heading = "Paths", value_name = "DIR")]
    library: Option<PathBuf>,

    /// Scratch directory for the cache, plan and undo manifests.
    #[arg(long, global = true, help_heading = "Paths", value_name = "DIR")]
    work_dir: Option<PathBuf>,

    /// GGUF model file for the llama backend.
    #[arg(long, global = true, help_heading = "Model", value_name = "FILE")]
    model: Option<PathBuf>,

    /// Which understanding backend to use.
    #[arg(long, global = true, value_enum, default_value_t = Backend::Auto, help_heading = "Model")]
    backend: Backend,

    /// Log level: error, warn, info, debug, trace.
    #[arg(long, global = true, default_value = "info", help_heading = "Model", value_name = "LEVEL")]
    log: String,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Backend {
    /// Use the model if one is available, otherwise fall back to rules.
    Auto,
    /// Local GGUF model on the GPU.
    Llama,
    /// Keyword rules. No model, no GPU.
    Heuristic,
}

#[derive(Subcommand)]
enum Command {
    /// Read the source tree and write a plan. Changes nothing on disk.
    Plan {
        /// How filed documents get into the library.
        #[arg(long, value_enum, default_value_t = LinkMode::Hardlink)]
        mode: LinkMode,

        /// Re-read every document, ignoring cached digests.
        #[arg(long)]
        rescan: bool,

        /// Re-read only the documents whose cached digest scored below this
        /// confidence. Use it to give text-poor PDFs a second look once the
        /// vision model is available, without discarding the whole cache.
        #[arg(long, value_name = "CONFIDENCE")]
        redigest_below: Option<f32>,

        /// Re-read only the documents the backend could not name.
        #[arg(long)]
        redigest_unknown: bool,

        /// Only consider the first N documents.
        #[arg(long)]
        limit: Option<usize>,
    },

    /// Read, plan and file in one go.
    ///
    /// The same work as `plan` followed by `apply`, with one confirmation in
    /// between. Use `plan` on its own when you want to study the proposal, or
    /// need --rescan and the other read-time options.
    Sort {
        /// How filed documents get into the library.
        #[arg(long, value_enum, default_value_t = LinkMode::Hardlink)]
        mode: LinkMode,

        /// Skip the confirmation prompt.
        #[arg(long, short = 'y')]
        yes: bool,

        /// Write the title, author and keywords into each PDF. Needs
        /// --mode copy or --mode move.
        #[arg(long)]
        metadata: bool,

        /// Give documents without one a chapter index. Needs
        /// --mode copy or --mode move.
        #[arg(long)]
        bookmarks: bool,

        /// Have the model judge which headings are real chapters.
        #[arg(long, requires = "bookmarks")]
        refine: bool,
    },

    /// Read a plan that has already been made.
    Show {
        /// List every document and its new name, not just folder counts.
        #[arg(long)]
        full: bool,

        /// Plan file to read. Defaults to the one in the work directory.
        #[arg(long, value_name = "FILE")]
        plan: Option<PathBuf>,
    },

    /// Carry out a plan.
    Apply {
        /// Plan file to carry out. Defaults to the one in the work directory.
        #[arg(long, value_name = "FILE")]
        plan: Option<PathBuf>,

        /// Skip the confirmation prompt.
        #[arg(long, short = 'y')]
        yes: bool,

        /// Place files differently than the plan recorded, without re-planning.
        /// The destinations are unaffected — only how each file gets there —
        /// so a tree you have already reviewed stays exactly as you saw it.
        #[arg(long, value_enum)]
        mode: Option<LinkMode>,

        /// Write the title, author, subject and keywords into each PDF, so the
        /// naming travels with the file. Needs a plan made with --mode copy or
        /// --mode move: a hard link or symlink is the same file as the
        /// original, and stamping one would rewrite your source PDFs.
        #[arg(long = "metadata", alias = "write-metadata")]
        write_metadata: bool,

        /// Build a chapter index (PDF bookmarks) for documents that have none,
        /// in the same rewrite as the metadata. Same mode requirement.
        #[arg(long = "bookmarks", alias = "write-bookmarks")]
        write_bookmarks: bool,

        /// Have the model judge which headings are real chapters. Without it,
        /// type size alone decides: instant and free, but it keeps sidebar
        /// titles and stat-block names that are not really chapters.
        #[arg(long, requires = "write_bookmarks")]
        refine: bool,

        /// Say what would happen, and write nothing.
        #[arg(long)]
        dry_run: bool,
    },

    /// Add a chapter index to an already-built library.
    ///
    /// Prefer `apply --bookmarks`, which reads the headings from the pristine
    /// sources and writes them in the same pass as everything else. This
    /// command rewrites files this tool has already rewritten, and lopdf cannot
    /// reliably re-read its own output — expect roughly a third to be skipped.
    #[command(hide = true)]
    Bookmark {
        /// Have the model judge which headings are real. Without it, type size
        /// alone decides: instant and free, but it keeps sidebar titles and
        /// stat-block names that are not really chapters.
        #[arg(long)]
        refine: bool,

        /// Rebuild the index even for documents that already have one.
        #[arg(long)]
        force: bool,

        /// Skip documents shorter than this.
        #[arg(long)]
        min_pages: Option<usize>,

        /// Report what would change without writing anything.
        #[arg(long)]
        dry_run: bool,

        /// Stop after this many documents.
        #[arg(long)]
        limit: Option<usize>,
    },

    /// Check this machine has what the tool needs.
    ///
    /// Reports the external tools, the models and the GPU, and says what is
    /// missing and what that costs you.
    Doctor,

    /// Show the chapter headings the font pass finds in one PDF.
    Headings {
        file: PathBuf,
        /// How much larger than body text a run must be to count.
        #[arg(long, default_value_t = 1.6)]
        min_ratio: f32,
        #[arg(long, default_value_t = 4)]
        max_depth: usize,
    },

    /// Reverse an applied run.
    Undo {
        /// Manifest to reverse. Defaults to the most recent one.
        manifest: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| cli.common.log.clone().into()),
        )
        .with_target(false)
        .without_time()
        .init();

    let cfg = resolve_config(&cli.common)?;

    match cli.command {
        Command::Plan { mode, rescan, limit, redigest_below, redigest_unknown } => {
            let opts = pipeline::Options { mode, rescan, limit, redigest_below, redigest_unknown };
            let plan = run_plan(&cfg, &cli.common, &opts)?;
            println!("\n{}", report::tree(&plan, false));
            println!("{}", report::summary(&plan));
            println!("  plan       {}\n", cfg.plan_path().display());
            println!("Review it, then run: sorting-hat apply");
        }

        Command::Sort { mode, yes, metadata, bookmarks, refine } => {
            let opts = pipeline::Options {
                mode,
                rescan: false,
                limit: None,
                redigest_below: None,
                redigest_unknown: false,
            };
            let plan = run_plan(&cfg, &cli.common, &opts)?;
            println!("\n{}", report::tree(&plan, false));
            println!("{}", report::summary(&plan));

            if metadata || bookmarks {
                metadata::may_write(plan.mode)?;
            }
            if !yes && !confirm("File these documents?")? {
                println!("Nothing done. The plan is saved; run `sorting-hat apply` when ready.");
                return Ok(());
            }

            let mut enrich = apply::Enrichment {
                outlines: Default::default(),
                max_growth: cfg.bookmarks.max_growth,
            };
            if bookmarks {
                enrich.outlines = prepare_outlines(&cfg, &plan, refine, &cli.common)?;
            }
            let outcome = apply::apply(&plan, &cfg.work_dir, metadata, &enrich)?;
            report_apply(&outcome, metadata, bookmarks);
        }

        Command::Show { full, plan } => {
            let plan = load_plan(plan.unwrap_or_else(|| cfg.plan_path()))?;
            println!("\n{}", report::tree(&plan, full));
            println!("{}", report::summary(&plan));
            if !plan.unfiled.is_empty() {
                println!("Unreadable:");
                for item in &plan.unfiled {
                    println!("  {} — {}", item.path.display(), item.reason);
                }
            }
        }

        Command::Apply { plan, yes, mode, write_metadata, write_bookmarks, refine, dry_run } => {
            let path = plan.unwrap_or_else(|| cfg.plan_path());
            let mut plan = load_plan(path)?;
            if let Some(mode) = mode {
                if mode != plan.mode {
                    println!("  Placing by {mode:?} instead of the planned {:?}.", plan.mode);
                }
                plan.mode = mode;
            }

            println!("\n{}", report::summary(&plan));
            if write_metadata || write_bookmarks {
                // Fail before the prompt rather than after the work.
                metadata::may_write(plan.mode)?;
            }
            if write_metadata {
                println!("  Each filed PDF will be stamped with its title, author and keywords.");
            }
            if write_bookmarks {
                println!("  Documents without a chapter index will be given one.");
            }
            println!();
            if plan.mode == LinkMode::Move {
                println!("  This MOVES the originals out of {}.\n", plan.source_root.display());
            }

            if dry_run {
                println!("Dry run: nothing was written.");
                println!("Drop --dry-run to carry this out.");
                return Ok(());
            }

            if !yes && !confirm("Apply this plan?")? {
                println!("Nothing done.");
                return Ok(());
            }

            let mut enrich = apply::Enrichment {
                outlines: Default::default(),
                max_growth: cfg.bookmarks.max_growth,
            };
            if write_bookmarks {
                // Prepared from the sources, before anything is written: the
                // headings are read from pristine files rather than from
                // copies this run has already rewritten.
                enrich.outlines = prepare_outlines(&cfg, &plan, refine, &cli.common)?;
            }

            let outcome = apply::apply(&plan, &cfg.work_dir, write_metadata, &enrich)?;
            report_apply(&outcome, write_metadata, write_bookmarks);
        }

        Command::Bookmark { refine, force, min_pages, dry_run, limit } => {
            let mut brain = if refine {
                let brain = select_backend(&cli.common, &cfg)?;
                tracing::info!(backend = brain.name(), "refining headings with the model");
                Some(brain)
            } else {
                None
            };
            bookmark_library(&cfg, brain.as_deref_mut(), refine, force, min_pages, dry_run, limit)?;
        }

        Command::Doctor => doctor(&cfg)?,

        Command::Headings { file, min_ratio, max_depth } => {
            outline::dump(&file, min_ratio, max_depth, cfg.extract.timeout_secs)?;
        }

        Command::Undo { manifest } => {
            let manifest = match manifest {
                Some(path) => path,
                None => latest_manifest(&cfg.work_dir)?,
            };
            println!("Reversing {}", manifest.display());
            let (reversed, problems) = apply::undo(&manifest)?;
            println!("removed {reversed} entries from the library");
            for problem in &problems {
                println!("  {problem}");
            }
        }
    }

    Ok(())
}

fn resolve_config(common: &Common) -> Result<config::Config> {
    let mut cfg = config::Config::load(common.config.as_deref())?;
    if let Some(source) = &common.source {
        cfg.source = source.clone();
    }
    if let Some(library) = &common.library {
        cfg.library = library.clone();
    }
    if let Some(work_dir) = &common.work_dir {
        cfg.work_dir = work_dir.clone();
        // The library defaults to living inside the work directory; keep that
        // relationship unless it was set explicitly.
        if common.library.is_none() && common.config.is_none() {
            cfg.library = work_dir.join("library");
        }
    }
    if let Some(model) = &common.model {
        cfg.model.path = model.clone();
    }
    Ok(cfg)
}

fn select_backend(common: &Common, #[cfg_attr(not(feature = "llama"), allow(unused_variables))] cfg: &config::Config) -> Result<Box<dyn Brain>> {
    match common.backend {
        Backend::Heuristic => Ok(Box::new(brain::heuristic::Heuristic)),

        #[cfg(feature = "llama")]
        Backend::Llama => Ok(Box::new(brain::llama::LlamaBrain::load(&cfg.model)?)),

        #[cfg(not(feature = "llama"))]
        Backend::Llama => bail!(
            "this binary was built without the llama backend; rebuild with --features cuda"
        ),

        Backend::Auto => {
            #[cfg(feature = "llama")]
            if cfg.model.path.exists() {
                match brain::llama::LlamaBrain::load(&cfg.model) {
                    Ok(brain) => return Ok(Box::new(brain)),
                    Err(err) => tracing::warn!(%err, "falling back to the heuristic backend"),
                }
            } else {
                tracing::warn!(
                    path = %cfg.model.path.display(),
                    "no model found; falling back to the heuristic backend"
                );
            }
            Ok(Box::new(brain::heuristic::Heuristic))
        }
    }
}

/// Load the vision backend, if it is configured and its files are present.
///
/// Its absence is not an error: without it, text-poor PDFs simply fall back to
/// being judged by file name.
#[cfg(feature = "llama")]
fn select_eyes(common: &Common, cfg: &config::Config) -> Option<Box<dyn pipeline::Eyes>> {
    if !cfg.vision.enabled || common.backend == Backend::Heuristic {
        return None;
    }
    if !cfg.vision.path.exists() || !cfg.vision.mmproj.exists() {
        tracing::warn!(
            path = %cfg.vision.path.display(),
            "no vision model; text-poor PDFs will be judged by file name alone"
        );
        return None;
    }
    match brain::vision::VisionBrain::load(&cfg.vision) {
        Ok(eyes) => Some(Box::new(eyes)),
        Err(err) => {
            tracing::warn!(%err, "vision backend unavailable");
            None
        }
    }
}

#[cfg(not(feature = "llama"))]
fn select_eyes(_common: &Common, _cfg: &config::Config) -> Option<Box<dyn pipeline::Eyes>> {
    None
}

/// Build a chapter index across the library.
#[allow(clippy::too_many_arguments)]
fn bookmark_library(
    cfg: &config::Config,
    mut brain: Option<&mut (dyn Brain + '_)>,
    refine: bool,
    force: bool,
    min_pages: Option<usize>,
    dry_run: bool,
    limit: Option<usize>,
) -> Result<()> {
    let min_pages = min_pages.unwrap_or(cfg.bookmarks.min_pages);
    let mut files = scan::find_pdfs(&cfg.library, &[])?;
    if files.is_empty() {
        anyhow::bail!(
            "no PDFs under {}; run `sorting-hat apply` first",
            cfg.library.display()
        );
    }
    if let Some(limit) = limit {
        files.truncate(limit);
    }

    let bar = indicatif::ProgressBar::new(files.len() as u64);
    bar.set_style(
        indicatif::ProgressStyle::with_template("indexing    [{bar:32}] {pos}/{len} {eta_precise}")
            .expect("static template")
            .progress_chars("=> "),
    );

    let mut indexed = 0usize;
    let mut bookmarks = 0usize;
    let mut skipped: Vec<(PathBuf, String)> = Vec::new();

    for file in &files {
        bar.inc(1);

        let pages = extract::page_count(file, cfg.extract.timeout_secs).unwrap_or(0);
        if pages < min_pages {
            skipped.push((file.clone(), outline::Skip::TooShort { pages }.to_string()));
            continue;
        }
        // Checked before the expensive part: reading a document only to find it
        // cannot be written wastes the whole effort.
        match lopdf::Document::load(file) {
            Ok(doc) => {
                if doc.is_encrypted() {
                    skipped.push((file.clone(), outline::Skip::Encrypted.to_string()));
                    continue;
                }
                if !force && outline::has_outline(&doc) {
                    skipped.push((file.clone(), outline::Skip::AlreadyIndexed.to_string()));
                    continue;
                }
            }
            Err(err) => {
                skipped.push((file.clone(), outline::Skip::Unreadable(err.to_string()).to_string()));
                continue;
            }
        }

        let found = match outline::candidates(file, cfg.bookmarks.min_ratio, cfg.extract.timeout_secs)
        {
            Ok(found) if !found.is_empty() => found,
            Ok(_) => {
                skipped.push((file.clone(), outline::Skip::NoHeadings.to_string()));
                continue;
            }
            Err(err) => {
                skipped.push((file.clone(), format!("{err:#}")));
                continue;
            }
        };

        let mut entries = outline::nest(&found, cfg.bookmarks.max_depth);

        if refine {
            let lines: Vec<(usize, String, usize)> = entries
                .iter()
                .enumerate()
                .map(|(i, e)| (i, e.title.clone(), e.page))
                .collect();
            match brain.as_deref_mut().map(|b| b.refine_headings(&lines, cfg.bookmarks.max_depth)) {
                Some(Ok(Some(picks))) if !picks.is_empty() => {
                    entries = picks
                        .into_iter()
                        .filter_map(|(i, level)| {
                            entries.get(i).map(|e| outline::Entry { level, ..e.clone() })
                        })
                        .collect();
                }
                Some(Err(err)) => {
                    tracing::warn!(path = %file.display(), %err, "refinement failed; keeping the type-size guess");
                }
                _ => {}
            }
        }

        if dry_run {
            indexed += 1;
            bookmarks += entries.len();
            continue;
        }

        match outline::write(file, &entries, cfg.bookmarks.max_growth) {
            Ok(n) => {
                indexed += 1;
                bookmarks += n;
            }
            Err(skip) => skipped.push((file.clone(), skip.to_string())),
        }
    }
    bar.finish_and_clear();

    println!();
    if dry_run {
        println!("would index {indexed} documents with {bookmarks} bookmarks");
    } else {
        println!("indexed {indexed} documents with {bookmarks} bookmarks");
    }

    if !skipped.is_empty() {
        // Grouped by reason: forty lines of "only 4 pages" tells the user less
        // than one line saying forty documents were too short.
        let mut reasons: std::collections::BTreeMap<String, Vec<&PathBuf>> = Default::default();
        for (path, why) in &skipped {
            // Group on the reason, not its particulars: "grew from 4 to 9 bytes"
            // and "grew from 5 to 11" are one story, told once.
            let key = why.split(" from ").next().unwrap_or(why).to_string();
            reasons.entry(key).or_default().push(path);
        }
        println!("\nleft alone ({}):", skipped.len());
        for (why, paths) in &reasons {
            println!("  {} — {why}", paths.len());
            for path in paths.iter().take(3) {
                println!("      {}", path.display());
            }
            if paths.len() > 3 {
                println!("      ... and {} more", paths.len() - 3);
            }
        }
    }
    Ok(())
}

/// Read chapter headings from every source document that wants an index.
///
/// Done before anything is written, and always from the source rather than the
/// filed copy, so the headings come out of a pristine file.
fn prepare_outlines(
    cfg: &config::Config,
    plan: &Plan,
    refine: bool,
    common: &Common,
) -> Result<std::collections::HashMap<PathBuf, Vec<outline::Entry>>> {
    let mut brain = if refine {
        let brain = select_backend(common, cfg)?;
        tracing::info!(backend = brain.name(), "refining headings with the model");
        Some(brain)
    } else {
        None
    };

    let bar = indicatif::ProgressBar::new(plan.assignments.len() as u64);
    bar.set_style(
        indicatif::ProgressStyle::with_template("headings    [{bar:32}] {pos}/{len} {eta_precise}")
            .expect("static template")
            .progress_chars("=> "),
    );

    let mut out = std::collections::HashMap::new();
    for assignment in &plan.assignments {
        bar.inc(1);
        let source = &assignment.source;

        let pages = extract::page_count(source, cfg.extract.timeout_secs).unwrap_or(0);
        if pages < cfg.bookmarks.min_pages {
            continue;
        }

        let found =
            match outline::candidates(source, cfg.bookmarks.min_ratio, cfg.extract.timeout_secs) {
                Ok(found) if !found.is_empty() => found,
                _ => continue,
            };
        let mut entries = outline::nest(&found, cfg.bookmarks.max_depth);

        if let Some(brain) = brain.as_deref_mut() {
            let lines: Vec<(usize, String, usize)> = entries
                .iter()
                .enumerate()
                .map(|(i, e)| (i, e.title.clone(), e.page))
                .collect();
            match brain.refine_headings(&lines, cfg.bookmarks.max_depth) {
                Ok(Some(picks)) if !picks.is_empty() => {
                    entries = picks
                        .into_iter()
                        .filter_map(|(i, level)| {
                            entries.get(i).map(|e| outline::Entry { level, ..e.clone() })
                        })
                        .collect();
                }
                Ok(_) => {}
                Err(err) => tracing::warn!(
                    path = %source.display(), %err,
                    "refinement failed; keeping the type-size guess"
                ),
            }
        }

        if !entries.is_empty() {
            out.insert(source.clone(), entries);
        }
    }
    bar.finish_and_clear();
    tracing::info!(documents = out.len(), "chapter headings prepared");
    Ok(out)
}

/// Print skip reasons grouped, so one story is told once.
fn summarise_skips(skipped: &[(PathBuf, String)]) {
    if skipped.is_empty() {
        return;
    }
    let mut reasons: std::collections::BTreeMap<&str, usize> = Default::default();
    for (_, why) in skipped {
        *reasons.entry(why.split(" from ").next().unwrap_or(why)).or_default() += 1;
    }
    println!("  left alone ({}):", skipped.len());
    for (why, count) in &reasons {
        println!("    {count} — {why}");
    }
}

/// Read the source tree, build a plan, and save it.
fn run_plan(
    cfg: &config::Config,
    common: &Common,
    opts: &pipeline::Options,
) -> Result<Plan> {
    let mut brain = select_backend(common, cfg)?;
    tracing::info!(backend = brain.name(), "starting");

    let mut eyes = select_eyes(common, cfg);
    if let Some(eyes) = &eyes {
        tracing::info!(vision = eyes.name(), "vision pass enabled for text-poor PDFs");
    }

    let plan = pipeline::build_plan(cfg, brain.as_mut(), eyes.as_deref_mut(), opts)?;

    std::fs::create_dir_all(&cfg.work_dir)?;
    let path = cfg.plan_path();
    std::fs::write(&path, serde_json::to_vec_pretty(&plan)?)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(plan)
}

/// Print what an apply actually did.
fn report_apply(outcome: &apply::Report, metadata: bool, bookmarks: bool) {
    println!(
        "\nfiled {}, already present {}, failed {}",
        outcome.filed,
        outcome.skipped,
        outcome.failed.len()
    );
    for (path, err) in outcome.failed.iter().take(10) {
        println!("  {} — {err}", path.display());
    }
    if outcome.failed.len() > 10 {
        println!("  ... and {} more", outcome.failed.len() - 10);
    }
    if metadata {
        println!(
            "stamped {} PDFs, {} could not be stamped",
            outcome.stamped,
            outcome.stamp_failed.len()
        );
        for (path, err) in outcome.stamp_failed.iter().take(10) {
            println!("  {} — {err}", path.display());
        }
        if outcome.stamp_failed.len() > 10 {
            println!("  ... and {} more", outcome.stamp_failed.len() - 10);
        }
    }
    if bookmarks {
        println!(
            "indexed {} documents with {} bookmarks",
            outcome.indexed, outcome.bookmarks
        );
        summarise_skips(&outcome.index_skipped);
    }
    if let Some(manifest) = &outcome.manifest {
        println!("\nundo with: sorting-hat undo {}", manifest.display());
    }
}

/// Report what is present, what is missing, and what missing costs.
fn doctor(cfg: &config::Config) -> Result<()> {
    fn found(label: &str, ok: bool, detail: &str) {
        println!("  [{}] {label:<26} {detail}", if ok { "ok" } else { "--" });
    }
    fn have(bin: &str) -> bool {
        std::process::Command::new(bin).arg("-v").output().is_ok()
    }

    println!("\nExternal tools");
    let poppler = ["pdftotext", "pdfinfo", "pdftoppm", "pdftohtml"];
    for bin in poppler {
        let ok = have(bin);
        found(
            bin,
            ok,
            match (bin, ok) {
                (_, true) => "",
                ("pdftotext" | "pdfinfo", false) => "needed to read PDFs at all",
                ("pdftoppm", false) => "without it, image-only PDFs cannot be looked at",
                _ => "without it, no chapter indexes can be built",
            },
        );
    }

    println!("\nModels");
    let llama_built = cfg!(feature = "llama");
    found(
        "llama backend",
        llama_built,
        if llama_built { "" } else { "rebuild with --features cuda" },
    );
    for (label, path) in [
        ("text model", &cfg.model.path),
        ("vision model", &cfg.vision.path),
        ("vision projector", &cfg.vision.mmproj),
    ] {
        let size = std::fs::metadata(path).map(|m| m.len()).ok();
        found(
            label,
            size.is_some(),
            &match size {
                Some(n) => format!("{:.1} GB  {}", n as f64 / 1e9, path.display()),
                None => format!("missing: {}", path.display()),
            },
        );
    }

    println!("\nGPU");
    let gpu = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total", "--format=csv,noheader"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    found(
        "nvidia-smi",
        gpu.is_some(),
        gpu.as_deref().unwrap_or("no NVIDIA GPU visible; the model will run on CPU"),
    );

    println!("\nPaths");
    let pdfs = scan::find_pdfs(&cfg.source, &[cfg.library.clone()]).map(|v| v.len());
    found(
        "source",
        pdfs.as_ref().is_ok_and(|n| *n > 0),
        &match &pdfs {
            Ok(n) => format!("{n} PDFs in {}", cfg.source.display()),
            Err(err) => format!("{}: {err}", cfg.source.display()),
        },
    );
    let library_files = scan::find_pdfs(&cfg.library, &[]).map(|v| v.len()).unwrap_or(0);
    found("library", true, &format!("{library_files} PDFs in {}", cfg.library.display()));
    found(
        "plan",
        cfg.plan_path().exists(),
        &if cfg.plan_path().exists() {
            format!("{}", cfg.plan_path().display())
        } else {
            "none yet; run `sorting-hat plan`".to_string()
        },
    );
    let cached = cfg.cache_dir().join("digests.jsonl");
    let entries = std::fs::read_to_string(&cached).map(|s| s.lines().count()).unwrap_or(0);
    found("digest cache", entries > 0, &format!("{entries} documents already read"));

    println!();
    Ok(())
}

fn load_plan(path: PathBuf) -> Result<Plan> {
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("reading {}; run `sorting-hat plan` first", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

fn latest_manifest(work_dir: &std::path::Path) -> Result<PathBuf> {
    let mut manifests: Vec<PathBuf> = std::fs::read_dir(work_dir)
        .with_context(|| format!("reading {}", work_dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .map(|n| {
                    let n = n.to_string_lossy();
                    n.starts_with("undo-") && n.ends_with(".json")
                })
                .unwrap_or(false)
        })
        .collect();

    manifests.sort();
    manifests.pop().context("no undo manifest found; nothing to reverse")
}

fn confirm(question: &str) -> Result<bool> {
    print!("{question} [y/N] ");
    std::io::stdout().flush()?;
    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer)?;
    Ok(matches!(answer.trim().to_lowercase().as_str(), "y" | "yes"))
}
