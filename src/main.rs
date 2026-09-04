//! sorting-hat — read a firehose of PDFs, work out what they are, and propose a
//! library worth living in. Nothing moves until you have read the plan.

mod apply;
mod brain;
mod cache;
mod config;
mod extract;
mod metadata;
mod naming;
mod pipeline;
mod report;
mod scan;
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
#[command(name = "sorting-hat", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,

    #[command(flatten)]
    common: Common,
}

#[derive(Args, Clone)]
struct Common {
    /// TOML config file. Command line flags win over it.
    #[arg(long, short, global = true)]
    config: Option<PathBuf>,

    /// Directory to read PDFs from, recursively.
    #[arg(long, short, global = true)]
    source: Option<PathBuf>,

    /// Where the sorted library is built.
    #[arg(long, short, global = true)]
    library: Option<PathBuf>,

    /// Scratch directory for the cache, plan and undo manifests.
    #[arg(long, global = true)]
    work_dir: Option<PathBuf>,

    /// GGUF model file for the llama backend.
    #[arg(long, global = true)]
    model: Option<PathBuf>,

    /// Which understanding backend to use.
    #[arg(long, global = true, value_enum, default_value_t = Backend::Auto)]
    backend: Backend,

    #[arg(long, global = true, default_value = "info")]
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

    /// Print an existing plan.
    Show {
        /// List every document, not just folder counts.
        #[arg(long)]
        full: bool,

        #[arg(long)]
        plan: Option<PathBuf>,
    },

    /// Carry out a plan.
    Apply {
        #[arg(long)]
        plan: Option<PathBuf>,

        /// Skip the confirmation prompt.
        #[arg(long, short = 'y')]
        yes: bool,

        /// Write the title, author, subject and keywords into each PDF, so the
        /// naming travels with the file. Needs a plan made with --mode copy or
        /// --mode move: a hard link or symlink is the same file as the
        /// original, and stamping one would rewrite your source PDFs.
        #[arg(long)]
        write_metadata: bool,
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
            let mut brain = select_backend(&cli.common, &cfg)?;
            tracing::info!(backend = brain.name(), "starting");

            let mut eyes = select_eyes(&cli.common, &cfg);
            if let Some(eyes) = &eyes {
                tracing::info!(vision = eyes.name(), "vision pass enabled for text-poor PDFs");
            }

            let opts = pipeline::Options { mode, rescan, limit, redigest_below, redigest_unknown };
            let plan = pipeline::build_plan(
                &cfg,
                brain.as_mut(),
                eyes.as_deref_mut(),
                &opts,
            )?;

            std::fs::create_dir_all(&cfg.work_dir)?;
            let path = cfg.plan_path();
            std::fs::write(&path, serde_json::to_vec_pretty(&plan)?)
                .with_context(|| format!("writing {}", path.display()))?;

            println!("\n{}", report::tree(&plan, false));
            println!("{}", report::summary(&plan));
            println!("  plan       {}\n", path.display());
            println!("Review it, then run: sorting-hat apply");
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

        Command::Apply { plan, yes, write_metadata } => {
            let path = plan.unwrap_or_else(|| cfg.plan_path());
            let plan = load_plan(path)?;

            println!("\n{}", report::summary(&plan));
            if write_metadata {
                // Fail before the prompt rather than after the work.
                metadata::may_write(plan.mode)?;
                println!("  Each filed PDF will be stamped with its title, author and keywords.\n");
            }
            if plan.mode == LinkMode::Move {
                println!("  This MOVES the originals out of {}.\n", plan.source_root.display());
            }

            if !yes && !confirm("Apply this plan?")? {
                println!("Nothing done.");
                return Ok(());
            }

            let outcome = apply::apply(&plan, &cfg.work_dir, write_metadata)?;
            println!(
                "\nfiled {}, already present {}, failed {}",
                outcome.filed,
                outcome.skipped,
                outcome.failed.len()
            );
            for (path, err) in &outcome.failed {
                println!("  {} — {err}", path.display());
            }
            if write_metadata {
                println!("stamped {} PDFs, {} could not be stamped", outcome.stamped, outcome.stamp_failed.len());
                for (path, err) in &outcome.stamp_failed {
                    println!("  {} — {err}", path.display());
                }
            }
            if let Some(manifest) = &outcome.manifest {
                println!("\nundo with: sorting-hat undo {}", manifest.display());
            }
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
