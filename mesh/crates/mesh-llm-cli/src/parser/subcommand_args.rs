//! The options of `mesh-llm`'s subcommands, one struct per subcommand.
//!
//! Like the top-level groups in `top_level_options`, each struct gets its own
//! clap function, so `Command`'s debug frame no longer holds every
//! subcommand's options while the nested subcommands are defined.

use clap::Args;

use super::commands::{BinaryFlavor, DoctorCommand, GpuCommand};

#[derive(Args, Debug)]
pub struct WalletArgs {
    #[arg(long, default_value_t = 3131)]
    pub port: u16,
    #[command(subcommand)]
    pub command: crate::wallet::WalletCommand,
}

#[derive(Args, Debug)]
pub struct DownloadArgs {
    /// Model name (e.g. "Qwen2.5-32B-Instruct-Q4_K_M" or just "32b")
    pub name: Option<String>,
    /// Also download the recommended draft model for speculative decoding
    #[arg(long)]
    pub draft: bool,
}

#[derive(Args, Debug)]
pub struct UpdateArgs {
    /// Install this specific release tag or version (e.g. v0.60.0 or 0.60.0-rc.1).
    #[arg(long)]
    pub version: Option<String>,
    /// Install this release bundle flavor instead of the default installed flavor.
    #[arg(long, value_enum, conflicts_with = "detect_flavor")]
    pub flavor: Option<BinaryFlavor>,
    /// Re-detect the best host backend flavor before selecting the release bundle.
    #[arg(long, conflicts_with = "flavor")]
    pub detect_flavor: bool,
    /// Turn off the default plugins this build lists: the updated node
    /// does not install them from its release until `mesh-llm plugins
    /// enable NAME`. Same as `mesh-llm plugins install-defaults --off`.
    #[arg(long)]
    pub no_default_plugins: bool,
}

#[derive(Args, Debug)]
pub struct GpusArgs {
    /// Print machine-readable JSON output.
    #[arg(long)]
    pub json: bool,
    #[command(subcommand)]
    pub command: Option<GpuCommand>,
}

#[derive(Args, Debug)]
pub struct DoctorArgs {
    /// Print machine-readable JSON for the default doctor report.
    #[arg(long)]
    pub json: bool,
    #[command(subcommand)]
    pub command: Option<DoctorCommand>,
}

#[derive(Args, Debug)]
pub struct SetupArgs {
    /// Automatically answer yes to prompts.
    #[arg(long)]
    pub yes: bool,
    /// Run without prompting for interactive input.
    #[arg(long = "no-interactive")]
    pub no_interactive: bool,
    /// Install and enable the mesh-llm service.
    #[arg(long, conflicts_with = "no_service")]
    pub service: bool,
    /// Skip installing and enabling the mesh-llm service.
    #[arg(long = "no-service", conflicts_with = "service")]
    pub no_service: bool,
    /// Skip downloading or configuring the native runtime.
    #[arg(long = "skip-runtime")]
    pub skip_runtime: bool,
    /// Print detailed setup paths, commands, and follow-up guidance.
    #[arg(long)]
    pub verbose: bool,
}

#[derive(Args, Debug)]
pub struct UninstallArgs {
    /// Print what would be removed without changing the machine.
    #[arg(long)]
    pub dry_run: bool,
    /// Do not prompt before removing files and services.
    #[arg(long)]
    pub yes: bool,
    /// Preserve native runtime caches.
    #[arg(long)]
    pub keep_cache: bool,
    /// Preserve setup-owned service helper files.
    #[arg(long)]
    pub keep_service_files: bool,
    /// Also remove ~/.mesh-llm configuration and identity data.
    #[arg(long, conflicts_with = "keep_config")]
    pub purge_config: bool,
    /// Explicitly preserve ~/.mesh-llm configuration and identity data.
    #[arg(long, conflicts_with = "purge_config")]
    pub keep_config: bool,
    /// Override the installed binary path to remove.
    #[arg(long)]
    pub binary_path: Option<std::path::PathBuf>,
    /// Print machine-readable JSON.
    #[arg(long)]
    pub json: bool,
    /// Print detailed cleanup steps and removed paths.
    #[arg(long)]
    pub verbose: bool,
}

#[derive(Args, Debug)]
pub struct LoadArgs {
    /// Model name/path/url to load
    pub name: String,
    /// Console/API port of the running mesh-llm instance (default: 3131)
    #[arg(long, default_value = "3131")]
    pub port: u16,
}

#[derive(Args, Debug)]
pub struct UnloadArgs {
    /// Model name to unload
    pub name: String,
    /// Console/API port of the running mesh-llm instance (default: 3131)
    #[arg(long, default_value = "3131")]
    pub port: u16,
}

#[derive(Args, Debug)]
pub struct StatusArgs {
    /// Console/API port of the running mesh-llm instance (default: 3131)
    #[arg(long, default_value = "3131")]
    pub port: u16,
}

#[derive(Args, Debug)]
pub struct DiscoverArgs {
    /// Filter by mesh name (case-insensitive exact match)
    #[arg(long)]
    pub name: Option<String>,
    /// Filter by model name (substring match)
    #[arg(long)]
    pub model: Option<String>,
    /// Filter by minimum VRAM (GB)
    #[arg(long)]
    pub min_vram: Option<f64>,
    /// Filter by region
    #[arg(long)]
    pub region: Option<String>,
    /// Print the invite token of the best match (for piping to --join)
    #[arg(long)]
    pub auto: bool,
    /// Nostr relay URLs (default: see DEFAULT_RELAYS)
    #[arg(long)]
    pub relay: Vec<String>,
}

#[derive(Args, Debug)]
pub struct GooseArgs {
    /// Model id to use from /v1/models (default: auto = mesh picks best)
    #[arg(long)]
    pub model: Option<String>,
    /// API port for mesh-llm (default: 9337)
    #[arg(long, default_value = "9337")]
    pub port: u16,
    /// Console port of the mesh-llm node, which serves the MCP tools the agent gets (default: 3131)
    #[arg(long, default_value = "3131")]
    pub console: u16,
}

#[derive(Args, Debug)]
pub struct ClaudeArgs {
    /// Model id to use from /v1/models (default: auto = mesh picks best)
    #[arg(long)]
    pub model: Option<String>,
    /// API port for mesh-llm (default: 9337)
    #[arg(long, default_value = "9337")]
    pub port: u16,
    /// Console port of the mesh-llm node, which serves the MCP tools the agent gets (default: 3131)
    #[arg(long, default_value = "3131")]
    pub console: u16,
}

#[derive(Args, Debug)]
pub struct PiArgs {
    /// Model id to use from /v1/models (default: auto = mesh picks best)
    #[arg(long)]
    pub model: Option<String>,
    /// mesh-llm host or URL for Pi (default: 127.0.0.1:9337)
    #[arg(long, default_value = "127.0.0.1:9337")]
    pub host: String,
    /// Write the mesh provider config to Pi's models.json instead of launching.
    #[arg(long)]
    pub write: bool,
}

#[derive(Args, Debug)]
pub struct OpencodeArgs {
    /// Model id to use from /v1/models (default: auto = mesh picks best)
    #[arg(long)]
    pub model: Option<String>,
    /// mesh-llm host or URL for OpenCode (default: 127.0.0.1:9337)
    #[arg(long, default_value = "127.0.0.1:9337")]
    pub host: String,
    /// Write the mesh provider config to opencode's config file instead of launching.
    #[arg(long)]
    pub write: bool,
}

#[derive(Args, Debug)]
pub struct ModelPrepareArgs {
    /// Source HuggingFace model ref (e.g. unsloth/Qwen3-235B-A22B-GGUF:UD-Q4_K_XL).
    pub source_repo: Option<String>,

    /// Quantization variant (deprecated; prefer source refs like repo:Q4_K_M).
    #[arg(long)]
    pub quant: Option<String>,

    /// Target repo for the layer package (auto-derived if omitted).
    #[arg(long)]
    pub target: Option<String>,

    /// Override model ID in the manifest.
    #[arg(long)]
    pub model_id: Option<String>,

    /// HF Job hardware flavor. Use auto for the default CPU splitter baseline.
    #[arg(long, default_value = "auto")]
    pub flavor: String,

    /// Requested job timeout; raised automatically by model-size minimums.
    #[arg(long, default_value = "1h")]
    pub timeout: String,

    /// Branch or tag of mesh-llm to build in the job [default: main].
    #[arg(long, default_value = "main")]
    pub mesh_llm_ref: String,

    /// Explicitly keep this as a dry run. This is the default unless --confirm is set.
    #[arg(long)]
    pub dry_run: bool,

    /// Actually submit the HF Job. Without this, the command only prints plan, spec, and max cost.
    #[arg(long)]
    pub confirm: bool,

    /// Stream job logs after submission until completion.
    #[arg(long)]
    pub follow: bool,

    /// Emit JSON output.
    #[arg(long)]
    pub json: bool,

    /// Check status of a previously submitted job.
    #[arg(long)]
    pub status: Option<String>,

    /// Fetch logs for a previously submitted job.
    #[arg(long)]
    pub logs: Option<String>,

    /// Cancel a running job.
    #[arg(long)]
    pub cancel: Option<String>,

    /// List recent skippy-model-package jobs.
    #[arg(long)]
    pub list: bool,

    /// Upload the latest job script to the meshllm bucket (requires org access).
    #[arg(long)]
    pub update_script: bool,
}
