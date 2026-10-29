#![forbid(unsafe_code)]

pub mod agent_config;
pub mod benchmark;
pub mod inventory;
pub mod models;
pub mod pager;
pub mod parser;
pub mod runtime;
pub mod shell;

pub use mesh_llm_events::LogFormat;

pub use inventory::{
    CLI_INVENTORY_SCHEMA_VERSION, CliInventory, InventoryNode, InventoryNodeKind,
    build_cli_inventory, check_cli_inventory, cli_inventory_json, serialize_cli_inventory,
    write_cli_inventory,
};

pub use parser::{
    AnalyticsCommand, AuthCommand, BinaryFlavor, ClaudeArgs, Cli, Command, ConfigCommand,
    DiscoverArgs, DiscoveryScope, DoctorArgs, DoctorCommand, DownloadArgs, GooseArgs, GpuCommand,
    GpusArgs, JoiningOptions, KvCacheCommand, KvCacheOptions, LoadArgs, LoggingOptions,
    MeshDiscoveryMode, MeshGuardrailCliMode, MeshPolicyOptions, ModelPrepareArgs, NetworkOptions,
    NodeModeOptions, NormalizedRuntimeArgs, OpencodeArgs, OwnerOptions, PiArgs, PlacementOptions,
    PluginCommand, RuntimeSurface, ServingOptions, ServingStrategyCli, SetupArgs, SkillAgentArg,
    SkillCommand, SpeculativeOptions, StatusArgs, TrustCommand, TrustPolicy, UninstallArgs,
    UnloadArgs, UpdateArgs, WalletArgs, legacy_runtime_surface_warning,
    normalize_runtime_surface_args, raw_args_invoke_analytics, validate_discovery_mode_args,
};

pub mod wallet;
