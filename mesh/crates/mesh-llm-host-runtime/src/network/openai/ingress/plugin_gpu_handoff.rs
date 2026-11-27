//! Exclusive GPU handoff acquisition before forwarding to an `exclusive_gpu` plugin.
use super::*;
use crate::runtime::native_model_lifecycle::ExclusiveGpuHandoffLease;

/// Stable OpenAI error code returned with HTTP 503 when exclusive GPU handoff fails.
pub(in crate::network::openai) const HANDOFF_FAILED_CODE: &str = "plugin_gpu_handoff_failed";

pub(in crate::network::openai) enum PluginGpuHandoffError {
    ClientDisconnected,
    Failed(&'static str),
}

/// Acquire an exclusive GPU handoff lease when `plugin_name` opts into `exclusive_gpu`.
/// Races acquisition against downstream client disconnect so disconnected callers do
/// not evict native models or dial the plugin; any unload already accepted by the
/// control loop remains owned by its detached task until completion.
pub(in crate::network::openai) async fn acquire_for_plugin(
    node: &mesh::Node,
    tcp_stream: &ClientStream,
    plugin_name: &str,
    model_name: Option<&str>,
) -> Result<Option<ExclusiveGpuHandoffLease>, PluginGpuHandoffError> {
    if !node.plugin_exclusive_gpu_enabled(plugin_name).await {
        return Ok(None);
    }
    let Some(lifecycle) = node.native_model_lifecycle.get() else {
        return Ok(None);
    };
    tokio::select! {
        biased;
        disconnected = tcp_stream.wait_for_response_disconnect() => {
            debug_assert!(disconnected);
            tracing::info!(
                plugin = %plugin_name,
                model = model_name.unwrap_or(""),
                "downstream client disconnected while waiting for exclusive_gpu handoff"
            );
            Err(PluginGpuHandoffError::ClientDisconnected)
        }
        result = lifecycle.acquire_exclusive_gpu_handoff() => {
            result.map(Some).map_err(|error| {
                tracing::warn!(
                    plugin = %plugin_name,
                    model = model_name.unwrap_or(""),
                    %error,
                    "exclusive_gpu handoff failed before forwarding to plugin endpoint"
                );
                PluginGpuHandoffError::Failed(HANDOFF_FAILED_CODE)
            })
        }
    }
}
