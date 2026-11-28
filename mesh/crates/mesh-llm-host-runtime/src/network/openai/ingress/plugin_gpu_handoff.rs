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

/// The handoff for a selected route; a route without a plugin provider takes none.
pub(in crate::network::openai) async fn acquire_for_route(
    node: &mesh::Node,
    tcp_stream: &ClientStream,
    provider: Option<&str>,
    model_name: Option<&str>,
) -> Result<Option<ExclusiveGpuHandoffLease>, PluginGpuHandoffError> {
    match provider {
        Some(plugin_name) => acquire_for_plugin(node, tcp_stream, plugin_name, model_name).await,
        None => Ok(None),
    }
}

impl PluginGpuHandoffError {
    /// The outcome a route returns instead of dialing the plugin.
    pub(in crate::network::openai) fn into_outcome(self) -> proxy::RouteDispatchOutcome {
        match self {
            Self::ClientDisconnected => proxy::RouteDispatchOutcome::Dropped("client_disconnected"),
            Self::Failed(reason) => proxy::RouteDispatchOutcome::Failed(reason),
        }
    }
}

pub(in crate::network::openai) fn is_handoff_failure(
    outcome: &proxy::RouteDispatchOutcome,
) -> bool {
    matches!(
        outcome,
        proxy::RouteDispatchOutcome::Failed(HANDOFF_FAILED_CODE)
    )
}

/// HTTP 503 carrying the stable `plugin_gpu_handoff_failed` code.
pub(in crate::network::openai) async fn send_handoff_failed(
    stream: ClientStream,
    model_name: &str,
    observer: OpenAiRouteObserver<'_>,
) -> std::io::Result<()> {
    Box::pin(proxy::send_503_with_code_observed(
        stream,
        &format!("exclusive GPU handoff failed ({HANDOFF_FAILED_CODE}) for model '{model_name}'"),
        HANDOFF_FAILED_CODE,
        observer,
    ))
    .await
}
