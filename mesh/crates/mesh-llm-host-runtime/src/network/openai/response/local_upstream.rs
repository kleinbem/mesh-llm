//! The upstream for a local routing attempt: admit the selected local instance,
//! or reload an on-demand managed model whose instance retired while the request
//! waited, and connect to it.

use tokio::net::TcpStream;

use crate::mesh;

pub(super) struct LocalAttemptUpstream {
    pub(super) port: u16,
    _residency: Option<crate::runtime::native_model_lifecycle::RequestLease>,
    _instance: Option<crate::runtime::InstanceRequestGuard>,
    pub(super) stream: TcpStream,
}

pub(super) async fn acquire_local_attempt_upstream(
    node: &mesh::Node,
    port: u16,
    model: Option<&str>,
) -> Result<LocalAttemptUpstream, ()> {
    let managed_model = node
        .native_model_lifecycle
        .get()
        .and_then(|service| service.managed_model_for(port, model));
    if let Ok(instance_request) = node.begin_runtime_instance_request(port).await {
        if let Ok(stream) = TcpStream::connect(format!("127.0.0.1:{port}")).await {
            return Ok(LocalAttemptUpstream {
                port,
                _residency: None,
                _instance: instance_request,
                stream,
            });
        }
        tracing::warn!("API proxy: can't reach local OpenAI surface on {port}");
    } else {
        tracing::debug!(port, "local runtime instance rejected new work");
    }

    // If a warm on-demand managed instance retired while waiting behind an
    // exclusive_gpu handoff or eviction, reload it by identity and refresh the port.
    let (Some(service), Some(managed_model)) = (node.native_model_lifecycle.get(), managed_model)
    else {
        return Err(());
    };
    // Box the cold reload path so `route_local_attempt`'s stack frame stays small
    // on Windows debug builds (2 MiB test thread stack).
    Box::pin(reacquire_managed_local_attempt(
        node,
        service,
        &managed_model,
    ))
    .await
}

async fn reacquire_managed_local_attempt(
    node: &mesh::Node,
    service: &std::sync::Arc<crate::runtime::native_model_lifecycle::NativeModelLifecycle>,
    managed_model: &str,
) -> Result<LocalAttemptUpstream, ()> {
    let reacquired = crate::network::openai::managed_reacquire::reacquire_managed_instance(
        node,
        service,
        managed_model,
    )
    .await?;
    let stream = TcpStream::connect(format!("127.0.0.1:{}", reacquired.port))
        .await
        .map_err(|error| {
            tracing::warn!(
                "API proxy: can't reach reacquired local OpenAI surface on {}: {error}",
                reacquired.port
            );
        })?;
    Ok(LocalAttemptUpstream {
        port: reacquired.port,
        _residency: Some(reacquired.lease),
        _instance: reacquired.instance,
        stream,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;

    use crate::network::openai::managed_reacquire::test_support::{
        MANAGED, load_managed_on, node_with_managed_model,
    };

    #[tokio::test]
    async fn reacquire_does_not_dial_a_refreshed_port_that_rejects_new_work() {
        let mut fixture = node_with_managed_model().await;
        let backend = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = backend.local_addr().unwrap().port();
        load_managed_on(
            &fixture,
            port,
            crate::runtime::InstanceLifecycleState::Stopped,
        );
        let (node, service, intents, _) = &mut fixture;
        assert_eq!(service.port_for(MANAGED), Some(port));
        assert!(
            Box::pin(reacquire_managed_local_attempt(node, service, MANAGED))
                .await
                .is_err()
        );
        assert!(intents.try_recv().is_err(), "a ready model must not reload");
        tokio::select! {
            biased;
            accepted = backend.accept() => {
                panic!("a rejecting instance must not be dialed: {accepted:?}");
            }
            _ = tokio::task::yield_now() => {}
        }
        node.endpoint.close().await;
    }

    #[tokio::test]
    async fn reacquire_reports_an_unreachable_refreshed_port_and_releases_its_admission() {
        let fixture = node_with_managed_model().await;
        // Bind and drop a listener for a loopback port nothing listens on.
        let port = TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let record = load_managed_on(
            &fixture,
            port,
            crate::runtime::InstanceLifecycleState::Serving,
        );
        let (node, service, _, _) = &fixture;
        assert!(
            Box::pin(reacquire_managed_local_attempt(node, service, MANAGED))
                .await
                .is_err()
        );
        assert_eq!(record.lock().await.in_flight_count(), 0);
        assert!(
            record.lock().await.is_accepting_work(),
            "a refused connect must leave the instance serving"
        );
        node.endpoint.close().await;
    }
}
