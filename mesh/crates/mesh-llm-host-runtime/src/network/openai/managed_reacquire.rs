//! Reacquire an on-demand managed native model whose selected local instance
//! retired while the request waited (an exclusive_gpu handoff, an eviction or an
//! owner unload): reload it by identity and admit the request on its new port.

use std::sync::Arc;

use crate::mesh;
use crate::runtime::InstanceRequestGuard;
use crate::runtime::native_model_lifecycle::{
    NativeModelLifecycle, RequestLease, with_request_lease,
};

/// The reloaded instance: keep `lease` and `instance` for the whole request.
pub(super) struct ReacquiredInstance {
    pub(super) port: u16,
    pub(super) lease: RequestLease,
    pub(super) instance: Option<InstanceRequestGuard>,
}

pub(super) async fn reacquire_managed_instance(
    node: &mesh::Node,
    service: &Arc<NativeModelLifecycle>,
    managed_model: &str,
) -> Result<ReacquiredInstance, ()> {
    let lease = service
        .acquire(managed_model)
        .await
        .map_err(|error| {
            tracing::warn!(model = %managed_model, %error, "reacquire of retired managed model failed");
        })?
        .ok_or(())?;
    let port = service.port_for(managed_model).ok_or(())?;
    let instance = with_request_lease(
        Some(lease.clone()),
        node.begin_runtime_instance_request(port),
    )
    .await
    .map_err(|error| {
        tracing::debug!(%error, port, "reacquired local runtime instance rejected new work");
    })?;
    Ok(ReacquiredInstance {
        port,
        lease,
        instance,
    })
}

#[cfg(test)]
pub(super) mod test_support {
    use super::*;
    use crate::inference::election::{InferenceTarget, ModelTargets};
    use crate::runtime::{InstanceLifecycleRecord, InstanceLifecycleState, ModelIntent};

    pub(in crate::network::openai) const MANAGED: &str = "native-on-demand";

    pub(in crate::network::openai) type ManagedFixture = (
        mesh::Node,
        Arc<NativeModelLifecycle>,
        tokio::sync::mpsc::Receiver<ModelIntent>,
        tokio::sync::watch::Sender<ModelTargets>,
    );

    pub(in crate::network::openai) async fn node_with_managed_model() -> ManagedFixture {
        let node = Box::pin(mesh::Node::new_for_tests(mesh::NodeRole::Worker))
            .await
            .unwrap();
        let config: crate::plugin::MeshConfig = toml::from_str(&format!(
            "[[models]]\nmodel = \"{MANAGED}\"\n[models.lifecycle]\nload = \"on_demand\"\nprefetch = false\n"
        ))
        .unwrap();
        let (service, intents, targets) = NativeModelLifecycle::test_service(&config);
        assert!(node.native_model_lifecycle.set(service.clone()).is_ok());
        (node, service, intents, targets)
    }

    /// Marks `port` as the loaded instance of `MANAGED` with a lifecycle record in `state`.
    pub(in crate::network::openai) fn load_managed_on(
        fixture: &ManagedFixture,
        port: u16,
        state: InstanceLifecycleState,
    ) -> Arc<tokio::sync::Mutex<InstanceLifecycleRecord>> {
        let (node, service, _, targets) = fixture;
        targets.send_modify(|t| {
            t.targets
                .insert(MANAGED.into(), vec![InferenceTarget::Local(port)]);
        });
        service.record_loaded(MANAGED, "runtime-1", MANAGED, port);
        let record = Arc::new(tokio::sync::Mutex::new(InstanceLifecycleRecord::new(
            state, 4,
        )));
        node.register_runtime_instance_lifecycle(port, record.clone());
        record
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;
    use crate::runtime::ModelIntent;
    use tokio::time::{Duration, timeout};

    #[tokio::test]
    async fn reacquire_of_a_retired_managed_model_fails_when_it_is_unmanaged_or_cannot_reload() {
        let (node, service, mut intents, targets) = node_with_managed_model().await;
        assert!(
            Box::pin(reacquire_managed_instance(&node, &service, "unknown"))
                .await
                .is_err()
        );
        assert!(intents.try_recv().is_err(), "unmanaged names must not load");

        let reacquire = {
            let (node, service) = (node.clone(), service.clone());
            tokio::spawn(async move {
                Box::pin(reacquire_managed_instance(&node, &service, MANAGED))
                    .await
                    .is_err()
            })
        };
        let ModelIntent::Load {
            config_model_id,
            completion: Some(completion),
            ..
        } = timeout(Duration::from_secs(10), intents.recv())
            .await
            .expect("reacquire should request a reload")
            .unwrap()
        else {
            panic!("expected a load intent");
        };
        assert_eq!(config_model_id.as_deref(), Some(MANAGED));
        completion
            .send(Err(anyhow::anyhow!("capacity exhausted")))
            .unwrap();
        assert!(
            timeout(Duration::from_secs(10), reacquire)
                .await
                .expect("reacquire should finish after the failed reload")
                .unwrap()
        );
        assert_eq!(service.port_for(MANAGED), None);
        drop(targets);
        node.endpoint.close().await;
    }
}
