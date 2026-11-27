use super::*;
use crate::api::RuntimeUnloadResponse;
use crate::runtime::ModelIntent;
use crate::runtime::native_model_lifecycle::NativeModelLifecycle;
use mesh_llm_node::serving::UnloadTarget;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const NATIVE_MODEL: &str = "native-4b";
const PLUGIN_NAME: &str = "layerstream";
const PLUGIN_MODEL: &str = "plugin-14b";

async fn node_with_plugin_and_on_demand_native(
    exclusive_gpu: Option<bool>,
) -> (
    mesh::Node,
    tokio::sync::mpsc::Receiver<ModelIntent>,
    tokio::sync::watch::Sender<election::ModelTargets>,
    tempfile::TempDir,
) {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .expect("test node should start");

    let exclusive_line = match exclusive_gpu {
        Some(true) => "exclusive_gpu = true\n",
        Some(false) => "exclusive_gpu = false\n",
        None => "",
    };
    let toml_cfg = format!(
        r#"
[[models]]
model = "{NATIVE_MODEL}"
[models.lifecycle]
load = "on_demand"
idle_timeout_secs = 300
prefetch = false

[[plugin]]
name = "{PLUGIN_NAME}"
{exclusive_line}"#
    );
    let config_dir = tempfile::tempdir().expect("tempdir for test config");
    let config_path = config_dir.path().join("config.toml");
    std::fs::write(&config_path, &toml_cfg).expect("write test config");
    node.replace_config_state_for_test(&config_path)
        .await
        .expect("load test config state");
    let config = node.config_state.lock().await.config().clone();

    let (lifecycle, intents, targets_tx) = NativeModelLifecycle::test_service(&config);
    targets_tx.send_modify(|t| {
        t.targets.insert(
            NATIVE_MODEL.into(),
            vec![election::InferenceTarget::Local(4321)],
        );
    });
    lifecycle.record_loaded(NATIVE_MODEL, "runtime-1", NATIVE_MODEL, 4321);
    let _ = node.native_model_lifecycle.set(lifecycle);

    (node, intents, targets_tx, config_dir)
}

async fn run_plugin_dispatch(
    node: &mesh::Node,
    backend_addr: std::net::SocketAddr,
) -> (String, proxy::RouteDispatchOutcome) {
    let targets = election::ModelTargets::default();
    let affinity = affinity::AffinityRouter::new();
    let plugin_manager = tests::empty_test_plugin_manager();
    plugin_manager
        .set_test_inference_endpoints(vec![crate::plugin::InferenceEndpointRoute {
            plugin_name: PLUGIN_NAME.to_string(),
            endpoint_id: "layerstream".to_string(),
            protocol: Some("openai_compatible".to_string()),
            address: format!("http://{backend_addr}"),
            models: vec![PLUGIN_MODEL.to_string()],
        }])
        .await;

    let ctx = IngressRouteContext {
        node,
        targets: &targets,
        affinity: &affinity,
        plugin_manager: Some(&plugin_manager),
        requested_by_node_id: None,
        automatic_retry: None,
        #[cfg(test)]
        exchange_channel: None,
    };

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (client_res, server_res) = tokio::join!(TcpStream::connect(addr), listener.accept());
    let mut client = client_res.unwrap();
    let (server, _) = server_res.unwrap();
    let request = tests::plugin_only_request(PLUGIN_MODEL);

    let outcome = try_route_plugin_model(
        &ctx,
        server.into(),
        &request,
        PLUGIN_MODEL,
        OpenAiRouteObserver::default(),
    )
    .await;

    let mut response = String::new();
    client.read_to_string(&mut response).await.unwrap();
    (response, outcome)
}

fn assert_responded_status(outcome: proxy::RouteDispatchOutcome, expected: u16) {
    let actual = match outcome {
        proxy::RouteDispatchOutcome::Responded(status)
        | proxy::RouteDispatchOutcome::RespondedWithDigests {
            status_code: status,
            ..
        }
        | proxy::RouteDispatchOutcome::RespondedWithUsage {
            status_code: status,
            ..
        } => Some(status),
        _ => None,
    };
    assert_eq!(
        actual,
        Some(expected),
        "unexpected route dispatch outcome: {outcome:?}"
    );
}

#[tokio::test]
async fn exclusive_gpu_plugin_unloads_native_model_before_forwarding_request() {
    let (node, mut intents, targets_tx, _config_dir) =
        node_with_plugin_and_on_demand_native(Some(true)).await;

    let plugin_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let plugin_addr = plugin_listener.local_addr().unwrap();

    let dispatch_node = node.clone();
    let dispatch =
        tokio::spawn(async move { run_plugin_dispatch(&dispatch_node, plugin_addr).await });

    let ModelIntent::Unload {
        target,
        completion: Some(completion),
        ..
    } = tokio::time::timeout(std::time::Duration::from_secs(5), intents.recv())
        .await
        .expect("unload intent should arrive")
        .expect("intent channel open")
    else {
        panic!("expected unload intent");
    };
    assert_eq!(target, UnloadTarget::Instance("runtime-1".into()));

    // Keep unload completion unsent and confirm dispatch stays parked on the
    // unload completion without dialing the plugin listener.
    tokio::task::yield_now().await;
    assert!(
        !completion.is_closed(),
        "handoff must still be awaiting unload completion"
    );
    assert!(
        !dispatch.is_finished(),
        "dispatch must not finish before unload completes"
    );
    tokio::select! {
        biased;
        accepted = plugin_listener.accept() => {
            panic!("plugin endpoint must not receive a connection before unload completes: {accepted:?}");
        }
        _ = tokio::task::yield_now() => {}
    }

    targets_tx.send_modify(|t| t.targets.clear());
    completion
        .send(Ok(RuntimeUnloadResponse {
            model: NATIVE_MODEL.into(),
            instance_id: "runtime-1".into(),
            unloaded: true,
        }))
        .unwrap();

    let (mut stream, _) =
        tokio::time::timeout(std::time::Duration::from_secs(5), plugin_listener.accept())
            .await
            .expect("dispatch should connect to plugin after unload completes")
            .unwrap();
    let _req = proxy::read_http_request(&mut stream).await.unwrap();
    stream
        .write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\nConnection: close\r\n\r\n{\"ok\":true}",
        )
        .await
        .unwrap();

    let (response, outcome) = dispatch.await.unwrap();
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    assert_responded_status(outcome, 200);
    node.endpoint.close().await;
}

#[tokio::test]
async fn default_plugin_leaves_on_demand_native_model_loaded() {
    let (node, mut intents, _targets_tx, _config_dir) =
        node_with_plugin_and_on_demand_native(None).await;

    let plugin_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let plugin_addr = plugin_listener.local_addr().unwrap();
    let plugin_task = tokio::spawn(async move {
        let (mut stream, _) = plugin_listener.accept().await.unwrap();
        let _req = proxy::read_http_request(&mut stream).await.unwrap();
        stream
            .write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\nConnection: close\r\n\r\n{\"ok\":true}",
            )
            .await
            .unwrap();
    });

    let (response, outcome) = run_plugin_dispatch(&node, plugin_addr).await;
    plugin_task.await.unwrap();
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    assert_responded_status(outcome, 200);
    assert!(
        intents.try_recv().is_err(),
        "non-exclusive plugin must not unload native models"
    );
    node.endpoint.close().await;
}

#[tokio::test]
async fn exclusive_gpu_handoff_failure_returns_503_without_calling_plugin() {
    let (node, mut intents, _targets_tx, _config_dir) =
        node_with_plugin_and_on_demand_native(Some(true)).await;

    let plugin_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let plugin_addr = plugin_listener.local_addr().unwrap();
    let dispatch_node = node.clone();
    let dispatch =
        tokio::spawn(async move { run_plugin_dispatch(&dispatch_node, plugin_addr).await });

    let ModelIntent::Unload {
        completion: Some(completion),
        ..
    } = tokio::time::timeout(std::time::Duration::from_secs(5), intents.recv())
        .await
        .expect("unload intent should arrive")
        .expect("intent channel open")
    else {
        panic!("expected unload intent");
    };
    completion
        .send(Err(anyhow::anyhow!("maintained owner intent")))
        .unwrap();

    let (response, outcome) = dispatch.await.unwrap();
    assert!(response.starts_with("HTTP/1.1 503"), "{response}");
    let (_, body) = response
        .split_once("\r\n\r\n")
        .expect("503 response should contain JSON body");
    let json: serde_json::Value =
        serde_json::from_str(body).expect("503 body should be valid JSON");
    assert_eq!(
        json["error"]["code"].as_str(),
        Some("plugin_gpu_handoff_failed"),
        "503 response body must include the stable plugin_gpu_handoff_failed code: {body}"
    );
    assert_responded_status(outcome, 503);
    tokio::select! {
        biased;
        accepted = plugin_listener.accept() => {
            panic!("plugin endpoint must not be dialed when exclusive_gpu handoff fails: {accepted:?}");
        }
        _ = tokio::task::yield_now() => {}
    }
    node.endpoint.close().await;
}

#[tokio::test]
async fn selected_route_denial_does_not_unload_native_model_for_exclusive_gpu_plugin() {
    let (node, mut intents, _targets_tx, _config_dir) =
        node_with_plugin_and_on_demand_native(Some(true)).await;

    let plugin_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let plugin_addr = plugin_listener.local_addr().unwrap();

    // Deny attempt 1 at backend_selected (HTTP 403) before any GPU handoff runs.
    let (plugin_manager, _seen, _terminals) =
        crate::plugin::selected_attempt_test_support::attempt_policy_manager(1, false).await;
    plugin_manager
        .set_test_inference_endpoints(vec![crate::plugin::InferenceEndpointRoute {
            plugin_name: PLUGIN_NAME.to_string(),
            endpoint_id: "layerstream".to_string(),
            protocol: Some("openai_compatible".to_string()),
            address: format!("http://{plugin_addr}"),
            models: vec![PLUGIN_MODEL.to_string()],
        }])
        .await;
    node.set_plugin_manager(plugin_manager.clone()).await;

    let targets = election::ModelTargets::default();
    let affinity = affinity::AffinityRouter::new();
    let ctx = IngressRouteContext {
        node: &node,
        targets: &targets,
        affinity: &affinity,
        plugin_manager: Some(&plugin_manager),
        requested_by_node_id: None,
        automatic_retry: None,
        #[cfg(test)]
        exchange_channel: None,
    };

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (client_res, server_res) = tokio::join!(TcpStream::connect(addr), listener.accept());
    let mut client = client_res.unwrap();
    let (server, _) = server_res.unwrap();
    let request = tests::plugin_only_request(PLUGIN_MODEL);

    let outcome = try_route_plugin_model(
        &ctx,
        server.into(),
        &request,
        PLUGIN_MODEL,
        OpenAiRouteObserver::default(),
    )
    .await;

    let mut response = String::new();
    client.read_to_string(&mut response).await.unwrap();
    assert!(response.starts_with("HTTP/1.1 403"), "{response}");
    assert_eq!(
        outcome,
        proxy::RouteDispatchOutcome::PolicyDenied { status_code: 403 }
    );
    assert!(
        intents.try_recv().is_err(),
        "selected-route policy denial must not unload native models"
    );
    tokio::select! {
        biased;
        accepted = plugin_listener.accept() => {
            panic!("denied request must not dial plugin endpoint: {accepted:?}");
        }
        _ = tokio::task::yield_now() => {}
    }
    node.endpoint.close().await;
}

#[tokio::test]
async fn client_disconnect_while_waiting_for_handoff_skips_eviction_and_plugin_dial() {
    let (node, mut intents, _targets_tx, _config_dir) =
        node_with_plugin_and_on_demand_native(Some(true)).await;
    let lifecycle = node.native_model_lifecycle.get().unwrap().clone();
    let active_lease = lifecycle.acquire(NATIVE_MODEL).await.unwrap().unwrap();
    let mut checkpoints = lifecycle.install_gate_wait_checkpoint();

    let plugin_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let plugin_addr = plugin_listener.local_addr().unwrap();

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (client_res, server_res) = tokio::join!(TcpStream::connect(addr), listener.accept());
    let client = client_res.unwrap();
    let (server, _) = server_res.unwrap();

    let dispatch_node = node.clone();
    let dispatch = tokio::spawn(async move {
        let targets = election::ModelTargets::default();
        let affinity = affinity::AffinityRouter::new();
        let plugin_manager = tests::empty_test_plugin_manager();
        plugin_manager
            .set_test_inference_endpoints(vec![crate::plugin::InferenceEndpointRoute {
                plugin_name: PLUGIN_NAME.to_string(),
                endpoint_id: "layerstream".to_string(),
                protocol: Some("openai_compatible".to_string()),
                address: format!("http://{plugin_addr}"),
                models: vec![PLUGIN_MODEL.to_string()],
            }])
            .await;
        let ctx = IngressRouteContext {
            node: &dispatch_node,
            targets: &targets,
            affinity: &affinity,
            plugin_manager: Some(&plugin_manager),
            requested_by_node_id: None,
            automatic_retry: None,
            #[cfg(test)]
            exchange_channel: None,
        };
        let request = tests::plugin_only_request(PLUGIN_MODEL);
        try_route_plugin_model(
            &ctx,
            server.into(),
            &request,
            PLUGIN_MODEL,
            OpenAiRouteObserver::default(),
        )
        .await
    });

    assert_eq!(
        tokio::time::timeout(std::time::Duration::from_secs(2), checkpoints.recv())
            .await
            .unwrap()
            .unwrap(),
        "exclusive_gpu_handoff_write"
    );

    // Reset the client TCP socket while handoff is still waiting on the active native lease.
    client.set_zero_linger().unwrap();
    drop(client);

    let outcome = tokio::time::timeout(std::time::Duration::from_secs(2), dispatch)
        .await
        .expect("dispatch must abort promptly on client disconnect")
        .unwrap();
    assert_eq!(
        outcome,
        proxy::RouteDispatchOutcome::Dropped("client_disconnected")
    );

    drop(active_lease);
    tokio::task::yield_now().await;
    assert!(
        intents.try_recv().is_err(),
        "disconnected client must not evict native models once cancelled before unload"
    );
    tokio::select! {
        biased;
        accepted = plugin_listener.accept() => {
            panic!("disconnected client must not dial plugin endpoint: {accepted:?}");
        }
        _ = tokio::task::yield_now() => {}
    }
    node.endpoint.close().await;
}

#[tokio::test]
async fn client_disconnect_after_unload_accepted_keeps_gate_until_unload_completes() {
    let (node, mut intents, _targets_tx, _config_dir) =
        node_with_plugin_and_on_demand_native(Some(true)).await;
    let lifecycle = node.native_model_lifecycle.get().unwrap().clone();

    let plugin_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let plugin_addr = plugin_listener.local_addr().unwrap();

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (client_res, server_res) = tokio::join!(TcpStream::connect(addr), listener.accept());
    let client = client_res.unwrap();
    let (server, _) = server_res.unwrap();

    let dispatch_node = node.clone();
    let dispatch = tokio::spawn(async move {
        let targets = election::ModelTargets::default();
        let affinity = affinity::AffinityRouter::new();
        let plugin_manager = tests::empty_test_plugin_manager();
        plugin_manager
            .set_test_inference_endpoints(vec![crate::plugin::InferenceEndpointRoute {
                plugin_name: PLUGIN_NAME.to_string(),
                endpoint_id: "layerstream".to_string(),
                protocol: Some("openai_compatible".to_string()),
                address: format!("http://{plugin_addr}"),
                models: vec![PLUGIN_MODEL.to_string()],
            }])
            .await;
        let ctx = IngressRouteContext {
            node: &dispatch_node,
            targets: &targets,
            affinity: &affinity,
            plugin_manager: Some(&plugin_manager),
            requested_by_node_id: None,
            automatic_retry: None,
            #[cfg(test)]
            exchange_channel: None,
        };
        let request = tests::plugin_only_request(PLUGIN_MODEL);
        try_route_plugin_model(
            &ctx,
            server.into(),
            &request,
            PLUGIN_MODEL,
            OpenAiRouteObserver::default(),
        )
        .await
    });

    let ModelIntent::Unload {
        completion: Some(completion),
        ..
    } = tokio::time::timeout(std::time::Duration::from_secs(2), intents.recv())
        .await
        .expect("unload intent should arrive")
        .expect("intent channel open")
    else {
        panic!("expected unload intent");
    };

    // Disconnect client after the control loop already accepted the unload intent.
    client.set_zero_linger().unwrap();
    drop(client);

    let outcome = tokio::time::timeout(std::time::Duration::from_secs(2), dispatch)
        .await
        .expect("dispatch must return promptly when client disconnects during unload")
        .unwrap();
    assert_eq!(
        outcome,
        proxy::RouteDispatchOutcome::Dropped("client_disconnected")
    );
    assert!(
        !completion.is_closed(),
        "detached unload owner must remain alive waiting for control-loop unload completion"
    );

    // While unload is in flight, a fresh handoff must block behind the detached unload owner.
    let mut second_handoff = std::pin::pin!(lifecycle.acquire_exclusive_gpu_handoff());
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    assert!(
        second_handoff.as_mut().poll(&mut cx).is_pending(),
        "write gate must remain held until the accepted unload completes"
    );

    completion
        .send(Ok(RuntimeUnloadResponse {
            model: NATIVE_MODEL.into(),
            instance_id: "runtime-1".into(),
            unloaded: true,
        }))
        .unwrap();

    // Completing the unload clears residency and releases the gate cleanly.
    let _lease = tokio::time::timeout(std::time::Duration::from_secs(2), second_handoff)
        .await
        .expect("second handoff should proceed once accepted unload completes")
        .unwrap();
    assert!(
        intents.try_recv().is_err(),
        "completed unload must have cleared instance residency"
    );
    tokio::select! {
        biased;
        accepted = plugin_listener.accept() => {
            panic!("disconnected client must not dial plugin endpoint: {accepted:?}");
        }
        _ = tokio::task::yield_now() => {}
    }
    node.endpoint.close().await;
}

#[tokio::test]
async fn warm_native_http_request_reloads_and_succeeds_after_handoff_unloads_stale_port() {
    let (node, mut intents, targets_tx, _config_dir) =
        node_with_plugin_and_on_demand_native(Some(true)).await;
    let lifecycle = node.native_model_lifecycle.get().unwrap().clone();

    // Replace the placeholder port 4321 with a real bound listener + lifecycle record.
    let old_backend = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let old_port = old_backend.local_addr().unwrap().port();
    let old_instance_lifecycle = Arc::new(tokio::sync::Mutex::new(
        crate::runtime::InstanceLifecycleRecord::new(
            crate::runtime::InstanceLifecycleState::Serving,
            4,
        ),
    ));
    node.register_runtime_instance_lifecycle(old_port, old_instance_lifecycle.clone());
    node.set_hosted_models(vec![NATIVE_MODEL.into()]).await;
    targets_tx.send_modify(|t| {
        t.targets.insert(
            NATIVE_MODEL.into(),
            vec![election::InferenceTarget::Local(old_port)],
        );
    });
    lifecycle.record_loaded(NATIVE_MODEL, "runtime-1", NATIVE_MODEL, old_port);

    // 1. Hold an active lease so the exclusive_gpu handoff queues its writer first.
    let active_lease = lifecycle.acquire(NATIVE_MODEL).await.unwrap().unwrap();
    let mut checkpoints = lifecycle.install_gate_wait_checkpoint();
    let handoff = {
        let lifecycle = lifecycle.clone();
        tokio::spawn(async move { lifecycle.acquire_exclusive_gpu_handoff().await.unwrap() })
    };
    assert_eq!(
        tokio::time::timeout(std::time::Duration::from_secs(2), checkpoints.recv())
            .await
            .unwrap()
            .unwrap(),
        "exclusive_gpu_handoff_write"
    );

    // 2. Send a warm native HTTP request that selects `old_port` and waits on admission.
    let ingress_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let ingress_addr = ingress_listener.local_addr().unwrap();
    let mut initial_targets = election::ModelTargets::default();
    initial_targets.targets.insert(
        NATIVE_MODEL.into(),
        vec![election::InferenceTarget::Local(old_port)],
    );
    let ingress_node = node.clone();
    let ingress_task = tokio::spawn(async move {
        let (stream, _) = ingress_listener.accept().await.unwrap();
        handle_api_proxy_connection(
            ingress_node,
            stream.into(),
            initial_targets,
            affinity::AffinityRouter::new(),
            crate::runtime::IngressType::LocalOpenAi,
            None,
        )
        .await;
    });
    let mut client = TcpStream::connect(ingress_addr).await.unwrap();
    let body =
        format!(r#"{{"model":"{NATIVE_MODEL}","messages":[{{"role":"user","content":"hi"}}]}}"#);
    client
        .write_all(
            format!(
                "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .as_bytes(),
        )
        .await
        .unwrap();

    assert_eq!(
        tokio::time::timeout(std::time::Duration::from_secs(2), checkpoints.recv())
            .await
            .unwrap()
            .unwrap(),
        "lease_port_read"
    );

    // 3. Release the active lease so handoff unloads `old_port`.
    drop(active_lease);
    let ModelIntent::Unload {
        target,
        completion: Some(unload_completion),
        ..
    } = tokio::time::timeout(std::time::Duration::from_secs(2), intents.recv())
        .await
        .unwrap()
        .unwrap()
    else {
        panic!("expected unload intent");
    };
    assert_eq!(target, UnloadTarget::Instance("runtime-1".into()));
    {
        let mut record = old_instance_lifecycle.lock().await;
        let _ = record.mark_draining(std::time::Instant::now());
        let _ = record.transition_to_unloading();
        let _ = record.transition_to(crate::runtime::InstanceLifecycleState::Stopped);
    }
    node.unregister_runtime_instance_lifecycle(old_port);
    drop(old_backend);
    targets_tx.send_modify(|t| t.targets.clear());
    lifecycle.record_unloaded("runtime-1");
    unload_completion
        .send(Ok(RuntimeUnloadResponse {
            model: NATIVE_MODEL.into(),
            instance_id: "runtime-1".into(),
            unloaded: true,
        }))
        .unwrap();

    // 4. Handoff finishes and drops its lease; the waiting warm request must reload on a new port.
    let handoff_lease = handoff.await.unwrap();
    drop(handoff_lease);

    let ModelIntent::Load {
        config_model_id,
        completion: Some(load_completion),
        ..
    } = tokio::time::timeout(std::time::Duration::from_secs(2), intents.recv())
        .await
        .expect("retired warm request should trigger reload")
        .unwrap()
    else {
        panic!("expected reload intent after handoff retired stale port");
    };
    assert_eq!(config_model_id.as_deref(), Some(NATIVE_MODEL));

    let new_backend = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let new_port = new_backend.local_addr().unwrap().port();
    let new_backend_task = tokio::spawn(async move {
        let (mut stream, _) = new_backend.accept().await.unwrap();
        let _req = proxy::read_http_request(&mut stream).await.unwrap();
        stream
            .write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\nConnection: close\r\n\r\n{\"ok\":true}",
            )
            .await
            .unwrap();
    });
    node.register_runtime_instance_lifecycle(
        new_port,
        Arc::new(tokio::sync::Mutex::new(
            crate::runtime::InstanceLifecycleRecord::new(
                crate::runtime::InstanceLifecycleState::Serving,
                4,
            ),
        )),
    );
    targets_tx.send_modify(|t| {
        t.targets.insert(
            NATIVE_MODEL.into(),
            vec![election::InferenceTarget::Local(new_port)],
        );
    });
    lifecycle.record_loaded(NATIVE_MODEL, "runtime-2", NATIVE_MODEL, new_port);
    load_completion
        .send(Ok(crate::api::RuntimeLoadResponse {
            model_ref: NATIVE_MODEL.into(),
            model: NATIVE_MODEL.into(),
            instance_id: "runtime-2".into(),
            profile: String::new(),
            backend: None,
            context_length: None,
        }))
        .unwrap();

    let mut response = String::new();
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        client.read_to_string(&mut response),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(
        response.starts_with("HTTP/1.1 200"),
        "warm native request must succeed after reloading on refreshed port: {response}"
    );
    new_backend_task.await.unwrap();
    ingress_task.await.unwrap();
    node.endpoint.close().await;
}
