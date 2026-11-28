//! A live session whose selected local instance retired before admission:
//! an on-demand managed model is reloaded on a new port, anything else is
//! refused as draining.

use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use crate::inference::election::{InferenceTarget, ModelTargets};
use crate::mesh::{Node, NodeRole};
use crate::runtime::native_model_lifecycle::NativeModelLifecycle;
use crate::runtime::{InstanceLifecycleRecord, InstanceLifecycleState, ModelIntent};

const MODEL: &str = "whisper-on-demand";
const OPENING: &[u8] = b"GET /v1/realtime?model=whisper-on-demand HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n\r\n";
const HANG_GUARD: Duration = Duration::from_secs(10);

fn instance_record(
    state: InstanceLifecycleState,
) -> Arc<tokio::sync::Mutex<InstanceLifecycleRecord>> {
    Arc::new(tokio::sync::Mutex::new(InstanceLifecycleRecord::new(
        state, 4,
    )))
}

/// A node whose selected port `old_port` has stopped accepting work. The old
/// backend keeps listening so a session relayed there would be observable.
async fn node_with_retired_port(
    managed: bool,
) -> (
    Node,
    Option<(
        Arc<NativeModelLifecycle>,
        tokio::sync::mpsc::Receiver<ModelIntent>,
        tokio::sync::watch::Sender<ModelTargets>,
    )>,
    TcpListener,
) {
    let node = Box::pin(Node::new_for_tests(NodeRole::Worker))
        .await
        .unwrap();
    let old_backend = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let old_port = old_backend.local_addr().unwrap().port();
    node.register_runtime_instance_lifecycle(
        old_port,
        instance_record(InstanceLifecycleState::Stopped),
    );
    let lifecycle = managed.then(|| {
        let config: crate::plugin::MeshConfig = toml::from_str(&format!(
            "[[models]]\nmodel = \"{MODEL}\"\n[models.lifecycle]\nload = \"on_demand\"\nprefetch = false\n"
        ))
        .unwrap();
        let (lifecycle, intents, targets) = NativeModelLifecycle::test_service(&config);
        assert!(node.native_model_lifecycle.set(lifecycle.clone()).is_ok());
        (lifecycle, intents, targets)
    });
    (node, lifecycle, old_backend)
}

/// Runs one session against `port` and returns the client end.
async fn start_session(node: &Node, port: u16) -> (TcpStream, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (client, accepted) = tokio::join!(TcpStream::connect(addr), listener.accept());
    let (server, _) = accepted.unwrap();
    let node = node.clone();
    let session = tokio::spawn(async move {
        Box::pin(super::run_session(
            &node,
            server.into(),
            OPENING,
            MODEL,
            InferenceTarget::Local(port),
        ))
        .await;
    });
    (client.unwrap(), session)
}

async fn read_all(client: &mut TcpStream) -> String {
    let mut response = Vec::new();
    tokio::time::timeout(HANG_GUARD, client.read_to_end(&mut response))
        .await
        .expect("session should answer the client")
        .unwrap();
    String::from_utf8_lossy(&response).into_owned()
}

async fn assert_not_dialed(old_backend: &TcpListener) {
    tokio::select! {
        biased;
        accepted = old_backend.accept() => {
            panic!("the retired port must not receive the session: {accepted:?}");
        }
        _ = tokio::task::yield_now() => {}
    }
}

#[tokio::test]
async fn a_session_on_a_retired_managed_port_reloads_the_model_and_relays_to_its_new_port() {
    let (node, lifecycle, old_backend) = node_with_retired_port(true).await;
    let (lifecycle, mut intents, targets) = lifecycle.unwrap();
    let old_port = old_backend.local_addr().unwrap().port();
    let (mut client, session) = start_session(&node, old_port).await;

    let ModelIntent::Load {
        config_model_id,
        completion: Some(completion),
        ..
    } = tokio::time::timeout(HANG_GUARD, intents.recv())
        .await
        .expect("the retired managed model should be reloaded")
        .unwrap()
    else {
        panic!("expected a load intent");
    };
    assert_eq!(config_model_id.as_deref(), Some(MODEL));

    let new_backend = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let new_port = new_backend.local_addr().unwrap().port();
    let backend = tokio::spawn(async move {
        let (mut stream, _) = new_backend.accept().await.unwrap();
        let mut received = Vec::new();
        let mut buf = [0u8; 4096];
        while !received.windows(4).any(|w| w == b"\r\n\r\n") {
            let n = stream.read(&mut buf).await.unwrap();
            assert_ne!(n, 0, "relay closed before sending the head");
            received.extend_from_slice(&buf[..n]);
        }
        stream
            .write_all(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n")
            .await
            .unwrap();
        stream.shutdown().await.unwrap();
        String::from_utf8_lossy(&received).into_owned()
    });
    node.register_runtime_instance_lifecycle(
        new_port,
        instance_record(InstanceLifecycleState::Serving),
    );
    targets.send_modify(|t| {
        t.targets
            .insert(MODEL.into(), vec![InferenceTarget::Local(new_port)]);
    });
    lifecycle.record_loaded(MODEL, "runtime-2", MODEL, new_port);
    completion
        .send(Ok(crate::api::RuntimeLoadResponse {
            model_ref: MODEL.into(),
            model: MODEL.into(),
            instance_id: "runtime-2".into(),
            profile: String::new(),
            backend: None,
            context_length: None,
        }))
        .unwrap();

    let response = read_all(&mut client).await;
    assert!(response.starts_with("HTTP/1.1 101"), "{response}");
    let head = tokio::time::timeout(HANG_GUARD, backend)
        .await
        .expect("the new port should receive the session")
        .unwrap();
    assert!(
        head.starts_with(&format!("GET /v1/realtime?model={MODEL} HTTP/1.1\r\n")),
        "{head}"
    );
    assert!(head.contains("Connection: Upgrade"), "{head}");
    assert_not_dialed(&old_backend).await;
    tokio::time::timeout(HANG_GUARD, session)
        .await
        .expect("session should end once the host closes")
        .unwrap();
    node.endpoint.close().await;
}

#[tokio::test]
async fn a_session_on_a_retired_port_that_cannot_be_reacquired_is_refused_as_draining() {
    for managed in [false, true] {
        let (node, lifecycle, old_backend) = node_with_retired_port(managed).await;
        let old_port = old_backend.local_addr().unwrap().port();
        let (mut client, session) = start_session(&node, old_port).await;
        if let Some((_lifecycle, mut intents, _targets)) = lifecycle {
            let ModelIntent::Load {
                completion: Some(completion),
                ..
            } = tokio::time::timeout(HANG_GUARD, intents.recv())
                .await
                .expect("the managed model should be reloaded")
                .unwrap()
            else {
                panic!("expected a load intent");
            };
            completion
                .send(Err(anyhow::anyhow!("capacity exhausted")))
                .unwrap();
        }
        let response = read_all(&mut client).await;
        assert!(
            response.starts_with("HTTP/1.1 503"),
            "{managed}: {response}"
        );
        assert!(
            response.contains("local realtime instance is draining"),
            "{managed}: {response}"
        );
        assert_not_dialed(&old_backend).await;
        tokio::time::timeout(HANG_GUARD, session)
            .await
            .expect("refused session should end")
            .unwrap();
        node.endpoint.close().await;
    }
}
