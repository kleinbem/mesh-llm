//! Request leases, cold loads and the exclusive_gpu handoff.
use super::*;

fn fixture(
    timeout: u64,
) -> (
    Arc<NativeModelLifecycle>,
    mpsc::Receiver<ModelIntent>,
    watch::Sender<ModelTargets>,
) {
    let (intents, receiver) = mpsc::channel(8);
    let (targets, snapshot) = watch::channel(ModelTargets::default());
    let residency = Arc::new(Residency {
        policy: ModelLifecycle {
            load: ModelLoadPolicy::OnDemand,
            idle_timeout_secs: timeout,
            prefetch: false,
        },
        spec: "test".into(),
        names: vec!["test".into()],
        loading: Mutex::new(None),
        gate: Arc::default(),
        operation: Arc::default(),
        idle_since: std::sync::Mutex::new(std::time::Instant::now() - Duration::from_secs(60)),
        instance: std::sync::Mutex::new(None),
    });
    (
        Arc::new(NativeModelLifecycle {
            models: BTreeMap::from([("test".into(), residency)]),
            targets: snapshot,
            intents,
            gate_wait_checkpoint: std::sync::Mutex::new(None),
        }),
        receiver,
        targets,
    )
}

fn finish_load(
    lifecycle: &NativeModelLifecycle,
    intent: ModelIntent,
    targets: &watch::Sender<ModelTargets>,
) {
    let ModelIntent::Load {
        completion: Some(completion),
        ..
    } = intent
    else {
        panic!("expected load")
    };
    targets.send_modify(|targets| {
        targets
            .targets
            .insert("loaded-test".into(), vec![InferenceTarget::Local(1234)]);
    });
    let response = crate::api::RuntimeLoadResponse {
        model_ref: "test".into(),
        model: "loaded-test".into(),
        instance_id: "runtime-1".into(),
        profile: String::new(),
        backend: None,
        context_length: None,
    };
    lifecycle.record_loaded("test", &response.instance_id, &response.model, 1234);
    completion.send(Ok(response)).unwrap();
}

#[tokio::test]
async fn simultaneous_requests_share_one_load_and_hold_stream_leases() {
    let (lifecycle, mut intents, targets) = fixture(1);
    let first = {
        let lifecycle = lifecycle.clone();
        tokio::spawn(async move { lifecycle.acquire("test").await.unwrap().unwrap() })
    };
    let second = {
        let lifecycle = lifecycle.clone();
        tokio::spawn(async move { lifecycle.acquire("test").await.unwrap().unwrap() })
    };
    finish_load(&lifecycle, intents.recv().await.unwrap(), &targets);
    let (first, second) = (first.await.unwrap(), second.await.unwrap());
    assert!(intents.try_recv().is_err());
    assert_eq!(lifecycle.execution_model("test"), "loaded-test");
    assert!(lifecycle.manages("loaded-test"));
    let alias = lifecycle.acquire("loaded-test").await.unwrap();
    assert!(intents.try_recv().is_err());
    drop(alias);
    lifecycle.evict_idle().await;
    assert!(intents.try_recv().is_err());
    drop(first);
    lifecycle.evict_idle().await;
    assert!(intents.try_recv().is_err());
    drop(second);
    assert!(
        lifecycle.models["test"]
            .idle_since
            .lock()
            .unwrap()
            .elapsed()
            < Duration::from_secs(1)
    );
}

#[tokio::test]
async fn idle_eviction_targets_exact_instance_and_next_request_reloads() {
    let (lifecycle, mut intents, targets) = fixture(1);
    *lifecycle.models["test"].instance.lock().unwrap() =
        Some(("runtime-1".into(), "loaded-test".into(), 1234));
    let eviction = {
        let lifecycle = lifecycle.clone();
        tokio::spawn(async move { lifecycle.evict_idle().await })
    };
    let ModelIntent::Unload {
        target,
        completion: Some(completion),
        options,
        ..
    } = intents.recv().await.unwrap()
    else {
        panic!("expected unload")
    };
    assert_eq!(target, UnloadTarget::Instance("runtime-1".into()));
    assert!(!options.force);
    completion
        .send(Ok(crate::api::RuntimeUnloadResponse {
            model: "loaded-test".into(),
            instance_id: "runtime-1".into(),
            unloaded: true,
        }))
        .unwrap();
    eviction.await.unwrap();
    let request = {
        let lifecycle = lifecycle.clone();
        tokio::spawn(async move { lifecycle.acquire("test").await.unwrap() })
    };
    finish_load(&lifecycle, intents.recv().await.unwrap(), &targets);
    assert!(request.await.unwrap().is_some());
}

#[tokio::test]
async fn exclusive_gpu_handoff_drains_native_lease_unloads_and_reloads_after_release() {
    // Large idle timeout (300s) proves handoff unloads immediately instead of waiting for idle eviction.
    let (lifecycle, mut intents, targets) = fixture(300);

    // 1. Load native model and hold an active request lease.
    let active = {
        let lifecycle = lifecycle.clone();
        tokio::spawn(async move { lifecycle.acquire("test").await.unwrap().unwrap() })
    };
    finish_load(&lifecycle, intents.recv().await.unwrap(), &targets);
    let active_lease = active.await.unwrap();

    let mut checkpoints = lifecycle.install_gate_wait_checkpoint();

    // 2. Start exclusive_gpu plugin handoff: must wait while `active_lease` is held.
    let handoff_task = {
        let lifecycle = lifecycle.clone();
        tokio::spawn(async move { lifecycle.acquire_exclusive_gpu_handoff().await.unwrap() })
    };
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), checkpoints.recv())
            .await
            .expect("handoff should reach write gate")
            .expect("checkpoint channel open"),
        "exclusive_gpu_handoff_write"
    );
    assert!(
        intents.try_recv().is_err(),
        "handoff must not unload while native RequestLease is active"
    );

    // 3. Dropping the active native lease lets handoff proceed and unload immediately.
    drop(active_lease);
    let ModelIntent::Unload {
        target,
        completion: Some(completion),
        options,
        ..
    } = intents.recv().await.unwrap()
    else {
        panic!("expected unload intent for exclusive_gpu handoff")
    };
    assert_eq!(target, UnloadTarget::Instance("runtime-1".into()));
    assert!(!options.force);
    targets.send_modify(|t| {
        t.targets.clear();
    });
    completion
        .send(Ok(crate::api::RuntimeUnloadResponse {
            model: "loaded-test".into(),
            instance_id: "runtime-1".into(),
            unloaded: true,
        }))
        .unwrap();
    let handoff_lease = handoff_task.await.unwrap();

    // 4. While the plugin holds `handoff_lease`, a new native request must block without loading.
    let reload_task = {
        let lifecycle = lifecycle.clone();
        tokio::spawn(async move { lifecycle.acquire("test").await.unwrap().unwrap() })
    };
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), checkpoints.recv())
            .await
            .expect("reload should reach read gate")
            .expect("checkpoint channel open"),
        "acquire_read"
    );
    assert!(
        intents.try_recv().is_err(),
        "native reload must wait until exclusive_gpu handoff lease is dropped"
    );

    // 5. Releasing `handoff_lease` allows the waiting native request to reload cleanly.
    drop(handoff_lease);
    finish_load(&lifecycle, intents.recv().await.unwrap(), &targets);
    let reloaded = reload_task.await.unwrap();
    assert_eq!(lifecycle.execution_model("test"), "loaded-test");
    drop(reloaded);
}

#[tokio::test]
async fn exclusive_gpu_handoff_queued_after_acquire_does_not_deadlock_selected_instance_admission()
{
    let (lifecycle, mut intents, targets) = fixture(300);
    let node = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .unwrap();
    assert!(node.native_model_lifecycle.set(lifecycle.clone()).is_ok());
    node.register_runtime_instance_lifecycle(
        1234,
        Arc::new(tokio::sync::Mutex::new(
            crate::runtime::InstanceLifecycleRecord::new(
                crate::runtime::InstanceLifecycleState::Serving,
                10,
            ),
        )),
    );

    // 1. Cold-load native model and obtain the outer `RequestLease`.
    let active = {
        let lifecycle = lifecycle.clone();
        tokio::spawn(async move { lifecycle.acquire("test").await.unwrap() })
    };
    finish_load(&lifecycle, intents.recv().await.unwrap(), &targets);
    let outer_lease = active.await.unwrap();

    let mut checkpoints = lifecycle.install_gate_wait_checkpoint();

    // 2. Queue an exclusive_gpu handoff write lock while the outer read lease is held.
    let mut handoff_fut = std::pin::pin!(lifecycle.acquire_exclusive_gpu_handoff());
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    assert!(
        handoff_fut.as_mut().poll(&mut cx).is_pending(),
        "handoff must wait while outer read lease is held"
    );
    assert_eq!(
        checkpoints
            .try_recv()
            .expect("handoff should queue write gate"),
        "exclusive_gpu_handoff_write"
    );
    // Tokio's write-preferring RwLock now blocks any fresh read lock on this gate.
    assert!(lifecycle.models["test"].gate.try_read().is_err());

    // 3. Selected-instance admission inside `with_request_lease` must reuse
    // the existing read lease instead of deadlocking behind the queued writer.
    let selected = with_request_lease(outer_lease, async {
        tokio::time::timeout(
            Duration::from_secs(2),
            node.begin_runtime_instance_request(1234),
        )
        .await
        .expect("selected-instance admission must not deadlock behind queued handoff")
        .expect("instance should accept work")
        .expect("registered instance should return guard")
    })
    .await;

    // 4. Even after the outer scope exits, polling `handoff_fut` directly proves
    // `InstanceRequestGuard` still holds the shared read lease and blocks unload.
    assert!(
        handoff_fut.as_mut().poll(&mut cx).is_pending(),
        "handoff must remain blocked while InstanceRequestGuard holds the reused lease"
    );
    assert!(
        intents.try_recv().is_err(),
        "handoff must not emit unload intent while InstanceRequestGuard holds the reused lease"
    );

    // 5. Dropping `selected` releases the last lease reference and lets handoff unload.
    drop(selected);
    assert!(
        handoff_fut.as_mut().poll(&mut cx).is_pending(),
        "handoff should now be waiting for unload completion"
    );
    let ModelIntent::Unload {
        target,
        completion: Some(completion),
        ..
    } = intents
        .try_recv()
        .expect("unload intent should arrive once selected request ends")
    else {
        panic!("expected unload intent");
    };
    assert_eq!(target, UnloadTarget::Instance("runtime-1".into()));
    completion
        .send(Ok(crate::api::RuntimeUnloadResponse {
            model: "loaded-test".into(),
            instance_id: "runtime-1".into(),
            unloaded: true,
        }))
        .unwrap();
    let _handoff_lease = handoff_fut.await.unwrap();
    node.endpoint.close().await;
}

#[tokio::test]
async fn exclusive_gpu_handoff_timeout_keeps_gate_until_unload_finishes_and_clears_instance() {
    let (lifecycle, mut intents, _targets) = fixture(300);
    *lifecycle.models["test"].instance.lock().unwrap() =
        Some(("runtime-1".into(), "loaded-test".into(), 1234));

    // 1. Start handoff with a short timeout and hold the unload open past timeout.
    let handoff_task = {
        let lifecycle = lifecycle.clone();
        tokio::spawn(async move {
            lifecycle
                .acquire_exclusive_gpu_handoff_with_timeout(Duration::from_millis(40))
                .await
        })
    };
    let ModelIntent::Unload {
        target,
        completion: Some(completion),
        ..
    } = tokio::time::timeout(Duration::from_secs(2), intents.recv())
        .await
        .expect("unload intent should arrive")
        .expect("intent channel open")
    else {
        panic!("expected unload intent");
    };
    assert_eq!(target, UnloadTarget::Instance("runtime-1".into()));

    // 2. Wait for the caller's handoff timeout to expire while unload is still in flight.
    let err = handoff_task
        .await
        .unwrap()
        .expect_err("handoff should time out while unload is still running");
    assert!(err.to_string().contains("timed out"), "{err:#}");

    // 3. The detached unload owner must keep both operation and write gate locked
    // until the control loop actually completes the unload.
    assert!(lifecycle.models["test"].operation.try_lock().is_err());
    assert!(lifecycle.models["test"].gate.try_read().is_err());
    assert!(lifecycle.models["test"].gate.try_write().is_err());

    // 4. Completing the unload clears the stale instance record and releases the gate.
    completion
        .send(Ok(crate::api::RuntimeUnloadResponse {
            model: "loaded-test".into(),
            instance_id: "runtime-1".into(),
            unloaded: true,
        }))
        .unwrap();
    let _gate = tokio::time::timeout(
        Duration::from_secs(2),
        lifecycle.models["test"].gate.write(),
    )
    .await
    .expect("unload owner should release write gate after completion");
    drop(_gate);
    assert!(
        lifecycle.models["test"].instance.lock().unwrap().is_none(),
        "completed unload must clear residency.instance even after caller timed out"
    );

    // 5. A subsequent handoff succeeds immediately without trying to unload a stale instance.
    let _handoff = lifecycle.acquire_exclusive_gpu_handoff().await.unwrap();
    assert!(
        intents.try_recv().is_err(),
        "cleared instance record must not trigger a duplicate unload"
    );
}

#[tokio::test]
async fn record_unloaded_clears_only_matching_instance_and_allows_subsequent_handoff() {
    let (lifecycle, mut intents, _targets) = fixture(300);
    lifecycle.record_loaded("test", "runtime-1", "loaded-test", 1234);

    // Teardown of a different instance must not disturb the active record.
    lifecycle.record_unloaded("runtime-other");
    assert!(lifecycle.models["test"].instance.lock().unwrap().is_some());

    // Teardown of the tracked instance clears the record so exclusive_gpu
    // handoff does not try to unload a vanished instance.
    lifecycle.record_unloaded("runtime-1");
    assert!(lifecycle.models["test"].instance.lock().unwrap().is_none());

    let _handoff = lifecycle.acquire_exclusive_gpu_handoff().await.unwrap();
    assert!(
        intents.try_recv().is_err(),
        "exited instance must not trigger a stale unload intent"
    );
}

#[tokio::test]
async fn exclusive_gpu_handoff_propagates_unload_failure_without_dropping_instance() {
    let (lifecycle, mut intents, _targets) = fixture(300);
    *lifecycle.models["test"].instance.lock().unwrap() =
        Some(("runtime-1".into(), "loaded-test".into(), 1234));

    let handoff_task = {
        let lifecycle = lifecycle.clone();
        tokio::spawn(async move { lifecycle.acquire_exclusive_gpu_handoff().await })
    };
    let ModelIntent::Unload {
        completion: Some(completion),
        ..
    } = intents.recv().await.unwrap()
    else {
        panic!("expected unload intent")
    };
    completion
        .send(Err(anyhow::anyhow!("maintained owner intent")))
        .unwrap();
    let err = handoff_task.await.unwrap().unwrap_err();
    assert!(err.to_string().contains("failed to unload native model"));
    assert!(lifecycle.models["test"].instance.lock().unwrap().is_some());
}

#[tokio::test]
async fn exclusive_gpu_handoff_succeeds_when_earlier_owner_unload_already_cleared_residency() {
    let (lifecycle, mut intents, _targets) = fixture(300);
    lifecycle.record_loaded("test", "runtime-1", "loaded-test", 1234);

    let handoff_task = {
        let lifecycle = lifecycle.clone();
        tokio::spawn(async move { lifecycle.acquire_exclusive_gpu_handoff().await })
    };
    let ModelIntent::Unload {
        target,
        completion: Some(completion),
        ..
    } = intents.recv().await.unwrap()
    else {
        panic!("expected unload intent")
    };
    assert_eq!(target, UnloadTarget::Instance("runtime-1".into()));

    // An earlier owner-requested unload finishes first and clears residency,
    // so the control loop rejects the duplicate unload as missing.
    lifecycle.record_unloaded("runtime-1");
    completion
        .send(Err(anyhow::anyhow!(
            "model or runtime instance 'runtime-1' is not loaded"
        )))
        .unwrap();

    let lease = handoff_task
        .await
        .unwrap()
        .expect("handoff should succeed once the matching instance no longer holds residency");
    assert!(
        lifecycle.models["test"].gate.try_read().is_err(),
        "handoff lease must remain held after succeeding"
    );
    drop(lease);
    assert!(lifecycle.models["test"].gate.try_read().is_ok());
}

/// Rejections and lost completions wait another idle interval before retrying.
#[tokio::test]
async fn deferred_idle_unloads_back_off_before_retrying() {
    for error in [
        Some("maintained owner intent"),
        Some("instance no longer exists"),
        None,
    ] {
        let (lifecycle, mut intents, _targets) = fixture(30);
        let residency = &lifecycle.models["test"];
        *residency.instance.lock().unwrap() =
            Some(("runtime-1".into(), "loaded-test".into(), 1234));
        let eviction = {
            let lifecycle = lifecycle.clone();
            tokio::spawn(async move { lifecycle.evict_idle().await })
        };
        let ModelIntent::Unload {
            completion: Some(completion),
            ..
        } = intents.recv().await.unwrap()
        else {
            panic!("expected unload")
        };
        if let Some(error) = error {
            completion.send(Err(anyhow::anyhow!(error))).unwrap();
        } else {
            drop(completion);
        }
        eviction.await.unwrap();
        assert!(residency.instance.lock().unwrap().is_some());
        lifecycle.evict_idle().await;
        assert!(
            intents.try_recv().is_err(),
            "must not retry on the next maintenance tick"
        );
        *residency.idle_since.lock().unwrap() = std::time::Instant::now() - Duration::from_secs(15);
        lifecycle.evict_idle().await;
        assert!(
            intents.try_recv().is_err(),
            "must wait the entire idle interval"
        );
        *residency.idle_since.lock().unwrap() = std::time::Instant::now() - Duration::from_secs(31);
        let retry = {
            let lifecycle = lifecycle.clone();
            tokio::spawn(async move { lifecycle.evict_idle().await })
        };
        let ModelIntent::Unload {
            completion: Some(completion),
            ..
        } = intents.recv().await.unwrap()
        else {
            panic!("expected retry")
        };
        completion
            .send(Ok(crate::api::RuntimeUnloadResponse {
                model: "loaded-test".into(),
                instance_id: "runtime-1".into(),
                unloaded: true,
            }))
            .unwrap();
        retry.await.unwrap();
        assert!(residency.instance.lock().unwrap().is_none());
    }
}

#[tokio::test]
async fn resident_policy_and_unowned_instances_are_never_evicted() {
    for timeout in [0, 1] {
        let (lifecycle, mut intents, _targets) = fixture(timeout);
        if timeout == 0 {
            *lifecycle.models["test"].instance.lock().unwrap() =
                Some(("runtime-1".into(), "loaded-test".into(), 1234));
        }
        lifecycle.evict_idle().await;
        assert!(intents.try_recv().is_err());
    }
}

#[tokio::test]
async fn failed_load_releases_waiters_and_allows_retry() {
    let (lifecycle, mut intents, targets) = fixture(1);
    let first = {
        let lifecycle = lifecycle.clone();
        tokio::spawn(async move { lifecycle.acquire("test").await })
    };
    let ModelIntent::Load {
        completion: Some(completion),
        ..
    } = intents.recv().await.unwrap()
    else {
        panic!("expected load")
    };
    completion
        .send(Err(anyhow::anyhow!("capacity exhausted")))
        .unwrap();
    assert!(first.await.unwrap().is_err());
    let retry = {
        let lifecycle = lifecycle.clone();
        tokio::spawn(async move { lifecycle.acquire("test").await })
    };
    finish_load(&lifecycle, intents.recv().await.unwrap(), &targets);
    assert!(retry.await.unwrap().is_ok());
}

#[tokio::test]
async fn cancelling_request_does_not_lose_load_ownership() {
    let (lifecycle, mut intents, targets) = fixture(1);
    let request = {
        let lifecycle = lifecycle.clone();
        tokio::spawn(async move { lifecycle.acquire("test").await })
    };
    let intent = intents.recv().await.unwrap();
    request.abort();
    finish_load(&lifecycle, intent, &targets);
    // Acquiring again synchronizes with the detached load job.
    let _lease = lifecycle.acquire("test").await.unwrap();
    assert!(lifecycle.models["test"].instance.lock().unwrap().is_some());
    assert!(intents.try_recv().is_err());
}

#[tokio::test]
async fn failed_load_is_shared_and_cancelled_waiters_do_not_queue_retries() {
    for cancel in [false, true] {
        let (lifecycle, mut intents, _targets) = fixture(0);
        let mut waiters = (0..16)
            .map(|_| Box::pin(lifecycle.acquire("test")))
            .collect::<Vec<_>>();
        std::future::poll_fn(|cx| {
            for waiter in &mut waiters {
                assert!(std::future::Future::poll(waiter.as_mut(), cx).is_pending());
            }
            std::task::Poll::Ready(())
        })
        .await;
        let ModelIntent::Load {
            completion: Some(completion),
            ..
        } = intents.recv().await.unwrap()
        else {
            panic!("expected load")
        };
        if cancel {
            drop(waiters);
            completion.send(Err(anyhow::anyhow!("capacity"))).unwrap();
        } else {
            completion.send(Err(anyhow::anyhow!("capacity"))).unwrap();
            for waiter in waiters {
                assert!(waiter.await.is_err());
            }
        }
        let _finished = lifecycle.models["test"].operation.lock().await;
        assert!(
            intents.try_recv().is_err(),
            "failed/cancelled waiters must not spawn sequential loads"
        );
    }
}

#[tokio::test]
async fn actual_instance_guard_protects_a_port_selected_under_another_model_name() {
    let (mut lifecycle, mut intents, targets) = fixture(1);
    let other = Arc::new(Residency {
        policy: ModelLifecycle {
            load: ModelLoadPolicy::OnDemand,
            idle_timeout_secs: 1,
            prefetch: false,
        },
        spec: "other".into(),
        names: vec!["other".into()],
        loading: Mutex::new(None),
        gate: Arc::default(),
        operation: Arc::default(),
        idle_since: std::sync::Mutex::new(std::time::Instant::now() - Duration::from_secs(60)),
        instance: std::sync::Mutex::new(Some(("runtime-2".into(), "shared".into(), 2345))),
    });
    Arc::get_mut(&mut lifecycle)
        .unwrap()
        .models
        .insert("other".into(), other.clone());
    *lifecycle.models["test"].instance.lock().unwrap() =
        Some(("runtime-1".into(), "shared".into(), 1234));
    targets.send_modify(|targets| {
        targets.targets.insert(
            "shared".into(),
            vec![InferenceTarget::Local(1234), InferenceTarget::Local(2345)],
        );
    });
    assert_eq!(
        lifecycle.targets_for("test").candidates("shared"),
        vec![InferenceTarget::Local(1234)]
    );
    assert_eq!(
        lifecycle.targets_for("other").candidates("shared"),
        vec![InferenceTarget::Local(2345)]
    );
    let node = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .unwrap();
    assert!(node.native_model_lifecycle.set(lifecycle.clone()).is_ok());
    node.register_runtime_instance_lifecycle(
        2345,
        Arc::new(tokio::sync::Mutex::new(
            crate::runtime::InstanceLifecycleRecord::new(
                crate::runtime::InstanceLifecycleState::Serving,
                10,
            ),
        )),
    );
    let outer = lifecycle.acquire("test").await.unwrap();
    let selected = node.begin_runtime_instance_request(2345).await.unwrap();
    lifecycle.evict_idle().await;
    assert!(intents.try_recv().is_err());
    drop(outer);
    lifecycle.evict_idle().await;
    assert!(
        intents.try_recv().is_err(),
        "selected port must remain leased after an unrelated model lease ends"
    );
    drop(selected);
    assert!(other.idle_since.lock().unwrap().elapsed() < Duration::from_secs(1));
    node.endpoint.close().await;
}

#[test]
fn managed_identity_falls_back_to_the_model_name_once_no_instance_owns_the_port() {
    let (lifecycle, _intents, _targets) = fixture(300);
    // A retired port is no longer owned, so the request's model name decides.
    assert_eq!(
        lifecycle.managed_model_for(4321, Some("test")).as_deref(),
        Some("test")
    );
    assert_eq!(lifecycle.managed_model_for(4321, Some("unknown")), None);
    assert_eq!(lifecycle.managed_model_for(4321, None), None);
    lifecycle.record_loaded("test", "runtime-1", "loaded-test", 1234);
    assert_eq!(
        lifecycle.managed_model_for(1234, None).as_deref(),
        Some("test")
    );
    assert_eq!(
        lifecycle
            .managed_model_for(4321, Some("loaded-test"))
            .as_deref(),
        Some("test")
    );
}

#[tokio::test]
async fn lease_port_leaves_ports_no_managed_instance_owns_unmanaged() {
    let (lifecycle, _intents, _targets) = fixture(300);
    lifecycle.record_loaded("test", "runtime-1", "loaded-test", 1234);
    assert!(matches!(
        lifecycle.lease_port(4321).await,
        PortLease::Unmanaged
    ));
    assert!(
        lifecycle.models["test"].gate.try_write().is_ok(),
        "an unmanaged port must not take the managed model's gate"
    );
    let PortLease::Leased(lease) = lifecycle.lease_port(1234).await else {
        panic!("the owned port should be leased");
    };
    assert!(lifecycle.models["test"].gate.try_write().is_err());
    drop(lease);
}

#[tokio::test]
async fn exclusive_gpu_handoff_reports_an_interrupted_unload_and_releases_the_gate() {
    let (lifecycle, mut intents, _targets) = fixture(300);
    lifecycle.record_loaded("test", "runtime-1", "loaded-test", 1234);
    let handoff = {
        let lifecycle = lifecycle.clone();
        tokio::spawn(async move { lifecycle.acquire_exclusive_gpu_handoff().await })
    };
    let ModelIntent::Unload {
        completion: Some(completion),
        ..
    } = tokio::time::timeout(Duration::from_secs(10), intents.recv())
        .await
        .expect("unload intent should arrive")
        .expect("intent channel open")
    else {
        panic!("expected unload intent");
    };
    // The control loop stops without answering the unload.
    drop(completion);
    let err = tokio::time::timeout(Duration::from_secs(10), handoff)
        .await
        .expect("handoff should finish once the completion is dropped")
        .unwrap()
        .expect_err("an interrupted unload must fail the handoff");
    assert_eq!(
        err.to_string(),
        "native model unload interrupted during plugin GPU handoff"
    );
    assert!(lifecycle.models["test"].instance.lock().unwrap().is_some());

    // A later handoff must not stay blocked behind the interrupted one.
    let retry = {
        let lifecycle = lifecycle.clone();
        tokio::spawn(async move { lifecycle.acquire_exclusive_gpu_handoff().await })
    };
    let ModelIntent::Unload {
        target,
        completion: Some(completion),
        ..
    } = tokio::time::timeout(Duration::from_secs(10), intents.recv())
        .await
        .expect("retried unload intent should arrive")
        .expect("intent channel open")
    else {
        panic!("expected unload intent");
    };
    assert_eq!(target, UnloadTarget::Instance("runtime-1".into()));
    completion
        .send(Ok(crate::api::RuntimeUnloadResponse {
            model: "loaded-test".into(),
            instance_id: "runtime-1".into(),
            unloaded: true,
        }))
        .unwrap();
    let _lease = tokio::time::timeout(Duration::from_secs(10), retry)
        .await
        .expect("retried handoff should finish")
        .unwrap()
        .expect("retried handoff should succeed");
}

#[tokio::test]
async fn acquire_inside_its_own_lease_never_waits_behind_a_queued_handoff() {
    let (lifecycle, mut intents, targets) = fixture(300);
    let active = {
        let lifecycle = lifecycle.clone();
        tokio::spawn(async move { lifecycle.acquire("test").await.unwrap() })
    };
    finish_load(&lifecycle, intents.recv().await.unwrap(), &targets);
    let outer_lease = active.await.unwrap();

    let mut checkpoints = lifecycle.install_gate_wait_checkpoint();
    let mut handoff_fut = std::pin::pin!(lifecycle.acquire_exclusive_gpu_handoff());
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    assert!(handoff_fut.as_mut().poll(&mut cx).is_pending());
    assert_eq!(
        checkpoints
            .try_recv()
            .expect("handoff should queue write gate"),
        "exclusive_gpu_handoff_write"
    );
    assert!(lifecycle.models["test"].gate.try_read().is_err());

    // While the model is ready the task's own lease is reused.
    let reused = with_request_lease(outer_lease.clone(), async {
        tokio::time::timeout(Duration::from_secs(10), lifecycle.acquire("test")).await
    })
    .await
    .expect("acquire inside its own lease must not wait behind the queued handoff")
    .expect("a ready model should reuse the lease");
    assert!(reused.is_some());
    drop(reused);

    // The instance retires while the lease is held (owner unload, crash): the
    // reacquire fails at once instead of reloading behind the handoff.
    lifecycle.record_unloaded("runtime-1");
    targets.send_modify(|targets| {
        targets.targets.remove("loaded-test");
    });
    let retired = with_request_lease(outer_lease.clone(), async {
        tokio::time::timeout(Duration::from_secs(10), lifecycle.acquire("test")).await
    })
    .await
    .expect("a retired model must fail fast inside its own lease");
    assert!(retired.is_err());
    assert!(
        intents.try_recv().is_err(),
        "no reload while the request holds its lease"
    );

    // Releasing the lease lets the handoff through; nothing is left to unload.
    drop(outer_lease);
    let _handoff = tokio::time::timeout(Duration::from_secs(10), handoff_fut)
        .await
        .expect("handoff should finish once the lease is released")
        .expect("handoff should succeed");
}

#[tokio::test]
async fn selected_instance_admission_on_an_unmanaged_port_takes_no_residency_lease() {
    let (lifecycle, _intents, _targets) = fixture(300);
    lifecycle.record_loaded("test", "runtime-1", "loaded-test", 1234);
    let node = Box::pin(crate::mesh::Node::new_for_tests(
        crate::mesh::NodeRole::Worker,
    ))
    .await
    .unwrap();
    assert!(node.native_model_lifecycle.set(lifecycle.clone()).is_ok());
    let record = Arc::new(tokio::sync::Mutex::new(
        crate::runtime::InstanceLifecycleRecord::new(
            crate::runtime::InstanceLifecycleState::Serving,
            10,
        ),
    ));
    node.register_runtime_instance_lifecycle(4321, record.clone());
    let guard = node
        .begin_runtime_instance_request(4321)
        .await
        .expect("an unmanaged serving port should accept work")
        .expect("a registered port should return a guard");
    assert_eq!(record.lock().await.in_flight_count(), 1);
    assert!(
        lifecycle.models["test"].gate.try_write().is_ok(),
        "an unmanaged port must not hold a managed model's gate"
    );
    drop(guard);
    assert_eq!(record.lock().await.in_flight_count(), 0);
    node.endpoint.close().await;
}

#[test]
fn assigned_peer_execution_never_uses_a_remote_ready_target() {
    let peer = iroh::SecretKey::from_bytes(&[7; 32]).public();
    let mut targets = ModelTargets::default();
    targets
        .targets
        .insert("test".into(), vec![InferenceTarget::Remote(peer)]);
    assert!(ready(&targets, "test", false));
    assert!(!ready(&targets, "test", true));
}
