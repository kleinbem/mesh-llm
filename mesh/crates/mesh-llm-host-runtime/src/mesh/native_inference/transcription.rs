use super::*;
use crate::sdk::native::transcription::{RealtimeTranscriptionConfig, TranscriptionSession};

struct SessionAdmission {
    _slot: tokio::sync::OwnedSemaphorePermit,
    _instance: crate::runtime::InstanceRequestGuard,
    _inflight: InflightRequestGuard,
    _cancel: CancelNativeOnDrop,
}

impl Node {
    pub(crate) async fn native_live_transcription(
        &self,
        mut config: RealtimeTranscriptionConfig,
    ) -> Result<TranscriptionSession> {
        validate(&config)?;
        if let AdmissionResult::Paused { reason, .. } = self
            .activity_policy_guard
            .check_admission(IngressType::LocalOpenAi)
        {
            anyhow::bail!("inference paused: {reason}");
        }
        if let Some(manager) = self.plugin_manager().await {
            anyhow::ensure!(
                !manager.has_exchange_hooks().await,
                "live transcription is unavailable on a node with exchange admission plugins"
            );
        }
        let models = self.native_models().await?;
        let local: Vec<_> = models
            .into_iter()
            .filter(|model| {
                model.target.provider_node_id == self.endpoint.id().to_string()
                    && model
                        .descriptor
                        .metadata
                        .as_ref()
                        .and_then(|meta| meta.workload_class.as_ref())
                        == Some(&ModelWorkloadClass::SpeechRecognition)
            })
            .collect();
        let callable: Vec<_> = local
            .iter()
            .map(|model| model.target.serving_id.clone())
            .collect();
        config.model = if config.model.is_empty() {
            callable
                .first()
                .context("no local model supports live transcription")?
                .clone()
        } else {
            let requested = self
                .native_model_lifecycle
                .get()
                .map(|service| service.execution_model(&config.model))
                .unwrap_or(config.model);
            let descriptors = self.served_model_descriptors().await;
            let resolved = crate::network::openai::request_parse::internal_model_for_public_id(
                &requested,
                &callable,
                &descriptors,
            )
            .unwrap_or(requested);
            resolved.split('#').next().unwrap_or_default().to_string()
        };
        anyhow::ensure!(
            callable.contains(&config.model),
            "no local model supports live transcription for {:?}; remote models require an explicit network session",
            config.model
        );
        let slot = self
            .native_inference
            .stream_slots
            .clone()
            .try_acquire_owned()
            .context("native live transcription capacity exhausted")?;
        let candidates: Vec<(String, u16, std::sync::Weak<dyn InferenceBackend>)> = self
            .native_inference
            .backends
            .read()
            .await
            .get(&config.model)
            .map(|instances| {
                instances
                    .iter()
                    .map(|(id, entry)| (id.clone(), entry.port, Arc::downgrade(&entry.backend)))
                    .collect()
            })
            .context("native speech model is no longer loaded")?;
        for (id, port, weak_backend) in candidates {
            let Ok(Some(instance)) = self.begin_runtime_instance_request(port).await else {
                continue;
            };
            // Acquire the backend only after admission so waiters parked behind
            // an unload/handoff gate never keep a retired GPU backend alive.
            let Some(backend) = self
                .native_inference
                .backends
                .read()
                .await
                .get(&config.model)
                .and_then(|instances| instances.get(&id))
                .filter(|active| {
                    active.port == port && weak_backend.ptr_eq(&Arc::downgrade(&active.backend))
                })
                .map(|active| active.backend.clone())
            else {
                continue;
            };
            let context = InferenceRequestContext::new();
            let admission = SessionAdmission {
                _slot: slot,
                _instance: instance,
                _inflight: self.begin_inflight_request(),
                _cancel: CancelNativeOnDrop(context.cancellation_token()),
            };
            let session = backend
                .realtime_transcription(config, context.clone())
                .await
                .map_err(|error| anyhow::anyhow!("start native live transcription: {error}"))?;
            return Ok(TranscriptionSession::new(session, context, admission));
        }
        anyhow::bail!("native speech model is draining or unavailable")
    }
}

fn validate(config: &RealtimeTranscriptionConfig) -> Result<()> {
    anyhow::ensure!(
        (8_000..=96_000).contains(&config.input_sample_rate),
        "input sample rate must be 8000 to 96000 Hz"
    );
    if let Some(step) = config.step_ms {
        anyhow::ensure!((200..=5000).contains(&step), "step_ms must be 200 to 5000");
    }
    if let Some(vad) = config.turn_detection {
        anyhow::ensure!(
            (0.0..=1.0).contains(&vad.threshold),
            "turn detection threshold must be 0 to 1"
        );
        anyhow::ensure!(
            vad.prefix_padding_ms <= 5000,
            "prefix padding must be at most 5000 ms"
        );
        anyhow::ensure!(
            (100..=30_000).contains(&vad.silence_duration_ms),
            "silence duration must be 100 to 30000 ms"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{InstanceLifecycleRecord, InstanceLifecycleState};
    use crate::sdk::native::transcription::{RealtimeInput, RealtimeOutput};
    use async_trait::async_trait;
    use skippy_inference_api::generation::{
        GenerationEventStream, GenerationRequest, GenerationResponse,
    };
    use skippy_inference_api::{InferenceError, InferenceResult, ModelObject};
    use std::sync::Mutex;

    #[test]
    fn native_options_reject_the_same_out_of_range_audio_and_cadence_settings() {
        let base = RealtimeTranscriptionConfig::new("speech");
        for config in [
            RealtimeTranscriptionConfig {
                input_sample_rate: 0,
                ..base.clone()
            },
            RealtimeTranscriptionConfig {
                input_sample_rate: 96_001,
                ..base.clone()
            },
            RealtimeTranscriptionConfig {
                step_ms: Some(199),
                ..base.clone()
            },
            RealtimeTranscriptionConfig {
                turn_detection: Some(crate::sdk::native::transcription::RealtimeTurnDetection {
                    threshold: f32::NAN,
                    ..Default::default()
                }),
                ..base.clone()
            },
        ] {
            assert!(validate(&config).is_err());
        }
        assert!(
            validate(&RealtimeTranscriptionConfig {
                input_sample_rate: 48_000,
                step_ms: Some(400),
                language: Some("de".into()),
                prompt: Some("jargon".into()),
                turn_detection: None,
                tentative: true,
                ..base
            })
            .is_ok()
        );
    }
    #[derive(Default)]
    struct SpeechBackend {
        options: Mutex<Option<RealtimeTranscriptionConfig>>,
    }

    #[async_trait]
    impl InferenceBackend for SpeechBackend {
        async fn models(&self) -> InferenceResult<Vec<ModelObject>> {
            Ok(vec![ModelObject::new("speech")])
        }
        async fn chat_completion(
            &self,
            _: GenerationRequest,
        ) -> InferenceResult<GenerationResponse> {
            Err(InferenceError::unsupported("speech only"))
        }
        async fn chat_completion_stream(
            &self,
            _: GenerationRequest,
            _: InferenceRequestContext,
        ) -> InferenceResult<GenerationEventStream> {
            Err(InferenceError::unsupported("speech only"))
        }
        async fn realtime_transcription(
            &self,
            config: RealtimeTranscriptionConfig,
            context: InferenceRequestContext,
        ) -> InferenceResult<skippy_inference_api::realtime::RealtimeTranscriptionSession> {
            *self.options.lock().unwrap() = Some(config);
            let (session, mut input, output) =
                skippy_inference_api::realtime::RealtimeTranscriptionSession::channels();
            tokio::spawn(async move {
                assert_eq!(input.recv().await, Some(RealtimeInput::Audio(vec![3, -4])));
                output
                    .send(RealtimeOutput::Delta {
                        turn: 0,
                        text: "hello".into(),
                    })
                    .await
                    .unwrap();
                assert_eq!(input.recv().await, Some(RealtimeInput::Commit));
                output
                    .send(RealtimeOutput::Completed {
                        turn: 0,
                        transcript: "hello world".into(),
                    })
                    .await
                    .unwrap();
                context.cancellation_token().cancelled().await;
            });
            Ok(session)
        }
    }

    #[tokio::test]
    async fn a_local_speech_backend_executes_without_any_http_server() {
        let node = Node::new_for_tests(NodeRole::Host { http_port: 9 })
            .await
            .unwrap();
        node.set_hosted_models(vec!["speech".into()]).await;
        node.set_serving_models(vec!["speech".into()]).await;
        node.upsert_served_model_descriptor(ServedModelDescriptor {
            identity: ServedModelIdentity {
                model_name: "speech".into(),
                is_primary: true,
                ..Default::default()
            },
            capabilities_known: true,
            metadata: Some(ServedModelMetadata {
                workload_class: Some(ModelWorkloadClass::SpeechRecognition),
                ..Default::default()
            }),
            ..Default::default()
        })
        .await;
        let lifecycle = Arc::new(tokio::sync::Mutex::new(InstanceLifecycleRecord::new(
            InstanceLifecycleState::Serving,
            1,
        )));
        node.register_runtime_instance_lifecycle(9, lifecycle);
        let backend = Arc::new(SpeechBackend::default());
        node.register_native_backend("speech", "speech-instance", 9, Some(backend.clone()), None)
            .await;
        let options = RealtimeTranscriptionConfig {
            model: "speech#fast".into(),
            input_sample_rate: 48_000,
            language: Some("de".into()),
            prompt: Some("jargon".into()),
            turn_detection: None,
            step_ms: Some(400),
            tentative: true,
        };
        let mut live = node
            .native_live_transcription(options.clone())
            .await
            .unwrap();
        let mut expected = options;
        expected.model = "speech".into();
        assert_eq!(*backend.options.lock().unwrap(), Some(expected));
        live.audio
            .send(RealtimeInput::Audio(vec![3, -4]))
            .await
            .unwrap();
        assert_eq!(
            live.events.next_output().await,
            Some(RealtimeOutput::Delta {
                turn: 0,
                text: "hello".into()
            })
        );
        live.audio.send(RealtimeInput::Commit).await.unwrap();
        assert_eq!(
            live.events.next_output().await,
            Some(RealtimeOutput::Completed {
                turn: 0,
                transcript: "hello world".into()
            })
        );
        drop(live);
        let error = node
            .native_live_transcription(RealtimeTranscriptionConfig::new("remote-only"))
            .await
            .err()
            .unwrap();
        assert!(error.to_string().contains("no local model"));
        node.endpoint.close().await;
    }

    struct DropTrackingSpeechBackend {
        dropped: Arc<std::sync::atomic::AtomicBool>,
    }

    impl Drop for DropTrackingSpeechBackend {
        fn drop(&mut self) {
            self.dropped
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    #[async_trait]
    impl InferenceBackend for DropTrackingSpeechBackend {
        async fn models(&self) -> InferenceResult<Vec<ModelObject>> {
            Ok(vec![ModelObject::new("speech")])
        }
        async fn chat_completion(
            &self,
            _: GenerationRequest,
        ) -> InferenceResult<GenerationResponse> {
            Err(InferenceError::unsupported("speech only"))
        }
        async fn chat_completion_stream(
            &self,
            _: GenerationRequest,
            _: InferenceRequestContext,
        ) -> InferenceResult<GenerationEventStream> {
            Err(InferenceError::unsupported("speech only"))
        }
        async fn realtime_transcription(
            &self,
            _: RealtimeTranscriptionConfig,
            _: InferenceRequestContext,
        ) -> InferenceResult<skippy_inference_api::realtime::RealtimeTranscriptionSession> {
            panic!("retired speech backend must not start transcription after unload");
        }
    }

    #[tokio::test]
    async fn waiting_transcription_does_not_keep_unloaded_speech_backend_alive() {
        let node = Node::new_for_tests(NodeRole::Host { http_port: 9 })
            .await
            .unwrap();
        node.set_hosted_models(vec!["speech".into()]).await;
        node.set_serving_models(vec!["speech".into()]).await;
        node.upsert_served_model_descriptor(ServedModelDescriptor {
            identity: ServedModelIdentity {
                model_name: "speech".into(),
                is_primary: true,
                ..Default::default()
            },
            capabilities_known: true,
            metadata: Some(ServedModelMetadata {
                workload_class: Some(ModelWorkloadClass::SpeechRecognition),
                ..Default::default()
            }),
            ..Default::default()
        })
        .await;

        let config: crate::plugin::MeshConfig = toml::from_str(
            r#"
[[models]]
model = "speech"
[models.lifecycle]
load = "on_demand"
idle_timeout_secs = 300
prefetch = false
"#,
        )
        .unwrap();
        let (service, mut intents, targets_tx) =
            crate::runtime::native_model_lifecycle::NativeModelLifecycle::test_service(&config);
        targets_tx.send_modify(|t| {
            t.targets.insert(
                "speech".into(),
                vec![crate::inference::election::InferenceTarget::Local(9)],
            );
        });
        service.record_loaded("speech", "speech-instance", "speech", 9);
        assert!(node.native_model_lifecycle.set(service.clone()).is_ok());

        let instance_lifecycle = Arc::new(tokio::sync::Mutex::new(InstanceLifecycleRecord::new(
            InstanceLifecycleState::Serving,
            4,
        )));
        node.register_runtime_instance_lifecycle(9, instance_lifecycle.clone());

        let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let backend: Arc<dyn InferenceBackend> = Arc::new(DropTrackingSpeechBackend {
            dropped: dropped.clone(),
        });
        node.register_native_backend("speech", "speech-instance", 9, Some(backend), None)
            .await;

        // Hold an active lease so the exclusive_gpu handoff queues its writer first.
        let active_lease = service.acquire("speech").await.unwrap().unwrap();
        let mut checkpoints = service.install_gate_wait_checkpoint();

        let handoff = {
            let service = service.clone();
            tokio::spawn(async move { service.acquire_exclusive_gpu_handoff().await.unwrap() })
        };
        assert_eq!(
            tokio::time::timeout(std::time::Duration::from_secs(2), checkpoints.recv())
                .await
                .unwrap()
                .unwrap(),
            "exclusive_gpu_handoff_write"
        );

        // Start a transcription request while handoff is queued: it must park on
        // admission without holding a strong clone of the backend Arc.
        let waiter = {
            let node = node.clone();
            tokio::spawn(async move {
                node.native_live_transcription(RealtimeTranscriptionConfig::new("speech"))
                    .await
            })
        };
        assert_eq!(
            tokio::time::timeout(std::time::Duration::from_secs(2), checkpoints.recv())
                .await
                .unwrap()
                .unwrap(),
            "lease_port_read"
        );

        drop(active_lease);
        let crate::runtime::ModelIntent::Unload {
            completion: Some(completion),
            ..
        } = tokio::time::timeout(std::time::Duration::from_secs(2), intents.recv())
            .await
            .unwrap()
            .unwrap()
        else {
            panic!("expected unload intent");
        };

        // Simulate teardown: unregister the backend and complete unload while the
        // waiter is still blocked behind the handoff write gate.
        {
            let mut record = instance_lifecycle.lock().await;
            let _ = record.mark_draining(std::time::Instant::now());
            let _ = record.transition_to_unloading();
            let _ = record.transition_to(InstanceLifecycleState::Stopped);
        }
        node.unregister_runtime_instance_lifecycle(9);
        node.unregister_native_backend("speech", "speech-instance")
            .await;
        service.record_unloaded("speech-instance");
        targets_tx.send_modify(|t| t.targets.clear());
        completion
            .send(Ok(crate::api::RuntimeUnloadResponse {
                model: "speech".into(),
                instance_id: "speech-instance".into(),
                unloaded: true,
            }))
            .unwrap();

        let handoff_lease = handoff.await.unwrap();
        assert!(
            dropped.load(std::sync::atomic::Ordering::SeqCst),
            "parked transcription waiter must not keep the unloaded speech backend alive during plugin handoff"
        );

        drop(handoff_lease);
        assert!(
            waiter.await.unwrap().is_err(),
            "waiter must reject the retired instance after handoff completes"
        );
        node.endpoint.close().await;
    }
}
