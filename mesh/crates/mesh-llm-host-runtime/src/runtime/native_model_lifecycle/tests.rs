use super::*;
#[cfg(test)]
mod requests;

#[cfg(test)]
mod intent_tests {
    use super::*;

    #[test]
    fn idle_eviction_respects_owner_ensure_and_refuses_split_controllers() {
        use crate::runtime::model_reconciliation::ModelTargetReconciliationState;
        use crate::runtime::run_auto::{RuntimeUnloadCandidate, RuntimeUnloadOwner};
        let mut state = ModelTargetReconciliationState::default();
        let mut candidate = RuntimeUnloadCandidate {
            owner: RuntimeUnloadOwner::Runtime,
            instance_id: "runtime-1".into(),
            model_name: "loaded-test".into(),
            profile: String::new(),
        };
        assert!(idle_unload_allowed(&state, Some(&candidate), Some("test")));
        state.add_desired("test", "", IntentSource::OwnerEnsure);
        assert!(!idle_unload_allowed(&state, Some(&candidate), Some("test")));
        candidate.owner = RuntimeUnloadOwner::Managed;
        assert!(!idle_unload_allowed(
            &ModelTargetReconciliationState::default(),
            Some(&candidate),
            Some("test")
        ));
        assert!(!idle_unload_allowed(&state, None, None));
    }
}

#[cfg(test)]
mod configuration_tests {
    use super::*;
    use crate::runtime::{RuntimeOptions, startup_models::build_startup_model_specs};

    /// Each uniquely named entry has either a startup spec or a residency.
    #[test]
    fn unique_on_demand_entries_are_managed_and_startup_entries_remain_eager() {
        let config: crate::plugin::MeshConfig = toml::from_str(
            r#"
[[models]]
model = "cold-model"
[models.lifecycle]
load = "on_demand"
[[models]]
model = "warm-model"
"#,
        )
        .unwrap();
        let residencies = configured_residencies(&config);
        let specs = build_startup_model_specs(&RuntimeOptions::default(), &config).unwrap();
        assert_eq!(
            residencies.keys().map(String::as_str).collect::<Vec<_>>(),
            ["cold-model"]
        );
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].config_model_id.as_deref(), Some("warm-model"));
    }

    /// Programmatic duplicate profiles must not fall between both launch paths.
    #[test]
    fn duplicate_model_profiles_keep_their_startup_specs() {
        let config: crate::plugin::MeshConfig = toml::from_str(
            r#"
[[models]]
model = "shared-model"
ctx_size = 1024
[models.lifecycle]
load = "on_demand"
[[models]]
model = "shared-model"
ctx_size = 2048
[models.lifecycle]
load = "on_demand"
"#,
        )
        .unwrap();
        assert!(configured_residencies(&config).is_empty());
        let specs = build_startup_model_specs(&RuntimeOptions::default(), &config).unwrap();
        assert_eq!(specs.len(), 2);
        assert_eq!(
            specs.iter().map(|spec| spec.ctx_size).collect::<Vec<_>>(),
            [Some(1024), Some(2048)]
        );
        assert_ne!(specs[0].profile, specs[1].profile);
    }
}

#[test]
fn provider_models_never_enter_native_residency_or_ambiguate_native_names() {
    let mut config: crate::plugin::MeshConfig = toml::from_str(
        r#"
[[plugin]]
name = "onnx"
command = "/opt/mesh-onnx"
[[models]]
model = "same-name"
[models.lifecycle]
load = "on_demand"
prefetch = false
[[models]]
model = "same-name"
provider = "onnx"
[models.lifecycle]
load = "on_demand"
prefetch = false
"#,
    )
    .unwrap();
    assert!(is_on_demand_managed(&config, &config.models[0]));
    assert!(!is_on_demand_managed(&config, &config.models[1]));
    // Programmatic configurations can bypass operator duplicate-name validation.
    config.models[0].lifecycle.prefetch = true;
    let managed = configured_residencies(&config);
    assert_eq!(managed.len(), 1);
    config.models.retain(|model| model.provider.is_some());
    assert!(configured_residencies(&config).is_empty());
    assert!(sources::validate_managed_sources(&config).is_ok());
    // A provider sharing a name must not inherit the native prefetch plan.
    assert!(sources::spawn_prefetch(&config, &managed).is_ok());
}
