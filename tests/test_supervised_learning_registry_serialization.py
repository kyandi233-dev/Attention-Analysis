import pytest

from attention_pipeline.supervised_learning.feature_registry import (
    FeatureRegistryContractError,
    load_registered_features,
)


def _entry(**overrides):
    base = {
        "feature_id": "behavior_signal",
        "scientific_feature_id": "behavior_signal",
        "columns": ["behavior_signal"],
        "role": "behavior",
        "modality": "behavior",
        "raw_source": "synthetic behavior",
        "source_namespace": "behavior",
        "required_devices": [],
        "behavior_reference_eligible": "true",
        "standalone_eligible": "false",
        "modality_model_eligible": "false",
        "full_model_eligible": "true",
        "full_leave_one_out_eligible": "false",
        "allowed_device_packages": [],
    }
    base.update(overrides)
    return base


def test_serialized_false_registry_flags_remain_false() -> None:
    features = load_registered_features({"features": [_entry()]})
    feature = features[0]

    assert feature.behavior_reference_eligible is True
    assert feature.standalone_eligible is False
    assert feature.modality_model_eligible is False
    assert feature.full_model_eligible is True
    assert feature.full_leave_one_out_eligible is False
    assert feature.modality == "behavior"
    assert feature.source_namespace == "behavior"
    assert feature.required_devices == ()


def test_invalid_registry_boolean_text_fails_closed() -> None:
    with pytest.raises(FeatureRegistryContractError, match="standalone_eligible must be a boolean"):
        load_registered_features({"features": [_entry(standalone_eligible="sometimes")]})


def test_registry_requires_explicit_scientific_modality() -> None:
    entry = _entry()
    entry.pop("modality")
    with pytest.raises(FeatureRegistryContractError, match="missing required fields"):
        load_registered_features({"features": [entry]})


def test_registry_requires_explicit_source_namespace() -> None:
    entry = _entry()
    entry.pop("source_namespace")
    with pytest.raises(FeatureRegistryContractError, match="missing required fields"):
        load_registered_features({"features": [entry]})
