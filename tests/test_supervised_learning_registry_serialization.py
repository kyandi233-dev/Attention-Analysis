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
        "raw_source": "synthetic behavior",
        "required_devices": ["behavior"],
        "behavior_reference_eligible": "true",
        "standalone_eligible": "false",
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
    assert feature.full_model_eligible is True
    assert feature.full_leave_one_out_eligible is False


def test_invalid_registry_boolean_text_fails_closed() -> None:
    with pytest.raises(FeatureRegistryContractError, match="standalone_eligible must be a boolean"):
        load_registered_features({"features": [_entry(standalone_eligible="sometimes")]})
