import pytest

from attention_pipeline.supervised_learning.time_legality import (
    BLOCKED_FUTURE_INFORMATION,
    PENDING_UPSTREAM_FREEZE,
    VERIFIED_PRE_PROBE_ONLY,
    TimeLegalityContractError,
    time_legality_audit,
    validate_feature_registry_time_legality,
)


def _feature(**overrides):
    base = {
        "feature_id": "behavior_rt_level",
        "scientific_feature_id": "rt_level",
        "columns": ["go_correct_rt_median_ms"],
        "role": "behavior",
        "modality": "behavior",
        "raw_source": "Behavior/formal_v3/probe_primary_30s.csv",
        "source_namespace": "behavior",
        "required_devices": [],
        "preprocessing_dependencies": [],
        "temporal_anchor": "probe_time_ms",
        "temporal_scope": "pre_probe_only",
        "time_legality_status": VERIFIED_PRE_PROBE_ONLY,
        "time_legality_evidence": "30 s pre-probe Behavior window; probe anchor trial excluded",
        "standalone_eligible": True,
        "behavior_increment_eligible": False,
        "behavior_reference_eligible": True,
        "modality_model_eligible": False,
        "full_model_eligible": True,
        "full_leave_one_out_eligible": True,
        "allowed_device_packages": [],
    }
    base.update(overrides)
    return base


def test_verified_pre_probe_behavior_feature_can_request_prediction():
    records = validate_feature_registry_time_legality({"features": [_feature()]})
    assert len(records) == 1
    assert records[0].requests_prediction is True
    assert records[0].time_legality_status == VERIFIED_PRE_PROBE_ONLY


def test_pending_candidate_is_allowed_only_when_all_prediction_roles_are_disabled():
    pending = _feature(
        temporal_scope="pending_upstream_freeze",
        time_legality_status=PENDING_UPSTREAM_FREEZE,
        time_legality_evidence="",
        standalone_eligible=False,
        behavior_reference_eligible=False,
        full_model_eligible=False,
        full_leave_one_out_eligible=False,
    )
    records = validate_feature_registry_time_legality({"features": [pending]})
    assert records[0].requests_prediction is False


def test_pending_feature_fails_closed_if_any_prediction_role_is_enabled():
    pending = _feature(
        temporal_scope="pending_upstream_freeze",
        time_legality_status=PENDING_UPSTREAM_FREEZE,
        time_legality_evidence="",
        standalone_eligible=False,
        behavior_reference_eligible=False,
        full_model_eligible=True,
        full_leave_one_out_eligible=False,
    )
    with pytest.raises(TimeLegalityContractError, match="fail-closed"):
        validate_feature_registry_time_legality({"features": [pending]})


def test_verified_feature_requires_nonblank_temporal_evidence():
    with pytest.raises(TimeLegalityContractError, match="requires nonblank time_legality_evidence"):
        validate_feature_registry_time_legality(
            {"features": [_feature(time_legality_evidence="")]}
        )


def test_verified_feature_rejects_non_preprobe_temporal_scope():
    with pytest.raises(TimeLegalityContractError, match="temporal_scope='pre_probe_only'"):
        validate_feature_registry_time_legality(
            {"features": [_feature(temporal_scope="whole_session")]}
        )


def test_blocked_future_information_cannot_enter_device_package():
    blocked = _feature(
        role="sensor",
        modality="ocular",
        source_namespace="nir",
        required_devices=["nir", "rgb"],
        temporal_scope="blocked_future_information",
        time_legality_status=BLOCKED_FUTURE_INFORMATION,
        time_legality_evidence="producer currently uses a whole-session reference",
        standalone_eligible=False,
        behavior_reference_eligible=False,
        full_model_eligible=False,
        full_leave_one_out_eligible=False,
        allowed_device_packages=["M5"],
    )
    with pytest.raises(TimeLegalityContractError, match="fail-closed"):
        validate_feature_registry_time_legality({"features": [blocked]})


def test_cross_device_pupil_can_pass_when_dependency_is_explicit_and_preprobe_only():
    pupil = _feature(
        feature_id="pupil_level_rgb_assisted",
        scientific_feature_id="pupil_level",
        columns=["pupil_level_median"],
        role="sensor",
        modality="ocular",
        source_namespace="nir",
        required_devices=["nir", "rgb"],
        preprocessing_dependencies=["RGB blink mask restricted to the same 30 s pre-probe window"],
        time_legality_evidence=(
            "NIR samples and RGB blink mask both restricted to [probe_time_ms-30000, probe_time_ms); "
            "no post-probe frames or participant whole-session statistics"
        ),
        behavior_reference_eligible=False,
        behavior_increment_eligible=True,
        modality_model_eligible=True,
        allowed_device_packages=["M5", "M7"],
    )
    records = validate_feature_registry_time_legality({"features": [pupil]})
    assert records[0].requests_prediction is True
    assert records[0].preprocessing_dependencies


def test_serialized_false_flags_do_not_become_truthy():
    pending = _feature(
        temporal_scope="pending_upstream_freeze",
        time_legality_status=PENDING_UPSTREAM_FREEZE,
        time_legality_evidence="",
        standalone_eligible="false",
        behavior_increment_eligible="false",
        behavior_reference_eligible="false",
        modality_model_eligible="false",
        full_model_eligible="false",
        full_leave_one_out_eligible="false",
    )
    records = validate_feature_registry_time_legality({"features": [pending]})
    assert records[0].requests_prediction is False


def test_time_legality_audit_archives_machine_readable_contract():
    audit = time_legality_audit({"features": [_feature()]})
    assert audit["schema_version"] == "1.16.10-time-legality-v1"
    assert audit["n_verified_pre_probe_only"] == 1
    assert audit["contract"]["prediction_requires_verified_pre_probe_only"] is True
    assert audit["features"][0]["feature_id"] == "behavior_rt_level"
