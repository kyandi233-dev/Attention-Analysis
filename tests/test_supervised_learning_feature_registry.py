from __future__ import annotations

import pytest

from attention_pipeline.supervised_learning.entrypoint import _resolve_model_plan
from attention_pipeline.supervised_learning.feature_registry import (
    DEVICE_PACKAGES,
    FeatureRegistryContractError,
    RegisteredFeature,
    build_feature_comparison_plan,
    validate_registered_features,
)


def _registry() -> list[RegisteredFeature]:
    return [
        RegisteredFeature(
            feature_id="rt_variability",
            scientific_feature_id="rt_variability",
            columns=("go_correct_rt_cv",),
            role="behavior",
            modality="behavior",
            raw_source="SART behavior",
            required_devices=(),
            behavior_reference_eligible=True,
            behavior_increment_eligible=False,
        ),
        RegisteredFeature(
            feature_id="omission",
            scientific_feature_id="go_omission",
            columns=("raw_go_omission_rate",),
            role="behavior",
            modality="behavior",
            raw_source="SART behavior",
            required_devices=(),
            behavior_reference_eligible=True,
            behavior_increment_eligible=False,
        ),
        RegisteredFeature(
            feature_id="pupil_variability_nir_only",
            scientific_feature_id="pupil_variability",
            columns=("pupil_sd_nir_only",),
            role="sensor",
            modality="ocular",
            feature_type="pupil",
            raw_source="NIR pupil",
            required_devices=("nir",),
            preprocessing_dependencies=("NIR-only blink/artifact QC",),
            standalone_eligible=False,
            behavior_increment_eligible=False,
            modality_model_eligible=False,
            full_model_eligible=False,
            full_leave_one_out_eligible=False,
            allowed_device_packages=("M1", "M4"),
        ),
        RegisteredFeature(
            feature_id="pupil_variability_rgb_assisted",
            scientific_feature_id="pupil_variability",
            columns=("pupil_sd_rgb_assisted",),
            role="sensor",
            modality="ocular",
            feature_type="pupil",
            raw_source="NIR pupil",
            required_devices=("nir", "rgb"),
            preprocessing_dependencies=("RGB blink mask",),
            standalone_eligible=True,
            behavior_increment_eligible=True,
            modality_model_eligible=True,
            full_model_eligible=True,
            full_leave_one_out_eligible=True,
            allowed_device_packages=("M5", "M7"),
        ),
        RegisteredFeature(
            feature_id="blink_rate",
            scientific_feature_id="blink_rate",
            columns=("blink_event_rate_per_min",),
            role="sensor",
            modality="ocular",
            feature_type="blink",
            raw_source="RGB eye landmarks",
            required_devices=("rgb",),
            behavior_increment_eligible=True,
            modality_model_eligible=True,
            allowed_device_packages=("M3", "M5", "M6", "M7"),
        ),
        RegisteredFeature(
            feature_id="body_motion",
            scientific_feature_id="body_motion_energy",
            columns=("body_motion_energy_median",),
            role="sensor",
            modality="movement",
            feature_type="body_motion",
            raw_source="RGB body landmarks",
            required_devices=("rgb",),
            behavior_increment_eligible=True,
            modality_model_eligible=True,
            allowed_device_packages=("M3", "M5", "M6", "M7"),
        ),
        RegisteredFeature(
            feature_id="breathing_rate",
            scientific_feature_id="breathing_rate",
            columns=("mmwave_breath_rate",),
            role="sensor",
            modality="cardiopulmonary",
            feature_type="breathing_rate",
            raw_source="mmWave",
            required_devices=("mmwave",),
            behavior_increment_eligible=True,
            modality_model_eligible=True,
            allowed_device_packages=("M2", "M4", "M6", "M7"),
        ),
    ]


def test_plan_separates_feature_modality_and_device_models() -> None:
    plan = build_feature_comparison_plan(_registry())
    models = plan.model_map()

    assert "behavior_reference" in models
    assert "standalone::pupil_variability_rgb_assisted" in models
    assert "behavior_plus::pupil_variability_rgb_assisted" in models
    assert "modality::ocular" in models
    assert "modality::movement" in models
    assert "modality::cardiopulmonary" in models
    assert "behavior_plus_modality::ocular" in models
    assert "full" in models
    assert "full_minus_modality::ocular" in models
    assert set(plan.device_package_model_ids) == set(DEVICE_PACKAGES)
    assert plan.unavailable_device_packages == {}

    assert set(plan.modality_model_ids) == {
        "behavior",
        "ocular",
        "movement",
        "cardiopulmonary",
    }
    assert ("behavior_reference", "behavior_plus_modality::ocular", "ocular") in plan.modality_increment_pairs


def test_cross_device_ocular_model_keeps_science_modality_distinct_from_devices() -> None:
    plan = build_feature_comparison_plan(_registry())
    models = plan.model_map()
    ocular = models["modality::ocular"]

    assert set(ocular.feature_ids) == {
        "pupil_variability_rgb_assisted",
        "blink_rate",
    }
    assert ocular.modalities == ("ocular",)
    assert set(ocular.required_devices) == {"nir", "rgb"}
    assert not ocular.includes_behavior_reference

    families = plan.to_runner_feature_schemes()
    scheme = families["modality::ocular"][0]
    assert scheme.modalities == ("ocular",)
    assert set(scheme.required_devices) == {"nir", "rgb"}
    assert scheme.modality_blocks == ()


def test_device_packages_are_hardware_only_but_explicitly_include_behavior_reference() -> None:
    plan = build_feature_comparison_plan(_registry())
    models = plan.model_map()

    assert DEVICE_PACKAGES["M0"] == frozenset()
    assert models["M0"].required_devices == ()
    assert models["M0"].modalities == ("behavior",)
    assert models["M0"].includes_behavior_reference
    assert set(models["M0"].feature_ids) == {"rt_variability", "omission"}

    assert "pupil_variability_nir_only" in models["M1"].feature_ids
    assert "pupil_variability_rgb_assisted" not in models["M1"].feature_ids
    assert models["M1"].required_devices == ("nir",)

    assert "pupil_variability_rgb_assisted" in models["M5"].feature_ids
    assert set(models["M5"].required_devices) == {"nir", "rgb"}
    assert "behavior" not in models["M5"].required_devices
    assert models["M5"].includes_behavior_reference


def test_rgb_device_package_can_contain_ocular_and_movement_without_conflating_them() -> None:
    plan = build_feature_comparison_plan(_registry())
    rgb = plan.model_map()["M3"]

    assert set(rgb.required_devices) == {"rgb"}
    assert set(rgb.modalities) == {"behavior", "ocular", "movement"}
    assert "blink_rate" in rgb.feature_ids
    assert "body_motion" in rgb.feature_ids
    assert "pupil_variability_rgb_assisted" not in rgb.feature_ids


def test_leave_one_modality_out_removes_predictors_not_other_features_production_dependencies() -> None:
    plan = build_feature_comparison_plan(_registry())
    models = plan.model_map()
    reduced = models["full_minus_modality::movement"]

    assert "body_motion" not in reduced.feature_ids
    assert "pupil_variability_rgb_assisted" in reduced.feature_ids
    assert "rgb" in reduced.required_devices
    assert "ocular" in reduced.modalities


def test_entrypoint_prefers_nonempty_frozen_registry_over_manual_feature_families() -> None:
    config = {
        "feature_registry": {
            "features": [feature.audit_dict() for feature in _registry()],
        },
        "feature_schemes": {
            "model_families": {
                "manual_should_not_win": {
                    "candidates": [
                        {
                            "feature_set_id": "manual",
                            "columns": ["go_correct_rt_cv"],
                            "modalities": ["behavior"],
                            "required_devices": [],
                        }
                    ]
                }
            }
        },
    }
    families, plan = _resolve_model_plan(config)

    assert plan is not None
    assert "manual_should_not_win" not in families
    assert "behavior_reference" in families
    assert set(plan.device_package_model_ids) == set(DEVICE_PACKAGES)


def test_registry_rejects_behavior_as_device() -> None:
    features = _registry()
    behavior = features[0]
    features[0] = RegisteredFeature(
        **{**behavior.__dict__, "required_devices": ("behavior",)}
    )
    with pytest.raises(FeatureRegistryContractError, match="unknown required_devices"):
        validate_registered_features(features)


def test_registry_rejects_impossible_cross_device_package_claim() -> None:
    features = _registry()
    pupil = features[3]
    features[3] = RegisteredFeature(
        **{**pupil.__dict__, "allowed_device_packages": ("M1",)}
    )
    with pytest.raises(FeatureRegistryContractError, match="lacks required devices"):
        validate_registered_features(features)


def test_plan_marks_sensor_packages_unavailable_when_a_device_has_no_frozen_feature() -> None:
    features = [feature for feature in _registry() if feature.feature_id != "breathing_rate"]
    plan = build_feature_comparison_plan(features)

    assert {"M0", "M1", "M3", "M5"}.issubset(plan.device_package_model_ids)
    for package_id in ("M2", "M4", "M6", "M7"):
        assert package_id not in plan.device_package_model_ids
        assert "mmwave" in plan.unavailable_device_packages[package_id]

    assert "cardiopulmonary" not in plan.modality_model_ids
    assert "cardiopulmonary" in plan.unavailable_modalities


def test_modality_model_eligibility_is_not_automatic_from_membership() -> None:
    features = _registry()
    nir_only = features[2]
    assert nir_only.modality == "ocular"
    assert not nir_only.modality_model_eligible

    plan = build_feature_comparison_plan(features)
    assert "pupil_variability_nir_only" not in plan.model_map()["modality::ocular"].feature_ids


def test_registry_rejects_two_modality_model_representations_of_same_scientific_feature() -> None:
    features = _registry()
    nir_only = features[2]
    features[2] = RegisteredFeature(
        **{**nir_only.__dict__, "modality_model_eligible": True}
    )
    with pytest.raises(FeatureRegistryContractError, match="multiple ocular modality-model representations"):
        validate_registered_features(features)


def test_registry_rejects_two_full_representations_of_same_scientific_feature() -> None:
    features = _registry()
    nir_only = features[2]
    features[2] = RegisteredFeature(
        **{
            **nir_only.__dict__,
            "full_model_eligible": True,
            "full_leave_one_out_eligible": True,
        }
    )
    with pytest.raises(FeatureRegistryContractError, match="multiple full-model representations"):
        validate_registered_features(features)
