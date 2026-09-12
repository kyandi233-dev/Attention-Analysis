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
            raw_source="SART behavior",
            required_devices=("behavior",),
            behavior_reference_eligible=True,
            behavior_increment_eligible=False,
            allowed_device_packages=(),
        ),
        RegisteredFeature(
            feature_id="omission",
            scientific_feature_id="go_omission",
            columns=("raw_go_omission_rate",),
            role="behavior",
            raw_source="SART behavior",
            required_devices=("behavior",),
            behavior_reference_eligible=True,
            behavior_increment_eligible=False,
        ),
        RegisteredFeature(
            feature_id="pupil_variability_nir_only",
            scientific_feature_id="pupil_variability",
            columns=("pupil_sd_nir_only",),
            role="sensor",
            raw_source="NIR pupil",
            required_devices=("nir",),
            preprocessing_dependencies=("NIR-only blink/artifact QC",),
            standalone_eligible=False,
            behavior_increment_eligible=False,
            full_model_eligible=False,
            full_leave_one_out_eligible=False,
            allowed_device_packages=("M1", "M4"),
        ),
        RegisteredFeature(
            feature_id="pupil_variability_rgb_assisted",
            scientific_feature_id="pupil_variability",
            columns=("pupil_sd_rgb_assisted",),
            role="sensor",
            raw_source="NIR pupil",
            required_devices=("nir", "rgb"),
            preprocessing_dependencies=("RGB blink mask",),
            standalone_eligible=True,
            behavior_increment_eligible=True,
            full_model_eligible=True,
            full_leave_one_out_eligible=True,
            allowed_device_packages=("M5", "M7"),
        ),
        RegisteredFeature(
            feature_id="blink_rate",
            scientific_feature_id="blink_rate",
            columns=("blink_event_rate_per_min",),
            role="sensor",
            raw_source="RGB eye landmarks",
            required_devices=("rgb",),
            behavior_increment_eligible=True,
            allowed_device_packages=("M3", "M5", "M6", "M7"),
        ),
        RegisteredFeature(
            feature_id="breathing_rate",
            scientific_feature_id="breathing_rate",
            columns=("mmwave_breath_rate",),
            role="sensor",
            raw_source="mmWave",
            required_devices=("mmwave",),
            behavior_increment_eligible=True,
            allowed_device_packages=("M2", "M4", "M6", "M7"),
        ),
    ]


def test_plan_generates_standalone_behavior_increment_full_minus_and_m0_m7() -> None:
    plan = build_feature_comparison_plan(_registry())
    models = plan.model_map()

    assert "behavior_reference" in models
    assert "standalone::rt_variability" in models
    assert "standalone::omission" in models
    assert "standalone::pupil_variability_rgb_assisted" in models
    assert "behavior_plus::pupil_variability_rgb_assisted" in models
    assert "behavior_plus::blink_rate" in models
    assert "behavior_plus::breathing_rate" in models
    assert "full" in models
    assert "full_minus::rt_variability" in models
    assert "full_minus::pupil_variability_rgb_assisted" in models
    assert set(plan.device_package_model_ids) == set(DEVICE_PACKAGES)

    assert set(models["behavior_reference"].feature_ids) == {"rt_variability", "omission"}
    assert models["standalone::rt_variability"].feature_ids == ("rt_variability",)

    assert set(models["behavior_plus::blink_rate"].feature_ids) == {
        "rt_variability",
        "omission",
        "blink_rate",
    }

    assert "pupil_variability_rgb_assisted" in models["full"].feature_ids
    assert "pupil_variability_rgb_assisted" not in models["full_minus::pupil_variability_rgb_assisted"].feature_ids


def test_cross_device_pupil_provenance_controls_m0_m7_membership() -> None:
    plan = build_feature_comparison_plan(_registry())
    models = plan.model_map()

    assert "pupil_variability_nir_only" in models["M1"].feature_ids
    assert "pupil_variability_rgb_assisted" not in models["M1"].feature_ids
    assert "pupil_variability_nir_only" in models["M4"].feature_ids

    assert "pupil_variability_rgb_assisted" in models["M5"].feature_ids
    assert "pupil_variability_nir_only" not in models["M5"].feature_ids
    assert "pupil_variability_rgb_assisted" in models["M7"].feature_ids

    assert set(models["standalone::pupil_variability_rgb_assisted"].required_devices) == {"nir", "rgb"}
    assert set(models["M5"].required_devices) == {"behavior", "nir", "rgb"}


def test_plan_converts_to_existing_runner_feature_scheme_interface() -> None:
    plan = build_feature_comparison_plan(_registry())
    families = plan.to_runner_feature_schemes()

    assert set(families) == set(plan.model_map())
    assert len(families["behavior_reference"]) == 1
    assert families["behavior_reference"][0].columns == ("go_correct_rt_cv", "raw_go_omission_rate")
    assert set(families["M7"][0].modality_blocks) == {"behavior", "mmwave", "nir", "rgb"}


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
                            "modality_blocks": ["behavior"],
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


def test_registry_rejects_impossible_cross_device_package_claim() -> None:
    features = _registry()
    features[3] = RegisteredFeature(
        **{
            **features[3].__dict__,
            "allowed_device_packages": ("M1",),
        }
    )
    with pytest.raises(FeatureRegistryContractError, match="lacks required devices"):
        validate_registered_features(features)


def test_plan_rejects_named_device_package_with_no_information_from_declared_device() -> None:
    features = _registry()
    features = [feature for feature in features if feature.feature_id != "breathing_rate"]
    with pytest.raises(FeatureRegistryContractError, match="declares sensor devices without registered scientific information"):
        build_feature_comparison_plan(features)


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
