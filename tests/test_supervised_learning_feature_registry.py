from __future__ import annotations

import pytest

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
            allowed_device_packages=(),  # Behavior defaults to every M0-M7 package.
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

    # Behavior is decomposed into concrete features for interpretation but is
    # also retained together as the B reference model.
    assert set(models["behavior_reference"].feature_ids) == {"rt_variability", "omission"}
    assert models["standalone::rt_variability"].feature_ids == ("rt_variability",)

    # B+x really contains the complete frozen Behavior reference plus x.
    assert set(models["behavior_plus::blink_rate"].feature_ids) == {
        "rt_variability",
        "omission",
        "blink_rate",
    }

    # Full-x removes only the requested exact scientific representation.
    assert "pupil_variability_rgb_assisted" in models["full"].feature_ids
    assert "pupil_variability_rgb_assisted" not in models["full_minus::pupil_variability_rgb_assisted"].feature_ids


def test_cross_device_pupil_provenance_controls_m0_m7_membership() -> None:
    plan = build_feature_comparison_plan(_registry())
    models = plan.model_map()

    # NIR-only packages can use only the NIR-only pupil representation.
    assert "pupil_variability_nir_only" in models["M1"].feature_ids
    assert "pupil_variability_rgb_assisted" not in models["M1"].feature_ids
    assert "pupil_variability_nir_only" in models["M4"].feature_ids

    # NIR+RGB packages use the explicitly registered RGB-assisted representation.
    assert "pupil_variability_rgb_assisted" in models["M5"].feature_ids
    assert "pupil_variability_nir_only" not in models["M5"].feature_ids
    assert "pupil_variability_rgb_assisted" in models["M7"].feature_ids

    # Required device metadata follows the representation actually used.
    assert set(models["standalone::pupil_variability_rgb_assisted"].required_devices) == {"nir", "rgb"}
    assert set(models["M5"].required_devices) == {"behavior", "nir", "rgb"}


def test_plan_converts_to_existing_runner_feature_scheme_interface() -> None:
    plan = build_feature_comparison_plan(_registry())
    families = plan.to_runner_feature_schemes()

    assert set(families) == set(plan.model_map())
    assert len(families["behavior_reference"]) == 1
    assert families["behavior_reference"][0].columns == ("go_correct_rt_cv", "raw_go_omission_rate")
    assert set(families["M7"][0].modality_blocks) == {"behavior", "mmwave", "nir", "rgb"}


def test_registry_rejects_impossible_cross_device_package_claim() -> None:
    features = _registry()
    features[3] = RegisteredFeature(
        **{
            **features[3].__dict__,
            "allowed_device_packages": ("M1",),  # NIR-only package cannot produce RGB-assisted pupil.
        }
    )
    with pytest.raises(FeatureRegistryContractError, match="lacks required devices"):
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


def test_registry_rejects_q2_or_other_forbidden_mainline_column() -> None:
    features = _registry()
    features.append(
        RegisteredFeature(
            feature_id="illegal_q2",
            scientific_feature_id="sleepiness",
            columns=("q2_ordinal_4level",),
            role="sensor",
            raw_source="probe",
            required_devices=("rgb",),
            behavior_increment_eligible=True,
            allowed_device_packages=("M3",),
        )
    )
    with pytest.raises(FeatureRegistryContractError, match="Q2 is interpretation/construct validation"):
        validate_registered_features(features)
