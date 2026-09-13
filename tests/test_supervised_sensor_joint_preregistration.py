"""Pre-registration tests for the sensor-only joint comparison model.

Report chapter 3 and section 4.6 require an explicit evaluation of the sensor
information combination that EXCLUDES task behavior, so that the report can answer how
much Q1-report prediction survives when the dedicated task behaviour is removed. The
comparison plan previously offered only single-category models, behaviour-conditioned
increments, full leave-one-out and device packages; device packages answer the hardware
configuration question and must never be substituted for this scientific comparison.

These tests lock the pre-registration contract:

* the model exists, is defined by frozen eligibility rather than a hard-coded modality
  list, and excludes task behaviour entirely;
* it participates in no paired comparison (it is nested with neither the behaviour
  reference nor the full model), so it can only ever be reported as a standalone
  descriptive result;
* it joins and leaves exactly with the frozen eligibility flags; and
* for the current freeze its column set coincides with
  ``full_minus_modality::behavior``, which is a fingerprint of this freeze, not a
  definition.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from attention_pipeline.supervised_learning.comparison_provenance import (
    build_paired_comparison_specs,
)
from attention_pipeline.supervised_learning.feature_registry import (
    build_feature_comparison_plan,
    load_registered_features,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "supervised_learning_v1.yaml"

SENSOR_JOINT_MODEL_ID = "sensor_only_joint"


def _registry_section() -> dict:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return config["feature_registry"]


def _plan(section: dict | None = None):
    return build_feature_comparison_plan(
        load_registered_features(section or _registry_section())
    )


def test_sensor_only_joint_model_is_pre_registered() -> None:
    plan = _plan()
    assert plan.sensor_joint_model_id == SENSOR_JOINT_MODEL_ID
    assert plan.sensor_joint_model_id in plan.model_map()
    assert plan.audit_dict()["sensor_joint_model_id"] == SENSOR_JOINT_MODEL_ID

    model = plan.model_map()[SENSOR_JOINT_MODEL_ID]
    assert model.comparison_role == "standalone_sensor_joint"
    assert model.includes_behavior_reference is False
    assert set(model.modalities) == {"ocular", "movement"}


def test_sensor_only_joint_columns_are_exactly_the_eligible_sensor_union() -> None:
    """The definition: every sensor feature holding formal modality-model qualification."""
    registry = load_registered_features(_registry_section())
    expected = tuple(
        feature.feature_id
        for feature in registry
        if feature.role == "sensor" and feature.modality_model_eligible
    )
    plan = _plan()
    model = plan.model_map()[SENSOR_JOINT_MODEL_ID]

    assert model.feature_ids == expected
    expected_columns = {
        column
        for feature in registry
        if feature.role == "sensor" and feature.modality_model_eligible
        for column in feature.columns
    }
    assert set(model.columns) == expected_columns


def test_sensor_only_joint_excludes_every_behavior_predictor() -> None:
    registry = load_registered_features(_registry_section())
    behavior_columns = {
        column
        for feature in registry
        if feature.role == "behavior"
        for column in feature.columns
    }
    assert behavior_columns, "fixture expects a non-empty Behavior reference"

    model = _plan().model_map()[SENSOR_JOINT_MODEL_ID]
    assert set(model.columns).isdisjoint(behavior_columns)
    assert model.required_devices == ("nir", "rgb")


def test_current_freeze_matches_full_minus_behavior_columns() -> None:
    """Fingerprint of this freeze, not a definition.

    It holds because every currently modality-model-eligible sensor feature is also
    eligible for the full model. If a future freeze admits a sensor feature that is
    modality-model eligible but not full-model eligible, the two models may legitimately
    diverge and this assertion is the place that records the change.
    """
    plan = _plan()
    models = plan.model_map()
    joint = models[SENSOR_JOINT_MODEL_ID]
    reduced = models["full_minus_modality::behavior"]
    assert set(joint.columns) == set(reduced.columns)
    assert tuple(sorted(joint.columns)) == tuple(sorted(reduced.columns))


def test_sensor_only_joint_is_not_part_of_any_paired_comparison() -> None:
    plan = _plan()
    pair_tuples = (
        plan.behavior_increment_pairs
        + plan.full_leave_one_out_pairs
        + plan.modality_increment_pairs
        + plan.full_leave_one_modality_out_pairs
    )
    assert [pair for pair in pair_tuples if SENSOR_JOINT_MODEL_ID in (pair[0], pair[1])] == []

    specs = build_paired_comparison_specs(plan, [model.model_id for model in plan.models])
    assert [
        spec
        for spec in specs
        if SENSOR_JOINT_MODEL_ID in (spec["baseline_model_id"], spec["added_model_id"])
    ] == []


def test_sensor_only_joint_absent_when_no_sensor_feature_is_eligible() -> None:
    section = copy.deepcopy(_registry_section())
    for entry in section["features"]:
        if entry["role"] == "sensor":
            entry["modality_model_eligible"] = False

    plan = _plan(section)
    assert plan.sensor_joint_model_id == ""
    assert SENSOR_JOINT_MODEL_ID not in plan.model_map()
    assert set(plan.audit_dict()["unavailable_modalities"]) == {"ocular", "movement", "cardiopulmonary"}


def test_cardiopulmonary_joins_only_through_the_eligibility_flag() -> None:
    """The model is derived from eligibility, so a future qualified modality joins.

    Here a synthetic cardiopulmonary entry is marked modality-model eligible and must
    join the joint model automatically, without touching any hard-coded modality list.
    """
    section = copy.deepcopy(_registry_section())
    section["features"].append(
        {
            "feature_id": "cardiopulmonary.synthetic_hr.v1",
            "scientific_feature_id": "cardiopulmonary.synthetic_heart_rate",
            "columns": ["synthetic_mmwave_hr_median"],
            "role": "sensor",
            "modality": "cardiopulmonary",
            "feature_type": "heart_rate",
            "raw_source": "synthetic/for_test_only",
            "source_namespace": "mmwave",
            "required_devices": ["mmwave"],
            "preprocessing_dependencies": [],
            "temporal_anchor": "probe_time_ms",
            "temporal_scope": "pre_probe_only",
            "time_legality_status": "verified_pre_probe_only",
            "time_legality_evidence": "synthetic eligibility test fixture",
            "standalone_eligible": True,
            "behavior_increment_eligible": True,
            "behavior_reference_eligible": False,
            "modality_model_eligible": True,
            "full_model_eligible": True,
            "full_leave_one_out_eligible": True,
            "allowed_device_packages": [],
        }
    )

    plan = _plan(section)
    model = plan.model_map()[SENSOR_JOINT_MODEL_ID]
    assert "cardiopulmonary.synthetic_hr.v1" in model.feature_ids
    assert "synthetic_mmwave_hr_median" in model.columns
    assert "mmwave" in model.required_devices
    assert set(model.modalities) == {"cardiopulmonary", "ocular", "movement"}
    # And the modality is no longer reported as unavailable.
    assert "cardiopulmonary" not in plan.audit_dict()["unavailable_modalities"]
