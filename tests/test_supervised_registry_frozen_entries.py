"""Contract tests for the frozen 2026-09-13 unified supervised feature registry.

These tests lock the two things that the Step-2/Step-3 gate depends on:

1. the shipped registry is a complete, time-legality-verified, structurally consistent
   frozen artefact copied from the single-modality producer handoffs; and
2. the structural half of the validator actually rejects a colliding registry, so the
   gate cannot be satisfied by entries that could never generate a legitimate model.

They also pin the two facts the report chapters depend on:
* ``cardiopulmonary`` is excluded from the modality model plan even though NO
  cardiopulmonary feature is registered (the plan iterates every sensor scientific
  modality, so no placeholder entry is needed); and
* device packages M0/M3/M5 are available while M7 is not, because no frozen
  package-eligible feature actually uses the mmWave device.
"""
from __future__ import annotations

import copy
import json
import runpy
from pathlib import Path

import pytest
import yaml

from attention_pipeline.supervised_learning.feature_registry import (
    FeatureRegistryContractError,
    build_feature_comparison_plan,
    load_registered_features,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "supervised_learning_v1.yaml"

REQUIRED_ENTRY_FIELDS = (
    "feature_id",
    "scientific_feature_id",
    "columns",
    "role",
    "modality",
    "feature_type",
    "raw_source",
    "source_namespace",
    "required_devices",
    "preprocessing_dependencies",
    "temporal_anchor",
    "temporal_scope",
    "time_legality_status",
    "time_legality_evidence",
    "standalone_eligible",
    "behavior_increment_eligible",
    "behavior_reference_eligible",
    "modality_model_eligible",
    "full_model_eligible",
    "full_leave_one_out_eligible",
    "allowed_device_packages",
)

EXPECTED_FEATURE_IDS = (
    "behavior.rt_level.median.v1",
    "behavior.rt_variability.cv.v1",
    "behavior.rt_trend.theilsen.v1",
    "behavior.go_omission.raw.v1",
    "behavior.nogo_commission.raw.v1",
    "ocular.pupil_level.rseg_hard.rgb_nir_qc.v1",
    "ocular.pupil_variability.rseg_hard.rgb_nir_qc.v1",
    "ocular.pupil_linear_trend.rseg_hard.rgb_nir_qc.v1",
    "ocular.pupil_quadratic_curvature.rseg_hard.rgb_nir_qc.v1",
    "ocular.blink_rate.rgb_event_rate.pre30s.v1",
    "movement.body_motion_energy.median.pre30s.v1",
)


def _registry_section() -> dict:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return config["feature_registry"]


def _structural_helper():
    namespace = runpy.run_path(str(ROOT / "scripts" / "validate_supervised_feature_registry.py"))
    return namespace["_structure_and_plan"]


def test_registry_is_frozen_and_lists_exactly_the_frozen_representations() -> None:
    section = _registry_section()
    assert section["status"] == "frozen"
    assert section["schema_version"] == "1.16.10-time-legality-v1"
    assert section["frozen_at"] == "2026-09-13"
    assert tuple(entry["feature_id"] for entry in section["features"]) == EXPECTED_FEATURE_IDS
    # frozen_at must stay a string: an unquoted YAML date parses to datetime.date and
    # breaks the config digest, which is computed with json.dumps.
    assert isinstance(section["frozen_at"], str)


def test_every_entry_records_the_full_field_set_and_verified_time_legality() -> None:
    for entry in _registry_section()["features"]:
        missing = [field for field in REQUIRED_ENTRY_FIELDS if field not in entry]
        assert missing == [], f"{entry['feature_id']}: missing fields {missing}"
        assert entry["temporal_anchor"] == "probe_time_ms"
        assert entry["temporal_scope"] == "pre_probe_only"
        assert entry["time_legality_status"] == "verified_pre_probe_only"
        assert str(entry["time_legality_evidence"]).strip() != ""
        assert entry["columns"], f"{entry['feature_id']}: no predictor column"
        assert entry["raw_source"].strip() != ""


def test_scientific_modality_source_namespace_and_devices_stay_independent() -> None:
    features = {entry["feature_id"]: entry for entry in _registry_section()["features"]}

    # Pupil features are scientific modality ocular, produced in the nir namespace, but
    # genuinely depend on both nir and rgb because RGB blink masking assists the track.
    for feature_id in (
        "ocular.pupil_level.rseg_hard.rgb_nir_qc.v1",
        "ocular.pupil_variability.rseg_hard.rgb_nir_qc.v1",
        "ocular.pupil_linear_trend.rseg_hard.rgb_nir_qc.v1",
        "ocular.pupil_quadratic_curvature.rseg_hard.rgb_nir_qc.v1",
    ):
        entry = features[feature_id]
        assert entry["modality"] == "ocular"
        assert entry["source_namespace"] == "nir"
        assert entry["required_devices"] == ["nir", "rgb"]
        assert entry["allowed_device_packages"] == ["M5", "M7"]

    # Blink is the ocular scientific modality produced by the RGB namespace with no nir
    # dependency at all. None of the three fields may be inferred from another.
    blink = features["ocular.blink_rate.rgb_event_rate.pre30s.v1"]
    assert blink["modality"] == "ocular"
    assert blink["source_namespace"] == "rgb"
    assert blink["required_devices"] == ["rgb"]

    # Behavior comes from task/probe records and must declare no sensor device at all.
    for feature_id in EXPECTED_FEATURE_IDS[:5]:
        assert features[feature_id]["modality"] == "behavior"
        assert features[feature_id]["source_namespace"] == "behavior"
        assert features[feature_id]["required_devices"] == []
        assert features[feature_id]["allowed_device_packages"] == []


def test_no_cardiopulmonary_feature_is_registered() -> None:
    """HR/BR stay physiology_qualification = LIMITED_SUPPORTING_ONLY.

    They must not be registered as formal predictors, so the modality simply has no
    entry. Its exclusion is still reported by the generated plan, which is asserted in
    the next test.
    """
    modalities = {entry["modality"] for entry in _registry_section()["features"]}
    assert modalities == {"behavior", "ocular", "movement"}
    namespaces = {entry["source_namespace"] for entry in _registry_section()["features"]}
    assert "mmwave" not in namespaces


def test_plan_excludes_cardiopulmonary_without_any_placeholder_entry() -> None:
    plan = build_feature_comparison_plan(load_registered_features(_registry_section()))
    audit = plan.audit_dict()
    assert audit["unavailable_modalities"] == {
        "cardiopulmonary": (
            "no frozen registered feature is eligible for the scientific modality model"
        )
    }
    assert "modality::cardiopulmonary" not in {model["model_id"] for model in audit["models"]}
    assert not any(
        model["comparison_role"] == "standalone_modality" and model["modalities"] == ["cardiopulmonary"]
        for model in audit["models"]
    )


def test_device_package_availability_follows_real_hardware_dependency() -> None:
    plan = build_feature_comparison_plan(load_registered_features(_registry_section()))
    audit = plan.audit_dict()

    # M0 = behavior only; M3 = rgb (blink + movement); M5 = nir + rgb (pupil + blink + movement).
    assert audit["device_package_model_ids"] == {"M0": "M0", "M3": "M3", "M5": "M5"}

    unavailable = audit["unavailable_device_packages"]
    # M1 (nir only) cannot serve the pupil representation, which really needs rgb as well.
    assert "nir" in unavailable["M1"]
    # M7 requires some frozen package-eligible feature to actually USE mmWave; none does,
    # because cardiopulmonary is not registered. This is why M7 is unavailable.
    assert "mmwave" in unavailable["M7"]
    for package_id in ("M2", "M4", "M6"):
        assert "mmwave" in unavailable[package_id]


def test_structural_helper_reports_ok_for_the_shipped_registry() -> None:
    _structure_and_plan = _structural_helper()
    structure, comparison_plan = _structure_and_plan(_registry_section())
    assert structure["status"] == "ok"
    assert structure["n_features"] == len(EXPECTED_FEATURE_IDS)
    assert structure["n_features"] == 11
    assert structure["n_models"] == len(comparison_plan["model_ids"]) == 40
    assert comparison_plan["status"] == "ok"
    # JSON round-trip proves the audit payload written to disk is serialisable.
    json.dumps({"registry_structure": structure, "comparison_plan": comparison_plan})


def test_structural_gate_rejects_duplicate_full_model_representation() -> None:
    _structure_and_plan = _structural_helper()
    section = copy.deepcopy(_registry_section())
    duplicate = copy.deepcopy(section["features"][0])
    duplicate["feature_id"] = "behavior.rt_level.median.duplicate.v1"
    duplicate["columns"] = ["go_correct_rt_median_duplicate_ms"]
    # Same scientific_feature_id still eligible for the full model => two full-model
    # representations of one scientific feature, which must never be planned.
    section["features"].append(duplicate)

    with pytest.raises(FeatureRegistryContractError, match="multiple full-model representations"):
        _structure_and_plan(section)


def test_structural_gate_rejects_reused_predictor_column() -> None:
    _structure_and_plan = _structural_helper()
    section = copy.deepcopy(_registry_section())
    clash = copy.deepcopy(section["features"][5])  # an ocular pupil entry
    clash["feature_id"] = "ocular.pupil_level.collision.v1"
    clash["scientific_feature_id"] = "ocular.pupil_level_collision"
    clash["columns"] = list(section["features"][0]["columns"])  # reuse the RT median column
    section["features"].append(clash)

    with pytest.raises(FeatureRegistryContractError, match="assigned to multiple registered features"):
        _structure_and_plan(section)


def test_structural_gate_rejects_device_package_missing_required_device() -> None:
    _structure_and_plan = _structural_helper()
    section = copy.deepcopy(_registry_section())
    for entry in section["features"]:
        if entry["feature_id"] == "ocular.blink_rate.rgb_event_rate.pre30s.v1":
            # Claim a package that does not contain the device the feature really needs.
            entry["allowed_device_packages"] = ["M2"]
            break
    else:  # pragma: no cover - guards against a renamed fixture
        raise AssertionError("blink fixture entry not found")

    with pytest.raises(FeatureRegistryContractError, match="lacks required devices"):
        _structure_and_plan(section)


def test_empty_registry_is_reported_not_crashed() -> None:
    _structure_and_plan = _structural_helper()
    structure, comparison_plan = _structure_and_plan({"features": []})
    assert structure["status"] == "empty_registry"
    assert comparison_plan["status"] == "empty_registry"
    assert comparison_plan["model_ids"] == []
