"""Contract tests for the supplementary feature pathway.

The supplementary pathway exists so that a feature whose qualification is incomplete can be reported
in a clearly labelled comparison **without** this repository ever asserting formal prediction
eligibility for it. These tests pin the three properties that make that safe:

1. ``supplementary_model_eligible`` is a separate axis from the six formal eligibility flags, and a
   feature may not hold both;
2. the supplementary models reuse the frozen Behavior reference and the frozen ``full`` model as
   baselines, so the paired increment is computed on identical rows;
3. enabling the pathway leaves the formal plan completely unchanged - no new formal modality model,
   no new device package, no new formal standalone model.
"""
from __future__ import annotations

import yaml

import pytest

from attention_pipeline.supervised_learning.feature_registry import (
    FeatureRegistryContractError,
    build_feature_comparison_plan,
    load_registered_features,
)
from attention_pipeline.supervised_learning.time_legality import (
    validate_feature_registry_time_legality,
)

FREEZE_CONFIRMING = ""


def _supplementary(**overrides) -> dict:
    entry = {
        "feature_id": "cardiopulmonary.hr.fused.v1",
        "scientific_feature_id": "radar_derived_heart_rate",
        "columns": ["mmwave_hr_fused_bpm_median"],
        "role": "sensor",
        "modality": "cardiopulmonary",
        "feature_type": "heart_rate",
        "raw_source": "mmwave/taskb_source.csv",
        "source_namespace": "mmwave",
        "required_devices": ["mmwave"],
        "preprocessing_dependencies": [],
        "temporal_anchor": "probe_time_ms",
        "temporal_scope": "pre_probe_only",
        "time_legality_status": "blocked_upstream_contract_mismatch",
        "time_legality_evidence": "metadata contract internally consistent; provenance open",
        "standalone_eligible": False,
        "behavior_increment_eligible": False,
        "behavior_reference_eligible": False,
        "modality_model_eligible": False,
        "full_model_eligible": False,
        "full_leave_one_out_eligible": False,
        "allowed_device_packages": [],
        "supplementary_model_eligible": True,
        "physiology_qualification": "LIMITED_SUPPORTING_ONLY",
        "report_role": "supporting_only",
    }
    entry.update(overrides)
    return entry


def _formal_behavior() -> dict:
    return {
        "feature_id": "behavior.rt_level.median.v1",
        "scientific_feature_id": "behavior_rt_level",
        "columns": ["go_correct_rt_median_ms"],
        "role": "behavior",
        "modality": "behavior",
        "raw_source": "Behavior/formal_v3/probe_primary_30s.csv",
        "source_namespace": "behavior",
        "required_devices": [],
        "temporal_anchor": "probe_time_ms",
        "temporal_scope": "pre_probe_only",
        "time_legality_status": "verified_pre_probe_only",
        "time_legality_evidence": "30 s pre-probe Behavior window",
        "standalone_eligible": True,
        "behavior_reference_eligible": True,
        "full_model_eligible": True,
        "full_leave_one_out_eligible": True,
    }


def _formal_ocular() -> dict:
    return {
        "feature_id": "ocular.blink_rate.v1",
        "scientific_feature_id": "blink_frequency",
        "columns": ["blink_event_rate_per_min"],
        "role": "sensor",
        "modality": "ocular",
        "raw_source": "FormalScience/Ocular/ocular_probe_features_wide.csv",
        "source_namespace": "rgb",
        "required_devices": ["rgb"],
        "temporal_anchor": "probe_time_ms",
        "temporal_scope": "pre_probe_only",
        "time_legality_status": "verified_pre_probe_only",
        "time_legality_evidence": "30 s pre-probe window",
        "standalone_eligible": True,
        "behavior_increment_eligible": True,
        "modality_model_eligible": True,
        "full_model_eligible": True,
        "full_leave_one_out_eligible": True,
        "allowed_device_packages": ["M3", "M5", "M7"],
    }


def _registry(extra: dict | None = None) -> dict:
    features = [_formal_behavior(), _formal_ocular()]
    if extra is not None:
        features.append(extra)
    return {"features": features}


def test_supplementary_feature_generates_three_models_and_two_paired_specs():
    plan = build_feature_comparison_plan(
        load_registered_features(_registry(_supplementary()))
    )
    assert plan.supplementary_model_ids == {
        "cardiopulmonary_only": "supplementary::cardiopulmonary_only",
        "behavior_plus_cardiopulmonary": "supplementary::behavior_plus_cardiopulmonary",
        "full_plus_cardiopulmonary": "supplementary::full_plus_cardiopulmonary",
    }
    assert plan.supplementary_pairs == (
        ("behavior_reference", "supplementary::behavior_plus_cardiopulmonary", "cardiopulmonary"),
        ("full", "supplementary::full_plus_cardiopulmonary", "cardiopulmonary"),
    )
    models = plan.model_map()
    assert len(models["supplementary::cardiopulmonary_only"].columns) == 1
    assert len(models["supplementary::behavior_plus_cardiopulmonary"].columns) == 2
    assert len(models["supplementary::full_plus_cardiopulmonary"].columns) == 3


def test_supplementary_models_reuse_the_frozen_baselines():
    """The baselines must be the frozen model ids, not newly invented reduced models.

    Reusing them is what makes the paired increment a like-for-like comparison: the runner refits
    the baseline on the same rows as the added model.
    """
    plan = build_feature_comparison_plan(
        load_registered_features(_registry(_supplementary()))
    )
    models = plan.model_map()
    assert models["supplementary::behavior_plus_cardiopulmonary"].feature_ids == (
        "behavior.rt_level.median.v1",
        "cardiopulmonary.hr.fused.v1",
    )
    assert models["supplementary::full_plus_cardiopulmonary"].feature_ids == (
        "behavior.rt_level.median.v1",
        "ocular.blink_rate.v1",
        "cardiopulmonary.hr.fused.v1",
    )
    # The formal full model must NOT have absorbed the supplementary feature.
    assert models["full"].feature_ids == ("behavior.rt_level.median.v1", "ocular.blink_rate.v1")


def test_supplementary_pathway_leaves_formal_structure_unchanged():
    """No formal modality model, standalone model or device package may appear."""
    without = build_feature_comparison_plan(load_registered_features(_registry()))
    with_supp = build_feature_comparison_plan(
        load_registered_features(_registry(_supplementary()))
    )
    formal_ids_without = {
        model.model_id for model in without.models if not model.model_id.startswith("supplementary::")
    }
    formal_ids_with = {
        model.model_id
        for model in with_supp.models
        if not model.model_id.startswith("supplementary::")
    }
    assert formal_ids_without == formal_ids_with
    assert with_supp.unavailable_modalities == without.unavailable_modalities
    assert with_supp.unavailable_device_packages == without.unavailable_device_packages
    assert with_supp.modality_model_ids == without.modality_model_ids
    assert with_supp.device_package_model_ids == without.device_package_model_ids


def test_supplementary_declaration_states_what_is_not_claimed():
    plan = build_feature_comparison_plan(
        load_registered_features(_registry(_supplementary()))
    )
    declaration = plan.supplementary_declaration
    assert declaration["prediction_eligibility_claimed"] is False
    assert declaration["device_package_eligibility_claimed"] is False
    assert declaration["preregistration"] == "post_hoc_declared_after_primary_results_observed"
    assert declaration["scientific_modality"] == "cardiopulmonary"
    assert declaration["feature_ids"] == ["cardiopulmonary.hr.fused.v1"]


def test_absent_flag_produces_no_supplementary_models():
    plan = build_feature_comparison_plan(load_registered_features(_registry()))
    assert plan.supplementary_model_ids == {}
    assert plan.supplementary_pairs == ()
    assert plan.supplementary_declaration == {}


@pytest.mark.parametrize(
    "leaked_field",
    [
        "standalone_eligible",
        "behavior_increment_eligible",
        "behavior_reference_eligible",
        "modality_model_eligible",
        "full_model_eligible",
        "full_leave_one_out_eligible",
    ],
)
def test_supplementary_feature_may_not_hold_any_formal_eligibility(leaked_field):
    """Every formal eligibility flag must be refused.

    Some of these fields also trip an *earlier* pre-existing invariant (for example
    ``full_leave_one_out_eligible`` requires ``full_model_eligible``, and
    ``behavior_reference_eligible`` requires ``role == "behavior"``), so the exact message depends on
    which guard fires first. Rejection is what matters; the specific message of the new guard is
    pinned separately below.
    """
    with pytest.raises(FeatureRegistryContractError):
        load_registered_features(_registry(_supplementary(**{leaked_field: True})))


def test_the_supplementary_separation_guard_reports_its_own_reason():
    """A field that reaches the new guard must be refused with the separation message."""
    with pytest.raises(FeatureRegistryContractError, match="requires every formal"):
        load_registered_features(_registry(_supplementary(standalone_eligible=True)))


def test_supplementary_feature_may_not_claim_a_device_package():
    with pytest.raises(FeatureRegistryContractError, match="forbids device-package claims"):
        load_registered_features(_registry(_supplementary(allowed_device_packages=["M2"])))


def test_supplementary_feature_requires_an_explicit_qualification_and_role():
    with pytest.raises(FeatureRegistryContractError, match="non-blank physiology_qualification"):
        load_registered_features(_registry(_supplementary(physiology_qualification="")))
    with pytest.raises(FeatureRegistryContractError, match="non-blank report_role"):
        load_registered_features(_registry(_supplementary(report_role="")))


def test_supplementary_feature_still_requests_no_prediction_eligibility():
    """The whole point: the time-legality gate must see no prediction request at all.

    If this ever reports ``requests_prediction is True``, the supplementary pathway has become a
    back door to formal eligibility and the fail-closed guarantee no longer holds.
    """
    registry = _registry(_supplementary())
    records = validate_feature_registry_time_legality(registry)
    by_id = {record.feature_id: record for record in records}
    record = by_id["cardiopulmonary.hr.fused.v1"]
    assert record.requests_prediction is False
    assert record.prediction_eligibility_fields_true == ()
    assert record.time_legality_status == "blocked_upstream_contract_mismatch"


def test_omitting_the_denials_would_be_read_as_a_prediction_request():
    """Documents why the config must spell out the three default-True denials."""
    entry = _supplementary()
    for field in ("standalone_eligible", "full_model_eligible", "full_leave_one_out_eligible"):
        entry.pop(field)
    with pytest.raises(FeatureRegistryContractError, match="requires every formal"):
        load_registered_features(_registry(entry))


def test_shipped_supplementary_config_is_v1_plus_two_entries():
    """The shipped config must be exactly the frozen v1 registry plus the two mmWave entries."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    v1 = yaml.safe_load((root / "configs/supervised_learning_v1.yaml").read_text(encoding="utf-8"))
    v2 = yaml.safe_load(
        (root / "configs/supervised_learning_v2_supplementary_cardiopulmonary.yaml").read_text(
            encoding="utf-8"
        )
    )
    frozen = v1["feature_registry"]["features"]
    extended = v2["feature_registry"]["features"]
    assert len(frozen) == 11
    assert len(extended) == 13
    assert extended[: len(frozen)] == frozen
    assert [entry["feature_id"] for entry in extended[len(frozen) :]] == [
        "cardiopulmonary.hr.fused.v1",
        "cardiopulmonary.br.v1",
    ]
    # The frozen v1 config itself must be untouched by this pathway.
    assert v1["feature_registry"]["status"] == "frozen"
    assert v1["feature_registry"].get("supplementary_extension") is None


def test_shipped_supplementary_config_builds_the_expected_models():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs/supervised_learning_v2_supplementary_cardiopulmonary.yaml").read_text(
            encoding="utf-8"
        )
    )
    plan = build_feature_comparison_plan(load_registered_features(config["feature_registry"]))
    assert sorted(plan.supplementary_model_ids) == [
        "behavior_plus_cardiopulmonary",
        "cardiopulmonary_only",
        "full_plus_cardiopulmonary",
    ]
    models = plan.model_map()
    ids = plan.supplementary_model_ids
    assert len(models["full"].columns) == 11
    assert len(models[ids["full_plus_cardiopulmonary"]].columns) == 13
    assert len(models[ids["behavior_plus_cardiopulmonary"]].columns) == 7
    assert len(models[ids["cardiopulmonary_only"]].columns) == 2
    assert plan.unavailable_modalities == {
        "cardiopulmonary": "no frozen registered feature is eligible for the scientific modality model"
    }
    assert sorted(plan.unavailable_device_packages) == ["M1", "M2", "M4", "M6", "M7"]
