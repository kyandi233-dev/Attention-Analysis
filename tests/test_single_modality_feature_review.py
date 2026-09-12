from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from attention_pipeline.formal_analysis.feature_qualification import (
    FeatureQualificationError, file_sha256, review_single_modality, write_single_modality_review,
)
from attention_pipeline.behavior_formal.behavior_supervised_contract import frozen_behavior_reference_columns
from attention_pipeline.behavior_formal.behavior_supervised_interface import build_behavior_supervised_feature_audit

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "configs/single_modality_features_v1.yaml"


@pytest.fixture
def catalog():
    return yaml.safe_load(CATALOG.read_text(encoding="utf-8"))


@pytest.fixture
def probes():
    # Unequal repeated-session counts make participant vs pooled coverage distinct.
    return pd.DataFrame({
        "participant_group_id": ["p1", "p1", "p1", "p2"],
        "session_id": ["s1", "s1", "s2", "s3"],
        "block_id": ["b1"] * 4, "probe_event_id": ["a", "b", "c", "d"],
        "q1_nominal_4class": [1, 2, 3, 4], "q2_ordinal_4level": [1, 2, 3, 4],
        "go_correct_rt_mean_ms": [400., 420., 410., 500.],
        "go_correct_rt_median_ms": [395., 415., 405., 490.],
        "go_correct_rt_cv": [.1, .2, .15, np.nan],
        "go_correct_rt_theilsen_slope_ms_per_s": [1., 2., 1., .5],
        "raw_go_omission_rate": [0., 0., .1, 0.], "commission_rate": [.1, .2, .1, .3],
        "dprime_loglinear": [2., 2., 2., 1.],
        "blink_event_rate_per_min": [0., 2., 4., np.nan],
        "body_motion_energy_median": [1., 2., 3., 4.],
        "exposure_change_abs_median": [1., 2., 3., 4.],
    })


def test_q1_q2_shuffle_and_unrelated_devices_cannot_change_review(catalog, probes):
    a = review_single_modality(probes, catalog, "behavior")
    changed = probes.drop(columns=["blink_event_rate_per_min", "body_motion_energy_median"])
    changed["q1_nominal_4class"] = "malformed-unused-label"
    changed["q2_ordinal_4level"] = np.nan
    b = review_single_modality(changed, catalog, "behavior")
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_frame_equal(a[1], b[1])
    assert a[2] == b[2]
    assert not a[2]["modality_primary_representations_frozen"]
    assert a[2]["qualified_primary_features"] == []


def test_descriptive_coverage_is_participant_aware_and_not_a_selection_gate(catalog, probes):
    review, _, state = review_single_modality(probes, catalog, "behavior")
    cv = review.set_index("column").loc["go_correct_rt_cv"]
    assert cv["finite_fraction"] == .75
    assert cv["participant_macro_finite_fraction"] == .5
    assert cv["participant_n"] == 2
    assert cv["session_n"] == 3
    assert not review["automatic_selection_applied"].any()
    assert state["next_gate"] == "upstream_measurement_freeze"


def test_representation_contrasts_do_not_subtract_incompatible_units(catalog, probes):
    _, contrasts, _ = review_single_modality(probes, catalog, "behavior")
    assert len(contrasts) == 1
    assert contrasts.iloc[0]["left_column"] == "go_correct_rt_mean_ms"
    assert contrasts.iloc[0]["right_column"] == "go_correct_rt_median_ms"
    assert contrasts.iloc[0]["median_difference"] == 5.
    _, blink_contrasts, _ = review_single_modality(probes, catalog, "ocular")
    assert blink_contrasts.empty


def test_behavior_five_dimensions_and_existing_audit_respect_latest_roles(catalog, probes):
    chosen = frozen_behavior_reference_columns("go_correct_rt_median_ms")
    assert len(chosen) == 5
    assert "dprime_loglinear" not in chosen
    with pytest.raises(ValueError, match="explicit mean/median"):
        frozen_behavior_reference_columns("best_Q1_predictor")
    audit = build_behavior_supervised_feature_audit(probes).set_index("field")
    assert audit.loc["go_correct_rt_cv", "main_reference_fixed_member"]
    assert audit.loc["commission_rate", "main_reference_fixed_member"]
    assert audit.loc["dprime_loglinear", "role"] == "descriptive_qc_sensitivity_only"


def _movement_decision():
    return {"movement_visible_level": {
        "measurement_status": "qualified", "selected_column": "body_motion_energy_median",
        "evidence_refs": ["synthetic-test-quality-evidence"],
    }}


def test_only_explicit_qualification_selects_primary_and_never_authorizes_fitting(catalog, probes):
    review, _, state = review_single_modality(probes, catalog, "movement", decisions=_movement_decision())
    assert state["modality_primary_representations_frozen"]
    assert not state["supervised_execution_authorized"]
    feature = state["qualified_primary_features"][0]
    assert feature["modality"] == "movement"
    assert feature["required_devices"] == ["rgb"]
    assert feature["columns"] == ["body_motion_energy_median"]
    assert not review.loc[review["column"].eq("exposure_change_abs_median"), "selected_primary"].any()


@pytest.mark.parametrize("edit", [
    {"selected_column": "exposure_change_abs_median"},
    {"measurement_status": "false"},
    {"evidence_refs": []},
    {"required_devices": ["nir"]},
    {"preprocessing_dependencies": []},
    {"outer_test_auc": .99},
])
def test_invalid_decisions_fail_closed(catalog, probes, edit):
    decisions = _movement_decision()
    decisions["movement_visible_level"].update(edit)
    with pytest.raises(FeatureQualificationError):
        review_single_modality(probes, catalog, "movement", decisions=decisions)


def test_corrupt_values_are_distinct_from_missing_and_prevent_execution(catalog, probes):
    probes["body_motion_energy_median"] = ["bad", "", np.inf, 0]
    review, _, state = review_single_modality(probes, catalog, "movement", decisions=_movement_decision())
    row = review.set_index("column").loc["body_motion_energy_median"]
    assert (row["malformed_n"], row["missing_n"], row["nonfinite_n"], row["finite_n"]) == (1, 1, 1, 1)
    assert not state["modality_primary_representations_frozen"]
    assert state["dimensions"][0]["measurement_status"] == "qualified"
    assert state["dimensions"][0]["execution_state"] == "malformed_or_nonfinite_source"


def test_blink_zero_is_observed_but_unfrozen_pupil_is_not_silently_dropped(catalog, probes):
    review, _, state = review_single_modality(probes, catalog, "ocular")
    blink = review.set_index("column").loc["blink_event_rate_per_min"]
    assert blink["finite_n"] == 3
    assert len(state["dimensions"]) == 5
    assert review["role"].eq("unresolved_representation").sum() == 4
    assert not state["modality_primary_representations_frozen"]


def test_cardiopulmonary_remains_pending_without_invented_devices(catalog, probes):
    review, _, state = review_single_modality(probes, catalog, "cardiopulmonary")
    assert len(review) == 2
    assert review["required_devices"].eq("null").all()
    assert not state["modality_primary_representations_frozen"]


def test_legacy_rt_cv_mask_is_reported_without_reconstructing_values(catalog, probes):
    probes["rt_cv_min_n"] = 20
    probes["rt_cv_status"] = ["estimable", "estimable", "estimable", "not_estimable_low_rt_n"]
    probes["correct_go_rt_opportunities"] = [25, 25, 25, 10]
    before = probes.copy(deep=True)
    decisions = {"behavior_rt_variability": {
        "measurement_status": "qualified", "selected_column": "go_correct_rt_cv", "evidence_refs": ["review"]}}
    _, _, state = review_single_modality(probes, catalog, "behavior", decisions=decisions)
    check = next(x for x in state["producer_contract_checks"] if x["scientific_feature_id"] == "behavior_rt_variability")
    assert check["status"] == "failed"
    assert "rt_cv_min_n=2" in check["reason"]
    dimension = next(x for x in state["dimensions"] if x["scientific_feature_id"] == "behavior_rt_variability")
    assert dimension["execution_state"] == "producer_contract_failed"
    assert state["qualified_primary_features"] == []
    pd.testing.assert_frame_equal(probes, before)


def test_ambiguous_probes_and_label_candidates_are_rejected(catalog, probes):
    with pytest.raises(FeatureQualificationError, match="duplicate probe"):
        review_single_modality(pd.concat([probes, probes.iloc[[0]]]), catalog, "behavior")
    bad = copy.deepcopy(catalog)
    bad["dimensions"][0]["primary_candidates"] = ["q1_nominal_4class"]
    bad["dimensions"][0]["comparison_pairs"] = []
    with pytest.raises(FeatureQualificationError, match="outcome/identity"):
        review_single_modality(probes, bad, "behavior")


def test_file_entry_preserves_inputs_hashes_and_requires_fresh_decision_provenance(tmp_path, probes):
    source = tmp_path / "probes.csv"
    probes.to_csv(source, index=False)
    source_hash = file_sha256(source)
    out = tmp_path / "review"
    first = write_single_modality_review(source, CATALOG, "movement", out)
    assert first["source_sha256"] == source_hash == file_sha256(source)
    assert not first["modality_primary_representations_frozen"]
    decision_path = tmp_path / "decisions.json"
    decision_path.write_text(json.dumps({"source_sha256": source_hash, "catalog_sha256": file_sha256(CATALOG),
                                         "modality": "movement", "decisions": _movement_decision()}))
    second = write_single_modality_review(source, CATALOG, "movement", tmp_path / "frozen-review", decisions_path=decision_path)
    assert second["modality_primary_representations_frozen"]
    with pytest.raises(FileExistsError):
        write_single_modality_review(source, CATALOG, "movement", out)
    source.write_text(source.read_text() + "\n")
    with pytest.raises(FeatureQualificationError, match="provenance"):
        write_single_modality_review(source, CATALOG, "movement", tmp_path / "stale", decisions_path=decision_path)
    assert not (tmp_path / "stale").exists()


def test_real_cli_writes_aggregate_artifacts_without_training(tmp_path, probes):
    source = tmp_path / "input.csv"
    probes.to_csv(source, index=False)
    import os
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run([sys.executable, str(ROOT / "scripts/single_modality_feature_review.py"),
                             "--input", str(source), "--catalog", str(CATALOG), "--modality", "behavior",
                             "--output-root", str(tmp_path / "out")], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "out/modality_release.json").read_text(encoding="utf-8"))
    assert manifest["label_values_read"] is False
    assert manifest["supervised_execution_authorized"] is False
    report = pd.read_csv(tmp_path / "out/feature_review.csv")
    assert "participant_group_id" not in report.columns
    assert not report["column"].str.startswith("q1").any()
