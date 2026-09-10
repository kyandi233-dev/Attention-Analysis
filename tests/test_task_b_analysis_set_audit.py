import numpy as np
import pandas as pd
import pytest

from attention_pipeline.multimodal_formal.quality_admission import audit_quality
from attention_pipeline.multimodal_formal.analysis_sets import build_analysis_sets
from attention_pipeline.multimodal_formal.prediction_archive import validate_prediction_archive


def _tables():
    behavior = pd.DataFrame({
        "session_id": ["s1", "s1", "s2", "s2"],
        "participant_group_id": ["p1", "p1", "p2", "p2"],
        "block_id": ["b1"] * 4,
        "probe_index_in_block": [1, 2, 1, 2],
        "b": [1.0, 2.0, 3.0, 4.0],
        "raw_go_omission_rate": [0.0, 0.1, 0.0, 0.2],
        "window_name": ["pre_30s"] * 4,
    })
    nir = behavior[["session_id", "participant_group_id", "block_id", "probe_index_in_block"]].copy()
    nir["source_present"] = True
    nir["source_readable"] = True
    nir["window_name"] = "pre_30s"
    nir["n"] = [10.0, np.nan, 12.0, 13.0]
    rgb = behavior[["session_id", "participant_group_id", "block_id", "probe_index_in_block"]].copy()
    rgb["source_present"] = True
    rgb["source_readable"] = True
    rgb["window_name"] = "pre_30s"
    rgb["r"] = [1.0, 2.0, 3.0, 4.0]
    return {"behavior": behavior, "nir": nir, "rgb": rgb}


def test_coverage_below_80_is_review_only_not_exclusion():
    t = _tables()
    t["nir"].loc[1:, "n"] = np.nan
    result = audit_quality(t, {"behavior": ["b"], "nir": ["n"]})
    row = result["feature_coverage"].query("modality == 'nir' and feature == 'n'").iloc[0]
    assert row.feature_computable_coverage == 0.25
    assert bool(row.coverage_below_0_80_review)
    assert not bool(row.automatic_exclusion)
    assert "main_candidate" not in result["feature_coverage"].columns


def test_structural_missing_and_single_feature_missing_are_distinct():
    t = _tables()
    t["nir"] = t["nir"].iloc[:3].copy()
    result = audit_quality(t, {"behavior": ["b"], "nir": ["n"]})
    status = result["probe_feature_status"].query("modality == 'nir'").sort_values(["session_id", "probe_index_in_block"])
    reasons = dict(zip(zip(status.session_id, status.probe_index_in_block), status.missing_kind))
    assert reasons[("s1", 2)] == "single_feature_missing"
    assert reasons[("s2", 2)] == "record_missing"
    single = status[(status.session_id == "s1") & (status.probe_index_in_block == 2)].iloc[0]
    structural = status[(status.session_id == "s2") & (status.probe_index_in_block == 2)].iloc[0]
    assert bool(single.eligible_for_missing_strategy)
    assert not bool(structural.eligible_for_missing_strategy)


def test_analysis_set_is_comparison_specific_and_ignores_unrequested_modality():
    t = _tables()
    t["rgb"] = t["rgb"].iloc[:1].copy()
    result = audit_quality(t, {"behavior": ["b"], "nir": ["n"], "rgb": ["r"]})
    specs = {
        "M0_vs_M1_nir": {
            "models": ["M0", "M1"],
            "required_features": {"behavior": ["b"], "nir": ["n"]},
        },
        "M0_vs_rgb": {
            "models": ["M0", "MRGB"],
            "required_features": {"behavior": ["b"], "rgb": ["r"]},
        },
    }
    sets, summary = build_analysis_sets(result["formal_probe_identity"], result["probe_feature_status"], specs)
    nir_set = sets[sets.analysis_set_id == "M0_vs_M1_nir"]
    rgb_set = sets[sets.analysis_set_id == "M0_vs_rgb"]
    assert nir_set.included_complete.sum() == 3
    assert nir_set.included_missing_aware.sum() == 4
    assert rgb_set.included_complete.sum() == 1
    assert summary.query("analysis_set_id == 'M0_vs_M1_nir' and membership == 'included_complete'").probe_n.iloc[0] == 3


def test_alignment_failure_not_allowed_in_missing_aware_set():
    t = _tables()
    t["nir"].loc[0, "window_name"] = "post_30s"
    result = audit_quality(t, {"behavior": ["b"], "nir": ["n"]})
    sets, _ = build_analysis_sets(
        result["formal_probe_identity"],
        result["probe_feature_status"],
        {"x": {"required_features": {"behavior": ["b"], "nir": ["n"]}}},
    )
    row = sets[(sets.session_id == "s1") & (sets.probe_index_in_block == 1)].iloc[0]
    assert not bool(row.included_complete)
    assert not bool(row.included_missing_aware)
    assert "structural_alignment_invalid" in row.missing_aware_exclusion_reason


def _prediction_rows(sets):
    included = sets[sets.included_complete].copy()
    rows = []
    for model in ["M0", "M1"]:
        for _, r in included.iterrows():
            rows.append({
                "session_id": r.session_id,
                "participant_group_id": r.participant_group_id,
                "block_id": r.block_id,
                "probe_index_in_block": r.probe_index_in_block,
                "analysis_set_id": r.analysis_set_id,
                "outer_fold_group": r.participant_group_id,
                "outcome": "q1_binary",
                "model_id": model,
                "y_true": 1,
                "y_pred": 1,
                "probability_positive": 0.75,
            })
    return pd.DataFrame(rows)


def test_prediction_archive_one_row_per_probe_and_complete_coverage():
    result = audit_quality(_tables(), {"behavior": ["b"], "nir": ["n"]})
    sets, _ = build_analysis_sets(
        result["formal_probe_identity"],
        result["probe_feature_status"],
        {"M0_vs_M1_nir": {"required_features": {"behavior": ["b"], "nir": ["n"]}}},
    )
    predictions = _prediction_rows(sets)
    audit = validate_prediction_archive(predictions, sets)
    assert audit["status"] == "PASS_PREDICTION_ARCHIVE"
    assert audit["prediction_n"] == 6
    bad = predictions.iloc[:-1].copy()
    with pytest.raises(ValueError, match="incomplete prediction coverage"):
        validate_prediction_archive(bad, sets)


def test_prediction_archive_rejects_wrong_fold_duplicate_and_outside_set():
    result = audit_quality(_tables(), {"behavior": ["b"], "nir": ["n"]})
    sets, _ = build_analysis_sets(
        result["formal_probe_identity"],
        result["probe_feature_status"],
        {"M0_vs_M1_nir": {"required_features": {"behavior": ["b"], "nir": ["n"]}}},
    )
    predictions = _prediction_rows(sets)
    wrong = predictions.copy()
    wrong.loc[0, "outer_fold_group"] = "other"
    with pytest.raises(ValueError, match="outer_fold_group"):
        validate_prediction_archive(wrong, sets)
    dup = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_prediction_archive(dup, sets)
    outside = predictions.copy()
    excluded = sets[~sets.included_complete].iloc[0]
    outside.loc[0, ["session_id", "participant_group_id", "block_id", "probe_index_in_block"]] = [excluded.session_id, excluded.participant_group_id, excluded.block_id, excluded.probe_index_in_block]
    with pytest.raises(ValueError, match="outside requested"):
        validate_prediction_archive(outside, sets, require_complete=False)


def test_finite_value_does_not_prove_unverified_source():
    t = _tables()
    t["rgb"] = t["rgb"].drop(columns=["source_present", "source_readable"])
    result = audit_quality(t, {"behavior": ["b"], "rgb": ["r"]})
    status = result["probe_feature_status"].query("modality == 'rgb'")
    assert status.feature_computable.sum() == 0
    assert set(status.missing_kind) == {"structural_source_missing"}
    availability = result["modality_availability"].query("modality == 'rgb'").iloc[0]
    assert availability.modality_availability_rate == 0


def test_native_qc_failure_is_not_imputation_scope():
    t = _tables()
    t["nir"]["native_qc_valid"] = True
    t["nir"].loc[0, "native_qc_valid"] = False
    result = audit_quality(t, {"behavior": ["b"], "nir": ["n"]})
    sets, _ = build_analysis_sets(
        result["formal_probe_identity"],
        result["probe_feature_status"],
        {"x": {"required_features": {"behavior": ["b"], "nir": ["n"]}}},
    )
    row = sets[(sets.session_id == "s1") & (sets.probe_index_in_block == 1)].iloc[0]
    assert not bool(row.included_missing_aware)
    assert "native_qc_invalid" in row.missing_aware_exclusion_reason
