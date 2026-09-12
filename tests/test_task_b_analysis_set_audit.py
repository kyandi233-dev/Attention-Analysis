import numpy as np
import pandas as pd
import pytest

from attention_pipeline.multimodal_formal.quality_admission import audit_quality
from attention_pipeline.multimodal_formal.analysis_sets import build_analysis_sets
from attention_pipeline.multimodal_formal.prediction_archive import (
    normalize_task_a_predictions,
    validate_prediction_archive,
)


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
    status = result["probe_feature_status"].query("modality == 'nir'").sort_values(
        ["session_id", "probe_index_in_block"]
    )
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
    sets, summary = build_analysis_sets(
        result["formal_probe_identity"], result["probe_feature_status"], specs
    )
    nir_set = sets[sets.analysis_set_id == "M0_vs_M1_nir"]
    rgb_set = sets[sets.analysis_set_id == "M0_vs_rgb"]
    assert nir_set.included_complete.sum() == 3
    assert nir_set.included_missing_aware.sum() == 4
    assert rgb_set.included_complete.sum() == 1
    assert (
        summary.query(
            "analysis_set_id == 'M0_vs_M1_nir' and membership == 'included_complete'"
        ).probe_n.iloc[0]
        == 3
    )


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


def _prediction_sets():
    result = audit_quality(_tables(), {"behavior": ["b"], "nir": ["n"]})
    sets, _ = build_analysis_sets(
        result["formal_probe_identity"],
        result["probe_feature_status"],
        {
            "M0_vs_M1_nir": {
                "models": ["M0", "M1"],
                "required_features": {"behavior": ["b"], "nir": ["n"]},
            }
        },
    )
    return sets


def test_prediction_archive_one_row_per_probe_and_complete_coverage():
    sets = _prediction_sets()
    predictions = _prediction_rows(sets)
    audit = validate_prediction_archive(predictions, sets)
    assert audit["status"] == "PASS_PREDICTION_ARCHIVE"
    assert audit["prediction_n"] == 6
    bad = predictions.iloc[:-1].copy()
    with pytest.raises(ValueError, match="incomplete prediction coverage"):
        validate_prediction_archive(bad, sets)


def test_prediction_archive_rejects_entire_missing_declared_model():
    sets = _prediction_sets()
    predictions = _prediction_rows(sets)
    only_m0 = predictions[predictions.model_id == "M0"].copy()
    with pytest.raises(ValueError, match="incomplete prediction coverage"):
        validate_prediction_archive(only_m0, sets)


def test_prediction_archive_rejects_wrong_fold_duplicate_and_outside_set():
    sets = _prediction_sets()
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
    outside.loc[
        0,
        ["session_id", "participant_group_id", "block_id", "probe_index_in_block"],
    ] = [
        excluded.session_id,
        excluded.participant_group_id,
        excluded.block_id,
        excluded.probe_index_in_block,
    ]
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


def test_behavior_explicit_nonestimable_sdt_is_not_residual_missingness():
    t = _tables()
    t["behavior"]["dprime_loglinear"] = [np.nan, 0.2, 0.3, 0.4]
    t["behavior"]["sdt_status"] = [
        "not_estimable_low_opportunity",
        "estimable",
        "estimable",
        "estimable",
    ]
    result = audit_quality(t, {"behavior": ["dprime_loglinear"]})
    status = result["probe_feature_status"].query("modality == 'behavior'")
    first = status[(status.session_id == "s1") & (status.probe_index_in_block == 1)].iloc[0]
    assert first.missing_kind == "feature_support_invalid"
    assert not bool(first.feature_support_valid)
    assert not bool(first.eligible_for_missing_strategy)


def test_behavior_rt_cv_support_does_not_reintroduce_n20_gate():
    t = _tables()
    t["behavior"]["correct_go_rt_opportunities"] = [2, 1, 2, 2]
    t["behavior"]["go_correct_rt_cv"] = [0.1, np.nan, 0.2, 0.3]
    result = audit_quality(t, {"behavior": ["go_correct_rt_cv"]})
    status = result["probe_feature_status"].query("modality == 'behavior'")
    two = status[(status.session_id == "s1") & (status.probe_index_in_block == 1)].iloc[0]
    one = status[(status.session_id == "s1") & (status.probe_index_in_block == 2)].iloc[0]
    assert bool(two.feature_support_valid)
    assert bool(two.feature_computable)
    assert two.feature_support_evidence == "correct_go_rt_opportunities>=2"
    assert one.missing_kind == "feature_support_invalid"
    assert not bool(one.eligible_for_missing_strategy)


def test_mmwave_qc_fail_is_not_missing_strategy_scope():
    t = _tables()
    mmwave = t["behavior"][[
        "session_id", "participant_group_id", "block_id", "probe_index_in_block"
    ]].copy()
    mmwave["mmwave_observed"] = True
    mmwave["mmwave_loadable"] = True
    mmwave["mmwave_state"] = ["QC_FAIL", "OBSERVED", "OBSERVED", "OBSERVED"]
    mmwave["window_name"] = "pre_30s"
    mmwave["m"] = [1.0, 2.0, 3.0, 4.0]
    t["mmwave"] = mmwave
    result = audit_quality(t, {"behavior": ["b"], "mmwave": ["m"]})
    status = result["probe_feature_status"].query("modality == 'mmwave'")
    first = status[(status.session_id == "s1") & (status.probe_index_in_block == 1)].iloc[0]
    assert first.missing_kind == "native_qc_invalid"
    assert not bool(first.native_qc_valid)
    assert not bool(first.eligible_for_missing_strategy)


def test_nir_no_valid_pupil_samples_are_explicit_support_failure():
    t = _tables()
    t["nir"]["n_nir_rows"] = [10, 10, 10, 10]
    t["nir"]["n_pupil_valid"] = [0, 2, 3, 4]
    t["nir"]["pupil_mean"] = [np.nan, 3.1, 3.2, 3.3]
    result = audit_quality(t, {"behavior": ["b"], "nir": ["pupil_mean"]})
    status = result["probe_feature_status"].query("modality == 'nir'")
    first = status[(status.session_id == "s1") & (status.probe_index_in_block == 1)].iloc[0]
    assert first.missing_kind == "feature_support_invalid"
    assert first.feature_support_evidence == "n_pupil_valid>=1:producer_math_support"
    assert not bool(first.eligible_for_missing_strategy)


def _task_a_native_predictions(sets):
    included = sets[sets.included_complete].copy()
    rows = []
    for model in ["M0", "M1"]:
        for _, r in included.iterrows():
            rows.append({
                "session_id": r.session_id,
                "participant_group_id": r.participant_group_id,
                "block_id": r.block_id,
                "probe_event_id": (
                    f"{r.session_id}|{r.block_id}|probe|{int(r.probe_index_in_block)}"
                ),
                "analysis_set_id": r.analysis_set_id,
                "outer_fold_group": r.participant_group_id,
                "model_id": model,
                "q1_binary": 1,
                "predicted_q1_binary": 1,
                "p_q1_equals_1": 0.75,
                "model_failed": False,
                "failure_reason": "",
            })
    return pd.DataFrame(rows)


def test_current_task_a_prediction_schema_normalizes_and_validates():
    sets = _prediction_sets()
    native = _task_a_native_predictions(sets)
    normalized = normalize_task_a_predictions(native)
    assert normalized.probe_index_in_block.notna().all()
    assert set(normalized.outcome) == {"q1_equals_1_vs_2_3_4"}
    audit = validate_prediction_archive(normalized, sets)
    assert audit["prediction_n"] == 6
    assert audit["failed_prediction_n"] == 0


def test_task_a_failed_rows_remain_auditable_without_fabricated_predictions():
    sets = _prediction_sets()
    native = _task_a_native_predictions(sets)
    native.loc[0, "model_failed"] = True
    native.loc[0, "failure_reason"] = "ModelSelectionError: synthetic failure"
    native.loc[0, "predicted_q1_binary"] = pd.NA
    native.loc[0, "p_q1_equals_1"] = np.nan
    normalized = normalize_task_a_predictions(native)
    audit = validate_prediction_archive(normalized, sets)
    assert audit["failed_prediction_n"] == 1
    assert audit["successful_prediction_n"] == 5
    bad = normalized.copy()
    bad.loc[0, "probability_positive"] = 0.5
    with pytest.raises(ValueError, match="fabricated"):
        validate_prediction_archive(bad, sets)
