import json

import pandas as pd
import pytest

from attention_pipeline.multimodal_formal.prediction_archive import (
    TASK_A_Q1_OUTCOME,
    normalize_task_a_predictions,
    validate_prediction_archive,
)


def _analysis_sets() -> pd.DataFrame:
    rows = []
    for set_id in ("set-a", "set-b"):
        for participant, session, probe, q1 in (
            ("p1", "s1", 1, 1),
            ("p2", "s2", 1, 2),
        ):
            rows.append(
                {
                    "session_id": session,
                    "participant_group_id": participant,
                    "block_id": "b1",
                    "probe_index_in_block": probe,
                    "analysis_set_id": set_id,
                    "comparison_models": json.dumps(["M0", "M1"]),
                    "required_outcomes": json.dumps(["q1_nominal_4class"]),
                    "q1_nominal_4class": q1,
                    "included_complete": True,
                    "included_missing_aware": True,
                }
            )
    return pd.DataFrame(rows)


def _predictions(set_id: str = "set-a") -> pd.DataFrame:
    rows = []
    for model in ("M0", "M1"):
        for participant, session, q1 in (("p1", "s1", 1), ("p2", "s2", 2)):
            rows.append(
                {
                    "run_id": "run-1",
                    "feature_set_id": model,
                    "membership_type": "included_complete",
                    "session_id": session,
                    "participant_group_id": participant,
                    "block_id": "b1",
                    "probe_index_in_block": 1,
                    "analysis_set_id": set_id,
                    "outer_fold_group": participant,
                    "outcome": TASK_A_Q1_OUTCOME,
                    "model_id": model,
                    "y_true": 1 if q1 == 1 else 0,
                    "y_pred": 1 if q1 == 1 else 0,
                    "probability_positive": 0.8 if q1 == 1 else 0.2,
                    "model_failed": False,
                    "failure_reason": "",
                }
            )
    return pd.DataFrame(rows)


def _task_a_native_predictions() -> pd.DataFrame:
    rows = []
    for model in ("M0", "M1"):
        for participant, session, q1 in (("p1", "s1", 1), ("p2", "s2", 2)):
            failed = model == "M1"
            rows.append(
                {
                    "run_id": "task-a-run-001",
                    "feature_set_id": None if failed else "behavior-reference-v1",
                    "membership_type": "included_complete",
                    "session_id": session,
                    "participant_group_id": participant,
                    "block_id": "b1",
                    "probe_order_in_block": 1,
                    "probe_event_id": f"{session}|b1|probe|1",
                    "analysis_set_id": "set-a",
                    "outer_fold_group": participant,
                    "model_id": model,
                    "q1_nominal_4class": q1,
                    "q1_binary": 1 if q1 == 1 else 0,
                    "predicted_q1_binary": pd.NA if failed else (1 if q1 == 1 else 0),
                    "p_q1_equals_1": float("nan") if failed else (0.8 if q1 == 1 else 0.2),
                    "model_failed": failed,
                    "failure_reason": "synthetic failed fold" if failed else "",
                }
            )
    return pd.DataFrame(rows)


def test_archive_rejects_y_true_that_disagrees_with_behavior_authority():
    sets = _analysis_sets()
    predictions = _predictions()
    predictions.loc[0, "y_true"] = 0
    with pytest.raises(ValueError, match="Behavior-authoritative Q1"):
        validate_prediction_archive(
            predictions,
            sets,
            requested_analysis_set_ids=["set-a"],
        )


def test_archive_expected_scope_starts_from_analysis_sets_not_observed_predictions():
    sets = _analysis_sets()
    predictions = _predictions("set-a")
    with pytest.raises(ValueError, match="incomplete prediction coverage"):
        validate_prediction_archive(predictions, sets)

    audit = validate_prediction_archive(
        predictions,
        sets,
        requested_analysis_set_ids=["set-a"],
    )
    assert audit["status"] == "PASS_PREDICTION_ARCHIVE"
    assert audit["requested_analysis_set_ids"] == ["set-a"]
    assert audit["authoritative_q1_checked"] is True
    assert len(audit["coverage"]) == 2


def test_archive_rejects_empty_predictions_and_entire_missing_model():
    sets = _analysis_sets()
    with pytest.raises(ValueError, match="archive is empty"):
        validate_prediction_archive(pd.DataFrame(), sets)

    predictions = _predictions()
    predictions = predictions[predictions["model_id"].eq("M0")].copy()
    with pytest.raises(ValueError, match="incomplete prediction coverage"):
        validate_prediction_archive(
            predictions,
            sets,
            requested_analysis_set_ids=["set-a"],
        )


def test_archive_rejects_membership_type_that_does_not_match_requested_membership():
    sets = _analysis_sets()
    predictions = _predictions()
    predictions["membership_type"] = "included_missing_aware"
    with pytest.raises(ValueError, match="membership_type"):
        validate_prediction_archive(
            predictions,
            sets,
            membership_column="included_complete",
            requested_analysis_set_ids=["set-a"],
        )


def test_archive_rejects_cross_set_authority_disagreement_for_same_probe():
    sets = _analysis_sets()
    mask = (
        sets["analysis_set_id"].eq("set-b")
        & sets["participant_group_id"].eq("p1")
    )
    sets.loc[mask, "q1_nominal_4class"] = 3
    with pytest.raises(ValueError, match="authoritative Q1 disagrees"):
        validate_prediction_archive(
            _predictions(),
            sets,
            requested_analysis_set_ids=["set-a", "set-b"],
        )


def test_task_a_adapter_preserves_audit_fields_and_failed_rows_end_to_end():
    native = _task_a_native_predictions()
    normalized = normalize_task_a_predictions(native)

    for column in ("run_id", "feature_set_id", "membership_type", "model_failed", "failure_reason"):
        assert column in normalized.columns
    assert set(normalized["run_id"]) == {"task-a-run-001"}
    assert set(normalized["membership_type"]) == {"included_complete"}
    assert normalized.loc[normalized["model_id"].eq("M0"), "feature_set_id"].eq(
        "behavior-reference-v1"
    ).all()

    failed = normalized[normalized["model_id"].eq("M1")]
    assert failed["model_failed"].all()
    assert failed["y_pred"].isna().all()
    assert failed["probability_positive"].isna().all()
    assert failed["failure_reason"].eq("synthetic failed fold").all()

    audit = validate_prediction_archive(
        normalized,
        _analysis_sets(),
        requested_analysis_set_ids=["set-a"],
    )
    assert audit["status"] == "PASS_PREDICTION_ARCHIVE"
    assert audit["failed_prediction_n"] == 2
    coverage = pd.DataFrame(audit["coverage"])
    assert coverage.loc[coverage["model_id"].eq("M1"), "failed_probe_n"].iloc[0] == 2
