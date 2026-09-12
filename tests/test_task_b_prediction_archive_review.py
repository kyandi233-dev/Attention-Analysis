from __future__ import annotations

import json

import pandas as pd
import pytest

from attention_pipeline.multimodal_formal.prediction_archive import (
    TASK_A_Q1_OUTCOME,
    validate_prediction_archive,
)


def _sets() -> pd.DataFrame:
    rows = []
    for participant, session, included, q1 in (
        ("p1", "s1", True, 1),
        ("p2", "s2", False, 2),
    ):
        rows.append(
            {
                "session_id": session,
                "participant_group_id": participant,
                "block_id": "b1",
                "probe_index_in_block": 1,
                "analysis_set_id": "set-a",
                "comparison_models": json.dumps(["M0"]),
                "required_outcomes": json.dumps(["q1_nominal_4class"]),
                "q1_nominal_4class": q1,
                "included_complete": "True" if included else "False",
                "included_missing_aware": "True" if included else "False",
            }
        )
    return pd.DataFrame(rows)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "feature_set_id": "M0",
                "membership_type": "included_complete",
                "model_failed": False,
                "failure_reason": "",
                "session_id": "s1",
                "participant_group_id": "p1",
                "block_id": "b1",
                "probe_index_in_block": 1,
                "analysis_set_id": "set-a",
                "outer_fold_group": "p1",
                "outcome": TASK_A_Q1_OUTCOME,
                "model_id": "M0",
                "y_true": 1,
                "y_pred": 1,
                "probability_positive": 0.8,
            }
        ]
    )


def test_serialized_false_membership_is_not_treated_as_true() -> None:
    audit = validate_prediction_archive(
        _predictions(),
        _sets(),
        membership_column="included_complete",
        requested_analysis_set_ids=["set-a"],
    )
    coverage = pd.DataFrame(audit["coverage"])
    assert coverage.loc[0, "expected_probe_n"] == 1
    assert coverage.loc[0, "predicted_probe_n"] == 1


def test_archive_rejects_mixed_run_ids() -> None:
    predictions = pd.concat([_predictions(), _predictions()], ignore_index=True)
    predictions.loc[1, "run_id"] = "run-2"
    predictions.loc[1, "session_id"] = "s-extra"
    predictions.loc[1, "participant_group_id"] = "p-extra"
    predictions.loc[1, "outer_fold_group"] = "p-extra"
    with pytest.raises(ValueError, match="exactly one run_id"):
        validate_prediction_archive(
            predictions,
            _sets(),
            membership_column="included_complete",
            requested_analysis_set_ids=["set-a"],
        )


def test_invalid_serialized_membership_fails_closed() -> None:
    sets = _sets()
    sets.loc[0, "included_complete"] = "not-a-bool"
    with pytest.raises(ValueError, match="invalid boolean"):
        validate_prediction_archive(
            _predictions(),
            sets,
            membership_column="included_complete",
            requested_analysis_set_ids=["set-a"],
        )
