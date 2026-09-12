import json

import pandas as pd
import pytest

from attention_pipeline.multimodal_formal.prediction_archive import (
    TASK_A_Q1_OUTCOME,
    validate_prediction_archive,
)


def _analysis_sets() -> pd.DataFrame:
    rows = []
    for set_id in ("set-a", "set-b"):
        for participant, session, q1 in (("p1", "s1", 1), ("p2", "s2", 2)):
            rows.append(
                {
                    "session_id": session,
                    "participant_group_id": participant,
                    "block_id": "b1",
                    "probe_index_in_block": 1,
                    "analysis_set_id": set_id,
                    "comparison_models": json.dumps(["M0", "M1"]),
                    "required_outcomes": json.dumps(["q1_nominal_4class"]),
                    "q1_nominal_4class": q1,
                    "included_complete": True,
                    "included_missing_aware": True,
                }
            )
    return pd.DataFrame(rows)


def _predictions(set_id: str, run_id: str) -> pd.DataFrame:
    rows = []
    for model in ("M0", "M1"):
        for participant, session, q1 in (("p1", "s1", 1), ("p2", "s2", 2)):
            y = int(q1 == 1)
            rows.append(
                {
                    "run_id": run_id,
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
                    "y_true": y,
                    "y_pred": y,
                    "probability_positive": 0.8 if y else 0.2,
                    "model_failed": False,
                    "failure_reason": "",
                }
            )
    return pd.DataFrame(rows)


def test_multi_set_archive_allows_distinct_task_a_run_per_analysis_set() -> None:
    predictions = pd.concat(
        [_predictions("set-a", "run-a"), _predictions("set-b", "run-b")],
        ignore_index=True,
    )
    audit = validate_prediction_archive(predictions, _analysis_sets())

    assert audit["run_ids"] == ["run-a", "run-b"]
    assert audit["run_id_by_analysis_set"] == {"set-a": "run-a", "set-b": "run-b"}
    assert audit["analysis_set_n"] == 2


def test_one_analysis_set_cannot_mix_rows_from_multiple_runs() -> None:
    predictions = _predictions("set-a", "run-a")
    predictions.loc[0, "run_id"] = "run-other"

    with pytest.raises(ValueError, match="analysis_set_id=set-a must contain exactly one run_id"):
        validate_prediction_archive(
            predictions,
            _analysis_sets(),
            requested_analysis_set_ids=["set-a"],
        )
