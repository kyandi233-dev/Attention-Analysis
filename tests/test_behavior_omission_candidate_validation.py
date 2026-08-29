from __future__ import annotations

import numpy as np
import pandas as pd

from attention_pipeline.behavior_formal.behavior_error_taxonomy import TAXONOMY_RATE_METRICS
from attention_pipeline.behavior_formal.omission_candidate_validation import validate_omission_candidates


def _frame() -> pd.DataFrame:
    rows = []
    for participant, base in (("P1", 0.02), ("P2", 0.08), ("P3", 0.14)):
        for visit in (1, 2):
            row = {
                "repeat_participant_id": participant,
                "session_id": f"{participant}-s{visit}",
                "block_id": "B1",
            }
            for i, metric in enumerate(TAXONOMY_RATE_METRICS):
                row[metric] = base + 0.01 * visit + 0.002 * i
            rows.append(row)
    return pd.DataFrame(rows)


def test_omission_candidates_get_within_between_and_no_pvalue_freeze() -> None:
    frame = _frame()
    validation, redundancy = validate_omission_candidates(
        {"session": frame, "block": frame.copy(), "cycle": frame.copy()},
        frame.copy(),
    )
    session = validation[validation["scale"].eq("session")]
    assert set(session["metric"]) == set(TAXONOMY_RATE_METRICS)
    assert session["between_participant_variance"].notna().all()
    assert session["within_participant_variance"].notna().all()
    assert session["endpoint_status"].eq("pending_task_semantics_and_real_data_review").all()
    assert session["selection_contract"].str.contains("never outcome p-value", regex=False).all()
    assert not redundancy.empty
    assert redundancy["automatic_drop_allowed"].eq(False).all()


def test_floor_effect_is_a_review_flag_not_automatic_exclusion() -> None:
    frame = _frame()
    frame[TAXONOMY_RATE_METRICS[0]] = 0.0
    validation, _ = validate_omission_candidates({"session": frame}, frame.iloc[0:0].copy())
    row = validation[(validation["scale"].eq("session")) & (validation["metric"].eq(TAXONOMY_RATE_METRICS[0]))].iloc[0]
    assert np.isclose(row["floor_fraction"], 1.0)
    assert "strong_floor_effect" in row["candidate_reasons"]
    assert row["endpoint_status"] == "pending_task_semantics_and_real_data_review"
