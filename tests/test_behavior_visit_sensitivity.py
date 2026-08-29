from __future__ import annotations

import pandas as pd

from attention_pipeline.behavior_formal.visit_sensitivity import (
    build_visit_sensitivity_status,
    subset_verified_first_visit,
)


def _session() -> pd.DataFrame:
    return pd.DataFrame([
        {"session_id": "s1", "participant_key": "P1", "visit_order": 1, "prior_same_stage_count": 0},
        {"session_id": "s2", "participant_key": "P1", "visit_order": 2, "prior_same_stage_count": 1},
        {"session_id": "s3", "participant_key": "P2", "visit_order": 2, "prior_same_stage_count": 0},
        {"session_id": "s4", "participant_key": pd.NA, "visit_order": pd.NA, "prior_same_stage_count": pd.NA},
    ])


def test_first_any_visit_and_first_same_stage_visit_are_not_conflated() -> None:
    frame = _session()
    first_any = subset_verified_first_visit(frame, same_stage=False)
    first_stage = subset_verified_first_visit(frame, same_stage=True)
    assert set(first_any["session_id"]) == {"s1"}
    assert set(first_stage["session_id"]) == {"s1", "s3"}


def test_partial_missing_questionnaire_identity_is_complete_case_sensitivity_not_imputed() -> None:
    status = build_visit_sensitivity_status(_session())
    first = status.loc[status["analysis"].eq("first_any_visit_only")].iloc[0]
    adjusted = status.loc[status["analysis"].eq("visit_order_adjusted")].iloc[0]
    assert first["status"] == "ready_complete_case_only"
    assert adjusted["status"] == "ready_complete_case_only"
    assert first["n_rows_with_verified_order"] == 3
