from __future__ import annotations

import pandas as pd
import pytest

from attention_pipeline.supervised_learning.trajectory import (
    build_probe_trajectory,
    session_discrimination_audit,
    summarize_session_discrimination,
)


def _prediction_rows() -> pd.DataFrame:
    rows = []
    specs = {
        "S-mixed": [(1000, 1, 4, 0.9), (2000, 2, 3, 0.2), (3000, 1, 4, 0.8)],
        "S-single": [(1000, 1, 2, 0.7), (2000, 1, 2, 0.6)],
    }
    for session_id, probes in specs.items():
        for index, (probe_time, q1, q2, probability) in enumerate(probes, start=1):
            rows.append(
                {
                    "run_id": "run-trajectory",
                    "analysis_set_id": "set-a",
                    "membership_type": "included_complete",
                    "model_id": "behavior",
                    "participant_group_id": "P01",
                    "session_id": session_id,
                    "block_id": "B1",
                    "probe_event_id": f"{session_id}|B1|P{index}",
                    "probe_order_in_block": index,
                    "probe_time_ms": probe_time,
                    "q1_nominal_4class": q1,
                    "q1_binary": 1 if q1 == 1 else 0,
                    "q2_ordinal_4level": q2,
                    "p_q1_equals_1": probability,
                    "predicted_q1_binary": int(probability >= 0.5),
                    "model_failed": False,
                    "failure_reason": "",
                }
            )
    return pd.DataFrame(rows)


def test_probe_trajectory_keeps_observed_four_class_q2_and_time_order() -> None:
    source = _prediction_rows().sample(frac=1.0, random_state=7).reset_index(drop=True)
    trajectory = build_probe_trajectory(source)

    assert "q1_nominal_4class" in trajectory.columns
    assert "q2_ordinal_4level" in trajectory.columns
    mixed = trajectory[trajectory["session_id"].eq("S-mixed")]
    assert mixed["probe_time_ms"].tolist() == [1000, 2000, 3000]
    assert mixed["q1_nominal_4class"].tolist() == [1, 2, 1]
    assert mixed["q2_ordinal_4level"].tolist() == [4, 3, 4]


def test_session_auroc_is_estimable_only_when_both_classes_occur() -> None:
    summary = summarize_session_discrimination(_prediction_rows())

    mixed = summary[summary["session_id"].eq("S-mixed")].iloc[0]
    single = summary[summary["session_id"].eq("S-single")].iloc[0]
    assert mixed["status"] == "estimable"
    assert mixed["auroc"] == pytest.approx(1.0)
    assert mixed["n_positive"] == 2
    assert mixed["n_negative"] == 1
    assert single["status"] == "not_estimable_single_class"
    assert pd.isna(single["auroc"])

    audit = session_discrimination_audit(summary)
    assert audit["interpretation"] == "within_session_state_discrimination_not_dynamic_tracking"
    assert audit["session_model_rows_total"] == 2
    assert audit["session_model_rows_estimable"] == 1
    assert audit["session_model_rows_not_estimable_single_class"] == 1
    assert audit["session_model_estimable_fraction"] == pytest.approx(0.5)
    assert audit["participant_model_rows_total"] == 1
    assert audit["participant_model_rows_with_estimable_session"] == 1
    assert audit["participants_total"] == 1
    assert audit["participants_with_estimable_session"] == 1


def test_failed_oof_row_makes_session_auroc_not_estimable() -> None:
    source = _prediction_rows()
    source.loc[source["session_id"].eq("S-mixed") & source["probe_order_in_block"].eq(2), "model_failed"] = True
    source.loc[source["session_id"].eq("S-mixed") & source["probe_order_in_block"].eq(2), "p_q1_equals_1"] = float("nan")
    summary = summarize_session_discrimination(source)
    mixed = summary[summary["session_id"].eq("S-mixed")].iloc[0]
    assert mixed["status"] == "not_estimable_model_failure"
    assert pd.isna(mixed["auroc"])
