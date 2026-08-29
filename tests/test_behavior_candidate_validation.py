from __future__ import annotations

import numpy as np
import pandas as pd

from attention_pipeline.behavior_formal.candidate_validation import (
    FORMAL_BEHAVIOR_ENDPOINT_METRICS,
    build_candidate_validation,
    build_sensitivity_status,
    decompose_within_between,
)


def _frame() -> pd.DataFrame:
    rows = []
    for p in ("p1", "p2", "p3"):
        for session_i in (1, 2):
            base = {"p1": 400.0, "p2": 500.0, "p3": 600.0}[p]
            raw = 0.02 * session_i
            clean = raw * 0.6
            ambiguous = raw - clean
            rows.append(
                {
                    "repeat_participant_id": p,
                    "session_id": f"{p}-s{session_i}",
                    "block_id": "B1",
                    "cycle_bin": session_i,
                    "go_correct_rt_mean_ms": base + session_i * 10,
                    "go_correct_rt_median_ms": base + session_i * 10,
                    "go_correct_rt_sd_ms": 40 + session_i,
                    "go_correct_rt_mad_ms": 25 + session_i,
                    "go_correct_rt_iqr_ms": 50 + session_i,
                    "go_correct_rt_cv": (40 + session_i) / (base + session_i * 10),
                    "go_correct_rt_theilsen_slope_ms_per_s": float(session_i),
                    "omission_rate": raw,  # compatibility alias only
                    "raw_go_omission_rate": raw,
                    "clean_go_omission_rate": clean,
                    "timing_ambiguous_go_omission_rate": ambiguous,
                    "commission_rate": 0.10 + 0.01 * session_i,
                    "dprime_loglinear": 2.0 - 0.1 * session_i,
                    "criterion_c": 0.1 * session_i,
                    "beta": 1.1 + 0.1 * session_i,
                }
            )
    return pd.DataFrame(rows)


def test_within_between_decomposition_is_participant_centered() -> None:
    frame = _frame()
    out = decompose_within_between(frame, ["go_correct_rt_median_ms"])
    centered = "go_correct_rt_median_ms__within_participant"
    assert np.allclose(out.groupby("repeat_participant_id")[centered].sum().to_numpy(), 0.0)
    assert out["go_correct_rt_median_ms__participant_mean"].notna().all()


def test_candidate_validation_emits_coverage_redundancy_and_pending_freeze() -> None:
    frame = _frame()
    validation, redundancy, decisions = build_candidate_validation(
        {"session": frame, "block": frame.copy()}, frame.copy()
    )
    assert {"coverage", "between_participant_variance", "within_participant_variance"}.issubset(validation.columns)
    pair = redundancy[
        ((redundancy["metric_a"] == "go_correct_rt_mean_ms") & (redundancy["metric_b"] == "go_correct_rt_median_ms"))
        | ((redundancy["metric_b"] == "go_correct_rt_mean_ms") & (redundancy["metric_a"] == "go_correct_rt_median_ms"))
    ]
    assert not pair.empty
    assert pair["redundant_flag"].all()
    assert decisions["selection_rule"].str.contains("never p-value", regex=False).all()
    formal_omission = decisions[decisions["metric"].isin({
        "raw_go_omission_rate", "clean_go_omission_rate", "timing_ambiguous_go_omission_rate"
    })]
    assert formal_omission["final_endpoint_freeze_status"].eq(
        "prespecified_formal_endpoint_pending_real_data_stability_review"
    ).all()


def test_legacy_omission_rate_is_compatibility_alias_not_second_formal_endpoint() -> None:
    assert "omission_rate" not in FORMAL_BEHAVIOR_ENDPOINT_METRICS
    assert "raw_go_omission_rate" in FORMAL_BEHAVIOR_ENDPOINT_METRICS
    validation, _, decisions = build_candidate_validation(
        {"session": _frame()}, _frame().iloc[0:0].copy()
    )
    assert "omission_rate" not in set(validation["metric"])
    assert "omission_rate" not in set(decisions["metric"])


def test_visit_sensitivity_fails_closed_without_verified_order() -> None:
    status = build_sensitivity_status(_frame())
    first = status.loc[status["analysis"] == "first_session_only"].iloc[0]
    assert first["status"] == "not_estimable"
    assert "infer from session_id" in first["reason"]


def test_visit_sensitivity_allows_verified_order_only() -> None:
    frame = _frame()
    frame["visit_order"] = frame.groupby("repeat_participant_id").cumcount() + 1
    status = build_sensitivity_status(frame)
    assert status.loc[status["analysis"] == "first_session_only", "status"].iloc[0] == "ready"
    assert status.loc[status["analysis"] == "visit_order_adjusted", "status"].iloc[0] == "ready"
