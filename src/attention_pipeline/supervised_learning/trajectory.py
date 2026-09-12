"""Exploratory OOF trajectory and within-session discrimination summaries.

These outputs are reporting/construct-comparison layers only. They do not change
training, preprocessing, model selection, or the formal participant-equal outer
estimand. Probe trajectories are discrete probe-sampled sequences, not continuous
real-time attention traces.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .task import Q1_BINARY_SPEC, SupervisedLearningContractError


_TRAJECTORY_CORE = (
    "run_id",
    "analysis_set_id",
    "membership_type",
    "model_id",
    "participant_group_id",
    "session_id",
    "block_id",
    "probe_event_id",
    "q1_nominal_4class",
    "q1_binary",
    Q1_BINARY_SPEC.positive_probability_name,
    "predicted_q1_binary",
    "model_failed",
    "failure_reason",
)
_TRAJECTORY_OPTIONAL = (
    "probe_order_in_block",
    "probe_index_in_block",
    "probe_index_global",
    "probe_time_ms",
    "probe_onset_unix_ms",
    "q2_ordinal_4level",
)
_SESSION_KEY = (
    "run_id",
    "analysis_set_id",
    "membership_type",
    "model_id",
    "participant_group_id",
    "session_id",
)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, context: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise SupervisedLearningContractError(f"{context} missing required columns: {missing}")


def build_probe_trajectory(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return one OOF row per model×probe, ordered within participant/session.

    The table retains observed Q1 four-category reports, the first-round binary Q1
    target/probability, and Q2 when upstream supplied it. Four-class *predicted*
    probabilities are not fabricated by the current binary Task A model.
    """
    _require_columns(predictions, _TRAJECTORY_CORE, context="OOF trajectory archive")
    if predictions.empty:
        raise SupervisedLearningContractError("OOF trajectory archive is empty")

    columns = list(_TRAJECTORY_CORE) + [c for c in _TRAJECTORY_OPTIONAL if c in predictions.columns]
    out = predictions.loc[:, columns].copy()

    sort_columns = ["participant_group_id", "session_id", "model_id"]
    if "probe_time_ms" in out.columns:
        sort_columns.append("probe_time_ms")
    elif "probe_onset_unix_ms" in out.columns:
        sort_columns.append("probe_onset_unix_ms")
    elif "probe_index_global" in out.columns:
        sort_columns.append("probe_index_global")
    elif "probe_order_in_block" in out.columns:
        sort_columns.extend(["block_id", "probe_order_in_block"])
    else:
        sort_columns.extend(["block_id", "probe_event_id"])

    return out.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def summarize_session_discrimination(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute within-session AUROC when both Q1 binary classes are observed.

    AUROC here is a within-session state-discrimination diagnostic. It does not use
    temporal order and must not be described as dynamic tracking performance.
    """
    required = [
        *_SESSION_KEY,
        "q1_binary",
        Q1_BINARY_SPEC.positive_probability_name,
        "model_failed",
    ]
    _require_columns(predictions, required, context="session discrimination archive")
    if predictions.empty:
        raise SupervisedLearningContractError("session discrimination archive is empty")

    rows: list[dict[str, object]] = []
    for key, frame in predictions.groupby(list(_SESSION_KEY), sort=True, dropna=False):
        record = dict(zip(_SESSION_KEY, key, strict=True))
        failed = frame["model_failed"].astype(bool)
        y = pd.to_numeric(frame["q1_binary"], errors="coerce")
        p = pd.to_numeric(frame[Q1_BINARY_SPEC.positive_probability_name], errors="coerce")
        n_positive = int(y.eq(1).sum())
        n_negative = int(y.eq(0).sum())
        record.update(
            {
                "n_probes": int(len(frame)),
                "n_positive": n_positive,
                "n_negative": n_negative,
                "auroc": np.nan,
                "status": "",
                "reason": "",
            }
        )

        if failed.any():
            record["status"] = "not_estimable_model_failure"
            record["reason"] = "one_or_more_oof_rows_failed"
        elif y.isna().any() or not y.isin([0, 1]).all():
            record["status"] = "not_estimable_invalid_label"
            record["reason"] = "q1_binary_not_complete_0_1"
        elif y.nunique() < 2:
            record["status"] = "not_estimable_single_class"
            record["reason"] = "session_contains_only_one_q1_binary_class"
        elif p.isna().any() or ((p < 0.0) | (p > 1.0)).any():
            record["status"] = "not_estimable_invalid_probability"
            record["reason"] = "oof_probability_missing_or_out_of_range"
        else:
            record["auroc"] = float(roc_auc_score(y.astype(int), p.astype(float)))
            record["status"] = "estimable"
            record["reason"] = ""
        rows.append(record)

    return pd.DataFrame(rows)


def session_discrimination_audit(summary: pd.DataFrame) -> dict[str, object]:
    """Return denominator-aware session and participant counts for AUROC availability."""
    _require_columns(
        summary,
        ["status", "model_id", "participant_group_id", "session_id"],
        context="session discrimination summary",
    )
    total = int(len(summary))
    estimable_mask = summary["status"].eq("estimable")
    estimable = int(estimable_mask.sum())
    single_class = int(summary["status"].eq("not_estimable_single_class").sum())

    participant_model_total = int(
        summary[["model_id", "participant_group_id"]].drop_duplicates().shape[0]
    )
    participant_model_estimable = int(
        summary.loc[estimable_mask, ["model_id", "participant_group_id"]]
        .drop_duplicates()
        .shape[0]
    )
    participant_total = int(summary["participant_group_id"].nunique())
    participant_estimable = int(summary.loc[estimable_mask, "participant_group_id"].nunique())

    return {
        "interpretation": "within_session_state_discrimination_not_dynamic_tracking",
        "session_model_rows_total": total,
        "session_model_rows_estimable": estimable,
        "session_model_rows_not_estimable_single_class": single_class,
        "session_model_estimable_fraction": float(estimable / total) if total else np.nan,
        "participant_model_rows_total": participant_model_total,
        "participant_model_rows_with_estimable_session": participant_model_estimable,
        "participants_total": participant_total,
        "participants_with_estimable_session": participant_estimable,
    }
