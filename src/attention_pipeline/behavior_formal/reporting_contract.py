"""Reporting/admission helpers for FocusWave behavior science v3.

This module does not invent a new prediction target.  It defines the minimum
contract that any behavior or multimodal classifier/result must satisfy before
it can be reported by the formal analysis line.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


class ReportingContractError(ValueError):
    pass


def _rank_auc(y: np.ndarray, score: np.ndarray) -> float:
    valid = np.isfinite(score)
    y, score = y[valid].astype(int), score[valid]
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = pd.Series(score).rank(method="average").to_numpy(dtype=float)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision(y: np.ndarray, score: np.ndarray) -> float:
    valid = np.isfinite(score)
    y, score = y[valid].astype(int), score[valid]
    if len(y) == 0 or not np.any(y == 1):
        return math.nan
    order = np.argsort(-score, kind="stable")
    ys = y[order]
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    precision = tp / np.maximum(1, tp + fp)
    return float(np.sum(precision[ys == 1]) / np.sum(ys == 1))


def binary_prediction_report(
    y_true: Sequence[Any],
    *,
    y_pred: Sequence[Any],
    score: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Return imbalance-aware binary metrics including the majority baseline."""
    y = pd.to_numeric(pd.Series(y_true), errors="coerce")
    pred = pd.to_numeric(pd.Series(y_pred), errors="coerce")
    valid = y.isin([0, 1]) & pred.isin([0, 1])
    yv = y[valid].to_numpy(dtype=int)
    pv = pred[valid].to_numpy(dtype=int)
    if len(yv) == 0:
        raise ReportingContractError("binary prediction report has no valid 0/1 rows")
    prevalence = float(np.mean(yv))
    majority_class = int(prevalence >= 0.5)
    majority_accuracy = float(np.mean(yv == majority_class))
    accuracy = float(np.mean(yv == pv))
    sensitivity = float(np.mean(pv[yv == 1] == 1)) if np.any(yv == 1) else math.nan
    specificity = float(np.mean(pv[yv == 0] == 0)) if np.any(yv == 0) else math.nan
    balanced_accuracy = float(np.nanmean([sensitivity, specificity]))
    report: dict[str, Any] = {
        "n": int(len(yv)),
        "positive_prevalence": prevalence,
        "majority_class": majority_class,
        "majority_accuracy": majority_accuracy,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "accuracy_minus_majority": accuracy - majority_accuracy,
        "class_imbalance_warning": bool(max(prevalence, 1.0 - prevalence) >= 0.80),
        "auroc": math.nan,
        "pr_auc": math.nan,
    }
    if score is not None:
        raw = pd.to_numeric(pd.Series(score), errors="coerce")
        if len(raw) != len(y):
            raise ReportingContractError("score length differs from y_true")
        sv = raw.loc[valid].to_numpy(dtype=float)
        report["auroc"] = _rank_auc(yv, sv)
        report["pr_auc"] = _average_precision(yv, sv)
    return report


def assert_participant_disjoint(
    rows: pd.DataFrame,
    *,
    group_col: str = "repeat_participant_id",
    fold_col: str = "fold_id",
) -> None:
    """One participant group may belong to one and only one outer test fold."""
    missing = {group_col, fold_col} - set(rows.columns)
    if missing:
        raise ReportingContractError(f"participant-disjoint audit missing {sorted(missing)}")
    if rows[group_col].isna().any() or rows[fold_col].isna().any():
        raise ReportingContractError("participant-disjoint audit contains missing group/fold values")
    per_group = rows.groupby(group_col)[fold_col].nunique(dropna=False)
    if per_group.gt(1).any():
        raise ReportingContractError("one participant group appears in multiple outer folds")


def forest_panel_keys(
    rows: pd.DataFrame,
    *,
    unit_col: str = "observation_unit",
    estimate_unit_col: str = "estimate_unit",
) -> pd.Series:
    """Return explicit panel keys so incompatible units cannot share one axis."""
    missing = {unit_col, estimate_unit_col} - set(rows.columns)
    if missing:
        raise ReportingContractError(f"forest rows missing {sorted(missing)}")
    if rows[[unit_col, estimate_unit_col]].isna().any().any():
        raise ReportingContractError("forest rows require non-missing observation and estimate units")
    return rows[unit_col].astype(str) + " | " + rows[estimate_unit_col].astype(str)


def assert_single_forest_panel(rows: pd.DataFrame) -> str:
    keys = forest_panel_keys(rows)
    unique = sorted(keys.unique())
    if len(unique) != 1:
        raise ReportingContractError(
            "mixed observation/estimate units cannot be plotted on one forest axis: " + "; ".join(unique)
        )
    return unique[0]


def validate_candidate_evidence(rows: pd.DataFrame) -> pd.DataFrame:
    """Reject an unexplained 0/1 candidate matrix; require auditable evidence weights."""
    required = {
        "candidate",
        "evidence_source",
        "evidence_weight",
        "prespecified_rule",
        "status",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ReportingContractError(f"candidate evidence table missing {sorted(missing)}")
    out = rows.copy()
    out["evidence_weight"] = pd.to_numeric(out["evidence_weight"], errors="coerce")
    if out["evidence_weight"].isna().any() or (~np.isfinite(out["evidence_weight"])).any():
        raise ReportingContractError("candidate evidence_weight must be finite")
    if out["evidence_source"].astype(str).str.strip().eq("").any():
        raise ReportingContractError("candidate evidence_source cannot be empty")
    if out["prespecified_rule"].astype(str).str.strip().eq("").any():
        raise ReportingContractError("candidate prespecified_rule cannot be empty")
    return out


def qc_count_table(
    counts: Mapping[str, tuple[int, str, int]],
) -> pd.DataFrame:
    """Create QC rows where every count declares its denominator and unit.

    Mapping value is ``(observed_count, denominator_name, denominator_count)``.
    """
    rows: list[dict[str, Any]] = []
    for name, (count, denominator_name, denominator_count) in counts.items():
        count_i = int(count)
        denom_i = int(denominator_count)
        rows.append(
            {
                "qc_axis": str(name),
                "count": count_i,
                "denominator_name": str(denominator_name),
                "denominator_count": denom_i,
                "fraction": (float(count_i / denom_i) if denom_i > 0 else math.nan),
            }
        )
    return pd.DataFrame(rows)


def label_dependency_relationships(
    pairs: Iterable[tuple[str, str]],
    *,
    mathematical_pairs: Iterable[frozenset[str]] = (
        frozenset({"omission_rate", "go_accuracy"}),
        frozenset({"commission_rate", "nogo_correct_inhibition_rate"}),
    ),
) -> pd.DataFrame:
    mathematical = set(mathematical_pairs)
    rows = []
    for left, right in pairs:
        relation = (
            "measurement_mathematical_dependency"
            if frozenset({str(left), str(right)}) in mathematical
            else "empirical_association_requires_interpretation"
        )
        rows.append({"left_metric": left, "right_metric": right, "relationship_role": relation})
    return pd.DataFrame(rows)
