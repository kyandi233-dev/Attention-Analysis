"""Descriptive audit for formal SART behavior candidate metrics.

This module reports full-cohort coverage, distribution, within/between variance,
time association and pairwise redundancy. These summaries do not authorize
full-cohort feature deletion or admission for supervised learning. Any empirical
selection must occur inside the relevant training boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .behavior_error_taxonomy import CURRENT_PRIMARY_OMISSION_ENDPOINT_METRICS
from .science_v3 import CANONICAL_METRICS


# ``omission_rate`` is retained in legacy tables only as a compatibility alias
# of ``raw_go_omission_rate``. Clean/timing omission are audited separately by
# omission_candidate_validation and are not current primary behavior endpoints.
_BASE_FORMAL_METRICS = tuple(m for m in CANONICAL_METRICS if m != "omission_rate")
FORMAL_BEHAVIOR_ENDPOINT_METRICS = tuple(dict.fromkeys(
    (*_BASE_FORMAL_METRICS, *CURRENT_PRIMARY_OMISSION_ENDPOINT_METRICS)
))


@dataclass(frozen=True)
class CandidateValidationConfig:
    # Historical/descriptive references only. They must never act as automatic
    # scientific inclusion/exclusion gates on the full formal cohort.
    min_coverage: float = 0.80
    redundancy_abs_r: float = 0.90
    rate_floor_ceiling_threshold: float = 0.95
    minimum_unique_values: int = 3


_PRIORITY = (
    "go_correct_rt_median_ms",
    "go_correct_rt_cv",
    "go_correct_rt_theilsen_slope_ms_per_s",
    "raw_go_omission_rate",
    "commission_rate",
    "dprime_loglinear",
    "criterion_c",
    "beta",
    "go_correct_rt_mean_ms",
    "go_correct_rt_mad_ms",
    "go_correct_rt_iqr_ms",
    "go_correct_rt_sd_ms",
)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _participant_column(frame: pd.DataFrame) -> str:
    """Prefer the canonical participant grouping interface for descriptive audits."""
    for column in ("participant_group_id", "repeat_participant_id", "participant_key"):
        if column in frame.columns:
            return column
    raise ValueError("descriptive behavior audit requires participant_group_id or a verified compatibility identity")


def decompose_within_between(
    frame: pd.DataFrame,
    metrics: Iterable[str],
    *,
    participant_col: str | None = None,
) -> pd.DataFrame:
    """Add participant mean and within-participant deviation for explanation/audit."""
    participant_col = participant_col or _participant_column(frame)
    if participant_col not in frame:
        raise ValueError(f"missing participant column: {participant_col}")
    out = frame.copy()
    if out[participant_col].isna().any():
        raise ValueError(f"{participant_col} contains missing values")
    for metric in metrics:
        x = _numeric(out, metric)
        mean_col = f"{metric}__participant_mean"
        within_col = f"{metric}__within_participant"
        out[mean_col] = x.groupby(out[participant_col]).transform("mean")
        out[within_col] = x - out[mean_col]
    return out


def _time_axis(frame: pd.DataFrame) -> tuple[str | None, pd.Series | None]:
    for column in ("cycle_bin", "probe_order_in_block", "time_in_block_sec"):
        if column in frame and _numeric(frame, column).notna().sum() >= 3:
            return column, _numeric(frame, column)
    return None, None


def _metric_row(
    frame: pd.DataFrame,
    *,
    scale: str,
    metric: str,
    cfg: CandidateValidationConfig,
) -> dict[str, object]:
    x = _numeric(frame, metric)
    finite = x[np.isfinite(x)]
    n_total = int(len(frame))
    n_valid = int(len(finite))
    participant_col = _participant_column(frame)
    participants = int(frame.loc[x.notna(), participant_col].nunique())
    sessions = int(frame.loc[x.notna(), "session_id"].nunique()) if "session_id" in frame else 0
    unique_n = int(finite.nunique())

    between_var = math.nan
    within_var = math.nan
    if n_valid:
        tmp = pd.DataFrame({"participant": frame[participant_col], "x": x}).dropna()
        pmeans = tmp.groupby("participant")["x"].mean()
        between_var = float(pmeans.var(ddof=1)) if len(pmeans) >= 2 else math.nan
        centered = tmp["x"] - tmp.groupby("participant")["x"].transform("mean")
        informative = tmp.groupby("participant")["x"].transform("count").ge(2)
        centered = centered[informative]
        within_var = float(centered.var(ddof=1)) if len(centered) >= 2 else math.nan

    floor_fraction = math.nan
    ceiling_fraction = math.nan
    if metric.endswith("_rate") and n_valid:
        floor_fraction = float((finite <= 0.02).mean())
        ceiling_fraction = float((finite >= 0.98).mean())

    time_column, t = _time_axis(frame)
    time_rho = math.nan
    if t is not None:
        mask = x.notna() & t.notna()
        if int(mask.sum()) >= 3 and x[mask].nunique() >= 2 and t[mask].nunique() >= 2:
            time_rho = float(x[mask].rank().corr(t[mask].rank()))

    coverage = float(n_valid / n_total) if n_total else 0.0
    review_flags: list[str] = []
    if coverage < cfg.min_coverage:
        review_flags.append("below_historical_80pct_coverage_reference")
    if unique_n < cfg.minimum_unique_values:
        review_flags.append("low_unique_values")
    if np.isfinite(floor_fraction) and floor_fraction >= cfg.rate_floor_ceiling_threshold:
        review_flags.append("floor_dominated")
    if np.isfinite(ceiling_fraction) and ceiling_fraction >= cfg.rate_floor_ceiling_threshold:
        review_flags.append("ceiling_dominated")
    if n_valid == 0:
        review_flags.append("not_computable")

    return {
        "scale": scale,
        "metric": metric,
        "n_rows": n_total,
        "n_valid": n_valid,
        "coverage": coverage,
        "below_historical_80pct_coverage_reference": bool(coverage < cfg.min_coverage),
        "participant_group_n": participants,
        "participant_group_column": participant_col,
        "session_n": sessions,
        "unique_value_n": unique_n,
        "mean": float(finite.mean()) if n_valid else math.nan,
        "sd": float(finite.std(ddof=1)) if n_valid >= 2 else math.nan,
        "q05": float(finite.quantile(0.05)) if n_valid else math.nan,
        "median": float(finite.median()) if n_valid else math.nan,
        "q95": float(finite.quantile(0.95)) if n_valid else math.nan,
        "floor_fraction": floor_fraction,
        "ceiling_fraction": ceiling_fraction,
        "between_participant_variance": between_var,
        "within_participant_variance": within_var,
        "time_axis": time_column,
        "spearman_like_time_rho": time_rho,
        "admission_status": "not_computable" if n_valid == 0 else "descriptive_audit_only",
        "admission_reasons": ";".join(review_flags) if review_flags else "",
        "review_flags": ";".join(review_flags) if review_flags else "",
        "selection_authority": "descriptive_only",
        "automatic_drop_allowed": False,
        "endpoint_role": (
            "current_primary_omission_endpoint"
            if metric in CURRENT_PRIMARY_OMISSION_ENDPOINT_METRICS
            else "formal_behavior_candidate"
        ),
        "decision_basis": (
            "full-cohort coverage/distribution/within-between/time-trend are descriptive only; "
            "empirical feature selection must occur inside the relevant training boundary"
        ),
    }


def build_candidate_validation(
    scale_tables: Mapping[str, pd.DataFrame],
    primary_probe: pd.DataFrame,
    *,
    config: CandidateValidationConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return descriptive validation, redundancy, and non-selecting decision records."""
    cfg = config or CandidateValidationConfig()
    frames: dict[str, pd.DataFrame] = {k: v for k, v in scale_tables.items() if v is not None}
    frames["probe"] = primary_probe

    rows: list[dict[str, object]] = []
    redundancy_rows: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []

    for scale, frame in frames.items():
        if frame is None or frame.empty:
            continue
        available = [m for m in FORMAL_BEHAVIOR_ENDPOINT_METRICS if m in frame]
        for metric in available:
            rows.append(_metric_row(frame, scale=scale, metric=metric, cfg=cfg))

        numeric = frame[available].apply(pd.to_numeric, errors="coerce") if available else pd.DataFrame()
        corr = numeric.corr(method="spearman", min_periods=3) if not numeric.empty else pd.DataFrame()
        for i, a in enumerate(available):
            for b in available[i + 1 :]:
                r = corr.loc[a, b] if a in corr.index and b in corr.columns else math.nan
                redundancy_rows.append({
                    "scale": scale,
                    "metric_a": a,
                    "metric_b": b,
                    "spearman_r": float(r) if np.isfinite(r) else math.nan,
                    "abs_r": abs(float(r)) if np.isfinite(r) else math.nan,
                    "redundant_flag": bool(np.isfinite(r) and abs(float(r)) >= cfg.redundancy_abs_r),
                    "threshold": cfg.redundancy_abs_r,
                    "threshold_role": "historical_descriptive_reference_only",
                    "selection_authority": "descriptive_only",
                    "automatic_drop_allowed": False,
                })

        validation_by_metric = {r["metric"]: r for r in rows if r["scale"] == scale}
        for metric in [m for m in _PRIORITY if m in available]:
            v = validation_by_metric[metric]
            is_primary_omission = metric in CURRENT_PRIMARY_OMISSION_ENDPOINT_METRICS
            decisions.append({
                "scale": scale,
                "metric": metric,
                "candidate_role_recommendation": (
                    "current_primary_omission_endpoint_descriptive_audit"
                    if is_primary_omission
                    else "scientific_candidate_requires_prespecified_or_training_boundary_decision"
                ),
                "reason": (
                    "full-cohort descriptive audit reports coverage/distribution/redundancy but does not select or drop this metric; "
                    f"review_flags={v['review_flags']}"
                ),
                "final_endpoint_freeze_status": (
                    "current_primary_omission_endpoint"
                    if is_primary_omission
                    else "pending_scientific_or_training_boundary_decision"
                ),
                "selection_authority": "descriptive_only",
                "automatic_drop_allowed": False,
                "selection_rule": (
                    "full-cohort audit never selects by coverage, redundancy or p-value; "
                    "data-dependent selection belongs inside training/inner-CV"
                ),
            })

    return pd.DataFrame(rows), pd.DataFrame(redundancy_rows), pd.DataFrame(decisions)


def build_sensitivity_status(session_metrics: pd.DataFrame) -> pd.DataFrame:
    """Register sensitivity tracks without guessing visit order from session ids."""
    rows = [{
        "analysis": "all_eligible_sessions",
        "status": "ready",
        "reason": "primary analysis uses all eligible sessions with participant clustering",
    }]
    order_col = next((c for c in ("visit_order", "session_order", "repeat_visit_order") if c in session_metrics), None)
    if order_col is None or pd.to_numeric(session_metrics[order_col], errors="coerce").isna().any():
        reason = "verified visit/session order unavailable; fail closed rather than infer from session_id"
        rows.extend([
            {"analysis": "first_session_only", "status": "not_estimable", "reason": reason},
            {"analysis": "visit_order_adjusted", "status": "not_estimable", "reason": reason},
        ])
    else:
        rows.extend([
            {"analysis": "first_session_only", "status": "ready", "reason": f"uses minimum verified {order_col} within participant"},
            {"analysis": "visit_order_adjusted", "status": "ready", "reason": f"uses verified {order_col}"},
        ])
    return pd.DataFrame(rows)
