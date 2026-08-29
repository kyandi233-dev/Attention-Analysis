from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def build_visit_sensitivity_status(session_metrics: pd.DataFrame) -> pd.DataFrame:
    """Declare exactly which verified visit-order tracks are calculable."""
    rows: list[dict[str, Any]] = [{
        "analysis": "all_eligible_sessions",
        "status": "ready",
        "n_rows_eligible": int(len(session_metrics)),
        "n_rows_with_verified_order": int(_numeric(session_metrics, "visit_order").notna().sum()),
        "reason": "primary analysis keeps all governed cohort sessions with participant clustering",
    }]
    visit = _numeric(session_metrics, "visit_order")
    prior_same_stage = _numeric(session_metrics, "prior_same_stage_count")
    participant_key_present = (
        session_metrics["participant_key"].notna()
        if "participant_key" in session_metrics.columns
        else pd.Series(False, index=session_metrics.index)
    )
    verified = visit.notna() & participant_key_present
    if not verified.any():
        reason = "no questionnaire-registry rows with participant_key and verified visit_order"
        for name in ("first_any_visit_only", "visit_order_adjusted"):
            rows.append({"analysis": name, "status": "not_estimable", "n_rows_eligible": 0,
                         "n_rows_with_verified_order": 0, "reason": reason})
    else:
        status = "ready" if bool(verified.all()) else "ready_complete_case_only"
        rows.append({
            "analysis": "first_any_visit_only", "status": status,
            "n_rows_eligible": int((verified & visit.eq(1)).sum()),
            "n_rows_with_verified_order": int(verified.sum()),
            "reason": "uses registry visit_order==1; missing questionnaire identity is never imputed",
        })
        rows.append({
            "analysis": "visit_order_adjusted", "status": status,
            "n_rows_eligible": int(verified.sum()),
            "n_rows_with_verified_order": int(verified.sum()),
            "reason": "uses verified questionnaire-registry visit_order as a sensitivity covariate",
        })
    if prior_same_stage.notna().any() and participant_key_present.any():
        stage_verified = prior_same_stage.notna() & participant_key_present
        rows.append({
            "analysis": "first_same_stage_visit_only",
            "status": "ready" if bool(stage_verified.all()) else "ready_complete_case_only",
            "n_rows_eligible": int((stage_verified & prior_same_stage.eq(0)).sum()),
            "n_rows_with_verified_order": int(stage_verified.sum()),
            "reason": "uses registry prior_same_stage_count==0; distinct from first-ever visit",
        })
    else:
        rows.append({
            "analysis": "first_same_stage_visit_only", "status": "not_estimable",
            "n_rows_eligible": 0, "n_rows_with_verified_order": 0,
            "reason": "prior_same_stage_count unavailable",
        })
    return pd.DataFrame(rows)


def subset_verified_first_visit(frame: pd.DataFrame, *, same_stage: bool = False) -> pd.DataFrame:
    """Return only rows whose first-visit status is explicitly verified in the registry."""
    if frame is None or frame.empty:
        return frame.copy() if frame is not None else pd.DataFrame()
    if "participant_key" not in frame.columns:
        return frame.iloc[0:0].copy()
    if same_stage:
        if "prior_same_stage_count" not in frame.columns:
            return frame.iloc[0:0].copy()
        mask = frame["participant_key"].notna() & _numeric(frame, "prior_same_stage_count").eq(0)
    else:
        if "visit_order" not in frame.columns:
            return frame.iloc[0:0].copy()
        mask = frame["participant_key"].notna() & _numeric(frame, "visit_order").eq(1)
    return frame.loc[mask].copy()


def fit_b1_b2_visit_order_sensitivity(
    pairs: pd.DataFrame,
    session_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adjust B2-B1 paired differences for verified repeat visit order.

    Visit order is centered at 1, so the intercept estimates the mean B2-B1
    difference at a first-ever visit and the slope estimates change per
    additional visit. This is a sensitivity analysis, never an endpoint-selection
    mechanism.
    """
    if pairs.empty:
        return pd.DataFrame(), pd.DataFrame([{
            "analysis": "b1_b2_visit_order_sensitivity", "metric": "all",
            "status": "not_estimable", "reason": "pair table empty",
        }])
    needed = ["session_id", "participant_key", "visit_order"]
    missing = [c for c in needed if c not in session_metrics.columns]
    if missing:
        return pd.DataFrame(), pd.DataFrame([{
            "analysis": "b1_b2_visit_order_sensitivity", "metric": "all",
            "status": "not_estimable", "reason": f"session identity fields missing: {missing}",
        }])
    metadata = session_metrics[needed].drop_duplicates("session_id")
    d0 = pairs.merge(metadata, on="session_id", how="left", validate="many_to_one")
    d0["visit_order"] = pd.to_numeric(d0["visit_order"], errors="coerce")
    d0["b2_minus_b1"] = pd.to_numeric(d0["b2_minus_b1"], errors="coerce")
    d0 = d0.dropna(subset=["participant_key", "visit_order", "b2_minus_b1"])
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for metric, current in d0.groupby("metric", sort=True):
        current = current.copy()
        if len(current) < 8 or current["participant_key"].nunique() < 6:
            failures.append({
                "analysis": "b1_b2_visit_order_sensitivity", "metric": metric,
                "status": "not_estimable", "n_rows": int(len(current)),
                "participant_group_n": int(current["participant_key"].nunique()),
                "session_n": int(current["session_id"].nunique()),
                "reason": "insufficient verified rows/participant groups",
            })
            continue
        current["visit_order_centered"] = current["visit_order"] - 1.0
        try:
            import statsmodels.formula.api as smf
            fit = smf.ols("b2_minus_b1 ~ visit_order_centered", data=current).fit(
                cov_type="cluster", cov_kwds={"groups": current["participant_key"].astype(str)}
            )
            for term in ("Intercept", "visit_order_centered"):
                estimate = float(fit.params[term])
                se = float(fit.bse[term])
                if not np.isfinite(estimate) or not np.isfinite(se):
                    raise ValueError("non-finite cluster-robust estimate/SE")
                rows.append({
                    "analysis": "b1_b2_visit_order_sensitivity",
                    "metric": metric,
                    "term": "first_visit_B2_minus_B1" if term == "Intercept" else "change_per_additional_visit",
                    "estimate": estimate,
                    "se": se,
                    "ci_low": estimate - 1.96 * se,
                    "ci_high": estimate + 1.96 * se,
                    "participant_group_n": int(current["participant_key"].nunique()),
                    "session_n": int(current["session_id"].nunique()),
                    "n_rows": int(len(current)),
                    "cluster_unit": "participant_key",
                    "visit_order_source": "questionnaire repeat registry",
                    "status": "estimable",
                    "selection_contract": "sensitivity only; never used to select endpoints by significance",
                })
        except Exception as exc:
            failures.append({
                "analysis": "b1_b2_visit_order_sensitivity", "metric": metric,
                "status": "not_estimable", "n_rows": int(len(current)),
                "participant_group_n": int(current["participant_key"].nunique()),
                "session_n": int(current["session_id"].nunique()),
                "reason": f"{type(exc).__name__}: {exc}",
            })
    return pd.DataFrame(rows), pd.DataFrame(failures)
