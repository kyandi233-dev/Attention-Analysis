"""Repeated-measure extensions required by the behavior science-v3 ledger."""
from __future__ import annotations

import json
import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .science_v3 import CANONICAL_METRICS, BehaviorContractError


def cluster_bootstrap_b1_b2(
    pairs: pd.DataFrame,
    *,
    iterations: int = 20000,
    seed: int = 20260829,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize session-internal B1-B2 differences with participant-cluster bootstrap.

    Repeated sessions first collapse to one participant-group mean difference so a
    participant with any number of sessions is not counted as multiple independent people.
    """
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    required = {"repeat_participant_id", "session_id", "metric", "b2_minus_b1"}
    if not required.issubset(pairs.columns):
        raise BehaviorContractError(f"B1-B2 pair table missing {sorted(required - set(pairs.columns))}")
    for metric, current in pairs.groupby("metric", sort=True):
        current = current.copy()
        current["b2_minus_b1"] = pd.to_numeric(current["b2_minus_b1"], errors="coerce")
        current = current.dropna(subset=["repeat_participant_id", "b2_minus_b1"])
        cluster = current.groupby("repeat_participant_id", as_index=False).agg(
            participant_mean_difference=("b2_minus_b1", "mean"),
            session_pair_n=("session_id", "nunique"),
        )
        values = cluster["participant_mean_difference"].to_numpy(dtype=float)
        if len(values) < 2:
            failures.append({
                "analysis": "B1_B2_cluster_bootstrap",
                "metric": metric,
                "status": "not_estimable",
                "participant_group_n": int(len(values)),
                "session_pair_n": int(current["session_id"].nunique()),
                "reason": "fewer than two participant groups with finite paired differences",
            })
            continue
        boots = np.empty(int(iterations), dtype=float)
        for i in range(int(iterations)):
            sample = rng.choice(values, size=len(values), replace=True)
            boots[i] = float(np.mean(sample))
        estimate = float(np.mean(values))
        rows.append({
            "metric": metric,
            "estimate_b2_minus_b1": estimate,
            "bootstrap_se": float(np.std(boots, ddof=1)),
            "ci_low": float(np.quantile(boots, 0.025)),
            "ci_high": float(np.quantile(boots, 0.975)),
            "participant_group_n": int(len(values)),
            "session_pair_n": int(current["session_id"].nunique()),
            "cluster_unit": "repeat_participant_id",
            "pairing_unit": "session",
            "bootstrap_iterations": int(iterations),
            "status": "estimable",
        })
    return pd.DataFrame(rows), pd.DataFrame(failures)


def build_error_trajectories(
    trials: pd.DataFrame,
    *,
    relative_trials: Iterable[int] = range(-3, 4),
) -> pd.DataFrame:
    """Build event-centered error trajectories with participant-centered Go RT.

    Commission and omission events remain separate. Neighbouring RT is populated
    only for correct Go trials. Centering uses each participant group's median
    correct-Go RT across its available sessions.
    """
    d = trials.copy()
    if "session_id" not in d:
        if "subject" not in d:
            raise BehaviorContractError("error trajectories require session_id or subject")
        d["session_id"] = d["subject"].astype(str)
    required = {
        "repeat_participant_id", "session_id", "block_num", "trial_num",
        "is_no_go", "omission", "commission", "correct", "rt",
    }
    missing = required - set(d.columns)
    if missing:
        raise BehaviorContractError(f"error trajectories missing {sorted(missing)}")
    d["rt"] = pd.to_numeric(d["rt"], errors="coerce")
    d["is_no_go"] = pd.to_numeric(d["is_no_go"], errors="coerce")
    d["correct"] = pd.to_numeric(d["correct"], errors="coerce")
    correct_go = d["is_no_go"].eq(0) & d["correct"].eq(1) & d["rt"].notna()
    baselines = (
        d.loc[correct_go]
        .groupby("repeat_participant_id")["rt"]
        .median()
        .rename("participant_correct_go_rt_median_ms")
    )
    offsets = sorted(set(int(x) for x in relative_trials))
    rows: list[dict[str, Any]] = []
    for (participant, session, block), current in d.groupby(
        ["repeat_participant_id", "session_id", "block_num"], sort=True, dropna=False
    ):
        current = current.sort_values("trial_num", kind="stable").reset_index(drop=True)
        omission = current["is_no_go"].eq(0) & pd.to_numeric(current["omission"], errors="coerce").eq(1)
        commission = current["is_no_go"].eq(1) & pd.to_numeric(current["commission"], errors="coerce").eq(1)
        events = [(int(i), "go_omission") for i in np.flatnonzero(omission.to_numpy())]
        events += [(int(i), "nogo_commission") for i in np.flatnonzero(commission.to_numpy())]
        baseline = baselines.get(participant, math.nan)
        for event_position, error_type in events:
            event_trial = int(current.iloc[event_position]["trial_num"])
            event_id = f"{session}|B{int(block)}|{error_type}|T{event_trial}"
            for offset in offsets:
                pos = event_position + offset
                if pos < 0 or pos >= len(current):
                    continue
                row = current.iloc[pos]
                is_correct_go = bool(float(row["is_no_go"]) == 0 and float(row["correct"]) == 1)
                raw_rt = float(row["rt"]) if is_correct_go and pd.notna(row["rt"]) else math.nan
                rows.append({
                    "repeat_participant_id": str(participant),
                    "session_id": str(session),
                    "block_num": int(block),
                    "error_event_id": event_id,
                    "error_type": error_type,
                    "error_trial_num": event_trial,
                    "relative_trial": int(offset),
                    "neighbor_trial_num": int(row["trial_num"]),
                    "neighbor_is_nogo": int(row["is_no_go"]),
                    "neighbor_correct": int(row["correct"]) if pd.notna(row["correct"]) else pd.NA,
                    "correct_go_rt_ms": raw_rt,
                    "participant_correct_go_rt_median_ms": baseline,
                    "correct_go_rt_centered_ms": (
                        raw_rt - float(baseline)
                        if np.isfinite(raw_rt) and np.isfinite(float(baseline))
                        else math.nan
                    ),
                    "clustering_contract": "error_event nested session nested repeat_participant_id",
                })
    return pd.DataFrame(rows)


def summarize_error_trajectories(events: pd.DataFrame) -> pd.DataFrame:
    """Participant-first descriptive summaries; not pseudo-independent trial SEs."""
    if events.empty:
        return pd.DataFrame()
    participant = (
        events.groupby(["error_type", "relative_trial", "repeat_participant_id"], as_index=False)
        .agg(
            centered_rt_mean_ms=("correct_go_rt_centered_ms", "mean"),
            event_n=("error_event_id", "nunique"),
            session_n=("session_id", "nunique"),
        )
    )
    rows: list[dict[str, Any]] = []
    for (error_type, relative_trial), current in participant.groupby(
        ["error_type", "relative_trial"], sort=True
    ):
        values = pd.to_numeric(current["centered_rt_mean_ms"], errors="coerce").dropna()
        rows.append({
            "error_type": error_type,
            "relative_trial": int(relative_trial),
            "participant_group_n": int(current["repeat_participant_id"].nunique()),
            "session_n": int(current["session_n"].sum()),
            "error_event_n": int(current["event_n"].sum()),
            "participant_centered_rt_mean_ms": float(values.mean()) if len(values) else math.nan,
            "participant_centered_rt_sem_ms": (
                float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) >= 2 else math.nan
            ),
            "observation_unit": "participant_group_summary",
        })
    return pd.DataFrame(rows)


def fit_block_cycle_gee(
    cycle_metrics: pd.DataFrame,
    *,
    metrics: Iterable[str] = (
        "go_correct_rt_median_ms",
        "go_correct_rt_cv",
        "omission_rate",
        "commission_rate",
        "dprime_loglinear",
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate block × cycle time-on-task trends with participant-clustered GEE."""
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if cycle_metrics.empty:
        return pd.DataFrame(), pd.DataFrame([{
            "analysis": "block_cycle_gee", "metric": "all", "status": "not_estimable",
            "reason": "cycle metrics are empty",
        }])
    required = {"repeat_participant_id", "session_id", "block_id", "cycle_bin"}
    missing = required - set(cycle_metrics.columns)
    if missing:
        raise BehaviorContractError(f"cycle GEE missing {sorted(missing)}")
    for metric in metrics:
        if metric not in cycle_metrics:
            continue
        d = cycle_metrics[["repeat_participant_id", "session_id", "block_id", "cycle_bin", metric]].copy()
        d[metric] = pd.to_numeric(d[metric], errors="coerce")
        d["cycle_bin"] = pd.to_numeric(d["cycle_bin"], errors="coerce")
        d["block2"] = d["block_id"].astype(str).eq("B2").astype(int)
        d = d.dropna(subset=[metric, "cycle_bin", "repeat_participant_id"])
        if len(d) < 12 or d["repeat_participant_id"].nunique() < 4 or d["cycle_bin"].nunique() < 2:
            failures.append({
                "analysis": "block_cycle_gee", "metric": metric, "status": "not_estimable",
                "reason": "insufficient rows/groups/cycle levels",
                "participant_group_n": int(d["repeat_participant_id"].nunique()),
                "session_n": int(d["session_id"].nunique()), "n_rows": int(len(d)),
            })
            continue
        try:
            import statsmodels.api as sm
            import statsmodels.formula.api as smf

            model = smf.gee(
                formula=f"{metric} ~ block2 * cycle_bin",
                groups="repeat_participant_id",
                data=d,
                cov_struct=sm.cov_struct.Exchangeable(),
                family=sm.families.Gaussian(),
            )
            fit = model.fit(maxiter=100)
            params = pd.Series(fit.params)
            bse = pd.Series(fit.bse)
            if params.empty or not np.isfinite(params.to_numpy(dtype=float)).all() or not np.isfinite(bse.to_numpy(dtype=float)).all():
                raise ValueError("empty/non-finite GEE parameter table")
            for term in params.index:
                estimate, se = float(params[term]), float(bse[term])
                results.append({
                    "analysis": "block_cycle_gee",
                    "metric": metric,
                    "term": str(term),
                    "estimate": estimate,
                    "se": se,
                    "ci_low": estimate - 1.96 * se,
                    "ci_high": estimate + 1.96 * se,
                    "participant_group_n": int(d["repeat_participant_id"].nunique()),
                    "session_n": int(d["session_id"].nunique()),
                    "n_rows": int(len(d)),
                    "correlation_structure": "Exchangeable within repeat_participant_id",
                    "status": "estimable",
                })
        except Exception as exc:
            failures.append({
                "analysis": "block_cycle_gee", "metric": metric, "status": "not_estimable",
                "reason": f"{type(exc).__name__}: {exc}",
                "participant_group_n": int(d["repeat_participant_id"].nunique()),
                "session_n": int(d["session_id"].nunique()), "n_rows": int(len(d)),
            })
    return pd.DataFrame(results), pd.DataFrame(failures)


def repeat_stability_boundary(session_metrics: pd.DataFrame) -> pd.DataFrame:
    """Describe unbalanced 1..N visits without treating visit count as reliability evidence."""
    groups = session_metrics[["repeat_participant_id", "session_id"]].drop_duplicates()
    counts = groups.groupby("repeat_participant_id")["session_id"].nunique().astype(int)
    repeated = counts[counts.gt(1)]
    distribution = {str(size): int(counts.eq(size).sum()) for size in sorted(counts.unique())}
    return pd.DataFrame([{
        "analysis": "repeat_session_stability",
        "status": "requires_metric_specific_reliability_design",
        "repeated_participant_group_n": int(len(repeated)),
        "max_sessions_per_participant": int(counts.max()) if len(counts) else 0,
        "group_size_distribution": json.dumps(distribution, ensure_ascii=False, sort_keys=True),
        "reason": "participant visit counts may be 1..N; counts alone do not authorize a broad reliability claim",
    }])
