"""Participant-aware scientific summaries for formal RGB downstream analysis.

This module deliberately stops short of outcome-driven endpoint selection.
It supplies the structures needed before formal inference/prediction: within-
participant decomposition, verified visit sensitivity, time-on-task trends,
participant-cluster uncertainty, participant-exclusive prediction folds, and
explicit deferred/failure registries.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import theilslopes


@dataclass(frozen=True)
class RGBScienceConfig:
    time_bin_seconds: float = 10.0
    bootstrap_replicates: int = 1000
    bootstrap_seed: int = 20260830
    prediction_folds: int = 5
    minimum_time_bins_for_slope: int = 3


ID_COLUMNS = {
    "subject", "session_id", "participant_key", "participant_group_id", "modality",
    "unix_ms", "video_frame_position", "capture_frame_idx", "phase", "block",
    "trial_num", "cycle_num", "position_in_cycle", "is_no_go", "response", "correct",
    "commission", "omission", "is_probe", "probe_response", "probe_vigilance",
    "absolute_onset_time", "probe_onset_time", "behavior_state", "dt_ms",
}


def _numeric_metrics(frame: pd.DataFrame) -> list[str]:
    excluded = ID_COLUMNS | {"gap_before", "irregular_dt", "motion_valid", "rgb_face_valid"}
    return [
        c for c in frame.columns
        if c not in excluded
        and pd.api.types.is_numeric_dtype(frame[c])
        and not str(c).endswith("_event_id")
    ]


def build_within_between(summary: pd.DataFrame) -> pd.DataFrame:
    """Decompose summary medians into participant mean + within-person deviation."""
    required = {"participant_group_id", "session_id", "scale", "metric", "median"}
    if summary.empty:
        return pd.DataFrame(columns=[*required, "participant_mean", "within_deviation"])
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"RGB within/between 缺少列: {sorted(missing)}")
    out = summary.copy()
    out["median"] = pd.to_numeric(out["median"], errors="coerce")
    keys = ["participant_group_id", "scale", "metric"]
    # Keep scale-specific meaning: block/probe values should not redefine a
    # participant mean together with unrelated session-level rows.
    out["participant_mean"] = out.groupby(keys, dropna=False)["median"].transform("mean")
    out["within_deviation"] = out["median"] - out["participant_mean"]
    out["between_component"] = out["participant_mean"]
    out["decomposition_status"] = np.where(
        out["participant_group_id"].notna() & out["median"].notna(), "estimable", "not_estimable"
    )
    return out


def build_repeat_visit_sensitivity(summary: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    """Use verified visit_order; never infer direction from a session number."""
    if summary.empty or identity.empty:
        return pd.DataFrame()
    required = {"session_id", "participant_group_id"}
    if required - set(identity.columns):
        raise ValueError("RGB visit sensitivity requires session_id and participant_group_id")
    if "visit_order" not in identity.columns:
        return pd.DataFrame([{
            "status": "not_estimable_without_verified_visit_order",
            "reason": "identity registry has no visit_order",
        }])
    meta_cols = [c for c in ("session_id", "participant_group_id", "participant_key", "visit_order", "prior_visit_count") if c in identity]
    meta = identity[meta_cols].drop_duplicates("session_id")
    current = summary.merge(meta, on=[c for c in ("session_id", "participant_group_id") if c in meta], how="left", validate="many_to_one")
    current["visit_order"] = pd.to_numeric(current["visit_order"], errors="coerce")
    # Only compare like with like. For block-level summaries, B1 changes are
    # compared with B1 changes and B2 with B2; other extra keys are retained.
    compare_keys = [c for c in ("scale", "metric", "block") if c in current.columns]
    rows: list[dict[str, object]] = []
    for keys, group in current.dropna(subset=["participant_group_id", "visit_order", "median"]).groupby(
        ["participant_group_id", *compare_keys], dropna=False, sort=True
    ):
        ordered = group.sort_values(["visit_order", "session_id"])
        if len(ordered) < 2:
            continue
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(["participant_group_id", *compare_keys], keys))
        prev = None
        for row in ordered.itertuples(index=False):
            if prev is not None:
                current_order = int(row.visit_order)
                previous_order = int(prev.visit_order)
                if current_order <= previous_order:
                    raise ValueError("verified visit_order is non-increasing within participant")
                rows.append({
                    **base,
                    "previous_session_id": prev.session_id,
                    "current_session_id": row.session_id,
                    "previous_visit_order": previous_order,
                    "current_visit_order": current_order,
                    "previous_value": float(prev.median),
                    "current_value": float(row.median),
                    "directional_change": float(row.median) - float(prev.median),
                    "absolute_change": abs(float(row.median) - float(prev.median)),
                    "direction_source": "verified_repeat_registry_visit_order",
                    "status": "estimable",
                })
            prev = row
    return pd.DataFrame(rows)


def build_time_on_task(features: pd.DataFrame, *, bin_seconds: float = 10.0, minimum_bins: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize native-rate features in time bins and estimate robust slopes.

    Native Face/Pose/Motion rates are preserved. Binning occurs only at the
    statistical summary layer, avoiding interpolation to a common frame rate.
    """
    if features.empty:
        return pd.DataFrame(), pd.DataFrame()
    required = {"session_id", "participant_group_id", "modality", "unix_ms"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"RGB time-on-task 缺少列: {sorted(missing)}")
    metrics = _numeric_metrics(features)
    rows: list[dict[str, object]] = []
    slope_rows: list[dict[str, object]] = []
    scope_cols = ["session_id", "participant_group_id", "modality"]
    if "block" in features.columns:
        scope_cols.append("block")
    current = features.copy()
    current["unix_ms"] = pd.to_numeric(current["unix_ms"], errors="coerce")
    for keys, group in current.dropna(subset=["unix_ms"]).groupby(scope_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(scope_cols, keys))
        start_ms = float(group["unix_ms"].min())
        elapsed = (group["unix_ms"] - start_ms) / 1000.0
        work = group.assign(_elapsed_sec=elapsed, _time_bin=(elapsed // float(bin_seconds)).astype("Int64"))
        for metric in metrics:
            numeric = pd.to_numeric(work[metric], errors="coerce")
            temp = work.assign(_value=numeric).dropna(subset=["_value", "_time_bin"])
            if temp.empty:
                continue
            bin_table = temp.groupby("_time_bin", sort=True).agg(
                time_sec=("_elapsed_sec", "median"), value=("_value", "median"), n_valid=("_value", "size")
            ).reset_index()
            for row in bin_table.itertuples(index=False):
                rows.append({
                    **base, "metric": metric, "time_bin": int(row._0) if hasattr(row, "_0") else int(getattr(row, "_time_bin")),
                    "time_sec": float(row.time_sec), "median": float(row.value), "n_valid": int(row.n_valid),
                    "bin_seconds": float(bin_seconds),
                })
            if len(bin_table) >= int(minimum_bins) and bin_table["time_sec"].nunique() >= int(minimum_bins):
                slope, intercept, low, high = theilslopes(bin_table["value"], bin_table["time_sec"], 0.95)
                slope_rows.append({
                    **base, "metric": metric, "n_bins": int(len(bin_table)),
                    "theilsen_slope_per_sec": float(slope), "theilsen_intercept": float(intercept),
                    "slope_ci95_low": float(low), "slope_ci95_high": float(high),
                    "status": "estimable", "method": "Theil-Sen over bin medians",
                })
            else:
                slope_rows.append({
                    **base, "metric": metric, "n_bins": int(len(bin_table)),
                    "theilsen_slope_per_sec": np.nan, "theilsen_intercept": np.nan,
                    "slope_ci95_low": np.nan, "slope_ci95_high": np.nan,
                    "status": "not_estimable_insufficient_time_bins", "method": "Theil-Sen over bin medians",
                })
    return pd.DataFrame(rows), pd.DataFrame(slope_rows)


def participant_cluster_bootstrap(
    summary: pd.DataFrame,
    *,
    replicates: int = 1000,
    seed: int = 20260830,
) -> pd.DataFrame:
    """Bootstrap participants as clusters, carrying all of their session rows."""
    if summary.empty:
        return pd.DataFrame()
    required = {"participant_group_id", "scale", "metric", "median"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"RGB cluster bootstrap 缺少列: {sorted(missing)}")
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, object]] = []
    strata = ["scale", "metric"]
    if "block" in summary.columns:
        # Block is meaningful only for block-scale rows; NaN remains a stratum.
        strata.append("block")
    for keys, group in summary.groupby(strata, dropna=False, sort=True):
        clean = group.dropna(subset=["participant_group_id"]).copy()
        clean["median"] = pd.to_numeric(clean["median"], errors="coerce")
        clean = clean.dropna(subset=["median"])
        participants = clean["participant_group_id"].astype(str).unique()
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(strata, keys))
        if len(participants) < 2:
            rows.append({**base, "participant_n": int(len(participants)), "session_n": int(clean["session_id"].nunique()) if "session_id" in clean else np.nan,
                         "estimate": float(clean["median"].mean()) if len(clean) else np.nan, "ci95_low": np.nan, "ci95_high": np.nan,
                         "bootstrap_replicates": 0, "status": "not_estimable_lt2_participants"})
            continue
        estimates = np.empty(int(replicates), dtype=float)
        grouped = {p: clean[clean["participant_group_id"].astype(str).eq(p)] for p in participants}
        for i in range(int(replicates)):
            sampled = rng.choice(participants, size=len(participants), replace=True)
            pieces = [grouped[p]["median"].to_numpy(float) for p in sampled]
            values = np.concatenate(pieces) if pieces else np.array([], dtype=float)
            estimates[i] = float(np.mean(values)) if len(values) else np.nan
        valid = estimates[np.isfinite(estimates)]
        rows.append({
            **base, "participant_n": int(len(participants)),
            "session_n": int(clean["session_id"].nunique()) if "session_id" in clean else np.nan,
            "estimate": float(clean["median"].mean()),
            "ci95_low": float(np.quantile(valid, .025)) if len(valid) else np.nan,
            "ci95_high": float(np.quantile(valid, .975)) if len(valid) else np.nan,
            "bootstrap_replicates": int(len(valid)), "status": "estimable",
            "resampling_unit": "participant_group_id_all_rows_together",
        })
    return pd.DataFrame(rows)


def participant_exclusive_folds(identity: pd.DataFrame, *, n_folds: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign deterministic folds at participant level; never split sessions of one person."""
    required = {"session_id", "participant_group_id"}
    missing = required - set(identity.columns)
    if missing:
        raise ValueError(f"RGB prediction fold audit 缺少列: {sorted(missing)}")
    current = identity.dropna(subset=["participant_group_id"]).copy()
    groups = sorted(current["participant_group_id"].astype(str).unique())
    k = max(2, min(int(n_folds), len(groups))) if len(groups) >= 2 else 1
    assignment: dict[str, int] = {}
    for group in groups:
        digest = hashlib.sha256(group.encode("utf-8")).hexdigest()
        assignment[group] = int(int(digest[:12], 16) % k) if k > 1 else 0
    current["prediction_fold"] = current["participant_group_id"].astype(str).map(assignment).astype("Int64")
    leakage = current.groupby("participant_group_id")["prediction_fold"].nunique().gt(1)
    audit = pd.DataFrame([{
        "participant_group_n": int(len(groups)), "session_n": int(current["session_id"].nunique()),
        "n_folds": int(k), "participant_cross_fold_leakage_n": int(leakage.sum()),
        "preprocessing_scope": "train_fold_only_required",
        "endpoint_selection_scope": "training_fold_only_after_endpoint_freeze",
        "prediction_model_status": "deferred_pending_endpoint_and_real_data_freeze",
        "status": "valid" if int(leakage.sum()) == 0 and len(groups) >= 2 else "not_estimable",
    }])
    return current, audit


def model_contract_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return explicit empty failure table plus deferred inference/prediction registry."""
    failures = pd.DataFrame(columns=[
        "model_name", "analysis_type", "outcome", "predictor", "participant_n", "session_n", "n_rows",
        "failure_type", "failure_reason",
    ])
    deferred = pd.DataFrame([
        {
            "model_family": "rgb_behavior_inference", "analysis_type": "inference",
            "status": "deferred_pending_rgb_and_behavior_endpoint_freeze",
            "reason": "avoid predictor-by-outcome Cartesian fishing before candidate validation",
        },
        {
            "model_family": "rgb_behavior_prediction", "analysis_type": "prediction",
            "status": "deferred_pending_endpoint_freeze_and_participant_exclusive_training",
            "reason": "prediction must use participant-exclusive folds and train-fold-only preprocessing",
        },
    ])
    return failures, deferred
