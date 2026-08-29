"""Scientific admission audit for pupil-only candidate metrics.

This stage consumes only candidate sidecars already written under
``10_analysis_ready``.  It never reads production files directly, never selects
metrics by outcome p-values, and never freezes a final pupil endpoint without a
real-data scientific review.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from attention_pipeline.config import Config, load_config
from attention_pipeline.nir_analysis_ready.candidate_metrics import PUPIL_CANDIDATE_METRICS

CANDIDATE_VALIDATION_VERSION = "nir-pupil-candidate-validation-v1"
BLOCK_MAP = {"block1": 1, "block2": 2}


def _resolve(config: Config, key: str) -> Path:
    raw = config.section("paths").get(key)
    if raw in (None, ""):
        raise KeyError(f"formal pupil config missing paths.{key}")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (config.path.parent.parent / path).resolve()
    return path


def _candidate_path(config: Config, session_id: str) -> Path:
    return (
        _resolve(config, "analysis_ready_root")
        / "candidate_frame_level"
        / session_id
        / f"{session_id}_nir_pupil_candidates.csv"
    )


def _discover_candidate_sessions(config: Config) -> list[str]:
    root = _resolve(config, "analysis_ready_root") / "candidate_frame_level"
    if not root.is_dir():
        return []
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and (p / f"{p.name}_nir_pupil_candidates.csv").is_file()
    )


def _selected(config: Config, override: Iterable[str] | None) -> list[str]:
    if override:
        return [str(x).strip() for x in override if str(x).strip()]
    include = config.section("sessions").get("include", [])
    if include:
        return [str(x).strip() for x in include if str(x).strip()]
    return _discover_candidate_sessions(config)


def _finite_median(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce")
    x = x[np.isfinite(x)]
    return float(x.median()) if len(x) else np.nan


def _fuse_eye_summary(left: float, right: float) -> tuple[float, str]:
    l = np.isfinite(left)
    r = np.isfinite(right)
    if l and r:
        return float((left + right) / 2.0), "binocular"
    if l:
        return float(left), "left_only"
    if r:
        return float(right), "right_only"
    return np.nan, "missing"


def summarize_session_block_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "session_id", "analysis_group_token", "phase", "eye",
        "pupil_valid_primary", "pupil_valid_strict",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"candidate sidecar missing columns: {missing}")
    out = frame.copy()
    out["block_num"] = out["phase"].astype(str).map(BLOCK_MAP)
    if out["block_num"].isna().any():
        raise ValueError("candidate sidecar contains non-formal phase rows")

    eye_rows: list[dict[str, Any]] = []
    for (session, group, block, eye), current in out.groupby(
        ["session_id", "analysis_group_token", "block_num", "eye"], sort=True
    ):
        for metric, spec in PUPIL_CANDIDATE_METRICS.items():
            raw_col = f"{metric}__raw"
            centered_col = f"{metric}__centered_primary"
            valid_col = f"{metric}__valid_primary"
            if raw_col not in current or valid_col not in current:
                continue
            valid = current[valid_col].fillna(False).astype(bool)
            raw = pd.to_numeric(current[raw_col], errors="coerce")
            centered = pd.to_numeric(current.get(centered_col), errors="coerce")
            eye_rows.append({
                "session_id": str(session),
                "analysis_group_token": str(group),
                "block_num": int(block),
                "eye": str(eye),
                "metric": metric,
                "metric_kind": spec["kind"],
                "unit": spec["unit"],
                "n_rows": int(len(current)),
                "n_valid": int(valid.sum()),
                "valid_fraction": float(valid.mean()) if len(valid) else np.nan,
                "raw_median": _finite_median(raw[valid]),
                "centered_median": _finite_median(centered[valid]),
            })
    eye = pd.DataFrame(eye_rows)
    if eye.empty:
        return eye

    rows: list[dict[str, Any]] = []
    key_cols = ["session_id", "analysis_group_token", "block_num", "metric", "metric_kind", "unit"]
    for key, current in eye.groupby(key_cols, sort=True, dropna=False):
        base = dict(zip(key_cols, key))
        by_eye = {str(r.eye): r for r in current.itertuples(index=False)}
        left = by_eye.get("left")
        right = by_eye.get("right")
        for value_name in ("raw_median", "centered_median"):
            lv = getattr(left, value_name) if left is not None else np.nan
            rv = getattr(right, value_name) if right is not None else np.nan
            fused, source = _fuse_eye_summary(lv, rv)
            base[f"binocular_{value_name}"] = fused
            base[f"binocular_{value_name}_source"] = source
        valid_fracs = [
            float(getattr(x, "valid_fraction"))
            for x in (left, right)
            if x is not None and np.isfinite(getattr(x, "valid_fraction"))
        ]
        base["eye_valid_fraction_mean"] = float(np.mean(valid_fracs)) if valid_fracs else np.nan
        base["left_raw_median"] = getattr(left, "raw_median") if left is not None else np.nan
        base["right_raw_median"] = getattr(right, "raw_median") if right is not None else np.nan
        rows.append(base)
    return pd.DataFrame(rows)


def add_within_between(summary: pd.DataFrame) -> pd.DataFrame:
    """Separate stable group means from within-participant deviations.

    The existing anonymous ``analysis_group_token`` is used only as the current
    grouping interface.  This function does not alter or infer the upstream
    participant mapping; that adapter remains frozen until the verified registry
    is supplied.
    """
    if summary.empty:
        return summary.copy()
    out = summary.copy()
    value = pd.to_numeric(out["binocular_raw_median"], errors="coerce")
    keys = [out["analysis_group_token"].astype(str), out["metric"].astype(str)]
    out["participant_metric_mean"] = value.groupby(keys).transform("mean")
    out["within_participant_deviation"] = value - out["participant_metric_mean"]
    out["participant_metric_observation_n"] = value.notna().groupby(keys).transform("sum").astype(int)
    out["within_participant_status"] = np.where(
        out["participant_metric_observation_n"].ge(2),
        "estimable",
        "not_estimable_single_observation",
    )
    out.loc[out["within_participant_status"].ne("estimable"), "within_participant_deviation"] = np.nan
    return out


def _metric_validation(summary: pd.DataFrame, requested_session_n: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected_block_rows = max(1, requested_session_n * 2)
    for metric, current in summary.groupby("metric", sort=False):
        values = pd.to_numeric(current["binocular_raw_median"], errors="coerce")
        finite = values[np.isfinite(values)]
        participant_means = (
            current.assign(_value=values)
            .groupby("analysis_group_token")["_value"]
            .mean()
            .dropna()
        )
        within = add_within_between(current)["within_participant_deviation"].dropna()
        observed = int(values.notna().sum())
        coverage = min(1.0, observed / expected_block_rows)
        reasons: list[str] = []
        if coverage < 0.80:
            reasons.append("low_session_block_coverage")
        if finite.nunique() < 3:
            reasons.append("low_unique_values")
        rows.append({
            "metric": metric,
            "unit": str(current["unit"].iloc[0]),
            "session_block_expected_n": expected_block_rows,
            "session_block_observed_n": observed,
            "coverage": coverage,
            "participant_group_n": int(current.loc[values.notna(), "analysis_group_token"].nunique()),
            "session_n": int(current.loc[values.notna(), "session_id"].nunique()),
            "raw_median": float(finite.median()) if len(finite) else np.nan,
            "raw_q05": float(finite.quantile(0.05)) if len(finite) else np.nan,
            "raw_q95": float(finite.quantile(0.95)) if len(finite) else np.nan,
            "between_participant_variance": float(participant_means.var(ddof=1)) if len(participant_means) >= 2 else np.nan,
            "within_participant_variance": float(within.var(ddof=1)) if len(within) >= 2 else np.nan,
            "candidate_status": "eligible_candidate" if not reasons else "needs_review",
            "candidate_reasons": ";".join(reasons),
            "scientific_winner_declared": False,
        })
    return pd.DataFrame(rows)


def _redundancy(summary: pd.DataFrame) -> pd.DataFrame:
    wide = summary.pivot_table(
        index=["analysis_group_token", "session_id", "block_num"],
        columns="metric",
        values="binocular_raw_median",
        aggfunc="first",
    )
    metrics = [m for m in PUPIL_CANDIDATE_METRICS if m in wide.columns]
    ranked = wide[metrics].rank(axis=0, method="average") if metrics else pd.DataFrame()
    corr = ranked.corr(method="pearson", min_periods=3) if not ranked.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(metrics):
        for b in metrics[i + 1:]:
            r = corr.loc[a, b] if a in corr.index and b in corr.columns else np.nan
            rows.append({
                "metric_a": a,
                "metric_b": b,
                "spearman_r": float(r) if np.isfinite(r) else np.nan,
                "abs_r": abs(float(r)) if np.isfinite(r) else np.nan,
                "high_redundancy_flag": bool(np.isfinite(r) and abs(float(r)) >= 0.90),
                "threshold": 0.90,
            })
    return pd.DataFrame(rows)


def _decisions(validation: pd.DataFrame, redundancy: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in validation.itertuples(index=False):
        redundant = redundancy[
            ((redundancy["metric_a"] == r.metric) | (redundancy["metric_b"] == r.metric))
            & redundancy["high_redundancy_flag"].fillna(False)
        ] if not redundancy.empty else pd.DataFrame()
        role = (
            "candidate_pending_redundancy_review"
            if not redundant.empty and r.candidate_status == "eligible_candidate"
            else str(r.candidate_status)
        )
        rows.append({
            "metric": r.metric,
            "candidate_role_recommendation": role,
            "high_redundancy_pair_n": int(len(redundant)),
            "final_endpoint_freeze_status": "pending_real_data_scientific_review",
            "selection_contract": "coverage + within/between + redundancy + scientific validity; never outcome p-value screening",
        })
    return pd.DataFrame(rows)


def _repeat_stability(summary: pd.DataFrame) -> pd.DataFrame:
    session = (
        summary.groupby(["analysis_group_token", "session_id", "metric"], as_index=False)["binocular_raw_median"]
        .median()
    )
    rows: list[dict[str, Any]] = []
    for (group, metric), current in session.groupby(["analysis_group_token", "metric"], sort=True):
        vals = pd.to_numeric(current["binocular_raw_median"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) == 2:
            rows.append({
                "analysis_group_token": str(group),
                "metric": str(metric),
                "session_n": 2,
                "repeat_stability_status": "descriptive_pair_available",
                "unordered_absolute_session_difference": float(abs(vals[1] - vals[0])),
                "directional_visit_change": np.nan,
                "directional_status": "not_estimable_without_verified_visit_order",
            })
    return pd.DataFrame(rows)


def run_candidate_validation(
    config_path: str | Path,
    *,
    subjects: Iterable[str] | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    sessions = _selected(config, subjects)
    if not sessions:
        raise ValueError(
            "No pupil candidate sidecars selected; rebuild 10_analysis_ready candidate layer first"
        )

    frames: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    for session_id in sessions:
        path = _candidate_path(config, session_id)
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
            actual = set(frame["session_id"].dropna().astype(str).unique())
            if actual != {session_id}:
                raise ValueError(f"candidate sidecar session mismatch: {sorted(actual)}")
            frames.append(frame)
        except Exception as exc:
            failures.append({
                "session_id": session_id,
                "stage": "candidate_validation_load",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

    root = _resolve(config, "output_root") / "candidate_validation"
    root.mkdir(parents=True, exist_ok=True)
    failure_table = pd.DataFrame(failures, columns=["session_id", "stage", "error_type", "error"])
    failure_table.to_csv(root / "nir_candidate_failures.csv", index=False, encoding="utf-8-sig")
    if failures:
        manifest = {
            "status": "blocked",
            "reason": "candidate sidecar missing/invalid for one or more requested sessions",
            "n_sessions_requested": len(sessions),
            "n_sessions_failed": len(failures),
            "scientific_inference_authorized": False,
        }
        (root / "nir_candidate_validation_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return manifest

    combined = pd.concat(frames, ignore_index=True)
    summary = summarize_session_block_candidates(combined)
    within_between = add_within_between(summary)
    validation = _metric_validation(summary, len(sessions))
    redundancy = _redundancy(summary)
    decisions = _decisions(validation, redundancy)
    repeat = _repeat_stability(summary)

    outputs = {
        "nir_candidate_session_block_metrics.csv": summary,
        "nir_candidate_within_between.csv": within_between,
        "nir_candidate_metric_validation.csv": validation,
        "nir_candidate_metric_redundancy.csv": redundancy,
        "nir_candidate_metric_decisions.csv": decisions,
        "nir_candidate_repeat_stability.csv": repeat,
    }
    for name, table in outputs.items():
        table.to_csv(root / name, index=False, encoding="utf-8-sig")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "pipeline_version": CANDIDATE_VALIDATION_VERSION,
        "n_sessions_requested": len(sessions),
        "n_sessions_failed": 0,
        "candidate_metrics": list(PUPIL_CANDIDATE_METRICS),
        "identity_contract": "uses existing analysis_group_token only; participant mapping adapter unchanged",
        "within_between_contract": "participant mean + within-participant deviation on session×block candidate summaries",
        "repeat_contract": "unordered absolute session difference only; directional visit change blocked without verified order",
        "endpoint_freeze": "pending_real_data_scientific_review",
        "scientific_inference_authorized": False,
        "outputs": {name: str(root / name) for name in outputs},
    }
    (root / "nir_candidate_validation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
