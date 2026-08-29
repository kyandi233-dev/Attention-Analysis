"""Compatibility surface for the formal RGB downstream package.

The only authoritative execution path is now
``scripts/rgb_formal_downstream.py -> rgb_formal.runner.run_rgb_formal_v2``.
Legacy feature helpers remain import-compatible, but the historical v1 orchestration is not
maintained as a second runner.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from attention_pipeline.config import Config
from attention_pipeline.formal_analysis.cohort import canonical_session_id
from .blink_candidates import derive_blink_candidates
from .motion_qc import CONTEXT_COLUMNS, derive_motion_qc
from .pose_direction import derive_pose_direction

PIPELINE_VERSION = "rgb-formal-downstream-v1-compatibility-only"


def discover_sessions(raw_root: Path) -> list[str]:
    if not raw_root.is_dir():
        return []
    return sorted(
        canonical_session_id(path.name)
        for path in raw_root.iterdir()
        if path.is_dir() and path.name.lower().startswith("sub-")
    )


def _subject_dir(raw_root: Path, session_id: str) -> Path:
    exact = raw_root / session_id
    if exact.is_dir():
        return exact
    matches = [
        path for path in raw_root.iterdir()
        if path.is_dir() and canonical_session_id(path.name) == session_id
    ] if raw_root.is_dir() else []
    if len(matches) != 1:
        raise FileNotFoundError(f"RGB subject directory unresolved for {session_id}: {matches}")
    return matches[0]


def _find_subject_file(subject_dir: Path, session_id: str, suffix: str) -> Path | None:
    exact = subject_dir / f"{session_id}{suffix}"
    if exact.is_file():
        return exact
    matches = sorted(subject_dir.glob(f"*{suffix}"))
    return matches[0] if len(matches) == 1 else None


def _load_optional(path: Path | None) -> pd.DataFrame:
    """Legacy generic loader; Face active-contract code must use projected reads instead."""
    if path is None or not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path)


def derive_motion_features(motion: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper around the active Motion QC module."""
    return derive_motion_qc(motion)[0]


def derive_pose_features(pose: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper around the active Pose confirmation module."""
    return derive_pose_direction(pose)[0]


def derive_face_features(face: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compatibility wrapper around algorithm-defined Blink candidates."""
    ocular = config.section("ocular")
    ref = ocular.get("open_reference", {})
    blink = ocular.get("blink_candidate", {})
    frames, events, status = derive_blink_candidates(
        face,
        preferred_phase=str(ref.get("preferred_phase", "baseline")),
        minimum_valid_frames=int(ref.get("minimum_valid_frames", 30)),
        relative_openness_threshold=float(blink.get("relative_openness_threshold", 0.20)),
        minimum_closed_duration_ms=float(blink.get("minimum_closed_duration_ms", 50)),
        maximum_closed_duration_ms=float(blink.get("maximum_closed_duration_ms", 1000)),
        gap_reset_ms=float(blink.get("gap_reset_ms", 250)),
        maximum_bilateral_relative_difference=float(blink.get("maximum_bilateral_relative_difference", 0.35)),
    )
    return frames, events, {
        **status,
        "status": status.get("blink_status", "not_estimable"),
        "primary_face_status": "formal_blink_candidate_primary_contract",
        "blink_threshold_status": "candidate_requires_real_data_freeze",
    }


def attach_behavior_context(native: pd.DataFrame, motion: pd.DataFrame, tolerance_ms: int = 150) -> pd.DataFrame:
    if native.empty or motion.empty or "unix_ms" not in native.columns or "unix_ms" not in motion.columns:
        return native.copy()
    context = motion[["unix_ms", *[c for c in CONTEXT_COLUMNS if c in motion.columns]]].drop_duplicates("unix_ms").sort_values("unix_ms")
    left = native.drop(columns=[c for c in CONTEXT_COLUMNS if c in native.columns], errors="ignore").sort_values("unix_ms")
    return pd.merge_asof(left, context, on="unix_ms", direction="nearest", tolerance=tolerance_ms)


def _mad(x: pd.Series) -> float:
    values = pd.to_numeric(x, errors="coerce").dropna()
    return float((values - values.median()).abs().median()) if len(values) else np.nan


def _summarize(frame: pd.DataFrame, group_cols: list[str], metric_cols: list[str], scale: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(group_cols, key))
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if not len(values):
                continue
            rows.append({
                **base,
                "scale": scale,
                "metric": metric,
                "n_valid": int(len(values)),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "sd": float(values.std(ddof=1)) if len(values) >= 2 else np.nan,
                "mad": _mad(values),
                "q10": float(values.quantile(0.1)),
                "q90": float(values.quantile(0.9)),
            })
    return pd.DataFrame(rows)


def build_multiscale(features: pd.DataFrame, probes: pd.DataFrame, probe_windows: Iterable[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retained historical helper; not executed by the lightweight default runner."""
    if features.empty:
        return pd.DataFrame(), pd.DataFrame()
    idcols = {"subject", "session_id", "participant_group_id", "participant_key", "unix_ms", *CONTEXT_COLUMNS}
    metrics = [
        column for column in features.columns
        if column not in idcols
        and pd.api.types.is_numeric_dtype(features[column])
        and column not in {"video_frame_position", "capture_frame_idx", "trial_num", "block", "cycle_num", "absolute_onset_time", "probe_onset_time"}
    ]
    parts = [_summarize(features, ["session_id", "participant_group_id"], metrics, "session")]
    if "block" in features.columns:
        parts.append(_summarize(features.dropna(subset=["block"]), ["session_id", "participant_group_id", "block"], metrics, "block"))
    summary = pd.concat([part for part in parts if not part.empty], ignore_index=True) if any(not part.empty for part in parts) else pd.DataFrame()
    return summary, pd.DataFrame()


def candidate_validation(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Historical compatibility helper; active lightweight execution does not call it."""
    if summary.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (scale, metric), group in summary.groupby(["scale", "metric"], sort=True):
        values = pd.to_numeric(group["median"], errors="coerce")
        finite = values.dropna()
        coverage = len(finite) / len(group) if len(group) else 0.0
        rows.append({
            "scale": scale,
            "metric": metric,
            "n_rows": int(len(group)),
            "n_valid": int(len(finite)),
            "coverage": float(coverage),
            "participant_group_n": int(group.loc[values.notna(), "participant_group_id"].nunique()),
            "session_n": int(group.loc[values.notna(), "session_id"].nunique()),
            "candidate_status": "historical_helper_not_active_default",
        })
    validation = pd.DataFrame(rows)
    decisions = validation[["scale", "metric", "candidate_status"]].copy()
    decisions["final_endpoint_freeze_status"] = "disabled_deferred_in_lightweight_runner"
    return validation, pd.DataFrame(), decisions


def run_rgb_formal_pipeline(
    config_path: str | Path,
    *,
    subjects: Iterable[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Compatibility alias to the unique authoritative ``run_rgb_formal_v2`` runner."""
    warnings.warn(
        "run_rgb_formal_pipeline is a compatibility alias; use run_rgb_formal_v2",
        DeprecationWarning,
        stacklevel=2,
    )
    if force:
        warnings.warn("legacy force flag is ignored by the governed v2 runner", RuntimeWarning, stacklevel=2)
    from .runner import run_rgb_formal_v2

    return run_rgb_formal_v2(config_path, subjects=subjects)
