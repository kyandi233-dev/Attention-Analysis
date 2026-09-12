from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from attention_pipeline.nir_formal_analysis.pupil_blink_measurement import audit_rgb_nir_sync


def _time_metrics(values: pd.Series) -> dict[str, float | int]:
    raw = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if raw.size == 0:
        return {
            "time_min_ms": np.nan,
            "time_max_ms": np.nan,
            "span_ms": np.nan,
            "nominal_interval_ms": np.nan,
            "nonpositive_step_n": 0,
            "large_gap_n": 0,
        }
    dt_raw = np.diff(raw)
    ordered = np.sort(np.unique(raw))
    dt = np.diff(ordered)
    positive = dt[dt > 0]
    nominal = float(np.median(positive)) if positive.size else np.nan
    gap_cut = max(1000.0, 10.0 * nominal) if np.isfinite(nominal) else 1000.0
    return {
        "time_min_ms": float(np.min(ordered)),
        "time_max_ms": float(np.max(ordered)),
        "span_ms": float(np.max(ordered) - np.min(ordered)),
        "nominal_interval_ms": nominal,
        "nonpositive_step_n": int(np.sum(dt_raw <= 0)),
        "large_gap_n": int(np.sum(dt > gap_cut)),
    }


def _nearest_signed(reference: np.ndarray, targets: np.ndarray) -> np.ndarray:
    reference = np.sort(np.unique(reference[np.isfinite(reference)]))
    targets = targets[np.isfinite(targets)]
    if reference.size == 0 or targets.size == 0:
        return np.array([], dtype=float)
    pos = np.searchsorted(reference, targets)
    left = reference[np.clip(pos - 1, 0, reference.size - 1)]
    right = reference[np.clip(pos, 0, reference.size - 1)]
    nearest = np.where(np.abs(right - targets) < np.abs(left - targets), right, left)
    return nearest - targets


def rgb_blink_source_availability(
    events: pd.DataFrame,
    rgb_frames: pd.DataFrame | None,
) -> tuple[bool, str]:
    """Classify whether RGB blink evidence exists without treating missing RGB as zero blinks.

    A non-empty RGB frame axis establishes that a zero-event session is an observed
    zero. Event boundaries alone are also usable, with weaker synchronization evidence.
    If neither exists, the RGB-assisted cleaning tracks are not estimable.
    """
    frame_axis_available = rgb_frames is not None and not rgb_frames.empty
    events_available = events is not None and not events.empty
    if frame_axis_available and events_available:
        return True, "rgb_frame_axis_plus_blink_events"
    if frame_axis_available:
        return True, "rgb_frame_axis_zero_detected_events"
    if events_available:
        return True, "blink_event_boundaries_only"
    return False, "rgb_blink_evidence_unavailable_or_ambiguous"


def audit_rgb_nir_sync_with_frames(
    timepoints: pd.DataFrame,
    events: pd.DataFrame,
    rgb_frames: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Extend event-boundary sync audit with the RGB frame time axis when available.

    Range deltas and frame-to-frame residuals are descriptive diagnostics only; they
    are not converted into an automatic synchronization pass/fail threshold.
    """
    base = audit_rgb_nir_sync(timepoints, events).copy()
    if len(base) != 1:
        raise ValueError("sync helper expects one session at a time")
    row = base.iloc[0].to_dict()
    nir_times = pd.to_numeric(timepoints["unix_ms"], errors="coerce")
    nir_metrics = _time_metrics(nir_times)
    for name, value in nir_metrics.items():
        row[f"nir_frame_{name}"] = value

    if rgb_frames is None or rgb_frames.empty:
        row.update(
            {
                "sync_evidence_level": "blink_event_boundaries_only",
                "rgb_frame_axis_available": False,
                "rgb_frame_time_min_ms": np.nan,
                "rgb_frame_time_max_ms": np.nan,
                "rgb_frame_span_ms": np.nan,
                "rgb_frame_nominal_interval_ms": np.nan,
                "rgb_frame_nonpositive_step_n": np.nan,
                "rgb_frame_large_gap_n": np.nan,
                "rgb_minus_nir_start_ms": np.nan,
                "rgb_minus_nir_end_ms": np.nan,
                "rgb_minus_nir_span_ms": np.nan,
                "frame_nearest_residual_signed_median_ms": np.nan,
                "frame_nearest_residual_abs_median_ms": np.nan,
                "frame_nearest_residual_abs_p95_ms": np.nan,
                "frame_nearest_residual_abs_max_ms": np.nan,
                "frame_nearest_residual_drift_ms_per_hour": np.nan,
            }
        )
        return pd.DataFrame([row])

    if "unix_ms" not in rgb_frames.columns:
        raise ValueError("RGB blink candidate frame table missing unix_ms")
    rgb_times = pd.to_numeric(rgb_frames["unix_ms"], errors="coerce")
    rgb_metrics = _time_metrics(rgb_times)
    for name, value in rgb_metrics.items():
        row[f"rgb_frame_{name}"] = value

    nir = nir_times.dropna().to_numpy(float)
    rgb = rgb_times.dropna().to_numpy(float)
    residual = _nearest_signed(nir, rgb)
    abs_res = np.abs(residual)
    drift = np.nan
    valid_rgb = rgb[np.isfinite(rgb)]
    if residual.size >= 3 and valid_rgb.size == residual.size and np.ptp(valid_rgb) > 0:
        x = (valid_rgb - valid_rgb.mean()) / 3_600_000.0
        drift = float(np.polyfit(x, residual, 1)[0])

    row.update(
        {
            "sync_evidence_level": "rgb_frame_axis_plus_blink_events",
            "rgb_frame_axis_available": True,
            "rgb_minus_nir_start_ms": (
                float(rgb_metrics["time_min_ms"] - nir_metrics["time_min_ms"])
                if np.isfinite(rgb_metrics["time_min_ms"]) and np.isfinite(nir_metrics["time_min_ms"])
                else np.nan
            ),
            "rgb_minus_nir_end_ms": (
                float(rgb_metrics["time_max_ms"] - nir_metrics["time_max_ms"])
                if np.isfinite(rgb_metrics["time_max_ms"]) and np.isfinite(nir_metrics["time_max_ms"])
                else np.nan
            ),
            "rgb_minus_nir_span_ms": (
                float(rgb_metrics["span_ms"] - nir_metrics["span_ms"])
                if np.isfinite(rgb_metrics["span_ms"]) and np.isfinite(nir_metrics["span_ms"])
                else np.nan
            ),
            "frame_nearest_residual_signed_median_ms": float(np.median(residual)) if residual.size else np.nan,
            "frame_nearest_residual_abs_median_ms": float(np.median(abs_res)) if abs_res.size else np.nan,
            "frame_nearest_residual_abs_p95_ms": float(np.quantile(abs_res, 0.95)) if abs_res.size else np.nan,
            "frame_nearest_residual_abs_max_ms": float(np.max(abs_res)) if abs_res.size else np.nan,
            "frame_nearest_residual_drift_ms_per_hour": drift,
        }
    )
    return pd.DataFrame([row])


def load_session_rgb_blink_frames(root: str | Path | None, session_id: str) -> pd.DataFrame | None:
    if root is None:
        return None
    path = Path(root).expanduser() / session_id / f"{session_id}_blink_candidate_frames.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path, columns=["unix_ms"])
