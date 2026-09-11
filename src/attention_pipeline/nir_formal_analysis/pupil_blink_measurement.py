from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


AUDIT_SCHEMA_VERSION = "pupil-blink-measurement-audit-v1"
FORMAL_PHASE_TO_BLOCK = {"block1": 1, "block2": 2}
GEOMETRY_SIGNAL = "pupil_geom_mean_diameter"
RSEG_HARD_SIGNAL = "seg_pupil_fraction_within_pupil_iris_hard"
RSEG_SOFT_SIGNAL = "seg_pupil_fraction_within_pupil_iris_soft"
SIGNAL_ROLES = {
    GEOMETRY_SIGNAL: "primary_candidate_geometry",
    RSEG_HARD_SIGNAL: "primary_candidate_hard_segmentation",
    RSEG_SOFT_SIGNAL: "sensitivity_candidate_soft_probability_mass",
}


@dataclass(frozen=True)
class BlinkBuffer:
    name: str
    pre_ms: float
    post_ms: float


DEFAULT_BLINK_BUFFERS: tuple[BlinkBuffer, ...] = (
    BlinkBuffer("pre50_post50", 50.0, 50.0),
    BlinkBuffer("pre100_post300", 100.0, 300.0),
    BlinkBuffer("pre200_post200", 200.0, 200.0),
)


def _as_bool(series: pd.Series | None, index: pd.Index, *, default: bool = False) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=bool)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default).astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y", "t"})


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _derive_block(frame: pd.DataFrame) -> pd.Series:
    if "block" in frame.columns:
        block = pd.to_numeric(frame["block"], errors="coerce")
    elif "block_num" in frame.columns:
        block = pd.to_numeric(frame["block_num"], errors="coerce")
    elif "phase" in frame.columns:
        phase = frame["phase"].astype("string").str.strip().str.lower()
        block = phase.map(FORMAL_PHASE_TO_BLOCK)
    else:
        raise ValueError("pupil rows require block/block_num or phase=block1/block2")
    if block.isna().any():
        raise ValueError("pupil rows contain missing/non-formal block identity")
    return block.astype(int)


def _reason_series(reason_masks: Sequence[tuple[str, pd.Series]], index: pd.Index) -> pd.Series:
    reasons: list[str] = []
    for idx in index:
        labels = [label for label, mask in reason_masks if bool(mask.loc[idx])]
        reasons.append("|".join(labels))
    return pd.Series(reasons, index=index, dtype="string")


def derive_eye_measurements(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive geometry and segmentation signals with independent validity.

    The function consumes canonical ``nir_pupil_only`` rows. Segmentation validity
    intentionally does not depend on ellipse/geometry fit validity.
    """
    required = {"session_id", "eye", "unix_ms", "hard_pupil_fraction", "hard_iris_fraction"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"pupil-only rows missing measurement-audit fields: {missing}")

    out = frame.copy()
    out["session_id"] = out["session_id"].astype(str)
    out["eye"] = out["eye"].astype("string").str.strip().str.lower()
    bad_eyes = sorted(set(out["eye"].dropna().astype(str)) - {"left", "right"})
    if bad_eyes:
        raise ValueError(f"unsupported eye labels: {bad_eyes}")
    out["block"] = _derive_block(out)
    out["unix_ms"] = pd.to_numeric(out["unix_ms"], errors="coerce")
    if out["unix_ms"].isna().any():
        raise ValueError("pupil-only rows contain non-numeric unix_ms")

    index = out.index
    source_observed = _as_bool(out.get("source_observed"), index)
    if "source_observed" not in out.columns and "source_eye_status" in out.columns:
        source_observed = out["source_eye_status"].astype("string").str.strip().str.lower().eq("observed")
    ritnet_missing = _as_bool(out.get("ritnet_missing"), index)
    if "ritnet_missing" not in out.columns and "ritnet_status" in out.columns:
        ritnet_missing = ~out["ritnet_status"].astype("string").str.strip().str.lower().eq("success")
    interpolation_only = _as_bool(out.get("interpolation_only"), index)
    temporal_flagged = _as_bool(out.get("temporal_flagged"), index)
    if "temporal_flagged" not in out.columns and "temporal_anomaly" in out.columns:
        temporal_flagged = _as_bool(out.get("temporal_anomaly"), index)

    geometry = _numeric(out, GEOMETRY_SIGNAL)
    geometry_value_ok = np.isfinite(geometry) & geometry.gt(0)
    geometry_valid = source_observed & ~ritnet_missing & ~interpolation_only & ~temporal_flagged & geometry_value_ok
    geometry_reasons = _reason_series(
        (
            ("source_not_observed", ~source_observed),
            ("ritnet_not_success", ritnet_missing),
            ("interpolation_only", interpolation_only),
            ("temporal_flagged", temporal_flagged),
            ("geometry_nonfinite_or_nonpositive", ~geometry_value_ok),
        ),
        index,
    )

    hard_pupil = _numeric(out, "hard_pupil_fraction")
    hard_iris = _numeric(out, "hard_iris_fraction")
    hard_pupil_ok = np.isfinite(hard_pupil) & hard_pupil.between(0.0, 1.0, inclusive="both")
    hard_iris_ok = np.isfinite(hard_iris) & hard_iris.between(0.0, 1.0, inclusive="both")
    hard_denom = hard_pupil + hard_iris
    hard_denom_ok = np.isfinite(hard_denom) & hard_denom.gt(0)
    hard_valid = (
        source_observed
        & ~ritnet_missing
        & ~interpolation_only
        & ~temporal_flagged
        & hard_pupil_ok
        & hard_iris_ok
        & hard_denom_ok
    )
    hard_ratio = pd.Series(np.nan, index=index, dtype=float)
    hard_ratio.loc[hard_valid] = (hard_pupil.loc[hard_valid] / hard_denom.loc[hard_valid]).astype(float)
    hard_reasons = _reason_series(
        (
            ("source_not_observed", ~source_observed),
            ("ritnet_not_success", ritnet_missing),
            ("interpolation_only", interpolation_only),
            ("temporal_flagged", temporal_flagged),
            ("hard_pupil_fraction_invalid", ~hard_pupil_ok),
            ("hard_iris_fraction_invalid", ~hard_iris_ok),
            ("pupil_iris_denominator_nonpositive", ~hard_denom_ok),
        ),
        index,
    )

    out[GEOMETRY_SIGNAL] = geometry
    out[f"{GEOMETRY_SIGNAL}__raw_computable"] = geometry_value_ok.astype(bool)
    out[f"{GEOMETRY_SIGNAL}__audit_valid"] = geometry_valid.astype(bool)
    out[f"{GEOMETRY_SIGNAL}__audit_invalid_reason"] = geometry_reasons
    hard_raw_ratio = pd.Series(np.nan, index=index, dtype=float)
    hard_raw_computable = hard_pupil_ok & hard_iris_ok & hard_denom_ok
    hard_raw_ratio.loc[hard_raw_computable] = hard_pupil.loc[hard_raw_computable] / hard_denom.loc[hard_raw_computable]
    out[f"{RSEG_HARD_SIGNAL}__raw_value"] = hard_raw_ratio
    out[f"{RSEG_HARD_SIGNAL}__raw_computable"] = hard_raw_computable.astype(bool)
    out[RSEG_HARD_SIGNAL] = hard_ratio
    out[f"{RSEG_HARD_SIGNAL}__audit_valid"] = hard_valid.astype(bool)
    out[f"{RSEG_HARD_SIGNAL}__audit_invalid_reason"] = hard_reasons

    if {"soft_pupil_fraction", "soft_iris_fraction"}.issubset(out.columns):
        soft_pupil = _numeric(out, "soft_pupil_fraction")
        soft_iris = _numeric(out, "soft_iris_fraction")
        soft_pupil_ok = np.isfinite(soft_pupil) & soft_pupil.between(0.0, 1.0, inclusive="both")
        soft_iris_ok = np.isfinite(soft_iris) & soft_iris.between(0.0, 1.0, inclusive="both")
        soft_denom = soft_pupil + soft_iris
        soft_denom_ok = np.isfinite(soft_denom) & soft_denom.gt(0)
        soft_valid = (
            source_observed
            & ~ritnet_missing
            & ~interpolation_only
            & ~temporal_flagged
            & soft_pupil_ok
            & soft_iris_ok
            & soft_denom_ok
        )
        soft_raw_computable = soft_pupil_ok & soft_iris_ok & soft_denom_ok
        soft_raw_ratio = pd.Series(np.nan, index=index, dtype=float)
        soft_raw_ratio.loc[soft_raw_computable] = soft_pupil.loc[soft_raw_computable] / soft_denom.loc[soft_raw_computable]
        soft_ratio = pd.Series(np.nan, index=index, dtype=float)
        soft_ratio.loc[soft_valid] = soft_pupil.loc[soft_valid] / soft_denom.loc[soft_valid]
        soft_reasons = _reason_series(
            (
                ("source_not_observed", ~source_observed),
                ("ritnet_not_success", ritnet_missing),
                ("interpolation_only", interpolation_only),
                ("temporal_flagged", temporal_flagged),
                ("soft_pupil_fraction_invalid", ~soft_pupil_ok),
                ("soft_iris_fraction_invalid", ~soft_iris_ok),
                ("pupil_iris_denominator_nonpositive", ~soft_denom_ok),
            ),
            index,
        )
        out[f"{RSEG_SOFT_SIGNAL}__raw_value"] = soft_raw_ratio
        out[f"{RSEG_SOFT_SIGNAL}__raw_computable"] = soft_raw_computable.astype(bool)
        out[RSEG_SOFT_SIGNAL] = soft_ratio
        out[f"{RSEG_SOFT_SIGNAL}__audit_valid"] = soft_valid.astype(bool)
        out[f"{RSEG_SOFT_SIGNAL}__audit_invalid_reason"] = soft_reasons

    return out


def _first_consistent_metadata(group: pd.DataFrame, name: str) -> object:
    if name not in group.columns:
        return pd.NA
    values = group[name].dropna()
    if values.empty:
        return pd.NA
    unique = values.astype(str).unique()
    if len(unique) > 1:
        raise ValueError(f"conflicting {name} values within binocular timepoint: {sorted(unique.tolist())}")
    return values.iloc[0]


def build_binocular_measurement_timepoints(frame: pd.DataFrame) -> pd.DataFrame:
    """Fuse valid eyes per timepoint separately for geometry and segmentation signals."""
    eye = derive_eye_measurements(frame)
    key = ["session_id", "block", "unix_ms"]
    if "frame_idx" in eye.columns:
        key.append("frame_idx")
    if eye.duplicated([*key, "eye"], keep=False).any():
        raise ValueError(f"duplicate eye-time key: {[*key, 'eye']}")

    metadata_cols = [
        name
        for name in ("participant_group_id", "analysis_group_token", "subject", "phase", "phase_segment")
        if name in eye.columns
    ]
    rows: list[dict[str, object]] = []
    for keys, group in eye.groupby(key, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(key, keys))
        for name in metadata_cols:
            row[name] = _first_consistent_metadata(group, name)
        signals = [GEOMETRY_SIGNAL, RSEG_HARD_SIGNAL]
        if RSEG_SOFT_SIGNAL in eye.columns:
            signals.append(RSEG_SOFT_SIGNAL)
        for signal in signals:
            valid_col = f"{signal}__audit_valid"
            raw_computable_col = f"{signal}__raw_computable"
            reason_col = f"{signal}__audit_invalid_reason"
            raw_value_col = signal if signal == GEOMETRY_SIGNAL else f"{signal}__raw_value"
            raw_values: dict[str, float] = {}
            qc_values: dict[str, float] = {}
            raw_ok_by_eye: dict[str, bool] = {}
            valid_by_eye: dict[str, bool] = {}
            reasons_by_eye: dict[str, str] = {}
            for eye_name in ("left", "right"):
                current = group[group["eye"].astype(str).eq(eye_name)]
                if current.empty:
                    raw_values[eye_name] = np.nan
                    qc_values[eye_name] = np.nan
                    raw_ok_by_eye[eye_name] = False
                    valid_by_eye[eye_name] = False
                    reasons_by_eye[eye_name] = "eye_row_missing"
                else:
                    item = current.iloc[0]
                    raw_values[eye_name] = float(item[raw_value_col]) if pd.notna(item[raw_value_col]) else np.nan
                    qc_values[eye_name] = float(item[signal]) if pd.notna(item[signal]) else np.nan
                    raw_ok_by_eye[eye_name] = bool(item[raw_computable_col])
                    valid_by_eye[eye_name] = bool(item[valid_col])
                    reasons_by_eye[eye_name] = str(item[reason_col]) if pd.notna(item[reason_col]) else ""

            def fuse(values: dict[str, float], flags: dict[str, bool]) -> tuple[float, str]:
                if flags["left"] and flags["right"]:
                    return (values["left"] + values["right"]) / 2.0, "binocular"
                if flags["left"]:
                    return values["left"], "left_only"
                if flags["right"]:
                    return values["right"], "right_only"
                return np.nan, "missing"

            raw_fused, raw_mode = fuse(raw_values, raw_ok_by_eye)
            qc_fused, qc_mode = fuse(qc_values, valid_by_eye)
            row[f"{signal}__raw"] = raw_fused
            row[f"{signal}__raw_source_mode"] = raw_mode
            row[signal] = qc_fused
            row[f"{signal}__source_mode"] = qc_mode
            row[f"{signal}__left_invalid_reason"] = reasons_by_eye["left"]
            row[f"{signal}__right_invalid_reason"] = reasons_by_eye["right"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(key, kind="stable").reset_index(drop=True)


def _normalize_blink_events(events: pd.DataFrame, session_ids: Sequence[str]) -> pd.DataFrame:
    required = {"start_unix_ms", "end_unix_ms"}
    missing = sorted(required - set(events.columns))
    if events.empty and missing:
        out = events.copy()
        for name in missing:
            out[name] = pd.Series(dtype=float)
    elif missing:
        raise ValueError(f"RGB blink events missing fields: {missing}")
    else:
        out = events.copy()
    out["start_unix_ms"] = pd.to_numeric(out["start_unix_ms"], errors="coerce")
    out["end_unix_ms"] = pd.to_numeric(out["end_unix_ms"], errors="coerce")
    if out[["start_unix_ms", "end_unix_ms"]].isna().any(axis=1).any():
        raise ValueError("RGB blink events contain non-numeric boundaries")
    if (out["end_unix_ms"] < out["start_unix_ms"]).any():
        raise ValueError("RGB blink event ends before it starts")
    if "session_id" not in out.columns:
        unique = sorted(set(str(x) for x in session_ids))
        if len(unique) != 1:
            raise ValueError("blink events without session_id can only align to one-session NIR input")
        out["session_id"] = unique[0]
    out["session_id"] = out["session_id"].astype(str)
    return out


def add_rgb_blink_mask(
    timepoints: pd.DataFrame,
    events: pd.DataFrame,
    *,
    pre_buffer_ms: float,
    post_buffer_ms: float,
) -> pd.DataFrame:
    """Add session-local RGB blink core and buffered masks; overlapping events union naturally."""
    if pre_buffer_ms < 0 or post_buffer_ms < 0:
        raise ValueError("blink buffers must be non-negative")
    required = {"session_id", "unix_ms"}
    missing = sorted(required - set(timepoints.columns))
    if missing:
        raise ValueError(f"NIR timepoints missing blink-alignment fields: {missing}")
    out = timepoints.copy()
    out["unix_ms"] = pd.to_numeric(out["unix_ms"], errors="coerce")
    if out["unix_ms"].isna().any():
        raise ValueError("NIR timepoints contain non-numeric unix_ms")
    blink = _normalize_blink_events(events, out["session_id"].astype(str).unique().tolist())
    core = pd.Series(False, index=out.index)
    buffered = pd.Series(False, index=out.index)
    cover_count = pd.Series(0, index=out.index, dtype=int)
    for session_id, group in blink.groupby("session_id", sort=False):
        idx = out.index[out["session_id"].astype(str).eq(str(session_id))]
        if len(idx) == 0:
            continue
        times = out.loc[idx, "unix_ms"]
        for _, event in group.iterrows():
            start = float(event["start_unix_ms"])
            end = float(event["end_unix_ms"])
            event_core = times.ge(start) & times.le(end)
            event_buffered = times.ge(start - pre_buffer_ms) & times.le(end + post_buffer_ms)
            core.loc[idx] |= event_core.to_numpy()
            buffered.loc[idx] |= event_buffered.to_numpy()
            cover_count.loc[idx] += event_buffered.astype(int).to_numpy()
    out["rgb_blink_core_mask"] = core.astype(bool)
    out["rgb_blink_mask"] = buffered.astype(bool)
    out["rgb_blink_buffer_pre_ms"] = float(pre_buffer_ms)
    out["rgb_blink_buffer_post_ms"] = float(post_buffer_ms)
    out["rgb_blink_event_coverage_count"] = cover_count
    return out


def _nearest_signed_residual(reference: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if reference.size == 0:
        return np.full(targets.shape, np.nan, dtype=float)
    reference = np.sort(reference[np.isfinite(reference)])
    if reference.size == 0:
        return np.full(targets.shape, np.nan, dtype=float)
    positions = np.searchsorted(reference, targets)
    left_pos = np.clip(positions - 1, 0, len(reference) - 1)
    right_pos = np.clip(positions, 0, len(reference) - 1)
    left = reference[left_pos]
    right = reference[right_pos]
    choose_right = np.abs(right - targets) < np.abs(left - targets)
    nearest = np.where(choose_right, right, left)
    return nearest - targets


def audit_rgb_nir_sync(timepoints: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Summarize event-boundary nearest-frame residuals and residual drift per session."""
    if "session_id" not in timepoints.columns or "unix_ms" not in timepoints.columns:
        raise ValueError("NIR timepoints require session_id/unix_ms for sync audit")
    blink = _normalize_blink_events(events, timepoints["session_id"].astype(str).unique().tolist())
    sessions = sorted(set(timepoints["session_id"].astype(str)) | set(blink["session_id"].astype(str)))
    rows: list[dict[str, object]] = []
    for session_id in sessions:
        nir = pd.to_numeric(
            timepoints.loc[timepoints["session_id"].astype(str).eq(session_id), "unix_ms"],
            errors="coerce",
        ).dropna().to_numpy(float)
        ev = blink[blink["session_id"].astype(str).eq(session_id)].copy()
        boundaries = np.concatenate(
            [
                pd.to_numeric(ev["start_unix_ms"], errors="coerce").to_numpy(float),
                pd.to_numeric(ev["end_unix_ms"], errors="coerce").to_numpy(float),
            ]
        ) if len(ev) else np.array([], dtype=float)
        residual = _nearest_signed_residual(np.unique(nir), boundaries) if len(boundaries) else np.array([], dtype=float)
        finite = residual[np.isfinite(residual)]
        abs_res = np.abs(finite)
        drift_per_hour = np.nan
        if len(finite) >= 3 and np.ptp(boundaries[np.isfinite(residual)]) > 0:
            x = boundaries[np.isfinite(residual)]
            x = (x - x.mean()) / 3_600_000.0
            drift_per_hour = float(np.polyfit(x, finite, 1)[0])
        nir_sorted = np.sort(nir[np.isfinite(nir)])
        dt = np.diff(nir_sorted)
        positive_dt = dt[dt > 0]
        nominal_dt = float(np.median(positive_dt)) if len(positive_dt) else np.nan
        large_gap_threshold = max(1000.0, 10.0 * nominal_dt) if np.isfinite(nominal_dt) else 1000.0
        rows.append(
            {
                "session_id": session_id,
                "nir_time_min_ms": float(np.min(nir_sorted)) if len(nir_sorted) else np.nan,
                "nir_time_max_ms": float(np.max(nir_sorted)) if len(nir_sorted) else np.nan,
                "nir_unique_time_n": int(len(np.unique(nir_sorted))),
                "nir_nonpositive_step_n": int(np.sum(dt <= 0)) if len(dt) else 0,
                "nir_large_gap_n": int(np.sum(dt > large_gap_threshold)) if len(dt) else 0,
                "blink_event_n": int(len(ev)),
                "blink_boundary_n": int(len(boundaries)),
                "nearest_residual_signed_median_ms": float(np.median(finite)) if len(finite) else np.nan,
                "nearest_residual_abs_median_ms": float(np.median(abs_res)) if len(abs_res) else np.nan,
                "nearest_residual_abs_p95_ms": float(np.quantile(abs_res, 0.95)) if len(abs_res) else np.nan,
                "nearest_residual_abs_max_ms": float(np.max(abs_res)) if len(abs_res) else np.nan,
                "nearest_residual_drift_ms_per_hour": drift_per_hour,
                "sync_status": "no_blink_events" if len(ev) == 0 else "audited",
            }
        )
    return pd.DataFrame(rows)


def select_probe_window(
    timepoints: pd.DataFrame,
    *,
    session_id: str,
    block_num: int,
    probe_onset_ms: float,
    window_sec: float = 30.0,
) -> pd.DataFrame:
    onset = float(probe_onset_ms)
    start = onset - float(window_sec) * 1000.0
    times = pd.to_numeric(timepoints["unix_ms"], errors="coerce")
    blocks = pd.to_numeric(timepoints["block"], errors="coerce")
    mask = (
        timepoints["session_id"].astype(str).eq(str(session_id))
        & blocks.eq(int(block_num))
        & times.ge(start)
        & times.lt(onset)
    )
    out = timepoints.loc[mask].copy()
    out["probe_relative_sec"] = (pd.to_numeric(out["unix_ms"], errors="coerce") - onset) / 1000.0
    return out.sort_values("unix_ms", kind="stable").reset_index(drop=True)


def signal_values_and_mask(window: pd.DataFrame, signal: str, cleaning_track: str) -> tuple[pd.Series, pd.Series]:
    raw_col = f"{signal}__raw"
    raw_values = pd.to_numeric(window[raw_col] if raw_col in window.columns else window[signal], errors="coerce")
    qc_values = pd.to_numeric(window[signal], errors="coerce")
    blink_valid = ~window.get("rgb_blink_mask", pd.Series(False, index=window.index)).fillna(False).astype(bool)
    if cleaning_track == "original_nir":
        return raw_values, pd.Series(np.isfinite(raw_values), index=window.index)
    if cleaning_track == "rgb_blink_only":
        return raw_values, pd.Series(np.isfinite(raw_values), index=window.index) & blink_valid
    if cleaning_track == "nir_qc_only":
        return qc_values, pd.Series(np.isfinite(qc_values), index=window.index)
    if cleaning_track == "rgb_plus_nir_qc":
        return qc_values, pd.Series(np.isfinite(qc_values), index=window.index) & blink_valid
    raise ValueError(f"unsupported cleaning_track: {cleaning_track}")


def fixed_probe_bins(
    window: pd.DataFrame,
    *,
    signal: str,
    window_sec: float,
    bin_width_sec: float,
    cleaning_track: str,
) -> pd.DataFrame:
    if bin_width_sec <= 0 or bin_width_sec > window_sec:
        raise ValueError("bin_width_sec must be >0 and <= window_sec")
    if "probe_relative_sec" not in window.columns:
        raise ValueError("window requires probe_relative_sec")
    rel = pd.to_numeric(window["probe_relative_sec"], errors="coerce")
    values, valid = signal_values_and_mask(window, signal, cleaning_track)
    valid = valid & rel.ge(-window_sec) & rel.lt(0)
    edges = np.arange(-float(window_sec), 0.0, float(bin_width_sec))
    if edges.size == 0 or not np.isclose(edges[0], -float(window_sec)):
        edges = np.insert(edges, 0, -float(window_sec))
    if not np.isclose(edges[-1], 0.0):
        edges = np.append(edges, 0.0)
    rows: list[dict[str, object]] = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = valid & rel.ge(lo) & rel.lt(hi)
        y = values[mask]
        rows.append(
            {
                "bin_index": i,
                "bin_start_sec": lo,
                "bin_end_sec": hi,
                "bin_center_sec": (lo + hi) / 2.0,
                "n_valid_samples": int(y.notna().sum()),
                "bin_value_median": float(y.median()) if y.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def fit_fixed_bin_dynamics(bins: pd.DataFrame, *, window_sec: float) -> dict[str, object]:
    valid = bins[np.isfinite(pd.to_numeric(bins["bin_value_median"], errors="coerce"))].copy()
    x = pd.to_numeric(valid["bin_center_sec"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(valid["bin_value_median"], errors="coerce").to_numpy(float)
    n = int(len(valid))
    result: dict[str, object] = {
        "n_valid_bins": n,
        "t_min_sec": float(np.min(x)) if n else np.nan,
        "t_max_sec": float(np.max(x)) if n else np.nan,
        "temporal_span_sec": float(np.max(x) - np.min(x)) if n >= 2 else 0.0 if n == 1 else np.nan,
        "has_early_support": bool(np.any(x < -float(window_sec) / 2.0)) if n else False,
        "has_late_support": bool(np.any(x >= -float(window_sec) / 2.0)) if n else False,
        "linear_slope_per_sec": np.nan,
        "linear_status": "not_estimable_low_valid_bins",
        "quadratic_curvature_per_sec2": np.nan,
        "quadratic_status": "not_estimable_low_valid_bins",
    }
    if n >= 2 and len(np.unique(x)) >= 2:
        result["linear_slope_per_sec"] = float(np.polyfit(x, y, 1)[0])
        result["linear_status"] = "computable"
    elif n >= 2:
        result["linear_status"] = "not_estimable_time_support"
    if n >= 3 and len(np.unique(x)) >= 3:
        midpoint = -float(window_sec) / 2.0
        centered = x - midpoint
        result["quadratic_curvature_per_sec2"] = float(np.polyfit(centered, y, 2)[0])
        result["quadratic_status"] = "computable"
    elif n >= 3:
        result["quadratic_status"] = "not_estimable_time_support"
    return result


def summarize_probe_signal(
    window: pd.DataFrame,
    *,
    signal: str,
    cleaning_track: str,
    window_sec: float,
    bin_width_sec: float,
) -> dict[str, object]:
    values, valid = signal_values_and_mask(window, signal, cleaning_track)
    x = values[valid & np.isfinite(values)]
    n_rows = int(len(window))
    n_valid = int(len(x))
    level_mean = float(x.mean()) if n_valid else np.nan
    level_median = float(x.median()) if n_valid else np.nan
    sd = float(x.std(ddof=1)) if n_valid >= 2 else np.nan
    median = float(x.median()) if n_valid else np.nan
    mad = float(np.median(np.abs(x.to_numpy(float) - median))) if n_valid else np.nan
    iqr = float(x.quantile(0.75) - x.quantile(0.25)) if n_valid else np.nan
    bins = fixed_probe_bins(
        window,
        signal=signal,
        window_sec=window_sec,
        bin_width_sec=bin_width_sec,
        cleaning_track=cleaning_track,
    )
    dynamics = fit_fixed_bin_dynamics(bins, window_sec=window_sec)
    return {
        "signal": signal,
        "signal_role": SIGNAL_ROLES.get(signal, "unspecified"),
        "cleaning_track": cleaning_track,
        "n_window_rows": n_rows,
        "n_valid_samples": n_valid,
        "valid_fraction": float(n_valid / n_rows) if n_rows else np.nan,
        "level_mean": level_mean,
        "level_median": level_median,
        "variability_sd": sd,
        "variability_mad": mad,
        "variability_iqr": iqr,
        "bin_width_sec": float(bin_width_sec),
        **dynamics,
    }


def audit_buffer_loss(
    timepoints: pd.DataFrame,
    events: pd.DataFrame,
    *,
    buffers: Iterable[BlinkBuffer] = DEFAULT_BLINK_BUFFERS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for buffer in buffers:
        masked = add_rgb_blink_mask(
            timepoints,
            events,
            pre_buffer_ms=buffer.pre_ms,
            post_buffer_ms=buffer.post_ms,
        )
        for session_id, group in masked.groupby("session_id", sort=True):
            for signal in (GEOMETRY_SIGNAL, RSEG_HARD_SIGNAL):
                finite = np.isfinite(pd.to_numeric(group[signal], errors="coerce"))
                blink = group["rgb_blink_mask"].fillna(False).astype(bool)
                valid_n = int(finite.sum())
                lost_n = int((finite & blink).sum())
                rows.append(
                    {
                        "session_id": str(session_id),
                        "buffer_id": buffer.name,
                        "pre_buffer_ms": buffer.pre_ms,
                        "post_buffer_ms": buffer.post_ms,
                        "signal": signal,
                        "finite_before_blink_mask_n": valid_n,
                        "finite_lost_to_blink_mask_n": lost_n,
                        "finite_lost_fraction": float(lost_n / valid_n) if valid_n else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def audit_signal_availability(eye_measurements: pd.DataFrame) -> pd.DataFrame:
    """Summarize eye-level availability and invalid-reason counts without thresholds."""
    rows: list[dict[str, object]] = []
    for (session_id, eye_name), group in eye_measurements.groupby(["session_id", "eye"], sort=True):
        signals = [GEOMETRY_SIGNAL, RSEG_HARD_SIGNAL]
        if RSEG_SOFT_SIGNAL in group.columns:
            signals.append(RSEG_SOFT_SIGNAL)
        for signal in signals:
            valid_col = f"{signal}__audit_valid"
            reason_col = f"{signal}__audit_invalid_reason"
            valid = group[valid_col].fillna(False).astype(bool)
            reason_counts: dict[str, int] = {}
            for value in group.loc[~valid, reason_col].fillna("").astype(str):
                for reason in [x for x in value.split("|") if x]:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
            row: dict[str, object] = {
                "session_id": str(session_id),
                "eye": str(eye_name),
                "signal": signal,
                "n_rows": int(len(group)),
                "n_valid": int(valid.sum()),
                "valid_fraction": float(valid.mean()) if len(group) else np.nan,
            }
            for reason, count in sorted(reason_counts.items()):
                row[f"invalid_reason__{reason}"] = int(count)
            rows.append(row)
    return pd.DataFrame(rows)


def audit_rseg_quality_associations(eye_measurements: pd.DataFrame) -> pd.DataFrame:
    """Descriptive R_seg associations with measurement/QC fields; no inferential selection."""
    candidates = (
        "hard_iris_fraction",
        "hard_pupil_fraction",
        "analysis_valid_pixel_fraction",
        "pupil_largest_component_fraction",
        "fullclass_ocular_aperture_ratio_median",
        "fullclass_ocular_aperture_ratio_p90",
        "soft_margin",
        "soft_entropy",
        "soft_max_probability",
    )
    rows: list[dict[str, object]] = []
    for session_id, group in eye_measurements.groupby("session_id", sort=True):
        rseg = pd.to_numeric(group[RSEG_HARD_SIGNAL], errors="coerce")
        for predictor in candidates:
            if predictor not in group.columns:
                continue
            x = pd.to_numeric(group[predictor], errors="coerce")
            mask = np.isfinite(rseg) & np.isfinite(x)
            n = int(mask.sum())
            rho = float(rseg[mask].corr(x[mask], method="spearman")) if n >= 3 else np.nan
            rows.append(
                {
                    "session_id": str(session_id),
                    "predictor": predictor,
                    "n_pair": n,
                    "spearman_rho": rho,
                    "role": "measurement_sensitivity_descriptive_only",
                }
            )
    return pd.DataFrame(rows)


def build_blink_recovery_bins(
    timepoints: pd.DataFrame,
    events: pd.DataFrame,
    *,
    pre_ms: float = 500.0,
    post_ms: float = 1000.0,
    bin_ms: float = 50.0,
    anchors: Sequence[str] = ("start", "end"),
) -> pd.DataFrame:
    """Build event-level aligned bins around blink onset/offset for recovery inspection.

    These bins are audit output only. They do not delete data or set the formal blink buffer.
    """
    if pre_ms < 0 or post_ms < 0 or bin_ms <= 0:
        raise ValueError("recovery pre/post must be non-negative and bin_ms must be positive")
    blink = _normalize_blink_events(events, timepoints["session_id"].astype(str).unique().tolist())
    rows: list[dict[str, object]] = []
    edges = np.arange(-float(pre_ms), float(post_ms) + float(bin_ms), float(bin_ms))
    if edges[-1] < post_ms:
        edges = np.append(edges, float(post_ms))
    for session_id, ev_group in blink.groupby("session_id", sort=True):
        session = timepoints[timepoints["session_id"].astype(str).eq(str(session_id))].copy()
        times = pd.to_numeric(session["unix_ms"], errors="coerce")
        for event_order, (_, event) in enumerate(ev_group.iterrows(), start=1):
            event_id = event.get("blink_event_id", event_order)
            for anchor in anchors:
                if anchor == "start":
                    anchor_ms = float(event["start_unix_ms"])
                elif anchor == "end":
                    anchor_ms = float(event["end_unix_ms"])
                else:
                    raise ValueError(f"unsupported recovery anchor: {anchor}")
                rel_ms = times - anchor_ms
                for i in range(len(edges) - 1):
                    lo, hi = float(edges[i]), float(edges[i + 1])
                    current = session.loc[rel_ms.ge(lo) & rel_ms.lt(hi)]
                    for signal in (GEOMETRY_SIGNAL, RSEG_HARD_SIGNAL):
                        raw_col = f"{signal}__raw"
                        raw_values = pd.to_numeric(
                            current[raw_col] if raw_col in current.columns else current.get(signal),
                            errors="coerce",
                        )
                        qc_values = pd.to_numeric(current.get(signal), errors="coerce")
                        n_rows = int(len(current))
                        raw_n = int(np.isfinite(raw_values).sum()) if n_rows else 0
                        qc_n = int(np.isfinite(qc_values).sum()) if n_rows else 0
                        rows.append(
                            {
                                "session_id": str(session_id),
                                "blink_event_id": event_id,
                                "anchor": anchor,
                                "relative_bin_start_ms": lo,
                                "relative_bin_end_ms": hi,
                                "relative_bin_center_ms": (lo + hi) / 2.0,
                                "signal": signal,
                                "n_rows": n_rows,
                                "raw_computable_n": raw_n,
                                "raw_computable_fraction": float(raw_n / n_rows) if n_rows else np.nan,
                                "raw_value_median": float(raw_values.median()) if raw_n else np.nan,
                                "nir_qc_valid_n": qc_n,
                                "nir_qc_valid_fraction": float(qc_n / n_rows) if n_rows else np.nan,
                                "nir_qc_value_median": float(qc_values.median()) if qc_n else np.nan,
                            }
                        )
    return pd.DataFrame(rows)
