"""Event-specific pupil response candidates for the formal pupil-only pipeline.

These features are deliberately separated from generic tonic/state windows.
They are computed only around an explicit trial onset from the analysis-ready
reference signal, with coverage/gap gates and a prespecified recovery rule.
Nothing in this module freezes a final pupil endpoint.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from attention_pipeline.config import Config, load_config
from .pupil_tables import selected_sessions

EVENT_RESPONSE_VERSION = "nir-event-response-candidates-v1"


@dataclass(frozen=True)
class EventResponseConfig:
    baseline_start_ms: float = -200.0
    baseline_end_ms: float = 0.0
    response_start_ms: float = 0.0
    response_end_ms: float = 1150.0
    late_recovery_window_ms: float = 200.0
    min_baseline_valid_n: int = 3
    min_response_valid_n: int = 10
    min_late_recovery_valid_n: int = 3
    max_gap_sec: float = 0.25
    recovery_fraction_of_peak: float = 0.20
    recovery_noise_mad_multiplier: float = 1.0


def _resolve(config: Config, key: str) -> Path:
    raw = config.section("paths").get(key)
    if raw in (None, ""):
        raise KeyError(f"formal pupil config missing paths.{key}")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (config.path.parent.parent / path).resolve()
    return path


def _frame_path(config: Config, session_id: str) -> Path:
    return (
        _resolve(config, "analysis_ready_root")
        / "frame_level"
        / session_id
        / f"{session_id}_nir_analysis_ready.csv"
    )


def _trial_path(config: Config, session_id: str) -> Path:
    return (
        _resolve(config, "output_root")
        / "sessions"
        / session_id
        / f"{session_id}_trial_level.csv"
    )


def _max_gap_sec(times_ms: np.ndarray) -> float:
    t = np.asarray(times_ms, dtype=float)
    t = t[np.isfinite(t)]
    if t.size < 2:
        return np.nan
    t = np.sort(t)
    return float(np.max(np.diff(t)) / 1000.0)


def _finite_segment(
    times_ms: np.ndarray,
    values: np.ndarray,
    start_ms: float,
    end_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(times_ms, dtype=float)
    v = np.asarray(values, dtype=float)
    mask = np.isfinite(t) & np.isfinite(v) & (t >= start_ms) & (t < end_ms)
    t = t[mask]
    v = v[mask]
    if t.size:
        order = np.argsort(t)
        t, v = t[order], v[order]
    return t, v


def _recovery_time_ms(
    response_t: np.ndarray,
    response_v: np.ndarray,
    *,
    onset_ms: float,
    baseline: float,
    baseline_mad: float,
    peak_index: int,
    peak_amplitude_abs: float,
    cfg: EventResponseConfig,
) -> tuple[float | None, float]:
    tolerance = max(
        cfg.recovery_fraction_of_peak * peak_amplitude_abs,
        cfg.recovery_noise_mad_multiplier * baseline_mad if np.isfinite(baseline_mad) else 0.0,
    )
    if not np.isfinite(tolerance) or tolerance <= 0:
        return None, tolerance
    for idx in range(int(peak_index) + 1, len(response_v)):
        if abs(float(response_v[idx]) - baseline) <= tolerance:
            return float(response_t[idx] - onset_ms), float(tolerance)
    return None, float(tolerance)


def event_response_features(
    times_ms: np.ndarray,
    values: np.ndarray,
    *,
    onset_ms: float,
    response_end_ms: float | None = None,
    config: EventResponseConfig | None = None,
) -> dict[str, Any]:
    """Compute explicitly gated event-response features for one trial event."""
    cfg = config or EventResponseConfig()
    response_end = onset_ms + cfg.response_end_ms
    if response_end_ms is not None and np.isfinite(response_end_ms):
        response_end = min(response_end, float(response_end_ms))
    baseline_start = onset_ms + cfg.baseline_start_ms
    baseline_end = onset_ms + cfg.baseline_end_ms
    response_start = onset_ms + cfg.response_start_ms

    bt, bv = _finite_segment(times_ms, values, baseline_start, baseline_end)
    rt, rv = _finite_segment(times_ms, values, response_start, response_end)
    late_start = max(response_start, response_end - cfg.late_recovery_window_ms)
    lt, lv = _finite_segment(times_ms, values, late_start, response_end)

    result: dict[str, Any] = {
        "event_response_version": EVENT_RESPONSE_VERSION,
        "baseline_valid_n": int(len(bv)),
        "response_valid_n": int(len(rv)),
        "late_recovery_valid_n": int(len(lv)),
        "baseline_max_gap_sec": _max_gap_sec(bt),
        "response_max_gap_sec": _max_gap_sec(rt),
        "baseline_start_ms": float(baseline_start),
        "baseline_end_ms": float(baseline_end),
        "response_start_ms": float(response_start),
        "response_end_ms": float(response_end),
        "recovery_fraction_of_peak": cfg.recovery_fraction_of_peak,
        "recovery_noise_mad_multiplier": cfg.recovery_noise_mad_multiplier,
    }
    reasons: list[str] = []
    if len(bv) < cfg.min_baseline_valid_n:
        reasons.append("low_baseline_valid_n")
    if len(rv) < cfg.min_response_valid_n:
        reasons.append("low_response_valid_n")
    if np.isfinite(result["baseline_max_gap_sec"]) and result["baseline_max_gap_sec"] > cfg.max_gap_sec:
        reasons.append("baseline_gap_exceeds_gate")
    if np.isfinite(result["response_max_gap_sec"]) and result["response_max_gap_sec"] > cfg.max_gap_sec:
        reasons.append("response_gap_exceeds_gate")
    if response_end <= response_start:
        reasons.append("response_interval_empty_or_truncated")

    if reasons:
        result.update({
            "event_response_status": "not_estimable",
            "event_response_reasons": ";".join(reasons),
            "baseline_median": np.nan,
            "baseline_mad": np.nan,
            "dilation_peak_amplitude": np.nan,
            "dilation_peak_latency_ms": np.nan,
            "constriction_peak_amplitude": np.nan,
            "constriction_peak_latency_ms": np.nan,
            "dominant_peak_direction": pd.NA,
            "dominant_peak_amplitude_abs": np.nan,
            "dominant_peak_latency_ms": np.nan,
            "recovery_tolerance": np.nan,
            "recovery_time_after_onset_ms": np.nan,
            "recovery_status": "not_estimable_event_response_failed",
            "late_recovery_residual": np.nan,
            "late_recovery_abs_residual": np.nan,
        })
        return result

    baseline = float(np.median(bv))
    baseline_mad = float(np.median(np.abs(bv - baseline)))
    delta = rv - baseline
    dilation_index = int(np.argmax(delta))
    constriction_index = int(np.argmin(delta))
    dilation_amp = max(0.0, float(delta[dilation_index]))
    constriction_amp = max(0.0, float(-delta[constriction_index]))
    if dilation_amp >= constriction_amp:
        direction = "dilation"
        peak_index = dilation_index
        peak_amp_abs = dilation_amp
    else:
        direction = "constriction"
        peak_index = constriction_index
        peak_amp_abs = constriction_amp

    recovery_time, tolerance = _recovery_time_ms(
        rt,
        rv,
        onset_ms=onset_ms,
        baseline=baseline,
        baseline_mad=baseline_mad,
        peak_index=peak_index,
        peak_amplitude_abs=peak_amp_abs,
        cfg=cfg,
    )
    if len(lv) >= cfg.min_late_recovery_valid_n:
        late_residual = float(np.median(lv) - baseline)
        late_abs = abs(late_residual)
        late_status = "estimable"
    else:
        late_residual = np.nan
        late_abs = np.nan
        late_status = "not_estimable_low_late_recovery_valid_n"

    result.update({
        "event_response_status": "estimable",
        "event_response_reasons": "",
        "baseline_median": baseline,
        "baseline_mad": baseline_mad,
        "dilation_peak_amplitude": dilation_amp,
        "dilation_peak_latency_ms": float(rt[dilation_index] - onset_ms),
        "constriction_peak_amplitude": constriction_amp,
        "constriction_peak_latency_ms": float(rt[constriction_index] - onset_ms),
        "dominant_peak_direction": direction,
        "dominant_peak_amplitude_abs": peak_amp_abs,
        "dominant_peak_latency_ms": float(rt[peak_index] - onset_ms),
        "recovery_tolerance": tolerance,
        "recovery_time_after_onset_ms": recovery_time if recovery_time is not None else np.nan,
        "recovery_status": (
            "recovered_within_response_window"
            if recovery_time is not None
            else "not_recovered_within_response_window"
        ),
        "late_recovery_residual": late_residual,
        "late_recovery_abs_residual": late_abs,
        "late_recovery_status": late_status,
    })
    return result


def _load_reference_signal(config: Config, session_id: str) -> pd.DataFrame:
    path = _frame_path(config, session_id)
    if not path.is_file():
        raise FileNotFoundError(path)
    use = ["session_id", "block", "unix_ms", "binocular_pupil"]
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    missing = sorted(set(use) - set(header.columns))
    if missing:
        raise ValueError(f"{session_id}: analysis-ready reference signal missing {missing}")
    frame = pd.read_csv(path, usecols=use, encoding="utf-8-sig", low_memory=False)
    frame["block"] = pd.to_numeric(frame["block"], errors="coerce")
    frame["unix_ms"] = pd.to_numeric(frame["unix_ms"], errors="coerce")
    frame["binocular_pupil"] = pd.to_numeric(frame["binocular_pupil"], errors="coerce")
    return frame


def build_session_event_responses(
    config: Config,
    session_id: str,
    *,
    feature_config: EventResponseConfig | None = None,
) -> pd.DataFrame:
    cfg = feature_config or EventResponseConfig()
    frame = _load_reference_signal(config, session_id)
    trial_path = _trial_path(config, session_id)
    if not trial_path.is_file():
        raise FileNotFoundError(trial_path)
    trials = pd.read_csv(trial_path, encoding="utf-8-sig", low_memory=False)
    required = {
        "session_id", "analysis_group_token", "block_num", "trial_num",
        "global_trial_index", "absolute_onset_time",
    }
    missing = sorted(required - set(trials.columns))
    if missing:
        raise ValueError(f"{session_id}: trial-level table missing {missing}")

    rows: list[dict[str, Any]] = []
    for block_num, block_trials in trials.groupby("block_num", sort=True):
        signal = frame[frame["block"].eq(int(block_num))].sort_values("unix_ms", kind="stable")
        times = signal["unix_ms"].to_numpy(dtype=float)
        values = signal["binocular_pupil"].to_numpy(dtype=float)
        ordered = block_trials.sort_values("absolute_onset_time", kind="stable").reset_index(drop=True)
        onsets = pd.to_numeric(ordered["absolute_onset_time"], errors="coerce")
        next_onsets = onsets.shift(-1)
        for idx, trial in ordered.iterrows():
            onset = float(onsets.iloc[idx])
            next_onset = next_onsets.iloc[idx]
            response_end = float(next_onset) if np.isfinite(next_onset) else onset + cfg.response_end_ms
            feature = event_response_features(
                times,
                values,
                onset_ms=onset,
                response_end_ms=response_end,
                config=cfg,
            )
            rows.append({
                "session_id": str(trial["session_id"]),
                "analysis_group_token": str(trial["analysis_group_token"]),
                "block_num": int(block_num),
                "trial_num": int(trial["trial_num"]),
                "global_trial_index": int(trial["global_trial_index"]),
                "trial_onset_ms": onset,
                "reference_signal": "binocular_primary_geom_mean_centered",
                "reference_signal_is_final_endpoint": False,
                **feature,
            })
    return pd.DataFrame(rows)


def run_event_response_candidates(
    config_path: str | Path,
    *,
    subjects: Iterable[str] | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    sessions = selected_sessions(config, subjects)
    root = _resolve(config, "output_root") / "event_response_candidates"
    root.mkdir(parents=True, exist_ok=True)
    tables: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    for session_id in sessions:
        try:
            tables.append(build_session_event_responses(config, session_id))
        except Exception as exc:
            failures.append({
                "session_id": session_id,
                "stage": "event_response_candidates",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
    combined = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    failure_table = pd.DataFrame(failures, columns=["session_id", "stage", "error_type", "error"])
    combined.to_csv(root / "trial_event_response_candidates.csv", index=False, encoding="utf-8-sig")
    failure_table.to_csv(root / "event_response_failures.csv", index=False, encoding="utf-8-sig")
    estimable = (
        int(combined["event_response_status"].eq("estimable").sum())
        if "event_response_status" in combined else 0
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if not failures else "partial",
        "pipeline_version": EVENT_RESPONSE_VERSION,
        "n_sessions_requested": len(sessions),
        "n_sessions_failed": len(failures),
        "n_trial_events": int(len(combined)),
        "n_event_response_estimable": estimable,
        "reference_signal_is_final_endpoint": False,
        "feature_role": "candidate_event_response_only",
        "baseline_window_ms": [-200, 0],
        "response_window_ms": [0, 1150],
        "response_end_is_capped_at_next_trial_onset": True,
        "recovery_rule": "first post-dominant-peak sample within max(20% peak amplitude, 1x baseline MAD)",
        "frequency_features_admitted": False,
        "endpoint_freeze": "pending_real_data_scientific_review",
        "scientific_inference_authorized_by_code_alone": False,
    }
    (root / "event_response_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
