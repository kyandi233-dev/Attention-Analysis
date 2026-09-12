from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from attention_pipeline.nir_behavior.alignment_v12 import (
    _estimate_sampling_rate_hz,
    _max_temporal_gap_sec,
)
from attention_pipeline.nir_formal_analysis.supervised_features import (
    summarize_supervised_window,
)


def _source_mode_audit(window: pd.DataFrame) -> dict[str, Any]:
    modes = (
        window["raw_binocular_source_mode"].fillna("missing").astype(str).str.strip().str.lower()
        if "raw_binocular_source_mode" in window.columns
        else pd.Series([], dtype="string")
    )
    n_rows = int(len(window))
    if n_rows == 0 or modes.empty:
        return {
            "source_mode_binocular_fraction": None,
            "source_mode_left_only_fraction": None,
            "source_mode_right_only_fraction": None,
            "source_mode_missing_fraction": None,
            "source_mode_summary": "no_rows",
        }

    fractions = {
        name: float(modes.eq(name).mean())
        for name in ("binocular", "left_only", "right_only", "missing")
    }
    observed = [name for name in ("binocular", "left_only", "right_only") if fractions[name] > 0]
    summary = observed[0] if len(observed) == 1 else "mixed" if observed else "missing_only"
    return {
        "source_mode_binocular_fraction": fractions["binocular"],
        "source_mode_left_only_fraction": fractions["left_only"],
        "source_mode_right_only_fraction": fractions["right_only"],
        "source_mode_missing_fraction": fractions["missing"],
        "source_mode_summary": summary,
    }


def audit_supervised_window_support(
    window: pd.DataFrame,
    *,
    requested_start_ms: float,
    requested_end_ms: float,
    available_start_ms: float | None = None,
    available_end_ms: float | None = None,
) -> dict[str, Any]:
    """Audit time support and candidate computability for one supervised NIR window.

    ``requested_*`` are the design window bounds. ``available_*`` may be narrower
    when the behavior-defined block starts after, or ends before, the requested
    window.  No empirical coverage threshold is used to delete a candidate.
    """

    requested_start = float(requested_start_ms)
    requested_end = float(requested_end_ms)
    if not np.isfinite(requested_start) or not np.isfinite(requested_end):
        raise ValueError("requested window bounds must be finite")
    if requested_end <= requested_start:
        raise ValueError("requested_end_ms must be greater than requested_start_ms")

    available_start = requested_start if available_start_ms is None else float(available_start_ms)
    available_end = requested_end if available_end_ms is None else float(available_end_ms)
    if not np.isfinite(available_start) or not np.isfinite(available_end):
        raise ValueError("available window bounds must be finite")
    available_start = max(requested_start, available_start)
    available_end = min(requested_end, available_end)

    requested_duration_sec = (requested_end - requested_start) / 1000.0
    available_duration_sec = max(0.0, (available_end - available_start) / 1000.0)

    if "unix_ms" not in window.columns or "raw_binocular_pupil" not in window.columns:
        raise ValueError("supervised NIR window must contain unix_ms and raw_binocular_pupil")

    times = pd.to_numeric(window["unix_ms"], errors="coerce").to_numpy(dtype=float)
    in_available = (
        np.isfinite(times)
        & (times >= available_start)
        & (times < available_end)
    )
    scoped = window.loc[in_available].copy().reset_index(drop=True)
    scoped_times = pd.to_numeric(scoped["unix_ms"], errors="coerce").to_numpy(dtype=float)
    pupil = pd.to_numeric(scoped["raw_binocular_pupil"], errors="coerce").to_numpy(dtype=float)
    n_rows = int(len(scoped))
    n_valid = int(np.isfinite(pupil).sum())

    sampling_rate = _estimate_sampling_rate_hz(scoped_times)
    expected_rows = (
        available_duration_sec * sampling_rate
        if sampling_rate is not None and available_duration_sec > 0
        else None
    )
    internal_coverage = (
        min(1.0, float(n_rows) / expected_rows)
        if expected_rows is not None and expected_rows > 0
        else None
    )
    max_gap = (
        _max_temporal_gap_sec(scoped_times, available_start, available_end)
        if available_duration_sec > 0
        else None
    )

    if available_duration_sec <= 0:
        support_status = "not_available_window_outside_block"
    elif n_rows == 0:
        support_status = "not_available_no_nir_rows"
    elif n_valid == 0:
        support_status = "available_no_valid_pupil_samples"
    else:
        support_status = "available"

    result: dict[str, Any] = {
        "requested_duration_sec": requested_duration_sec,
        "available_duration_sec": available_duration_sec,
        "available_duration_fraction": available_duration_sec / requested_duration_sec,
        "window_truncated_by_available_start": bool(available_start > requested_start),
        "window_truncated_by_available_end": bool(available_end < requested_end),
        "n_nir_rows": n_rows,
        "n_pupil_valid": n_valid,
        "pupil_valid_fraction": float(n_valid / n_rows) if n_rows else None,
        "sampling_rate_hz_estimate": sampling_rate,
        "expected_nir_rows_available": expected_rows,
        "internal_coverage_fraction": internal_coverage,
        "max_temporal_gap_sec": max_gap,
        "window_support_status": support_status,
        "automatic_coverage_drop_allowed": False,
    }
    result.update(_source_mode_audit(scoped))
    result.update(summarize_supervised_window(scoped))
    return result
