from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def coerce_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y"})


def mad(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    center = float(np.median(values))
    return float(np.median(np.abs(values - center)))


def iqr(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    q25, q75 = np.percentile(values, [25, 75])
    return float(q75 - q25)


def robust_binned_slope_per_sec(
    times_ms: np.ndarray,
    values: np.ndarray,
    *,
    max_bins: int = 8,
) -> float | None:
    """Scalable robust-ish slope using temporal-bin medians.

    Direct Theil/Siegel slopes are too expensive for 30-Hz windows across the
    full cohort. Median values within a small number of equal-duration bins
    suppress single-frame segmentation spikes before a simple linear fit.
    """
    times = np.asarray(times_ms, dtype=float)
    vals = np.asarray(values, dtype=float)
    valid = np.isfinite(times) & np.isfinite(vals)
    times = times[valid]
    vals = vals[valid]
    if vals.size < 3:
        return None
    order = np.argsort(times)
    times = times[order]
    vals = vals[order]
    duration = float(times[-1] - times[0])
    if duration <= 0:
        return None

    n_bins = int(max(2, min(max_bins, vals.size // 3 if vals.size >= 6 else 2)))
    edges = np.linspace(times[0], times[-1], n_bins + 1)
    bin_t: list[float] = []
    bin_v: list[float] = []
    for idx in range(n_bins):
        if idx == n_bins - 1:
            mask = (times >= edges[idx]) & (times <= edges[idx + 1])
        else:
            mask = (times >= edges[idx]) & (times < edges[idx + 1])
        if not mask.any():
            continue
        bin_t.append(float(np.median(times[mask])))
        bin_v.append(float(np.median(vals[mask])))
    if len(bin_v) < 2:
        return None
    x = (np.asarray(bin_t) - bin_t[0]) / 1000.0
    slope = np.polyfit(x, np.asarray(bin_v), 1)[0]
    return float(slope)


def successive_diff_mad(values: np.ndarray) -> float | None:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return None
    return mad(np.diff(vals))


def successive_diff_rate_mad_per_sec(
    times_ms: np.ndarray, values: np.ndarray
) -> float | None:
    times = np.asarray(times_ms, dtype=float)
    vals = np.asarray(values, dtype=float)
    valid = np.isfinite(times) & np.isfinite(vals)
    times = times[valid]
    vals = vals[valid]
    if vals.size < 2:
        return None
    order = np.argsort(times)
    times = times[order]
    vals = vals[order]
    dt = np.diff(times) / 1000.0
    dv = np.diff(vals)
    valid_dt = np.isfinite(dt) & (dt > 0)
    if not valid_dt.any():
        return None
    return mad(dv[valid_dt] / dt[valid_dt])


def summarize_signal(
    times_ms: np.ndarray, values: np.ndarray, prefix: str
) -> dict[str, Any]:
    times = np.asarray(times_ms, dtype=float)
    vals = np.asarray(values, dtype=float)
    valid = np.isfinite(times) & np.isfinite(vals)
    times = times[valid]
    vals = vals[valid]
    result: dict[str, Any] = {f"n_{prefix}_valid": int(vals.size)}
    if vals.size == 0:
        for suffix in (
            "median",
            "mean",
            "mad",
            "iqr",
            "sd",
            "p10",
            "p90",
            "slope_per_sec",
            "diff_mad",
            "diff_rate_mad_per_sec",
        ):
            result[f"{prefix}_{suffix}"] = None
        return result

    order = np.argsort(times)
    times = times[order]
    vals = vals[order]
    result.update(
        {
            f"{prefix}_median": float(np.median(vals)),
            f"{prefix}_mean": float(np.mean(vals)),
            f"{prefix}_mad": mad(vals),
            f"{prefix}_iqr": iqr(vals),
            f"{prefix}_sd": float(np.std(vals, ddof=1)) if vals.size >= 2 else 0.0,
            f"{prefix}_p10": float(np.percentile(vals, 10)),
            f"{prefix}_p90": float(np.percentile(vals, 90)),
            f"{prefix}_slope_per_sec": robust_binned_slope_per_sec(times, vals),
            f"{prefix}_diff_mad": successive_diff_mad(vals),
            f"{prefix}_diff_rate_mad_per_sec": successive_diff_rate_mad_per_sec(
                times, vals
            ),
        }
    )
    return result
