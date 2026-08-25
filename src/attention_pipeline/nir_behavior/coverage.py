from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


_GROUP_COLUMNS = [
    "subject",
    "block_num",
    "eye",
    "window_family",
    "window_name",
    "window_start_offset_ms",
    "window_end_offset_ms",
]


def _finite(values: pd.Series) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return array[np.isfinite(array)]


def _quantile(values: pd.Series, q: float) -> float | None:
    array = _finite(values)
    return float(np.quantile(array, q)) if array.size else None


def _median(values: pd.Series) -> float | None:
    return _quantile(values, 0.5)


def _minimum(values: pd.Series) -> float | None:
    array = _finite(values)
    return float(np.min(array)) if array.size else None


def _maximum(values: pd.Series) -> float | None:
    array = _finite(values)
    return float(np.max(array)) if array.size else None


def build_window_coverage_report(
    windows: pd.DataFrame,
    *,
    level: str,
) -> pd.DataFrame:
    """Summarize schema-v2 coverage without imposing exclusion thresholds.

    Boundary/design truncation is reported separately from internal NIR
    missingness. ``oar_available_fraction`` means that a finite OAR value exists;
    it is not a validated blink/eye-state quality label.
    """
    if level not in {"trial", "probe"}:
        raise ValueError("level must be 'trial' or 'probe'")
    required = set(_GROUP_COLUMNS) | {
        "n_nir_rows",
        "requested_duration_sec",
        "available_duration_sec",
        "available_duration_fraction",
        "window_truncated_by_block_start",
        "window_truncated_by_block_end",
        "sampling_rate_hz_estimate",
        "expected_nir_rows_available",
        "internal_coverage_fraction",
        "max_temporal_gap_sec",
        "pir_valid_fraction",
        "oar_available_fraction",
        "roi_clipped_fraction",
        "ritnet_found_fraction",
    }
    missing = required - set(windows.columns)
    if missing:
        raise ValueError(f"coverage input missing schema-v2 columns: {sorted(missing)}")

    frame = windows.copy()
    n_rows = pd.to_numeric(frame["n_nir_rows"], errors="coerce")
    available_duration = pd.to_numeric(
        frame["available_duration_sec"], errors="coerce"
    )
    frame["nir_rows_per_available_sec"] = n_rows / available_duration.replace(0, np.nan)
    frame["boundary_truncated"] = (
        frame["window_truncated_by_block_start"].fillna(False).astype(bool)
        | frame["window_truncated_by_block_end"].fillna(False).astype(bool)
    )

    records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(_GROUP_COLUMNS, dropna=False, sort=True):
        record = dict(zip(_GROUP_COLUMNS, keys, strict=True))
        record["level"] = level
        record["n_windows"] = int(len(group))

        truncated_start = group["window_truncated_by_block_start"].fillna(False).astype(bool)
        truncated_end = group["window_truncated_by_block_end"].fillna(False).astype(bool)
        truncated_any = group["boundary_truncated"].fillna(False).astype(bool)
        record["n_boundary_truncated_start"] = int(truncated_start.sum())
        record["n_boundary_truncated_end"] = int(truncated_end.sum())
        record["n_boundary_truncated_windows"] = int(truncated_any.sum())
        record["boundary_truncated_window_fraction"] = (
            float(truncated_any.mean()) if len(group) else None
        )

        zero_rows = pd.to_numeric(group["n_nir_rows"], errors="coerce").fillna(0).eq(0)
        record["n_zero_nir_windows"] = int(zero_rows.sum())
        record["zero_nir_window_fraction"] = float(zero_rows.mean()) if len(group) else None

        for column, prefix in (
            ("requested_duration_sec", "requested_duration_sec"),
            ("available_duration_sec", "available_duration_sec"),
            ("available_duration_fraction", "available_duration_fraction"),
            ("n_nir_rows", "nir_rows"),
            ("nir_rows_per_available_sec", "nir_rows_per_available_sec"),
            ("sampling_rate_hz_estimate", "sampling_rate_hz_estimate"),
            ("expected_nir_rows_available", "expected_nir_rows_available"),
            ("internal_coverage_fraction", "internal_coverage_fraction"),
            ("pir_valid_fraction", "pir_valid_fraction"),
            ("oar_available_fraction", "oar_available_fraction"),
            ("roi_clipped_fraction", "roi_clipped_fraction"),
            ("ritnet_found_fraction", "ritnet_found_fraction"),
        ):
            record[f"{prefix}_median"] = _median(group[column])
            record[f"{prefix}_p10"] = _quantile(group[column], 0.10)
            record[f"{prefix}_min"] = _minimum(group[column])

        record["max_temporal_gap_sec_median"] = _median(group["max_temporal_gap_sec"])
        record["max_temporal_gap_sec_p90"] = _quantile(group["max_temporal_gap_sec"], 0.90)
        record["max_temporal_gap_sec_max"] = _maximum(group["max_temporal_gap_sec"])
        records.append(record)

    return pd.DataFrame(records)


def coverage_overview(report: pd.DataFrame) -> dict[str, Any]:
    """Small JSON-friendly overview for one subject alignment summary."""
    if report.empty:
        return {"groups": 0}
    return {
        "groups": int(len(report)),
        "pir_valid_fraction_group_median": _median(report["pir_valid_fraction_median"]),
        "pir_valid_fraction_group_p10": _quantile(report["pir_valid_fraction_p10"], 0.10),
        "oar_available_fraction_group_median": _median(
            report["oar_available_fraction_median"]
        ),
        "oar_available_fraction_group_p10": _quantile(
            report["oar_available_fraction_p10"], 0.10
        ),
        "internal_coverage_fraction_group_median": _median(
            report["internal_coverage_fraction_median"]
        ),
        "internal_coverage_fraction_group_p10": _quantile(
            report["internal_coverage_fraction_p10"], 0.10
        ),
        "nir_rows_per_available_sec_group_median": _median(
            report["nir_rows_per_available_sec_median"]
        ),
        "boundary_truncated_window_fraction_group_max": _maximum(
            report["boundary_truncated_window_fraction"]
        ),
        "zero_nir_window_fraction_group_max": _maximum(
            report["zero_nir_window_fraction"]
        ),
        "max_temporal_gap_sec_group_max": _maximum(
            report["max_temporal_gap_sec_max"]
        ),
    }
