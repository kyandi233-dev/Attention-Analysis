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


def build_window_coverage_report(
    windows: pd.DataFrame,
    *,
    level: str,
) -> pd.DataFrame:
    """Summarize descriptive NIR coverage without imposing exclusion thresholds.

    The report is grouped by subject × block × eye × configured window. It keeps
    both row-density and metric-validity information so later cohort QC can
    distinguish "few NIR rows exist" from "rows exist but PIR normalization is
    invalid". No group is classified as good/bad here.
    """
    if level not in {"trial", "probe"}:
        raise ValueError("level must be 'trial' or 'probe'")
    required = set(_GROUP_COLUMNS) | {
        "n_nir_rows",
        "pir_valid_fraction",
        "oar_valid_fraction",
        "roi_clipped_fraction",
        "ritnet_found_fraction",
    }
    missing = required - set(windows.columns)
    if missing:
        raise ValueError(f"coverage input missing columns: {sorted(missing)}")

    frame = windows.copy()
    duration_sec = (
        pd.to_numeric(frame["window_end_offset_ms"], errors="coerce")
        - pd.to_numeric(frame["window_start_offset_ms"], errors="coerce")
    ) / 1000.0
    n_rows = pd.to_numeric(frame["n_nir_rows"], errors="coerce")
    frame["nir_rows_per_sec"] = n_rows / duration_sec.replace(0, np.nan)

    records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(_GROUP_COLUMNS, dropna=False, sort=True):
        record = dict(zip(_GROUP_COLUMNS, keys, strict=True))
        record["level"] = level
        record["n_windows"] = int(len(group))
        zero_rows = pd.to_numeric(group["n_nir_rows"], errors="coerce").fillna(0).eq(0)
        record["n_zero_nir_windows"] = int(zero_rows.sum())
        record["zero_nir_window_fraction"] = float(zero_rows.mean()) if len(group) else None

        for column, prefix in (
            ("n_nir_rows", "nir_rows"),
            ("nir_rows_per_sec", "nir_rows_per_sec"),
            ("pir_valid_fraction", "pir_valid_fraction"),
            ("oar_valid_fraction", "oar_valid_fraction"),
            ("roi_clipped_fraction", "roi_clipped_fraction"),
            ("ritnet_found_fraction", "ritnet_found_fraction"),
        ):
            record[f"{prefix}_median"] = _median(group[column])
            record[f"{prefix}_p10"] = _quantile(group[column], 0.10)
            record[f"{prefix}_min"] = _minimum(group[column])

        records.append(record)

    return pd.DataFrame(records)


def coverage_overview(report: pd.DataFrame) -> dict[str, Any]:
    """Small JSON-friendly overview for the subject alignment summary."""
    if report.empty:
        return {"groups": 0}
    return {
        "groups": int(len(report)),
        "pir_valid_fraction_group_median": _median(report["pir_valid_fraction_median"]),
        "pir_valid_fraction_group_p10": _quantile(report["pir_valid_fraction_p10"], 0.10),
        "oar_valid_fraction_group_median": _median(report["oar_valid_fraction_median"]),
        "oar_valid_fraction_group_p10": _quantile(report["oar_valid_fraction_p10"], 0.10),
        "nir_rows_per_sec_group_median": _median(report["nir_rows_per_sec_median"]),
        "zero_nir_window_fraction_group_max": (
            float(pd.to_numeric(report["zero_nir_window_fraction"], errors="coerce").max())
            if pd.to_numeric(report["zero_nir_window_fraction"], errors="coerce").notna().any()
            else None
        ),
    }
