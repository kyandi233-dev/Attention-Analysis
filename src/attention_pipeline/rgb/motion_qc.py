from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from attention_pipeline.config import Config
from attention_pipeline.rgb.paths import RGBOutputLayout


MOTION_QC_SCHEMA_VERSION = "rgb-motion-qc-v0.1"
_QUANTILES = (0.50, 0.75, 0.90, 0.95, 0.99)


def _number(value: object) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    if number.is_integer():
        return int(number)
    return number


def _quantile_summary(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    values = values[np.isfinite(values)]
    if values.empty:
        return {"count": 0, "min": None, "p50": None, "p75": None, "p90": None, "p95": None, "p99": None, "max": None}
    quantiles = values.quantile(list(_QUANTILES))
    return {
        "count": int(len(values)),
        "min": _number(values.min()),
        "p50": _number(quantiles.loc[0.50]),
        "p75": _number(quantiles.loc[0.75]),
        "p90": _number(quantiles.loc[0.90]),
        "p95": _number(quantiles.loc[0.95]),
        "p99": _number(quantiles.loc[0.99]),
        "max": _number(values.max()),
    }


def _correlation(x: pd.Series, y: pd.Series) -> float | None:
    pair = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return None
    value = pair["x"].corr(pair["y"], method="pearson")
    return _number(value)  # type: ignore[return-value]


def _records(table: pd.DataFrame, columns: list[str], limit: int | None = None) -> list[dict[str, object]]:
    available = [column for column in columns if column in table.columns]
    frame = table[available]
    if limit is not None:
        frame = frame.head(limit)
    rows: list[dict[str, object]] = []
    for record in frame.to_dict(orient="records"):
        clean: dict[str, object] = {}
        for key, value in record.items():
            if isinstance(value, (np.integer,)):
                clean[key] = int(value)
            elif isinstance(value, (np.floating,)):
                clean[key] = None if not np.isfinite(value) else float(value)
            elif pd.isna(value):
                clean[key] = None
            else:
                clean[key] = value
        rows.append(clean)
    return rows


def summarize_motion_table(table: pd.DataFrame, *, subject: str) -> dict[str, object]:
    if table.empty:
        raise ValueError("Motion parquet is empty")

    required = {"dt_ms", "motion_valid", "global_motion_energy", "gray_mean_delta", "irregular_dt", "gap_before"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Motion parquet missing required columns: {missing}")

    dt = pd.to_numeric(table["dt_ms"], errors="coerce")
    positive_dt = dt[dt > 0]
    valid_mask = table["motion_valid"].fillna(False).astype(bool)
    irregular_mask = table["irregular_dt"].fillna(False).astype(bool)
    gap_mask = table["gap_before"].fillna(False).astype(bool)
    valid = table.loc[valid_mask].copy()

    dt_bands = {
        "le_40_ms": int((positive_dt <= 40).sum()),
        "41_to_48_ms": int(((positive_dt > 40) & (positive_dt <= 48)).sum()),
        "49_to_66_ms": int(((positive_dt > 48) & (positive_dt <= 66)).sum()),
        "67_to_100_ms": int(((positive_dt > 66) & (positive_dt <= 100)).sum()),
        "gt_100_ms": int((positive_dt > 100).sum()),
    }

    valid_motion = pd.to_numeric(valid["global_motion_energy"], errors="coerce")
    abs_brightness = pd.to_numeric(valid["gray_mean_delta"], errors="coerce").abs()
    changed_ratio = pd.to_numeric(valid.get("changed_pixel_ratio"), errors="coerce") if "changed_pixel_ratio" in valid else pd.Series(dtype=float)
    motion_rate = pd.to_numeric(valid.get("global_motion_energy_per_sec"), errors="coerce") if "global_motion_energy_per_sec" in valid else pd.Series(dtype=float)

    sample_columns = [
        "video_frame_position",
        "capture_frame_idx",
        "unix_ms",
        "dt_ms",
        "phase",
        "block",
        "trial_num",
        "behavior_state",
        "global_motion_energy",
        "global_motion_energy_per_sec",
        "changed_pixel_ratio",
        "gray_mean_delta",
        "gray_mean",
        "irregular_dt",
        "gap_before",
        "gap_reason",
    ]

    top_motion = valid.assign(_sort=valid_motion).sort_values("_sort", ascending=False, kind="stable").drop(columns="_sort")
    top_brightness = valid.assign(_sort=abs_brightness).sort_values("_sort", ascending=False, kind="stable").drop(columns="_sort")
    gap_rows = table.loc[gap_mask].sort_values("dt_ms", ascending=False, kind="stable")

    phase_counts = {str(key): int(value) for key, value in table["phase"].value_counts(dropna=False).to_dict().items()} if "phase" in table else {}

    return {
        "schema_version": MOTION_QC_SCHEMA_VERSION,
        "subject": subject,
        "rows": int(len(table)),
        "motion_valid_rows": int(valid_mask.sum()),
        "motion_valid_fraction": float(valid_mask.mean()),
        "gap_rows": int(gap_mask.sum()),
        "irregular_dt_rows": int(irregular_mask.sum()),
        "irregular_dt_fraction": float(irregular_mask.mean()),
        "phase_rows": phase_counts,
        "dt_ms": _quantile_summary(positive_dt),
        "dt_bands": dt_bands,
        "global_motion_energy": _quantile_summary(valid_motion),
        "global_motion_energy_per_sec": _quantile_summary(motion_rate),
        "changed_pixel_ratio": _quantile_summary(changed_ratio),
        "abs_gray_mean_delta": _quantile_summary(abs_brightness),
        "correlations_valid_motion_rows": {
            "motion_vs_abs_gray_mean_delta": _correlation(valid_motion, abs_brightness),
            "motion_vs_changed_pixel_ratio": _correlation(valid_motion, changed_ratio),
            "motion_vs_dt_ms": _correlation(valid_motion, pd.to_numeric(valid["dt_ms"], errors="coerce")),
        },
        "samples": {
            "largest_timestamp_gaps": _records(gap_rows, sample_columns, limit=20),
            "highest_motion_energy": _records(top_motion, sample_columns, limit=20),
            "largest_abs_brightness_change": _records(top_brightness, sample_columns, limit=20),
        },
    }


def run_motion_qc(config: Config, subject: str) -> dict[str, object]:
    """Summarize an existing motion-test parquet without rereading the RGB video."""
    layout = RGBOutputLayout.from_config(config)
    source = layout.test_file(f"{subject}_motion-test.parquet")
    if not source.exists():
        raise FileNotFoundError(f"Motion pilot output not found: {source}")

    table = pd.read_parquet(source)
    summary = summarize_motion_table(table, subject=subject)
    summary["source_parquet"] = str(source)

    output = layout.test_file(f"{subject}_motion-qc.json")
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["output"] = str(output)
    return summary
