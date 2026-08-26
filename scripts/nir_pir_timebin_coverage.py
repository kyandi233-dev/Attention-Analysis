"""Calibrate PIR time-bin coverage before time-on-task inference.

This script does not choose a coverage cutoff and does not run inferential models.
It exports per-eye 30-s/60-s bin features plus retention sensitivity for candidate
coverage thresholds so a fixed analysis-window policy can be chosen before testing
scientific effects.

Example:
    PYTHONPATH=src python scripts/nir_pir_timebin_coverage.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from attention_pipeline.config import load_config
from attention_pipeline.nir_behavior.contract import PIR_COLUMN, PIR_VALID_COLUMN
from attention_pipeline.nir_behavior.discovery import find_nir_source
from attention_pipeline.nir_behavior.features import coerce_bool_series
from attention_pipeline.nir_behavior_cohort.io import (
    alignment_config,
    cohort_output_root,
    selected_cohort_subjects,
)

TIME_COLUMN = "phase_time_ms"
REQUIRED = {"subject", "phase", "eye", "unix_ms", PIR_COLUMN, PIR_VALID_COLUMN}
CANDIDATE_THRESHOLDS = (0.25, 0.50, 0.75, 0.80, 0.90)
BIN_SECONDS = (60, 30)
SUBBIN_SECONDS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate 30/60-s PIR time-bin coverage")
    parser.add_argument("--config", default="configs/nir_behavior_cohort.yaml")
    parser.add_argument("--subjects", help="Optional comma-separated subject override")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def _read_subject(config, subject: str) -> pd.DataFrame:
    aconfig = alignment_config(config)
    source = find_nir_source(aconfig, subject)
    header = pd.read_csv(source.csv_path, nrows=0, encoding="utf-8-sig")
    available = set(header.columns)
    missing = sorted(REQUIRED - available)
    if missing:
        raise ValueError(f"{source.csv_path}: missing required columns {missing}")

    usecols = sorted(REQUIRED | ({TIME_COLUMN} if TIME_COLUMN in available else set()))
    df = pd.read_csv(source.csv_path, usecols=usecols, encoding="utf-8-sig")
    df = df[df["phase"].isin(["block1", "block2"])].copy()
    df["block_num"] = df["phase"].map({"block1": 1, "block2": 2}).astype(int)
    df["eye"] = df["eye"].astype(str).str.lower().str.replace("frame_", "", regex=False)
    df["unix_ms"] = pd.to_numeric(df["unix_ms"], errors="coerce")
    df[PIR_COLUMN] = pd.to_numeric(df[PIR_COLUMN], errors="coerce")
    df[PIR_VALID_COLUMN] = coerce_bool_series(df[PIR_VALID_COLUMN])
    if TIME_COLUMN in df.columns:
        df[TIME_COLUMN] = pd.to_numeric(df[TIME_COLUMN], errors="coerce")
    else:
        df[TIME_COLUMN] = np.nan

    # If phase_time_ms is absent/invalid, use the common subject×Block start so left
    # and right eyes share identical bin boundaries.
    for block_num, idx in df.groupby("block_num").groups.items():
        block_idx = list(idx)
        phase_time = df.loc[block_idx, TIME_COLUMN]
        if phase_time.notna().sum() < max(1, int(0.9 * len(block_idx))):
            start = df.loc[block_idx, "unix_ms"].min()
            df.loc[block_idx, TIME_COLUMN] = df.loc[block_idx, "unix_ms"] - start

    df = df[df["unix_ms"].notna() & df[TIME_COLUMN].notna()].copy()
    df["pir_finite"] = np.isfinite(df[PIR_COLUMN].to_numpy(dtype=float))
    df["pir_usable"] = df[PIR_VALID_COLUMN].fillna(False).to_numpy(dtype=bool) & df["pir_finite"]

    # Freeze subject×eye baselines across both Blocks so Block-level shifts are preserved.
    df["pir_centered"] = np.nan
    df["pir_robust_z"] = np.nan
    for eye, idx in df.groupby("eye").groups.items():
        eye_idx = list(idx)
        usable = df.loc[eye_idx, "pir_usable"].to_numpy(dtype=bool)
        values = df.loc[eye_idx, PIR_COLUMN].to_numpy(dtype=float)
        valid_values = values[usable]
        if valid_values.size == 0:
            continue
        median = float(np.median(valid_values))
        mad = float(np.median(np.abs(valid_values - median)))
        robust_sigma = 1.4826 * mad
        df.loc[eye_idx, "pir_centered"] = values - median
        if np.isfinite(robust_sigma) and robust_sigma > 0:
            df.loc[eye_idx, "pir_robust_z"] = (values - median) / robust_sigma

    return df


def _sampling_rate_ms(group: pd.DataFrame) -> tuple[float, float]:
    times = np.sort(group["unix_ms"].dropna().unique().astype(float))
    if times.size < 2:
        return float("nan"), float("nan")
    diffs = np.diff(times)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return float("nan"), float("nan")
    median_dt = float(np.median(diffs))
    return 1000.0 / median_dt, median_dt


def _safe_median(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _summarize_eye_bin(
    group: pd.DataFrame,
    *,
    subject: str,
    block_num: int,
    eye: str,
    bin_sec: int,
    bin_idx: int,
    block_end_ms: float,
    sampling_rate_hz: float,
    median_dt_ms: float,
) -> dict[str, object]:
    start_ms = bin_idx * bin_sec * 1000.0
    end_ms = start_ms + bin_sec * 1000.0
    is_full = bool(end_ms <= block_end_ms + max(median_dt_ms, 0.0)) if np.isfinite(median_dt_ms) else bool(end_ms <= block_end_ms)

    n_rows = int(len(group))
    usable = group["pir_usable"].to_numpy(dtype=bool)
    n_usable = int(usable.sum())
    expected_n = float(bin_sec * sampling_rate_hz) if np.isfinite(sampling_rate_hz) else float("nan")
    row_expected_fraction = n_rows / expected_n if np.isfinite(expected_n) and expected_n > 0 else float("nan")
    usable_expected_fraction = n_usable / expected_n if np.isfinite(expected_n) and expected_n > 0 else float("nan")

    # Distribution of usable data across the window, using 5-s subbins.
    expected_subbins = int(bin_sec // SUBBIN_SECONDS)
    usable_group = group.loc[group["pir_usable"]].copy()
    if usable_group.empty:
        occupied_subbins = 0
        median_subbin_usable_fraction = 0.0 if np.isfinite(sampling_rate_hz) else float("nan")
    else:
        usable_group["subbin_idx"] = np.floor((usable_group[TIME_COLUMN] - start_ms) / (SUBBIN_SECONDS * 1000.0)).astype(int)
        usable_group = usable_group[(usable_group["subbin_idx"] >= 0) & (usable_group["subbin_idx"] < expected_subbins)]
        counts = usable_group.groupby("subbin_idx").size()
        occupied_subbins = int(counts.index.nunique())
        expected_subbin_n = SUBBIN_SECONDS * sampling_rate_hz if np.isfinite(sampling_rate_hz) else float("nan")
        if np.isfinite(expected_subbin_n) and expected_subbin_n > 0:
            fractions = np.zeros(expected_subbins, dtype=float)
            for sub_idx, count in counts.items():
                fractions[int(sub_idx)] = min(float(count) / expected_subbin_n, 1.0)
            median_subbin_usable_fraction = float(np.median(fractions))
        else:
            median_subbin_usable_fraction = float("nan")

    return {
        "subject": subject,
        "block_num": block_num,
        "eye": eye,
        "bin_sec": bin_sec,
        "bin_idx": bin_idx,
        "bin_start_ms": start_ms,
        "bin_end_ms": end_ms,
        "is_full_duration_bin": is_full,
        "sampling_rate_hz": sampling_rate_hz,
        "n_rows": n_rows,
        "n_usable": n_usable,
        "expected_n": expected_n,
        "row_expected_fraction": min(row_expected_fraction, 1.0) if np.isfinite(row_expected_fraction) else row_expected_fraction,
        "usable_expected_fraction": min(usable_expected_fraction, 1.0) if np.isfinite(usable_expected_fraction) else usable_expected_fraction,
        "usable_subbin_fraction": occupied_subbins / expected_subbins if expected_subbins else float("nan"),
        "median_subbin_usable_fraction": median_subbin_usable_fraction,
        "raw_pir_median": _safe_median(group.loc[group["pir_usable"], PIR_COLUMN]),
        "centered_pir_median": _safe_median(group.loc[group["pir_usable"], "pir_centered"]),
        "robust_z_pir_median": _safe_median(group.loc[group["pir_usable"], "pir_robust_z"]),
    }


def build_eye_bins(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    subject = str(df["subject"].dropna().iloc[0]) if "subject" in df.columns and df["subject"].notna().any() else "unknown"
    for block_num, block in df.groupby("block_num", sort=True):
        block_end_ms = float(block[TIME_COLUMN].max())
        for eye, eye_df in block.groupby("eye", sort=True):
            sr, median_dt = _sampling_rate_ms(eye_df)
            for bin_sec in BIN_SECONDS:
                work = eye_df.copy()
                work["bin_idx"] = np.floor(work[TIME_COLUMN] / (bin_sec * 1000.0)).astype(int)
                for bin_idx, group in work.groupby("bin_idx", sort=True):
                    rows.append(
                        _summarize_eye_bin(
                            group,
                            subject=subject,
                            block_num=int(block_num),
                            eye=str(eye),
                            bin_sec=int(bin_sec),
                            bin_idx=int(bin_idx),
                            block_end_ms=block_end_ms,
                            sampling_rate_hz=sr,
                            median_dt_ms=median_dt,
                        )
                    )
    return pd.DataFrame(rows)


def coverage_distribution(eye_bins: pd.DataFrame) -> pd.DataFrame:
    rows = []
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    metrics = ["row_expected_fraction", "usable_expected_fraction", "usable_subbin_fraction", "median_subbin_usable_fraction"]
    full = eye_bins[eye_bins["is_full_duration_bin"]].copy()
    for bin_sec, sub in full.groupby("bin_sec"):
        for metric in metrics:
            values = pd.to_numeric(sub[metric], errors="coerce").dropna()
            if values.empty:
                continue
            q = values.quantile(quantiles)
            rows.append({
                "bin_sec": int(bin_sec),
                "metric": metric,
                "n": int(len(values)),
                "min": float(values.min()),
                "p05": float(q.loc[0.05]),
                "p10": float(q.loc[0.10]),
                "p25": float(q.loc[0.25]),
                "median": float(q.loc[0.50]),
                "p75": float(q.loc[0.75]),
                "p90": float(q.loc[0.90]),
                "p95": float(q.loc[0.95]),
                "max": float(values.max()),
            })
    return pd.DataFrame(rows)


def threshold_sensitivity(eye_bins: pd.DataFrame) -> pd.DataFrame:
    full = eye_bins[eye_bins["is_full_duration_bin"]].copy()
    key = ["subject", "block_num", "bin_sec", "bin_idx"]
    pivot = full.pivot_table(index=key, columns="eye", values="usable_expected_fraction", aggfunc="first").reset_index()
    for eye in ("left", "right"):
        if eye not in pivot.columns:
            pivot[eye] = np.nan

    rows = []
    for bin_sec, sub in pivot.groupby("bin_sec"):
        for threshold in CANDIDATE_THRESHOLDS:
            left_ok = pd.to_numeric(sub["left"], errors="coerce") >= threshold
            right_ok = pd.to_numeric(sub["right"], errors="coerce") >= threshold
            both = left_ok & right_ok
            either = left_ok | right_ok
            one_only = left_ok ^ right_ok
            rows.append({
                "bin_sec": int(bin_sec),
                "candidate_coverage_threshold": float(threshold),
                "n_time_bins": int(len(sub)),
                "n_both_eyes_pass": int(both.sum()),
                "fraction_both_eyes_pass": float(both.mean()),
                "n_either_eye_pass": int(either.sum()),
                "fraction_either_eye_pass": float(either.mean()),
                "n_single_eye_fallback": int(one_only.sum()),
                "fraction_single_eye_fallback": float(one_only.mean()),
                "n_neither_eye_pass": int((~either).sum()),
                "fraction_neither_eye_pass": float((~either).mean()),
            })
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] if args.subjects else None
    selected = selected_cohort_subjects(config, subjects)

    pieces = []
    failures = []
    for subject in selected:
        try:
            df = _read_subject(config, subject)
            pieces.append(build_eye_bins(df))
        except Exception as exc:  # noqa: BLE001 - audit script must record failures
            failures.append({"subject": subject, "error": f"{type(exc).__name__}: {exc}"})

    eye_bins = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    dist = coverage_distribution(eye_bins) if not eye_bins.empty else pd.DataFrame()
    sensitivity = threshold_sensitivity(eye_bins) if not eye_bins.empty else pd.DataFrame()
    failures_df = pd.DataFrame(failures, columns=["subject", "error"])

    output_dir = Path(args.output_dir) if args.output_dir else cohort_output_root(config) / "03_time_on_task" / "coverage_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    eye_bins.to_csv(output_dir / "time_bin_eye_features.csv", index=False, encoding="utf-8-sig")
    dist.to_csv(output_dir / "time_bin_coverage_distribution.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(output_dir / "time_bin_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    failures_df.to_csv(output_dir / "subject_load_failures.csv", index=False, encoding="utf-8-sig")

    readme = """# PIR time-bin coverage calibration

This folder calibrates data coverage for 60-s primary and 30-s sensitivity time-on-task bins.
It does **not** select a final coverage threshold and does not run inferential models.

- `time_bin_eye_features.csv`: per subject×Block×eye×bin features and coverage diagnostics.
- `time_bin_coverage_distribution.csv`: cohort quantiles for row/usable/subbin coverage.
- `time_bin_threshold_sensitivity.csv`: retention under candidate coverage thresholds 0.25/0.50/0.75/0.80/0.90.
- `subject_load_failures.csv`: subject-level read failures.

Coverage uses expected frame count estimated from each subject×Block×eye median sampling interval. `usable_subbin_fraction` describes how broadly usable data are distributed across 5-s subbins, so a window with many frames concentrated in one short segment can be distinguished from a temporally representative window.

The candidate thresholds are sensitivity values only. They must not be selected based on downstream p-values or effect sizes.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"subjects_requested={len(selected)}")
    print(f"subjects_failed={len(failures_df)}")
    print(f"eye_bin_rows={len(eye_bins)}")
    print(f"output={output_dir}")
    return 0 if failures_df.empty else 2


if __name__ == "__main__":
    raise SystemExit(main())
