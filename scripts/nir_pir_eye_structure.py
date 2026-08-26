"""Evaluate left/right PIR measurement structure and candidate standardizations.

Phase-2 exploratory analysis for the NIR cohort. This script does not choose an eye
fusion policy or a primary standardization automatically. It only produces the
measurement evidence needed to freeze those decisions before formal effect models.

Example:
    PYTHONPATH=src python scripts/nir_pir_eye_structure.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from attention_pipeline.config import load_config
from attention_pipeline.nir_behavior.contract import PIR_COLUMN, PIR_VALID_COLUMN
from attention_pipeline.nir_behavior_cohort.io import (
    cohort_output_root,
    load_nir_qc_frame,
    selected_cohort_subjects,
)

ROBUST_SIGMA_SCALE = 1.4826
MIN_CORR_PAIRS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate left/right PIR structure and standardization candidates"
    )
    parser.add_argument("--config", default="configs/nir_behavior_cohort.yaml")
    parser.add_argument(
        "--subjects",
        help="Optional comma-separated subject override, e.g. sub-031,sub-032",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Optional output directory. Default: "
            "<cohort output>/02_standardization/eye_structure"
        ),
    )
    return parser.parse_args()


def _finite_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < MIN_CORR_PAIRS:
        return float("nan")
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < MIN_CORR_PAIRS:
        return float("nan")
    xr = pd.Series(x[mask]).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y[mask]).rank(method="average").to_numpy(dtype=float)
    return _finite_corr(xr, yr)


def _mad(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


def _iqr(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    q25, q75 = np.quantile(values, [0.25, 0.75])
    return float(q75 - q25)


def _icc_two_way(matrix: np.ndarray, absolute: bool) -> float:
    """ICC(A,1) if absolute=True, else ICC(C,1), for complete n×k matrix."""
    matrix = np.asarray(matrix, dtype=float)
    matrix = matrix[np.all(np.isfinite(matrix), axis=1)]
    n, k = matrix.shape if matrix.ndim == 2 else (0, 0)
    if n < 2 or k < 2:
        return float("nan")

    grand = float(matrix.mean())
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    ss_rows = k * float(np.sum((row_means - grand) ** 2))
    ss_cols = n * float(np.sum((col_means - grand) ** 2))
    residual = matrix - row_means[:, None] - col_means[None, :] + grand
    ss_error = float(np.sum(residual**2))

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    if absolute:
        denom = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    else:
        denom = ms_rows + (k - 1) * ms_error
    if denom == 0:
        return float("nan")
    return float((ms_rows - ms_error) / denom)


def _agreement_stats(left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    mask = np.isfinite(left) & np.isfinite(right)
    left = left[mask]
    right = right[mask]
    n = int(len(left))
    if n == 0:
        return {
            "n_pairs": 0,
            "pearson_r": float("nan"),
            "spearman_rho": float("nan"),
            "mean_left_minus_right": float("nan"),
            "median_left_minus_right": float("nan"),
            "mean_abs_difference": float("nan"),
            "median_abs_difference": float("nan"),
            "bland_altman_sd_diff": float("nan"),
            "bland_altman_loa_low": float("nan"),
            "bland_altman_loa_high": float("nan"),
        }
    diff = left - right
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1)) if n >= 2 else float("nan")
    return {
        "n_pairs": n,
        "pearson_r": _finite_corr(left, right),
        "spearman_rho": _spearman_corr(left, right),
        "mean_left_minus_right": mean_diff,
        "median_left_minus_right": float(np.median(diff)),
        "mean_abs_difference": float(np.mean(np.abs(diff))),
        "median_abs_difference": float(np.median(np.abs(diff))),
        "bland_altman_sd_diff": sd_diff,
        "bland_altman_loa_low": mean_diff - 1.96 * sd_diff if np.isfinite(sd_diff) else float("nan"),
        "bland_altman_loa_high": mean_diff + 1.96 * sd_diff if np.isfinite(sd_diff) else float("nan"),
    }


def _usable_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[["subject", "block_num", "eye", "unix_ms", PIR_COLUMN, PIR_VALID_COLUMN]].copy()
    out[PIR_COLUMN] = pd.to_numeric(out[PIR_COLUMN], errors="coerce")
    valid = out[PIR_VALID_COLUMN].fillna(False).to_numpy(dtype=bool)
    pir = out[PIR_COLUMN].to_numpy(dtype=float)
    out["pir_usable"] = valid & np.isfinite(pir)
    return out


def _build_baselines(all_frames: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (subject, eye), group in all_frames.groupby(["subject", "eye"], sort=True):
        values = group.loc[group["pir_usable"], PIR_COLUMN].to_numpy(dtype=float)
        med = float(np.median(values)) if len(values) else float("nan")
        mad = _mad(values)
        robust_sigma = ROBUST_SIGMA_SCALE * mad if np.isfinite(mad) else float("nan")
        rows.append(
            {
                "subject": subject,
                "eye": eye,
                "n_usable_frames": int(len(values)),
                "subject_eye_median_pir": med,
                "subject_eye_mad_pir": mad,
                "subject_eye_robust_sigma": robust_sigma,
                "robust_z_available": bool(np.isfinite(robust_sigma) and robust_sigma > 0),
            }
        )
    return pd.DataFrame(rows)


def _with_standardizations(all_frames: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    frame = all_frames.merge(baselines, on=["subject", "eye"], how="left", validate="many_to_one")
    frame["pir_raw"] = frame[PIR_COLUMN].where(frame["pir_usable"])
    frame["pir_subject_eye_centered"] = (
        frame["pir_raw"] - frame["subject_eye_median_pir"]
    )
    denom = frame["subject_eye_robust_sigma"].where(frame["subject_eye_robust_sigma"] > 0)
    frame["pir_subject_eye_robust_z"] = frame["pir_subject_eye_centered"] / denom

    usable = frame[frame["pir_usable"]].copy()
    block_medians = (
        usable.groupby(["subject", "block_num", "eye"], sort=True)["pir_raw"]
        .median()
        .rename("subject_eye_block_median_pir")
        .reset_index()
    )
    frame = frame.merge(
        block_medians,
        on=["subject", "block_num", "eye"],
        how="left",
        validate="many_to_one",
    )
    frame["pir_block_centered"] = frame["pir_raw"] - frame["subject_eye_block_median_pir"]
    return frame


def _eye_block_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (subject, block_num), block in frame.groupby(["subject", "block_num"], sort=True):
        row: dict[str, object] = {"subject": subject, "block_num": int(block_num)}
        series: dict[str, pd.DataFrame] = {}
        for eye in ("left", "right"):
            eye_frame = block[block["eye"] == eye].copy()
            total = len(eye_frame)
            usable = eye_frame[eye_frame["pir_usable"]].copy()
            values = usable["pir_raw"].to_numpy(dtype=float)
            row[f"{eye}_n_rows"] = int(total)
            row[f"{eye}_n_usable"] = int(len(usable))
            row[f"{eye}_usable_fraction"] = float(len(usable) / total) if total else float("nan")
            row[f"{eye}_median_raw"] = float(np.median(values)) if len(values) else float("nan")
            row[f"{eye}_mad_raw"] = _mad(values)
            row[f"{eye}_iqr_raw"] = _iqr(values)
            series[eye] = usable[
                [
                    "unix_ms",
                    "pir_raw",
                    "pir_subject_eye_centered",
                    "pir_subject_eye_robust_z",
                    "pir_block_centered",
                ]
            ].drop_duplicates(subset=["unix_ms"])

        row["raw_median_left_minus_right"] = (
            row["left_median_raw"] - row["right_median_raw"]
            if np.isfinite(row["left_median_raw"]) and np.isfinite(row["right_median_raw"])
            else float("nan")
        )
        row["raw_median_abs_difference"] = (
            abs(float(row["raw_median_left_minus_right"]))
            if np.isfinite(row["raw_median_left_minus_right"])
            else float("nan")
        )

        paired = series["left"].merge(
            series["right"], on="unix_ms", how="inner", suffixes=("_left", "_right"), validate="one_to_one"
        )
        row["exact_timestamp_usable_pairs"] = int(len(paired))
        denom_pairs = min(int(row["left_n_usable"]), int(row["right_n_usable"]))
        row["exact_pair_fraction_of_smaller_eye"] = (
            float(len(paired) / denom_pairs) if denom_pairs else float("nan")
        )

        versions = {
            "raw": "pir_raw",
            "subject_eye_centered": "pir_subject_eye_centered",
            "subject_eye_robust_z": "pir_subject_eye_robust_z",
            "block_centered": "pir_block_centered",
        }
        for name, column in versions.items():
            stats = _agreement_stats(
                paired[f"{column}_left"].to_numpy(dtype=float),
                paired[f"{column}_right"].to_numpy(dtype=float),
            )
            for key, value in stats.items():
                row[f"{name}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _cohort_standardization_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name in ("raw", "subject_eye_centered", "subject_eye_robust_z", "block_centered"):
        pearson = pd.to_numeric(summary[f"{name}_pearson_r"], errors="coerce")
        spearman = pd.to_numeric(summary[f"{name}_spearman_rho"], errors="coerce")
        absdiff = pd.to_numeric(summary[f"{name}_median_abs_difference"], errors="coerce")
        meandiff = pd.to_numeric(summary[f"{name}_mean_left_minus_right"], errors="coerce")
        rows.append(
            {
                "standardization": name,
                "n_subject_blocks": int(len(summary)),
                "n_with_defined_pearson": int(pearson.notna().sum()),
                "median_within_block_pearson_r": float(pearson.median()),
                "median_within_block_spearman_rho": float(spearman.median()),
                "median_subject_block_abs_paired_difference": float(absdiff.median()),
                "median_subject_block_mean_left_minus_right": float(meandiff.median()),
            }
        )
    return pd.DataFrame(rows)


def _cohort_summary(summary: pd.DataFrame) -> dict[str, object]:
    left = pd.to_numeric(summary["left_median_raw"], errors="coerce").to_numpy(dtype=float)
    right = pd.to_numeric(summary["right_median_raw"], errors="coerce").to_numpy(dtype=float)
    complete = np.isfinite(left) & np.isfinite(right)
    matrix = np.column_stack([left[complete], right[complete]]) if complete.any() else np.empty((0, 2))

    offset = summary.pivot(index="subject", columns="block_num", values="raw_median_left_minus_right")
    if 1 in offset.columns and 2 in offset.columns:
        b1 = pd.to_numeric(offset[1], errors="coerce").to_numpy(dtype=float)
        b2 = pd.to_numeric(offset[2], errors="coerce").to_numpy(dtype=float)
        stable_mask = np.isfinite(b1) & np.isfinite(b2)
        offset_pearson = _finite_corr(b1[stable_mask], b2[stable_mask])
        offset_spearman = _spearman_corr(b1[stable_mask], b2[stable_mask])
        offset_abs_change_median = (
            float(np.median(np.abs(b2[stable_mask] - b1[stable_mask])))
            if stable_mask.any()
            else float("nan")
        )
        n_offset_subjects = int(stable_mask.sum())
    else:
        offset_pearson = offset_spearman = offset_abs_change_median = float("nan")
        n_offset_subjects = 0

    raw_agreement = _agreement_stats(left[complete], right[complete])
    return {
        "n_subject_blocks": int(len(summary)),
        "n_complete_subject_block_median_pairs": int(complete.sum()),
        "cohort_subject_block_median_raw": raw_agreement,
        "icc_a1_absolute_agreement_raw_subject_block_medians": _icc_two_way(matrix, absolute=True),
        "icc_c1_consistency_raw_subject_block_medians": _icc_two_way(matrix, absolute=False),
        "eye_offset_block_stability": {
            "n_subjects_with_both_blocks": n_offset_subjects,
            "pearson_r_b1_vs_b2_left_minus_right_offset": offset_pearson,
            "spearman_rho_b1_vs_b2_left_minus_right_offset": offset_spearman,
            "median_absolute_change_in_offset_b1_to_b2": offset_abs_change_median,
        },
        "interpretation_boundary": (
            "Descriptive measurement evidence only. Do not choose eye fusion or primary "
            "standardization from significance/effect results."
        ),
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    subjects = (
        [item.strip() for item in args.subjects.split(",") if item.strip()]
        if args.subjects
        else None
    )
    selected = selected_cohort_subjects(config, subjects)

    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for subject in selected:
        try:
            frame, _ = load_nir_qc_frame(config, subject)
            frames.append(_usable_frame(frame))
        except Exception as exc:  # diagnostic output; do not silently drop failures
            failures.append({"subject": subject, "error": f"{type(exc).__name__}: {exc}"})

    if not frames:
        raise RuntimeError("No subject NIR frames could be loaded")

    all_frames = pd.concat(frames, ignore_index=True)
    baselines = _build_baselines(all_frames)
    standardized = _with_standardizations(all_frames, baselines)
    eye_block = _eye_block_summary(standardized)
    comparison = _cohort_standardization_comparison(eye_block)
    cohort_summary = _cohort_summary(eye_block)
    cohort_summary["subjects_requested"] = len(selected)
    cohort_summary["subjects_loaded"] = len(frames)
    cohort_summary["subject_load_failures"] = failures

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = cohort_output_root(config) / "02_standardization" / "eye_structure"
    output_dir.mkdir(parents=True, exist_ok=True)

    baselines.to_csv(output_dir / "subject_eye_baselines.csv", index=False, encoding="utf-8-sig")
    eye_block.to_csv(output_dir / "subject_block_eye_structure.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output_dir / "standardization_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(failures).to_csv(output_dir / "subject_load_failures.csv", index=False, encoding="utf-8-sig")
    (output_dir / "cohort_eye_structure_summary.json").write_text(
        json.dumps(cohort_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = """# PIR eye structure and standardization review

This folder contains Phase-2 exploratory measurement analysis. It does **not**
automatically choose an eye-fusion policy or a primary standardization.

Inputs use only frame-level usable PIR as frozen in `docs/020-nir/029-*`:
`fullclass_normalization_valid == True` plus finite PIR.

Outputs:

- `subject_eye_baselines.csv`: subject×eye all-block median, MAD and robust-sigma baseline.
- `subject_block_eye_structure.csv`: left/right usable fractions, raw medians, exact-timestamp paired correlations, differences and Bland–Altman quantities for four candidate transformations.
- `standardization_comparison.csv`: cohort summary across subject×Block units for raw, subject×eye centered, subject×eye robust-z, and block-centered PIR.
- `cohort_eye_structure_summary.json`: cohort-level raw left/right agreement, ICC(A,1), ICC(C,1), and Block1→Block2 stability of the left-right median offset.
- `subject_load_failures.csv`: any subject-level read errors; expected empty.

Candidate transformations:

1. raw PIR;
2. subject×eye median-centered PIR;
3. subject×eye robust-z PIR, using 1.4826×MAD as robust sigma;
4. subject×eye×Block median-centered PIR, retained only as a sensitivity candidate because it removes Block-level location differences by construction.

Important boundary: the retrospective subject×eye baseline in this exploratory measurement analysis may use both blocks. Future real-time/predictive models must estimate normalization from training/past data only and must not use future observations.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"subjects_requested={len(selected)}")
    print(f"subjects_loaded={len(frames)}")
    print(f"subject_block_rows={len(eye_block)}")
    print(f"output={output_dir}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
