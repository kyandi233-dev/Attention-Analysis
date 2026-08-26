"""Audit the six frozen frame-level PIR screening gates across the cohort.

This script is descriptive QC only. It does not define subject/eye/Block exclusion
thresholds and does not modify production NIR files.

Example:
    PYTHONPATH=src python scripts/nir_pir_gate_failure_qc.py
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


GATE_COLUMNS = {
    "pupil_fit": "fullclass_pupil_fit_valid",
    "iris_outer_fit": "fullclass_iris_outer_fit_valid",
    "pupil_not_edge": "fullclass_pupil_touches_roi_edge",
    "iris_outer_not_edge": "fullclass_iris_outer_touches_roi_edge",
    "pupil_center_in_iris": "fullclass_pupil_center_in_iris_outer",
    "iris_larger_than_pupil": None,
}

PUPIL_DIAMETER_COLUMN = "fullclass_pupil_geom_mean_diameter"
IRIS_DIAMETER_COLUMN = "fullclass_iris_outer_geom_mean_diameter"

REQUIRED_COLUMNS = {
    "subject",
    "phase",
    "eye",
    PIR_COLUMN,
    PIR_VALID_COLUMN,
    "fullclass_pupil_fit_valid",
    "fullclass_iris_outer_fit_valid",
    "fullclass_pupil_touches_roi_edge",
    "fullclass_iris_outer_touches_roi_edge",
    "fullclass_pupil_center_in_iris_outer",
    PUPIL_DIAMETER_COLUMN,
    IRIS_DIAMETER_COLUMN,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the six frozen PIR normalization gates"
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
            "<cohort output>/01_qc/screening_review/pir_gate_failure_qc"
        ),
    )
    return parser.parse_args()


def _read_subject(config, subject: str) -> pd.DataFrame:
    aconfig = alignment_config(config)
    source = find_nir_source(aconfig, subject)
    header = pd.read_csv(source.csv_path, nrows=0, encoding="utf-8-sig")
    available = set(header.columns)
    missing = sorted(REQUIRED_COLUMNS - available)
    if missing:
        raise ValueError(f"{source.csv_path}: missing PIR gate columns {missing}")

    frame = pd.read_csv(
        source.csv_path,
        usecols=sorted(REQUIRED_COLUMNS),
        encoding="utf-8-sig",
    )
    frame = frame[frame["phase"].isin(["block1", "block2"])].copy()
    frame["block_num"] = frame["phase"].map({"block1": 1, "block2": 2}).astype(int)
    frame["eye"] = frame["eye"].astype(str).str.strip().str.lower()

    for column in (
        PIR_VALID_COLUMN,
        "fullclass_pupil_fit_valid",
        "fullclass_iris_outer_fit_valid",
        "fullclass_pupil_touches_roi_edge",
        "fullclass_iris_outer_touches_roi_edge",
        "fullclass_pupil_center_in_iris_outer",
    ):
        frame[column] = coerce_bool_series(frame[column])

    for column in (PIR_COLUMN, PUPIL_DIAMETER_COLUMN, IRIS_DIAMETER_COLUMN):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def _gate_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    pupil_d = frame[PUPIL_DIAMETER_COLUMN].to_numpy(dtype=float)
    iris_d = frame[IRIS_DIAMETER_COLUMN].to_numpy(dtype=float)
    diameter_order = np.isfinite(pupil_d) & np.isfinite(iris_d) & (iris_d > pupil_d)

    return {
        "pupil_fit": frame["fullclass_pupil_fit_valid"].fillna(False).to_numpy(dtype=bool),
        "iris_outer_fit": frame["fullclass_iris_outer_fit_valid"].fillna(False).to_numpy(dtype=bool),
        "pupil_not_edge": ~frame["fullclass_pupil_touches_roi_edge"].fillna(True).to_numpy(dtype=bool),
        "iris_outer_not_edge": ~frame["fullclass_iris_outer_touches_roi_edge"].fillna(True).to_numpy(dtype=bool),
        "pupil_center_in_iris": frame["fullclass_pupil_center_in_iris_outer"].fillna(False).to_numpy(dtype=bool),
        "iris_larger_than_pupil": diameter_order,
    }


def summarize_group(frame: pd.DataFrame) -> dict[str, object]:
    gates = _gate_masks(frame)
    all_gate_pass = np.logical_and.reduce(list(gates.values()))
    recorded_valid = frame[PIR_VALID_COLUMN].fillna(False).to_numpy(dtype=bool)
    pir = frame[PIR_COLUMN].to_numpy(dtype=float)
    pir_finite = np.isfinite(pir)

    result: dict[str, object] = {
        "subject": str(frame["subject"].iloc[0]),
        "block_num": int(frame["block_num"].iloc[0]),
        "eye": str(frame["eye"].iloc[0]),
        "n_rows": int(len(frame)),
        "pir_numeric_finite_fraction": float(np.mean(pir_finite)),
        "pir_normalization_valid_fraction": float(np.mean(recorded_valid)),
        "pir_usable_fraction": float(np.mean(recorded_valid & pir_finite)),
        "recomputed_six_gate_pass_fraction": float(np.mean(all_gate_pass)),
        "normalization_gate_mismatch_fraction": float(np.mean(recorded_valid != all_gate_pass)),
    }
    for name, passed in gates.items():
        result[f"gate_{name}_pass_fraction"] = float(np.mean(passed))
        result[f"gate_{name}_failure_fraction"] = float(np.mean(~passed))
    return result


def _distribution_table(detail: pd.DataFrame) -> pd.DataFrame:
    failure_columns = [
        column for column in detail.columns if column.startswith("gate_") and column.endswith("_failure_fraction")
    ]
    rows = []
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    for column in failure_columns:
        values = pd.to_numeric(detail[column], errors="coerce").dropna()
        q = values.quantile(quantiles)
        rows.append(
            {
                "gate_failure_metric": column,
                "min": float(values.min()),
                "p05": float(q.loc[0.05]),
                "p10": float(q.loc[0.10]),
                "p25": float(q.loc[0.25]),
                "median": float(q.loc[0.50]),
                "p75": float(q.loc[0.75]),
                "p90": float(q.loc[0.90]),
                "p95": float(q.loc[0.95]),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def _top_failure_records(detail: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    pieces = []
    failure_columns = [
        column for column in detail.columns if column.startswith("gate_") and column.endswith("_failure_fraction")
    ]
    for column in failure_columns:
        selected = detail.nlargest(n, column)[
            [
                "subject",
                "block_num",
                "eye",
                column,
                "pir_numeric_finite_fraction",
                "pir_normalization_valid_fraction",
                "pir_usable_fraction",
            ]
        ].copy()
        selected.insert(0, "gate_failure_metric", column)
        selected = selected.rename(columns={column: "failure_fraction"})
        pieces.append(selected)
    return pd.concat(pieces, ignore_index=True)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    subjects = (
        [item.strip() for item in args.subjects.split(",") if item.strip()]
        if args.subjects
        else None
    )
    selected = selected_cohort_subjects(config, subjects)

    rows: list[dict[str, object]] = []
    for subject in selected:
        frame = _read_subject(config, subject)
        grouped = frame.groupby(["block_num", "eye"], sort=True)
        for (_, _), group in grouped:
            rows.append(summarize_group(group))

    detail = pd.DataFrame(rows).sort_values(["subject", "block_num", "eye"]).reset_index(drop=True)
    distribution = _distribution_table(detail)
    top = _top_failure_records(detail)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = cohort_output_root(config) / "01_qc" / "screening_review" / "pir_gate_failure_qc"
    output_dir.mkdir(parents=True, exist_ok=True)

    detail.to_csv(output_dir / "pir_gate_failure_by_eye_block.csv", index=False, encoding="utf-8-sig")
    distribution.to_csv(output_dir / "pir_gate_failure_distribution.csv", index=False, encoding="utf-8-sig")
    top.to_csv(output_dir / "pir_gate_failure_top_records.csv", index=False, encoding="utf-8-sig")

    mismatch_max = float(detail["normalization_gate_mismatch_fraction"].max()) if not detail.empty else float("nan")
    readme = f"""# PIR gate failure QC

This folder is a descriptive audit of the six frozen frame-level PIR screening gates.
It does not define subject/eye/Block exclusion thresholds.

Outputs:

- `pir_gate_failure_by_eye_block.csv`: one row per subject × eye × Block, including each gate pass/failure fraction.
- `pir_gate_failure_distribution.csv`: cohort distribution of each gate failure fraction.
- `pir_gate_failure_top_records.csv`: ten highest-failure eye×Block records for each gate.

Frozen six-gate contract:

1. pupil ellipse fit valid;
2. iris-outer ellipse fit valid;
3. pupil mask does not touch the 320×160 analysis ROI edge;
4. iris-outer mask does not touch the 320×160 analysis ROI edge;
5. pupil center lies inside/on the iris-outer contour;
6. iris geometric-mean diameter is larger than pupil geometric-mean diameter.

Final frame-level PIR usable additionally requires a finite PIR numeric value.
Gate failures can overlap, so failure fractions must not be summed as mutually exclusive causes.

`normalization_gate_mismatch_fraction` compares the recomputed six-gate conjunction against the stored `fullclass_normalization_valid`. The maximum observed mismatch in this run is {mismatch_max:.8f}. A non-zero mismatch requires code/data provenance review before interpretation.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"subjects={len(selected)}")
    print(f"eye_block_rows={len(detail)}")
    print(f"output={output_dir}")
    print(f"max_normalization_gate_mismatch_fraction={mismatch_max:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
