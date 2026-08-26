"""Read-only audit of frame-level RITnet edge and segmentation quality.

This script audits the frozen production full-class CSVs and the bounded sparse
QC label images.  It never edits production NIR files, changes a gate, deletes
rows, or selects an exclusion threshold.

The production edge flags are available for every block1/block2 frame in the
CSV.  Largest-contour and stray-component edge flags require a saved label map;
therefore those exact mask-level results are reported with their sparse-QC
denominator rather than being presented as all-frame estimates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
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


ANALYSIS_SIZE = (320, 160)
FORMAL_PHASES = ("block1", "block2")
EYE_ORDER = ("frame_left", "frame_right")
LABEL_COLORS_BGR = {
    "sclera": (255, 0, 0),
    "iris": (0, 255, 0),
    "pupil": (0, 0, 255),
}

BASE_COLUMNS = {
    "subject": "subject",
    "phase": "phase",
    "phase_segment": "phase_segment",
    "frame_idx": "frame_idx",
    "unix_ms": "unix_ms",
    "eye": "eye",
    "normalization_valid": "fullclass_normalization_valid",
    "pupil_edge": "fullclass_pupil_touches_roi_edge",
    "iris_edge": "fullclass_iris_outer_touches_roi_edge",
    "pupil_fit": "fullclass_pupil_fit_valid",
    "iris_fit": "fullclass_iris_outer_fit_valid",
    "center_in": "fullclass_pupil_center_in_iris_outer",
    "pupil_diameter": "fullclass_pupil_geom_mean_diameter",
    "iris_diameter": "fullclass_iris_outer_geom_mean_diameter",
    "video": "video",
    "roi_x1": "roi_x1",
    "roi_y1": "roi_y1",
    "roi_x2": "roi_x2",
    "roi_y2": "roi_y2",
}

QUALITY_COLUMNS = {
    "pupil_confidence": "fullclass_pupil_confidence",
    "ocular_component_count": "fullclass_ocular_component_count",
    "ocular_largest_component_fraction": "fullclass_ocular_largest_component_fraction",
    "iris_outer_fill_ratio": "fullclass_iris_outer_fill_ratio",
    "ellipse_area_ratio": "fullclass_pupil_to_iris_ellipse_area_ratio",
    "contour_area_ratio": "fullclass_pupil_to_iris_contour_area_ratio",
    "center_offset_px": "fullclass_pupil_center_offset_px",
    "center_offset_norm": "fullclass_pupil_center_offset_norm",
    "pir": PIR_COLUMN,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/nir_behavior_cohort.yaml")
    parser.add_argument("--subjects", help="Comma-separated subject override")
    parser.add_argument("--output-dir", help="External audit output directory")
    parser.add_argument("--example-limit", type=int, default=12)
    return parser.parse_args()


def _to_bool_series(series: pd.Series, *, default: bool = False) -> pd.Series:
    values = coerce_bool_series(series)
    return values.fillna(default).astype(bool)


def _bool_value(value: Any, default: bool = False) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", "", "nan", "none"}:
        return False
    return default


def _read_csv_with_named_columns(path: Path, names: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    available = set(header.columns)
    missing = [column for column in names.values() if column not in available]
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")
    usecols = list(dict.fromkeys(names.values()))
    frame = pd.read_csv(path, usecols=usecols, encoding="utf-8-sig", low_memory=False)
    return frame, sorted(missing)


def _load_subject_frame(config: Any, subject: str) -> tuple[pd.DataFrame, Any]:
    source = find_nir_source(alignment_config(config), subject)
    names = {**BASE_COLUMNS, **QUALITY_COLUMNS}
    frame, _ = _read_csv_with_named_columns(source.csv_path, names)
    rename = {source_name: key for key, source_name in names.items()}
    frame = frame.rename(columns=rename)
    frame["pir_valid"] = frame["normalization_valid"]
    frame["phase"] = frame["phase"].astype(str).str.strip()
    frame = frame[frame["phase"].isin(FORMAL_PHASES)].copy()
    frame["subject"] = frame["subject"].astype(str).str.strip()
    frame["eye"] = frame["eye"].astype(str).str.strip()
    frame["block_num"] = frame["phase"].map({"block1": 1, "block2": 2}).astype("int64")
    for column in frame.columns:
        if column in {"subject", "phase", "eye", "video"}:
            continue
        if column in {"normalization_valid", "pupil_edge", "iris_edge", "pupil_fit", "iris_fit", "center_in", "pir_valid"}:
            frame[column] = coerce_bool_series(frame[column])
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["normalization_valid"] = frame["normalization_valid"].fillna(False).astype(bool)
    # Missing edge flags are conservative failures, matching the existing gate audit.
    frame["pupil_edge"] = frame["pupil_edge"].fillna(True).astype(bool)
    frame["iris_edge"] = frame["iris_edge"].fillna(True).astype(bool)
    frame["pupil_fit"] = frame["pupil_fit"].fillna(False).astype(bool)
    frame["iris_fit"] = frame["iris_fit"].fillna(False).astype(bool)
    frame["center_in"] = frame["center_in"].fillna(False).astype(bool)
    frame["pir_valid"] = frame["pir_valid"].fillna(False).astype(bool)
    frame["row_key"] = (
        frame["phase"].astype(str)
        + "|"
        + frame["phase_segment"].fillna(-1).astype(int).astype(str)
        + "|"
        + frame["frame_idx"].fillna(-1).astype(int).astype(str)
        + "|"
        + frame["eye"].astype(str)
    )
    frame = frame.sort_values(["phase", "phase_segment", "eye", "frame_idx", "unix_ms"]).reset_index(drop=True)
    # A jump is only defined for adjacent finite PIR values within an eye/block sequence.
    frame["pir_jump_abs"] = frame.groupby(["phase", "phase_segment", "eye"], sort=False)["pir"].diff().abs()
    return frame, source


def _qc_index_path(source: Any, subject: str) -> Path | None:
    candidates = sorted(source.csv_path.parent.glob(f"{subject}_ritnet_fullclass_v1-2-fast-qc_qc_index.csv"))
    return candidates[0] if candidates else None


def _resolve_qc_path(qc_index_path: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return qc_index_path.parent / Path(str(value).replace("\\", os.sep))


def _mask_edge(mask: np.ndarray) -> bool:
    return bool(mask.size and mask.any() and (mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any()))


def _largest_contour_edge(mask: np.ndarray) -> bool:
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    contour = max(contours, key=cv2.contourArea)
    points = contour.reshape(-1, 2)
    height, width = mask.shape
    return bool(
        np.any(points[:, 0] <= 0)
        or np.any(points[:, 0] >= width - 1)
        or np.any(points[:, 1] <= 0)
        or np.any(points[:, 1] >= height - 1)
    )


def _parse_label_masks(path: Path) -> dict[str, np.ndarray]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read label image: {path}")
    native_masks = {
        name: np.all(image == np.asarray(color, dtype=np.uint8), axis=2)
        for name, color in LABEL_COLORS_BGR.items()
    }
    native_class = np.zeros(image.shape[:2], dtype=np.uint8)
    native_class[native_masks["sclera"]] = 1
    native_class[native_masks["iris"]] = 2
    native_class[native_masks["pupil"]] = 3
    resized = cv2.resize(native_class, ANALYSIS_SIZE, interpolation=cv2.INTER_NEAREST)
    pupil = resized == 3
    iris_outer = (resized == 2) | pupil
    return {"pupil": pupil, "iris_outer": iris_outer}


def _mask_audit(label_path: Path) -> dict[str, Any]:
    masks = _parse_label_masks(label_path)
    result: dict[str, Any] = {"label_path": str(label_path)}
    for name, mask in masks.items():
        whole = _mask_edge(mask)
        largest = _largest_contour_edge(mask)
        result[f"{name}_whole_mask_touches_edge"] = whole
        result[f"{name}_largest_contour_touches_edge"] = largest
        result[f"{name}_stray_component_edge_only"] = bool(whole and not largest)
    result["any_whole_mask_edge"] = bool(result["pupil_whole_mask_touches_edge"] or result["iris_outer_whole_mask_touches_edge"])
    result["any_largest_contour_edge"] = bool(result["pupil_largest_contour_touches_edge"] or result["iris_outer_largest_contour_touches_edge"])
    result["any_stray_component_edge_only"] = bool(result["pupil_stray_component_edge_only"] or result["iris_outer_stray_component_edge_only"])
    return result


def _quality_metrics(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = [
        "pupil_confidence",
        "ocular_component_count",
        "ocular_largest_component_fraction",
        "iris_outer_fill_ratio",
        "ellipse_area_ratio",
        "contour_area_ratio",
        "center_offset_px",
        "center_offset_norm",
        "pir_jump_abs",
    ]
    grouping = frame.groupby(group_cols, sort=True, dropna=False)
    for keys, group in grouping:
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for status, subset in (("normalization_valid", group[group["normalization_valid"]]), ("normalization_invalid", group[~group["normalization_valid"]])):
            row = {**base, "status": status, "n_rows": int(len(subset))}
            for metric in metrics:
                values = pd.to_numeric(subset[metric], errors="coerce")
                finite = values[np.isfinite(values)]
                row[f"{metric}_finite_n"] = int(len(finite))
                if len(finite):
                    quantiles = finite.quantile([0.05, 0.50, 0.95])
                    row[f"{metric}_p05"] = float(quantiles.loc[0.05])
                    row[f"{metric}_median"] = float(quantiles.loc[0.50])
                    row[f"{metric}_p95"] = float(quantiles.loc[0.95])
                    row[f"{metric}_min"] = float(finite.min())
                    row[f"{metric}_max"] = float(finite.max())
                else:
                    for suffix in ("p05", "median", "p95", "min", "max"):
                        row[f"{metric}_{suffix}"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _existing_suspicious(frame_row: pd.Series) -> bool:
    count = frame_row.get("ocular_component_count")
    fraction = frame_row.get("ocular_largest_component_fraction")
    if pd.notna(count) and pd.notna(fraction):
        # This is the already-frozen sparse QC diagnostic condition, reused as
        # a label for examples only; it is not introduced as a formal cutoff.
        if float(count) > 1 and float(fraction) < 0.90:
            return True
    return False


def _blur_variance(row: pd.Series) -> float | None:
    video = row.get("video")
    if pd.isna(video) or not str(video).strip():
        return None
    video_path = Path(str(video))
    if not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    try:
        frame_idx = int(float(row.get("frame_idx", -1)))
        if frame_idx < 0 or not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx):
            return None
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        x1, y1, x2, y2 = [int(float(row.get(key))) for key in ("roi_x1", "roi_y1", "roi_x2", "roi_y2")]
        h, w = frame.shape[:2]
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return None
        gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        if gray.size == 0:
            return None
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except (TypeError, ValueError, OverflowError, cv2.error):
        return None
    finally:
        cap.release()


def _copy_example(source_path: Path, destination_dir: Path, prefix: str) -> Path | None:
    if not source_path.exists():
        return None
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{prefix}_{source_path.name}"
    shutil.copy2(source_path, destination)
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    subjects = [item.strip() for item in args.subjects.split(",") if item.strip()] if args.subjects else None
    selected = selected_cohort_subjects(config, subjects)
    output_dir = Path(args.output_dir) if args.output_dir else cohort_output_root(config) / "04_frame_quality_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    errors: list[dict[str, str]] = []
    subject_rows: list[pd.DataFrame] = []
    quality_rows: list[pd.DataFrame] = []
    mask_rows: list[dict[str, Any]] = []
    example_candidates: dict[str, list[dict[str, Any]]] = {"a_main_contour_edge": [], "b_stray_edge_only": [], "c_valid_suspicious": [], "d_valid_control": []}

    for subject in selected:
        try:
            frame, source = _load_subject_frame(config, subject)
            frame["current_edge_fail"] = frame["pupil_edge"] | frame["iris_edge"]
            diameter_order = np.isfinite(frame["pupil_diameter"]) & np.isfinite(frame["iris_diameter"]) & (frame["iris_diameter"] > frame["pupil_diameter"])
            frame["other_six_gates_pass"] = frame["pupil_fit"] & frame["iris_fit"] & frame["center_in"] & diameter_order
            frame["edge_only_invalid_candidate"] = (~frame["normalization_valid"]) & frame["current_edge_fail"] & frame["other_six_gates_pass"]
            frame["source_csv"] = str(source.csv_path)
            subject_rows.append(frame)
            quality_rows.append(_quality_metrics(frame, ["subject", "block_num", "eye"]))

            qc_index_path = _qc_index_path(source, subject)
            if qc_index_path is None:
                warnings.append(f"{subject}: sparse QC index not found")
                continue
            qc = pd.read_csv(qc_index_path, encoding="utf-8-sig")
            qc = qc[qc["phase"].isin(FORMAL_PHASES)].copy()
            if qc.empty:
                warnings.append(f"{subject}: sparse QC index has no block1/block2 rows")
                continue
            qc["phase_segment"] = pd.to_numeric(qc["phase_segment"], errors="coerce")
            qc["frame_idx"] = pd.to_numeric(qc["frame_idx"], errors="coerce")
            qc["row_key"] = qc["phase"].astype(str) + "|" + qc["phase_segment"].fillna(-1).astype(int).astype(str) + "|" + qc["frame_idx"].fillna(-1).astype(int).astype(str) + "|" + qc["eye"].astype(str)
            keys = frame[["row_key"]].value_counts()
            duplicate_keys = int((keys > 1).sum())
            if duplicate_keys:
                warnings.append(f"{subject}: {duplicate_keys} duplicate fullclass frame keys; first row used for sparse join")
            lookup = frame.drop_duplicates("row_key", keep="first").set_index("row_key")
            for _, qc_row in qc.sort_values(["phase", "phase_segment", "frame_idx", "eye"]).iterrows():
                row_key = qc_row["row_key"]
                if row_key not in lookup.index:
                    warnings.append(f"{subject}: QC label key not found in fullclass CSV: {row_key}")
                    continue
                source_row = lookup.loc[row_key]
                label_path = _resolve_qc_path(qc_index_path, qc_row.get("labels_file"))
                try:
                    mask = _mask_audit(label_path)
                except Exception as exc:
                    errors.append({"subject": subject, "row_key": row_key, "error": f"{type(exc).__name__}: {exc}"})
                    continue
                record = {
                    "subject": subject,
                    "phase": source_row["phase"],
                    "block_num": int(source_row["block_num"]),
                    "phase_segment": int(source_row["phase_segment"]),
                    "frame_idx": int(source_row["frame_idx"]),
                    "eye": source_row["eye"],
                    "normalization_valid": bool(source_row["normalization_valid"]),
                    "current_edge_fail": bool(source_row["current_edge_fail"]),
                    "edge_only_invalid_candidate": bool(source_row["edge_only_invalid_candidate"]),
                    "existing_fragmentation_flag": bool(_existing_suspicious(source_row)),
                    "csv_pupil_whole_mask_touches_edge": bool(source_row["pupil_edge"]),
                    "csv_iris_outer_whole_mask_touches_edge": bool(source_row["iris_edge"]),
                    "qc_reason": str(qc_row.get("reason", "")),
                    "overlay_file": str(_resolve_qc_path(qc_index_path, qc_row.get("overlay_file"))),
                    **mask,
                }
                record["csv_any_whole_mask_edge"] = bool(record["csv_pupil_whole_mask_touches_edge"] or record["csv_iris_outer_whole_mask_touches_edge"])
                record["pupil_whole_edge_mismatch"] = bool(record["csv_pupil_whole_mask_touches_edge"] != record["pupil_whole_mask_touches_edge"])
                record["iris_outer_whole_edge_mismatch"] = bool(record["csv_iris_outer_whole_mask_touches_edge"] != record["iris_outer_whole_mask_touches_edge"])
                record["any_whole_edge_mismatch"] = bool(record["csv_any_whole_mask_edge"] != record["any_whole_mask_edge"])
                mask_rows.append(record)
                sampled_current_edge = record["current_edge_fail"]
                largest = record["any_largest_contour_edge"]
                stray_only = record["any_stray_component_edge_only"] and not largest
                suspicious = (bool(source_row["normalization_valid"]) and _existing_suspicious(source_row))
                candidate = {**record, "label_file": str(label_path), "source_row_key": row_key, "blur_laplacian_variance": None}
                if sampled_current_edge and largest:
                    example_candidates["a_main_contour_edge"].append(candidate)
                if sampled_current_edge and stray_only:
                    example_candidates["b_stray_edge_only"].append(candidate)
                if suspicious:
                    example_candidates["c_valid_suspicious"].append(candidate)
                if bool(source_row["normalization_valid"]) and not suspicious:
                    example_candidates["d_valid_control"].append(candidate)
        except Exception as exc:
            errors.append({"subject": subject, "row_key": "", "error": f"{type(exc).__name__}: {exc}"})

    all_frames = pd.concat(subject_rows, ignore_index=True) if subject_rows else pd.DataFrame()
    quality = pd.concat(quality_rows, ignore_index=True) if quality_rows else pd.DataFrame()
    masks = pd.DataFrame(mask_rows)

    if not all_frames.empty:
        all_frames["cohort"] = "cohort"
        all_frames["existing_fragmentation_flag"] = (
            pd.to_numeric(all_frames["ocular_component_count"], errors="coerce").gt(1)
            & pd.to_numeric(all_frames["ocular_largest_component_fraction"], errors="coerce").lt(0.90)
        )
        cohort_quality = _quality_metrics(all_frames, ["cohort"])
        quality = pd.concat([quality, cohort_quality], ignore_index=True)
        grouped = all_frames.groupby(["subject", "block_num", "eye"], sort=True)
        edge_rows = []
        for (subject, block, eye), group in grouped:
            invalid = ~group["normalization_valid"]
            edge_rows.append({
                "subject": subject,
                "block_num": int(block),
                "eye": eye,
                "n_rows": int(len(group)),
                "normalization_invalid_n": int(invalid.sum()),
                "current_edge_fail_n": int(group["current_edge_fail"].sum()),
                "current_edge_fail_fraction": float(group["current_edge_fail"].mean()),
                "pupil_whole_mask_edge_n": int(group["pupil_edge"].sum()),
                "pupil_whole_mask_edge_fraction": float(group["pupil_edge"].mean()),
                "iris_outer_whole_mask_edge_n": int(group["iris_edge"].sum()),
                "iris_outer_whole_mask_edge_fraction": float(group["iris_edge"].mean()),
                "edge_only_invalid_candidate_n": int(group["edge_only_invalid_candidate"].sum()),
                "normalization_valid_n": int(group["normalization_valid"].sum()),
            })
        edge_detail = pd.DataFrame(edge_rows)
    else:
        edge_detail = pd.DataFrame()

    cohort_rows: list[dict[str, Any]] = []
    if not all_frames.empty:
        invalid = ~all_frames["normalization_valid"]
        for label, mask in {
            "all_block_frames": np.ones(len(all_frames), dtype=bool),
            "normalization_invalid": invalid.to_numpy(),
            "edge_only_invalid_candidate": all_frames["edge_only_invalid_candidate"].to_numpy(),
            "valid_existing_fragmentation_flag": (all_frames["normalization_valid"] & all_frames["existing_fragmentation_flag"]).to_numpy(),
            "current_pupil_whole_mask_edge": all_frames["pupil_edge"].to_numpy(),
            "current_iris_outer_whole_mask_edge": all_frames["iris_edge"].to_numpy(),
            "current_any_whole_mask_edge": all_frames["current_edge_fail"].to_numpy(),
            "invalid_current_pupil_whole_mask_edge": (invalid & all_frames["pupil_edge"]).to_numpy(),
            "invalid_current_iris_outer_whole_mask_edge": (invalid & all_frames["iris_edge"]).to_numpy(),
        }.items():
            cohort_rows.append({"scope": label, "n_frames": int(mask.sum()), "denominator": int(len(all_frames)), "fraction_of_all_block_frames": float(mask.mean())})
    if not masks.empty:
        for label, column in (
            ("sampled_label_rows", None),
            ("sampled_pupil_whole_mask_edge", "pupil_whole_mask_touches_edge"),
            ("sampled_iris_outer_whole_mask_edge", "iris_outer_whole_mask_touches_edge"),
            ("sampled_pupil_largest_contour_edge", "pupil_largest_contour_touches_edge"),
            ("sampled_iris_outer_largest_contour_edge", "iris_outer_largest_contour_touches_edge"),
            ("sampled_pupil_stray_component_edge_only", "pupil_stray_component_edge_only"),
            ("sampled_iris_outer_stray_component_edge_only", "iris_outer_stray_component_edge_only"),
            ("sampled_whole_mask_edge", "any_whole_mask_edge"),
            ("sampled_largest_contour_edge", "any_largest_contour_edge"),
            ("sampled_stray_component_edge_only", "any_stray_component_edge_only"),
        ):
            count = len(masks) if column is None else int(masks[column].sum())
            cohort_rows.append({"scope": label, "n_frames": count, "denominator": int(len(masks)), "fraction_of_sampled_label_rows": float(count / len(masks))})
        for label, column in (
            ("sampled_pupil_whole_edge_mismatch", "pupil_whole_edge_mismatch"),
            ("sampled_iris_outer_whole_edge_mismatch", "iris_outer_whole_edge_mismatch"),
            ("sampled_any_whole_edge_mismatch", "any_whole_edge_mismatch"),
            ("sampled_existing_fragmentation_flag", "existing_fragmentation_flag"),
        ):
            count = int(masks[column].sum())
            cohort_rows.append({"scope": label, "n_frames": count, "denominator": int(len(masks)), "fraction_of_sampled_label_rows": float(count / len(masks))})
        invalid_sample = masks[~masks["normalization_valid"]]
        edge_only_sample = invalid_sample[invalid_sample["edge_only_invalid_candidate"]]
        for label, count in (
            ("sampled_invalid", len(invalid_sample)),
            ("sampled_invalid_edge_only_candidate", len(edge_only_sample)),
            ("sampled_invalid_stray_only_candidate", int((edge_only_sample["any_stray_component_edge_only"] & ~edge_only_sample["any_largest_contour_edge"]).sum())),
        ):
            cohort_rows.append({"scope": label, "n_frames": count, "denominator": int(len(invalid_sample)), "fraction_of_sampled_invalid": float(count / len(invalid_sample)) if len(invalid_sample) else np.nan})
        stray_only_count = int((edge_only_sample["any_stray_component_edge_only"] & ~edge_only_sample["any_largest_contour_edge"]).sum())
        cohort_rows.append({"scope": "sampled_invalid_stray_only_among_edge_only_candidate", "n_frames": stray_only_count, "denominator": int(len(edge_only_sample)), "fraction_of_sampled_edge_only_candidate": float(stray_only_count / len(edge_only_sample)) if len(edge_only_sample) else np.nan})

    examples: list[dict[str, Any]] = []
    for category, candidates in example_candidates.items():
        candidates = sorted(candidates, key=lambda row: (row["subject"], row["phase"], row["frame_idx"], row["eye"]))[: max(0, args.example_limit)]
        category_dir = output_dir / "examples" / category
        for idx, candidate in enumerate(candidates, start=1):
            label_src = Path(candidate["label_file"])
            overlay_src = Path(candidate["overlay_file"])
            prefix = f"{idx:02d}_{candidate['subject']}_{candidate['phase']}_f{candidate['frame_idx']:08d}_{candidate['eye']}"
            label_dst = _copy_example(label_src, category_dir, prefix)
            overlay_dst = _copy_example(overlay_src, category_dir, prefix)
            candidate = dict(candidate)
            candidate["category"] = category
            candidate["label_copy"] = str(label_dst) if label_dst else ""
            candidate["overlay_copy"] = str(overlay_dst) if overlay_dst else ""
            source_matches = all_frames[
                (all_frames["subject"].astype(str) == str(candidate["subject"]))
                & (all_frames["row_key"].astype(str) == str(candidate["source_row_key"]))
            ]
            candidate["blur_laplacian_variance"] = _blur_variance(source_matches.iloc[0]) if not source_matches.empty else None
            examples.append(candidate)

    output_files = {
        "edge_detail": output_dir / "frame_edge_by_subject_eye_block.csv",
        "quality": output_dir / "segmentation_quality_by_status.csv",
        "mask_sample": output_dir / "sampled_mask_edge_audit.csv",
        "cohort_summary": output_dir / "frame_quality_cohort_summary.csv",
        "examples": output_dir / "bounded_qc_examples.csv",
        "errors": output_dir / "scan_errors.csv",
        "warnings": output_dir / "scan_warnings.txt",
    }
    edge_detail.to_csv(output_files["edge_detail"], index=False, encoding="utf-8-sig")
    quality.to_csv(output_files["quality"], index=False, encoding="utf-8-sig")
    masks.to_csv(output_files["mask_sample"], index=False, encoding="utf-8-sig")
    pd.DataFrame(cohort_rows).to_csv(output_files["cohort_summary"], index=False, encoding="utf-8-sig")
    pd.DataFrame(examples).to_csv(output_files["examples"], index=False, encoding="utf-8-sig")
    pd.DataFrame(errors, columns=["subject", "row_key", "error"]).to_csv(output_files["errors"], index=False, encoding="utf-8-sig")
    output_files["warnings"].write_text("\n".join(warnings) + ("\n" if warnings else ""), encoding="utf-8")

    metadata = {
        "subjects_requested": len(selected),
        "subjects_loaded": int(all_frames["subject"].nunique()) if not all_frames.empty else 0,
        "formal_phases": list(FORMAL_PHASES),
        "analysis_size": list(ANALYSIS_SIZE),
        "sparse_label_rows": int(len(masks)),
        "production_csvs_are_read_only": True,
        "largest_contour_denominator_is_sparse_qc_only": True,
        "warnings_n": len(warnings),
        "errors_n": len(errors),
        "output_sha256": {name: _sha256(path) for name, path in output_files.items() if path.exists()},
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = f"""# Frame-level edge and segmentation-quality audit

This is a descriptive, read-only audit of the frozen full-class outputs for formal `block1` and `block2` only. It does not replace the current production edge gate, set a QC threshold, exclude data, or run alignment/statistics.

- Subjects requested/loaded: {len(selected)}/{metadata['subjects_loaded']}
- All-frame current edge flags and existing diagnostics use the production CSV rows.
- Largest-contour and stray-component edge flags are reconstructed from the available sparse QC labels only (`{len(masks)}` block rows). The denominator is explicitly reported and is not an all-frame estimate.
- Labels are parsed with the frozen palette and resized from native QC resolution to 320x160 with nearest-neighbor interpolation before edge checks.
- `blur_laplacian_variance` is present only for bounded examples when the source video/ROI can be opened; it is diagnostic only and has no cutoff.
- `scan_errors.csv` and `scan_warnings.txt` list all read/join/output limitations.

The existing sparse fragmentation condition (`ocular_component_count > 1` and `ocular_largest_component_fraction < 0.90`) is reused only to select the `valid_suspicious` example class. It is not introduced as a formal exclusion rule.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"subjects_requested={len(selected)}")
    print(f"subjects_loaded={metadata['subjects_loaded']}")
    print(f"formal_block_frames={len(all_frames)}")
    print(f"sampled_label_rows={len(masks)}")
    print(f"scan_errors={len(errors)}")
    print(f"scan_warnings={len(warnings)}")
    print(f"output_dir={output_dir}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
