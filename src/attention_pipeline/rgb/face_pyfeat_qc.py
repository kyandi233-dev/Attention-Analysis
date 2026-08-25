from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from attention_pipeline.config import Config
from attention_pipeline.rgb.paths import RGBOutputLayout


PYFEAT_QC_SCHEMA_VERSION = "rgb-face-benchmark-pyfeat-qc-v0.2"
FACEBOX_COLUMNS = ["FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight", "FaceScore"]
# Detectorv2/v2.4 native emotion schema is title-cased and differs from the
# legacy Detectorv1 lowercase names.
EMOTION_COLUMNS_V2 = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]
VA_COLUMNS = ["valence", "arousal"]
GAZE_COLUMNS = ["gaze_pitch", "gaze_yaw", "gaze_angle"]
HEADPOSE_COLUMNS = ["Pitch", "Roll", "Yaw", "X", "Y", "Z"]


def _numeric_summary(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {"count": 0, "min": None, "p10": None, "p50": None, "p90": None, "max": None, "mean": None}
    q = values.quantile([0.10, 0.50, 0.90])
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "p10": float(q.loc[0.10]),
        "p50": float(q.loc[0.50]),
        "p90": float(q.loc[0.90]),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def _group_summary(table: pd.DataFrame, columns: list[str]) -> dict[str, object]:
    present = [c for c in columns if c in table.columns]
    if not present:
        return {"columns": [], "column_count": 0, "row_any_valid_fraction": None, "cell_valid_fraction": None}
    subset = table[present]
    valid = subset.notna()
    return {
        "columns": present,
        "column_count": len(present),
        "row_any_valid_fraction": float(valid.any(axis=1).mean()) if len(subset) else None,
        "cell_valid_fraction": float(valid.to_numpy().mean()) if subset.size else None,
    }


def _detect_input_column(raw: pd.DataFrame) -> str:
    for candidate in ("input", "image_path", "file", "filename"):
        if candidate in raw.columns:
            return candidate
    raise ValueError("Py-Feat raw output has no recognizable input-path column")


def summarize_pyfeat_benchmark(sample: pd.DataFrame, raw: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    if sample.empty or "image_path" not in sample.columns:
        raise ValueError("Benchmark frame manifest is empty or missing image_path")
    if raw.empty:
        raise ValueError("Py-Feat raw output is empty")

    original_output_columns = int(len(raw.columns))
    input_col = _detect_input_column(raw)
    sample = sample.copy()
    raw = raw.copy()
    sample["_image_name"] = sample["image_path"].astype(str).map(lambda p: Path(p).name)
    raw["_image_name"] = raw[input_col].astype(str).map(lambda p: Path(p).name)

    counts = raw.groupby("_image_name").size().rename("face_count")
    per_image = sample.merge(counts, left_on="_image_name", right_index=True, how="left")
    per_image["face_count"] = per_image["face_count"].fillna(0).astype(int)

    if "FaceScore" in raw.columns:
        max_score = pd.to_numeric(raw["FaceScore"], errors="coerce").groupby(raw["_image_name"]).max().rename("max_face_score")
        per_image = per_image.merge(max_score, left_on="_image_name", right_index=True, how="left")
    else:
        per_image["max_face_score"] = np.nan

    phase_summary: dict[str, object] = {}
    if "phase" in per_image.columns:
        for phase, group in per_image.groupby("phase", dropna=False):
            phase_summary[str(phase)] = {
                "input_images": int(len(group)),
                "images_with_face": int((group["face_count"] > 0).sum()),
                "images_without_face": int((group["face_count"] == 0).sum()),
                "images_with_multiple_faces": int((group["face_count"] > 1).sum()),
                "face_detection_image_fraction": float((group["face_count"] > 0).mean()),
            }

    au_columns = [c for c in raw.columns if re.fullmatch(r"AU\d+", str(c))]
    # Detectorv2 exposes both a dlib-68 compatibility landmark block and the
    # full native 478x3 mesh. Keep these groups distinct in QC.
    landmark68_columns = [c for c in raw.columns if re.fullmatch(r"[xy]_\d+", str(c))]
    mesh_columns = [c for c in raw.columns if re.fullmatch(r"mesh_[xyz]_\d+", str(c))]
    identity_columns = [
        c for c in raw.columns
        if str(c) == "Identity" or re.fullmatch(r"Identity_\d+", str(c))
    ]
    excluded_for_blendshape = set(
        au_columns
        + landmark68_columns
        + mesh_columns
        + identity_columns
        + EMOTION_COLUMNS_V2
        + VA_COLUMNS
        + GAZE_COLUMNS
        + HEADPOSE_COLUMNS
        + FACEBOX_COLUMNS
    )
    blendshape_candidates = [
        c for c in raw.columns
        if any(token in str(c).lower() for token in ("brow", "cheek", "eye", "jaw", "mouth", "nose", "tongue"))
        and c not in excluded_for_blendshape
    ]

    constant_numeric_columns = []
    for col in raw.select_dtypes(include=[np.number]).columns:
        if col == "_image_name":
            continue
        values = pd.to_numeric(raw[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not values.empty and values.nunique(dropna=True) <= 1:
            constant_numeric_columns.append(str(col))

    multi = per_image[per_image["face_count"] > 1].copy()
    multi_records = [
        {
            "benchmark_index": int(row.benchmark_index) if hasattr(row, "benchmark_index") else None,
            "image_path": str(row.image_path),
            "phase": str(row.phase) if hasattr(row, "phase") else None,
            "face_count": int(row.face_count),
            "max_face_score": None if pd.isna(row.max_face_score) else float(row.max_face_score),
        }
        for row in multi.itertuples(index=False)
    ]

    summary = {
        "schema_version": PYFEAT_QC_SCHEMA_VERSION,
        "expected_input_images": int(len(sample)),
        "result_rows": int(len(raw)),
        "input_column": input_col,
        "images_with_face": int((per_image["face_count"] > 0).sum()),
        "images_without_face": int((per_image["face_count"] == 0).sum()),
        "images_with_exactly_one_face": int((per_image["face_count"] == 1).sum()),
        "images_with_multiple_faces": int((per_image["face_count"] > 1).sum()),
        "extra_face_rows_above_one_per_input": int((per_image["face_count"] - 1).clip(lower=0).sum()),
        "face_detection_image_fraction": float((per_image["face_count"] > 0).mean()),
        "face_score": _numeric_summary(raw["FaceScore"]) if "FaceScore" in raw.columns else None,
        "phase": phase_summary,
        "field_groups": {
            "facebox": _group_summary(raw, FACEBOX_COLUMNS),
            "action_units": _group_summary(raw, au_columns),
            "emotion_v2": _group_summary(raw, EMOTION_COLUMNS_V2),
            "valence_arousal": _group_summary(raw, VA_COLUMNS),
            "gaze": _group_summary(raw, GAZE_COLUMNS),
            "head_pose": _group_summary(raw, HEADPOSE_COLUMNS),
            "landmark68_xy": _group_summary(raw, landmark68_columns),
            "mesh478_xyz": _group_summary(raw, mesh_columns),
            "blendshapes": _group_summary(raw, blendshape_candidates),
            "identity": _group_summary(raw, identity_columns),
        },
        "output_columns": original_output_columns,
        "au_columns": au_columns,
        "landmark68_xy_column_count": len(landmark68_columns),
        "mesh478_xyz_column_count": len(mesh_columns),
        "identity_column_count": len(identity_columns),
        "constant_numeric_column_count": len(constant_numeric_columns),
        "constant_numeric_columns_first_50": constant_numeric_columns[:50],
        "multi_face_inputs": multi_records,
        "interpretation": (
            "This first-round benchmark tests installation, coverage, output availability and speed on sparse phase-stratified frames. "
            "It does not establish temporal smoothness or scientific accuracy; those require a contiguous-window review and/or ground truth."
        ),
    }
    return summary, per_image.drop(columns=["_image_name"], errors="ignore")


def run_pyfeat_benchmark_qc(config: Config, subject: str) -> dict[str, object]:
    layout = RGBOutputLayout.from_config(config)
    root = layout.test_dir() / "face-benchmark" / subject
    frame_manifests = sorted(root.glob("*_face-benchmark_frames.csv"))
    if len(frame_manifests) != 1:
        raise RuntimeError(f"Expected one benchmark frame manifest in {root}, found {len(frame_manifests)}")
    raw_path = root / "pyfeat_raw.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"Py-Feat raw benchmark output not found: {raw_path}")

    sample = pd.read_csv(frame_manifests[0])
    raw = pd.read_parquet(raw_path)
    summary, per_image = summarize_pyfeat_benchmark(sample, raw)
    summary["subject"] = subject
    summary["frame_manifest"] = str(frame_manifests[0])
    summary["raw_parquet"] = str(raw_path)

    per_image_path = root / "pyfeat_qc_per_image.csv"
    json_path = root / "pyfeat_qc.json"
    per_image.to_csv(per_image_path, index=False, encoding="utf-8-sig")
    summary["per_image_output"] = str(per_image_path)
    summary["output"] = str(json_path)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
