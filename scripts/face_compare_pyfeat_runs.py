from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _numeric(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def _metric(a: np.ndarray, b: np.ndarray) -> dict[str, float | int | None]:
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return {"n": 0, "mae": None, "max_abs": None, "pearson_r": None}
    x, y = a[mask], b[mask]
    diff = np.abs(x - y)
    if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
        r = float(np.corrcoef(x, y)[0, 1])
    else:
        r = None
    return {
        "n": int(len(x)),
        "mae": float(diff.mean()),
        "max_abs": float(diff.max()),
        "pearson_r": r,
    }


def _group(ref: pd.DataFrame, cand: pd.DataFrame, columns: list[str]) -> dict[str, object]:
    common = [c for c in columns if c in ref.columns and c in cand.columns]
    if not common:
        return {"columns": 0, "aggregate": None}
    aa = np.concatenate([_numeric(ref[c]) for c in common])
    bb = np.concatenate([_numeric(cand[c]) for c in common])
    return {"columns": len(common), "aggregate": _metric(aa, bb)}


def _bbox_iou(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, float | int | None]:
    cols = ["rf_bbox_x1", "rf_bbox_y1", "rf_bbox_x2", "rf_bbox_y2"]
    if not set(cols).issubset(a.columns) or not set(cols).issubset(b.columns):
        return {"n": 0, "mean": None, "min": None}
    av = a[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    bv = b[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    valid = np.isfinite(av).all(axis=1) & np.isfinite(bv).all(axis=1)
    if not valid.any():
        return {"n": 0, "mean": None, "min": None}
    av, bv = av[valid], bv[valid]
    ix1 = np.maximum(av[:, 0], bv[:, 0])
    iy1 = np.maximum(av[:, 1], bv[:, 1])
    ix2 = np.minimum(av[:, 2], bv[:, 2])
    iy2 = np.minimum(av[:, 3], bv[:, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = np.maximum(0.0, av[:, 2] - av[:, 0]) * np.maximum(0.0, av[:, 3] - av[:, 1])
    area_b = np.maximum(0.0, bv[:, 2] - bv[:, 0]) * np.maximum(0.0, bv[:, 3] - bv[:, 1])
    iou = inter / np.maximum(area_a + area_b - inter, 1e-12)
    return {"n": int(len(iou)), "mean": float(iou.mean()), "min": float(iou.min())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare legacy JPEG and optimized direct-AVI Py-Feat dry-run outputs")
    parser.add_argument("--reference-raw", required=True)
    parser.add_argument("--candidate-raw", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ref = pd.read_parquet(Path(args.reference_raw).resolve())
    cand = pd.read_parquet(Path(args.candidate_raw).resolve())
    key = ["benchmark_index", "face_rank"]
    if not set(key).issubset(ref.columns) or not set(key).issubset(cand.columns):
        raise ValueError("Both raw files must contain benchmark_index and face_rank")

    ref_counts = ref.groupby("benchmark_index").size()
    cand_counts = cand.groupby("benchmark_index").size()
    all_frames = ref_counts.index.union(cand_counts.index)
    face_count_agreement = float(
        (ref_counts.reindex(all_frames, fill_value=0) == cand_counts.reindex(all_frames, fill_value=0)).mean()
    )

    merged = ref.merge(cand, on=key, how="inner", suffixes=("__ref", "__cand"))
    ref_match = pd.DataFrame({c: merged[f"{c}__ref"] for c in ref.columns if f"{c}__ref" in merged.columns})
    cand_match = pd.DataFrame({c: merged[f"{c}__cand"] for c in cand.columns if f"{c}__cand" in merged.columns})

    groups = {
        "face_score": ["FaceScore"],
        "au20": [c for c in ref.columns if c.startswith("AU") and c[2:].isdigit()],
        "emotion7": [c for c in ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"] if c in ref.columns],
        "valence_arousal": ["valence", "arousal"],
        "pose6d": ["Pitch", "Roll", "Yaw", "X", "Y", "Z"],
        "gaze": ["gaze_pitch", "gaze_yaw", "gaze_angle"],
        "blendshapes": [c for c in ref.columns if c in cand.columns and (
            c.startswith("eye") or c.startswith("brow") or c.startswith("cheek") or c.startswith("jaw")
            or c.startswith("mouth") or c.startswith("nose") or c == "_neutral"
        )],
        "mesh_norm_478x3": [c for c in ref.columns if c.startswith("mesh_norm_")],
        "mesh_original_xy": [c for c in ref.columns if c.startswith("mesh_x_") or c.startswith("mesh_y_")],
    }
    result = {
        "schema_version": "rgb-face-formal-dryrun-optimization-parity-v0.1",
        "reference_raw": str(Path(args.reference_raw).resolve()),
        "candidate_raw": str(Path(args.candidate_raw).resolve()),
        "reference_rows": int(len(ref)),
        "candidate_rows": int(len(cand)),
        "matched_rows": int(len(merged)),
        "face_count_agreement": face_count_agreement,
        "bbox_iou": _bbox_iou(ref_match, cand_match),
        "groups": {name: _group(ref_match, cand_match, cols) for name, cols in groups.items()},
        "notes": [
            "Reference uses JPEG-quality-95 dry-run frames; candidate reads the original AVI directly.",
            "Small non-zero drift can therefore reflect removal of the JPEG round-trip rather than a model-graph change.",
        ],
    }
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
