from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _find_frame_manifest(root: Path) -> Path:
    candidates = sorted(root.glob("*_face-continuous_frames.csv")) + sorted(root.glob("*_face-benchmark_frames.csv"))
    unique = list(dict.fromkeys(candidates))
    if len(unique) != 1:
        raise RuntimeError(f"Expected exactly one shared Face frame manifest in {root}, found {len(unique)}")
    return unique[0]


def _mediapipe_gaze_features(aligned_paths: list[str]) -> tuple[np.ndarray, list[str | None]]:
    import mediapipe as mp

    rows: list[np.ndarray] = []
    errors: list[str | None] = []
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
    try:
        for path_str in aligned_paths:
            image_bgr = cv2.imread(path_str, cv2.IMREAD_COLOR)
            if image_bgr is None:
                rows.append(np.full((1404,), np.nan, dtype=np.float32))
                errors.append("cv2.imread failed")
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]
            result = face_mesh.process(image_rgb)
            if not result.multi_face_landmarks:
                rows.append(np.full((1404,), np.nan, dtype=np.float32))
                errors.append("MediaPipe FaceMesh returned no face")
                continue
            lm = result.multi_face_landmarks[0].landmark[:468]
            feat = np.empty((468, 3), dtype=np.float32)
            for i, p in enumerate(lm):
                feat[i, 0] = float(p.x) * float(w)
                feat[i, 1] = float(p.y) * float(h)
                feat[i, 2] = float(p.z) * float(w)
            rows.append(feat.reshape(-1))
            errors.append(None)
    finally:
        face_mesh.close()
    return np.stack(rows, axis=0), errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fresh CPU-side LibreFace preprocessing for the shared real 300-frame DirectML validation."
    )
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root = Path(args.benchmark_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir = out_dir / "aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = _find_frame_manifest(root)
    frames = pd.read_csv(manifest_path)
    required = {"benchmark_index", "image_path"}
    if frames.empty or not required.issubset(frames.columns):
        raise ValueError(f"Invalid frame manifest: {manifest_path}")

    from libreface import get_aligned_image

    alignment_rows: list[dict[str, object]] = []
    successful_paths: list[str] = []
    successful_indices: list[int] = []

    t0 = time.perf_counter()
    for row in frames.itertuples(index=False):
        idx = int(row.benchmark_index)
        image_path = str(Path(row.image_path))
        try:
            # Use a dedicated output dir so this is a fresh real-run timing rather than cached CPU reference reuse.
            aligned_path, headpose, landmark_data = get_aligned_image(
                image_path, temp_dir=str(aligned_dir), verbose=False
            )
            if not aligned_path:
                raise RuntimeError("get_aligned_image returned no aligned path")
            aligned_path = str(Path(aligned_path).resolve())
            successful_paths.append(aligned_path)
            successful_indices.append(idx)
            alignment_rows.append({
                "benchmark_index": idx,
                "image_path": image_path,
                "alignment_success": True,
                "alignment_error": None,
                "aligned_image_path": aligned_path,
                "headpose_json": json.dumps(headpose, ensure_ascii=False, default=str),
                "landmarks_json": json.dumps(landmark_data, ensure_ascii=False, default=str),
            })
        except Exception as exc:
            alignment_rows.append({
                "benchmark_index": idx,
                "image_path": image_path,
                "alignment_success": False,
                "alignment_error": f"{type(exc).__name__}: {exc}",
                "aligned_image_path": None,
                "headpose_json": None,
                "landmarks_json": None,
            })
    alignment_sec = time.perf_counter() - t0

    alignment = pd.DataFrame(alignment_rows)
    alignment_path = out_dir / "libreface_dml_alignment.parquet"
    alignment.to_parquet(alignment_path, index=False, engine="pyarrow", compression="zstd")

    gaze_t0 = time.perf_counter()
    if successful_paths:
        gaze_features, gaze_errors = _mediapipe_gaze_features(successful_paths)
    else:
        gaze_features = np.empty((0, 1404), dtype=np.float32)
        gaze_errors = []
    gaze_feature_sec = time.perf_counter() - gaze_t0

    gaze_path = out_dir / "libreface_dml_gaze_features.npy"
    np.save(gaze_path, gaze_features)
    gaze_index = pd.DataFrame({
        "benchmark_index": successful_indices,
        "aligned_image_path": successful_paths,
        "gaze_feature_success": [e is None for e in gaze_errors],
        "gaze_feature_error": gaze_errors,
    })
    gaze_index_path = out_dir / "libreface_dml_gaze_feature_index.parquet"
    gaze_index.to_parquet(gaze_index_path, index=False, engine="pyarrow", compression="zstd")

    try:
        libreface_version = importlib.metadata.version("libreface")
    except Exception:
        libreface_version = None
    try:
        mediapipe_version = importlib.metadata.version("mediapipe")
    except Exception:
        mediapipe_version = None

    summary = {
        "schema_version": "rgb-face-real300-libreface-prep-v0.1",
        "candidate": "libreface2",
        "benchmark_frame_manifest": str(manifest_path),
        "expected_frames": int(len(frames)),
        "aligned_frames": int(len(successful_paths)),
        "alignment_valid_fraction": len(successful_paths) / len(frames) if len(frames) else None,
        "gaze_feature_valid": int(sum(e is None for e in gaze_errors)),
        "timing_sec": {
            "fresh_alignment": alignment_sec,
            "mediapipe_gaze_feature_extraction": gaze_feature_sec,
            "cpu_preprocess_total": alignment_sec + gaze_feature_sec,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "libreface": libreface_version,
            "mediapipe": mediapipe_version,
        },
        "outputs": {
            "alignment": str(alignment_path),
            "gaze_features": str(gaze_path),
            "gaze_feature_index": str(gaze_index_path),
        },
        "notes": [
            "This stage intentionally reruns only the CPU-side preprocessing needed by the new DirectML real-input pipeline; it does not rerun the saved CPU model benchmark.",
            "Gaze features follow the saved LibreFace contract: refine_landmarks=True; first 468 landmarks flattened as x*w,y*h,z*w.",
        ],
    }
    summary_path = out_dir / "libreface_dml_prep_manifest.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
