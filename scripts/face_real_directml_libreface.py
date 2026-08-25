from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LIBREFACE_AU_INT = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
LIBREFACE_AU_DET = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
LIBREFACE_EXPRESSIONS = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _session(model: Path):
    import onnxruntime as ort

    if "DmlExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(f"DmlExecutionProvider unavailable: {ort.get_available_providers()}")
    so = ort.SessionOptions()
    so.enable_mem_pattern = False
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        str(model),
        sess_options=so,
        providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        enable_fallback=0,
    )
    sess.disable_fallback()
    return sess


def _imagenet_nchw(rgb_float01: np.ndarray) -> np.ndarray:
    x = (rgb_float01 - IMAGENET_MEAN[None, None, :]) / IMAGENET_STD[None, None, :]
    return np.transpose(x, (2, 0, 1)).astype(np.float32, copy=False)


def _pil_resize_shorter_center_crop(path: str, shorter: int = 256, crop: int = 224) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = float(shorter) / float(min(w, h))
        nw, nh = int(round(w * scale)), int(round(h * scale))
        im = im.resize((nw, nh), Image.Resampling.BILINEAR)
        left = max(0, (nw - crop) // 2)
        top = max(0, (nh - crop) // 2)
        im = im.crop((left, top, left + crop, top + crop))
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return _imagenet_nchw(arr)


def _pil_resize_224(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return _imagenet_nchw(arr)


def _run_batched(session: Any, input_name: str, values: np.ndarray, batch_size: int) -> list[np.ndarray]:
    output_chunks: list[list[np.ndarray]] = []
    for start in range(0, len(values), batch_size):
        batch = values[start:start + batch_size]
        out = session.run(None, {input_name: batch})
        output_chunks.append([np.asarray(v) for v in out])
    if not output_chunks:
        return []
    n_outputs = len(output_chunks[0])
    return [np.concatenate([chunk[i] for chunk in output_chunks], axis=0) for i in range(n_outputs)]


def run_libreface(args: argparse.Namespace) -> None:
    prep_dir = Path(args.prep_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prep_manifest = json.loads((prep_dir / "libreface_dml_prep_manifest.json").read_text(encoding="utf-8"))
    alignment = pd.read_parquet(prep_dir / "libreface_dml_alignment.parquet")
    gaze_index = pd.read_parquet(prep_dir / "libreface_dml_gaze_feature_index.parquet")
    gaze_features = np.load(prep_dir / "libreface_dml_gaze_features.npy")
    ok = alignment[alignment["alignment_success"].fillna(False).astype(bool)].copy()
    aligned_paths = ok["aligned_image_path"].astype(str).tolist()
    indices = ok["benchmark_index"].astype(int).tolist()
    if len(gaze_features) != len(gaze_index) or gaze_index["benchmark_index"].astype(int).tolist() != indices:
        raise RuntimeError("LibreFace prep alignment/gaze index mismatch")

    au_model = model_dir / "libreface2_au_joint.onnx"
    expr_model = model_dir / "libreface2_expression.onnx"
    gaze_model = model_dir / "libreface2_gaze_mlp.onnx"
    au_sess, expr_sess, gaze_sess = _session(au_model), _session(expr_model), _session(gaze_model)

    t0 = time.perf_counter()
    au_inputs = np.stack([_pil_resize_shorter_center_crop(p) for p in aligned_paths], axis=0)
    au_pre_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    au_int_prob, au_det_prob = _run_batched(
        au_sess, au_sess.get_inputs()[0].name, au_inputs, args.batch_size
    )
    au_infer_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    expr_inputs = np.stack([_pil_resize_224(p) for p in aligned_paths], axis=0)
    expr_pre_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    expr_scores = _run_batched(
        expr_sess, expr_sess.get_inputs()[0].name, expr_inputs, args.batch_size
    )[0]
    expr_infer_sec = time.perf_counter() - t0

    valid_gaze = np.isfinite(gaze_features).all(axis=1)
    gaze_out = np.full((len(indices), 2), np.nan, dtype=np.float32)
    t0 = time.perf_counter()
    if valid_gaze.any():
        gaze_out[valid_gaze] = _run_batched(
            gaze_sess,
            gaze_sess.get_inputs()[0].name,
            gaze_features[valid_gaze].astype(np.float32),
            args.batch_size,
        )[0]
    gaze_infer_sec = time.perf_counter() - t0

    # Retain native ONNX probabilities before scientific/convenience post-processing.
    au_int_raw = pd.DataFrame(au_int_prob, columns=[f"AU{x:02d}" for x in LIBREFACE_AU_INT])
    au_int_raw.insert(0, "benchmark_index", indices)
    au_det_raw = pd.DataFrame(au_det_prob, columns=[f"AU{x:02d}" for x in LIBREFACE_AU_DET])
    au_det_raw.insert(0, "benchmark_index", indices)

    au_int = pd.DataFrame(au_int_prob * 5.0, columns=[f"AU{x:02d}" for x in LIBREFACE_AU_INT])
    au_int.insert(0, "benchmark_index", indices)
    au_det = pd.DataFrame(
        (au_det_prob >= 0.5).astype(np.int8),
        columns=[f"AU{x:02d}" for x in LIBREFACE_AU_DET],
    )
    au_det.insert(0, "benchmark_index", indices)
    expr = pd.DataFrame(expr_scores, columns=[f"score__{x}" for x in LIBREFACE_EXPRESSIONS])
    expr.insert(
        0,
        "expression_label",
        [LIBREFACE_EXPRESSIONS[i] for i in np.argmax(expr_scores, axis=1)],
    )
    expr.insert(0, "benchmark_index", indices)
    gaze = pd.DataFrame(gaze_out, columns=["yaw", "pitch"])
    gaze.insert(0, "benchmark_index", indices)

    au_int_raw_path = out_dir / "libreface_dml_au_intensity_probability.parquet"
    au_det_raw_path = out_dir / "libreface_dml_au_detection_probability.parquet"
    au_int_path = out_dir / "libreface_dml_au_intensity.parquet"
    au_det_path = out_dir / "libreface_dml_au_detection.parquet"
    expr_path = out_dir / "libreface_dml_expression.parquet"
    gaze_path = out_dir / "libreface_dml_gaze.parquet"

    au_int_raw.to_parquet(au_int_raw_path, index=False, engine="pyarrow", compression="zstd")
    au_det_raw.to_parquet(au_det_raw_path, index=False, engine="pyarrow", compression="zstd")
    au_int.to_parquet(au_int_path, index=False, engine="pyarrow", compression="zstd")
    au_det.to_parquet(au_det_path, index=False, engine="pyarrow", compression="zstd")
    expr.to_parquet(expr_path, index=False, engine="pyarrow", compression="zstd")
    gaze.to_parquet(gaze_path, index=False, engine="pyarrow", compression="zstd")

    cpu_pre = float(prep_manifest["timing_sec"]["cpu_preprocess_total"])
    dml_stage = au_pre_sec + au_infer_sec + expr_pre_sec + expr_infer_sec + gaze_infer_sec
    total = cpu_pre + dml_stage
    manifest = {
        "schema_version": "rgb-face-real300-libreface-dml-v0.2",
        "candidate": "libreface2",
        "expected_frames": int(prep_manifest["expected_frames"]),
        "aligned_frames": len(indices),
        "batch_size": int(args.batch_size),
        "timing_sec": {
            "cpu_preprocess_from_fresh_prep": cpu_pre,
            "au_input_preprocess": au_pre_sec,
            "au_dml_inference": au_infer_sec,
            "expression_input_preprocess": expr_pre_sec,
            "expression_dml_inference": expr_infer_sec,
            "gaze_dml_inference": gaze_infer_sec,
            "dml_stage_total": dml_stage,
            "component_summed_end_to_end": total,
        },
        "input_frames_per_sec_component_summed": (
            prep_manifest["expected_frames"] / total if total > 0 else None
        ),
        "models": {
            p.stem: {"path": str(p), "sha256": _sha256(p)}
            for p in (au_model, expr_model, gaze_model)
        },
        "outputs": {
            "au_intensity_probability_raw": str(au_int_raw_path),
            "au_detection_probability_raw": str(au_det_raw_path),
            "au_intensity": str(au_int_path),
            "au_detection": str(au_det_path),
            "expression_scores_and_label": str(expr_path),
            "gaze": str(gaze_path),
            "fresh_alignment_headpose_landmarks": str(prep_dir / "libreface_dml_alignment.parquet"),
            "gaze_features_1404": str(prep_dir / "libreface_dml_gaze_features.npy"),
        },
        "notes": [
            "End-to-end is component-summed across the fresh CPU prep process and DirectML process; no saved CPU-model inference time is reused.",
            "Head pose/alignment landmarks remain in the fresh prep parquet; 1404 MediaPipe gaze features remain in the prep output.",
            "Both native AU probabilities and derived intensity/detection outputs are retained so later analyses do not require rerunning learned heads.",
        ],
    }
    (out_dir / "libreface_dml_real300_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Real 300-frame LibreFace DirectML runner")
    parser.add_argument("--prep-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    run_libreface(parser.parse_args())


if __name__ == "__main__":
    main()
