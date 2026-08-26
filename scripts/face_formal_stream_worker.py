from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

import face_formal_directml as formal
import face_formal_dryrun_directml_v02 as optimized
import face_real_directml_pyfeat as core
from attention_pipeline.config import load_config
from attention_pipeline.rgb.paths import RGBOutputLayout


SCHEMA_VERSION = "rgb-face-formal-pyfeat-dml-stream-v1.0"


def _read_exact(stream, n: int) -> bytes:
    chunks: list[bytes] = []
    remaining = n
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"Unexpected EOF while reading {n} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _iter_frames(height: int, width: int):
    stream = sys.stdin.buffer
    frame_bytes = int(height) * int(width) * 3
    while True:
        header = stream.read(4)
        if not header:
            return
        if len(header) != 4:
            raise EOFError("Truncated metadata-length header")
        meta_len = struct.unpack("<I", header)[0]
        meta = json.loads(_read_exact(stream, meta_len).decode("utf-8"))
        raw = _read_exact(stream, frame_bytes)
        bgr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
        yield meta, bgr


def run_stream(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    layout = RGBOutputLayout.from_config(config)
    out_path = layout.subject_file(args.subject, "face_raw.parquet")
    manifest_path = layout.subject_file(args.subject, "face_raw_manifest.json")
    frames_csv = layout.subject_file(args.subject, "face_frames.csv")
    prepare_manifest = layout.subject_file(args.subject, "face_prepare_manifest.json")

    if not frames_csv.is_file() or not prepare_manifest.is_file():
        raise FileNotFoundError("Face frame/prepare manifest missing; run face_formal_prepare.py first")
    expected_frames = pd.read_csv(frames_csv)
    if expected_frames.empty:
        raise ValueError(f"Empty Face frame manifest: {frames_csv}")

    if not args.force:
        existing = formal._existing_complete(out_path, manifest_path)
        if existing is not None:
            print(json.dumps({
                "status": "skipped_complete",
                "subject": args.subject,
                "raw_output": str(out_path),
                "manifest": str(manifest_path),
            }, ensure_ascii=False, indent=2))
            return existing
        if out_path.exists() or manifest_path.exists():
            raise RuntimeError(
                "Partial formal Face output exists. Inspect first or rerun with --force."
            )

    prepare = json.loads(prepare_manifest.read_text(encoding="utf-8"))
    source_video = str(prepare["source_video"])
    model_dir = Path(args.model_dir).expanduser().resolve()
    rf_model = model_dir / "pyfeat211_retinaface_r34.onnx"
    mt_model = model_dir / "pyfeat211_multitask_scientific_core.onnx"
    for model in (rf_model, mt_model):
        if not model.is_file():
            raise FileNotFoundError(model)

    face_cfg = config.section("face")
    retinaface_batch = int(args.retinaface_batch or face_cfg.get("retinaface_batch", 16))
    multitask_batch = int(args.multitask_batch or face_cfg.get("multitask_batch", 32))
    postprocess_inflight = int(face_cfg.get("postprocess_inflight", 2))
    face_threshold = float(face_cfg.get("detection_threshold", 0.5))
    nms_threshold = float(face_cfg.get("nms_threshold", 0.4))
    max_candidates = int(face_cfg.get("max_candidates_before_nms", 5000))
    max_faces_per_frame = int(face_cfg.get("max_faces_per_frame", 750))

    rf_sess, mt_sess = core._session(rf_model), core._session(mt_model)
    rf_input_name = rf_sess.get_inputs()[0].name
    mt_input_name = mt_sess.get_inputs()[0].name

    stage = {
        "stream_read_convert_cpu_sec": 0.0,
        "retinaface_dml_sec": 0.0,
        "decode_nms_crop_cpu_sec": 0.0,
        "multitask_preprocess_cpu_sec": 0.0,
        "multitask_dml_sec": 0.0,
        "postprocess_cpu_sec": 0.0,
        "parquet_write_sec": 0.0,
    }
    counters = {
        "retinaface_calls": 0,
        "retinaface_high_score_candidates": 0,
        "retinaface_nms_input_candidates": 0,
        "retinaface_faces_after_nms": 0,
        "multitask_full_batch_calls": 0,
        "multitask_partial_batch_calls": 0,
        "faces_sent_to_multitask": 0,
        "stream_frames_received": 0,
    }

    rows: list[dict[str, Any]] = []
    pending_chips: list[np.ndarray] = []
    pending_meta: list[dict[str, Any]] = []
    priors: np.ndarray | None = None

    def flush_multitask(n: int) -> None:
        chips = pending_chips[:n]
        metas = pending_meta[:n]
        t0 = time.perf_counter()
        mt_inputs = np.stack([core._preprocess_pyfeat_chip(c) for c in chips], axis=0)
        stage["multitask_preprocess_cpu_sec"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        outputs = [np.asarray(v) for v in mt_sess.run(None, {mt_input_name: mt_inputs})]
        stage["multitask_dml_sec"] += time.perf_counter() - t0
        counters["faces_sent_to_multitask"] += n
        if n == multitask_batch:
            counters["multitask_full_batch_calls"] += 1
        else:
            counters["multitask_partial_batch_calls"] += 1
        t0 = time.perf_counter()
        for i, meta in enumerate(metas):
            rows.append(optimized._emit_detected(meta, outputs, i))
        stage["postprocess_cpu_sec"] += time.perf_counter() - t0
        del pending_chips[:n]
        del pending_meta[:n]

    def consume(future: Future) -> None:
        result = future.result()
        stage["decode_nms_crop_cpu_sec"] += float(result["elapsed_sec"])
        counters["retinaface_high_score_candidates"] += int(result["high_score_candidates"])
        counters["retinaface_nms_input_candidates"] += int(result["nms_input_candidates"])
        counters["retinaface_faces_after_nms"] += int(result["faces_after_nms"])
        rows.extend(result["no_face_rows"])
        pending_chips.extend(result["chips"])
        pending_meta.extend(result["metas"])
        while len(pending_chips) >= multitask_batch:
            flush_multitask(multitask_batch)

    wall_started = time.perf_counter()
    batch_meta: list[dict[str, Any]] = []
    batch_rgb: list[np.ndarray] = []
    futures: deque[Future] = deque()

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="retina-post") as executor:
        for meta, bgr in _iter_frames(args.height, args.width):
            t0 = time.perf_counter()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            batch_meta.append(meta)
            batch_rgb.append(rgb)
            counters["stream_frames_received"] += 1
            stage["stream_read_convert_cpu_sec"] += time.perf_counter() - t0

            if len(batch_rgb) < retinaface_batch:
                continue

            batch_df = pd.DataFrame(batch_meta)
            t0 = time.perf_counter()
            rf_in = optimized._rf_tensor(batch_rgb)
            stage["stream_read_convert_cpu_sec"] += time.perf_counter() - t0
            if priors is None:
                h, w = batch_rgb[0].shape[:2]
                priors = core._generate_priors(h, w)

            t0 = time.perf_counter()
            loc, conf, landm = rf_sess.run(None, {rf_input_name: rf_in})
            stage["retinaface_dml_sec"] += time.perf_counter() - t0
            counters["retinaface_calls"] += 1
            futures.append(
                executor.submit(
                    formal._postprocess_retina_batch,
                    batch_df,
                    batch_rgb,
                    np.asarray(loc),
                    np.asarray(conf),
                    np.asarray(landm),
                    priors,
                    face_threshold=face_threshold,
                    nms_threshold=nms_threshold,
                    max_candidates=max_candidates,
                    max_faces_per_frame=max_faces_per_frame,
                )
            )
            batch_meta, batch_rgb = [], []
            if len(futures) >= postprocess_inflight:
                consume(futures.popleft())

        if batch_rgb:
            batch_df = pd.DataFrame(batch_meta)
            t0 = time.perf_counter()
            rf_in = optimized._rf_tensor(batch_rgb)
            stage["stream_read_convert_cpu_sec"] += time.perf_counter() - t0
            if priors is None:
                h, w = batch_rgb[0].shape[:2]
                priors = core._generate_priors(h, w)
            t0 = time.perf_counter()
            loc, conf, landm = rf_sess.run(None, {rf_input_name: rf_in})
            stage["retinaface_dml_sec"] += time.perf_counter() - t0
            counters["retinaface_calls"] += 1
            futures.append(
                executor.submit(
                    formal._postprocess_retina_batch,
                    batch_df,
                    batch_rgb,
                    np.asarray(loc),
                    np.asarray(conf),
                    np.asarray(landm),
                    priors,
                    face_threshold=face_threshold,
                    nms_threshold=nms_threshold,
                    max_candidates=max_candidates,
                    max_faces_per_frame=max_faces_per_frame,
                )
            )

        while futures:
            consume(futures.popleft())

    if pending_chips:
        flush_multitask(len(pending_chips))

    received = int(counters["stream_frames_received"])
    if received != len(expected_frames):
        raise RuntimeError(
            f"Shared-decode Face stream incomplete: received={received}, expected={len(expected_frames)}"
        )

    pipeline_wall_sec = time.perf_counter() - wall_started
    raw = pd.DataFrame(rows)
    if not raw.empty and {"benchmark_index", "face_rank"}.issubset(raw.columns):
        raw = raw.sort_values(["benchmark_index", "face_rank"]).reset_index(drop=True)

    t0 = time.perf_counter()
    raw.to_parquet(out_path, index=False, engine="pyarrow", compression="zstd")
    stage["parquet_write_sec"] = time.perf_counter() - t0
    total_wall = time.perf_counter() - wall_started
    detected = raw.get("detected", pd.Series(dtype=bool)).fillna(False).astype(bool)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "face-formal-directml",
        "output_mode": "formal",
        "completion_status": "complete",
        "subject": args.subject,
        "candidate": "pyfeat_detectorv2_scientific_core",
        "source_video": source_video,
        "source_frame_manifest": str(frames_csv),
        "source_prepare_manifest": str(prepare_manifest),
        "input_mode": "shared_decode_raw_bgr_pipe",
        "expected_input_frames": int(len(expected_frames)),
        "unique_output_frames": int(raw["benchmark_index"].nunique()) if not raw.empty else 0,
        "output_rows": int(len(raw)),
        "detected_rows": int(detected.sum()),
        "retinaface_batch": retinaface_batch,
        "multitask_batch": multitask_batch,
        "postprocess_inflight": postprocess_inflight,
        "detection_threshold": face_threshold,
        "nms_threshold": nms_threshold,
        "requested_inference_fps": float(face_cfg.get("inference_fps", 15.0)),
        "execution_provider": str(face_cfg.get("execution_provider", "DmlExecutionProvider")),
        "config_path": str(config.path),
        "config_digest": config.digest,
        "timing_sec": {
            **stage,
            "pipeline_wall_before_parquet_write": pipeline_wall_sec,
            "total_wall_with_parquet_write": total_wall,
        },
        "input_frames_per_sec_pipeline": len(expected_frames) / pipeline_wall_sec,
        "input_frames_per_sec_including_write": len(expected_frames) / total_wall,
        "counters": counters,
        "models": {
            rf_model.stem: {"path": str(rf_model), "sha256": core._sha256(rf_model)},
            mt_model.stem: {"path": str(mt_model), "sha256": core._sha256(mt_model)},
        },
        "raw_output": str(out_path),
        "notes": [
            "Frames are decoded once by the RGB shared-decoder parent and transferred losslessly as raw BGR bytes; there is no JPEG round-trip.",
            "RetinaFace final threshold is applied before bbox/landmark decode and NMS.",
            "RetinaFace DML overlaps with one CPU postprocess worker.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Py-Feat DirectML worker for shared-decoded raw BGR frames")
    parser.add_argument("--config", default="configs/rgb_analysis.yaml")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--model-dir", default=os.environ.get("ATTENTION_FACE_MODEL_DIR"))
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--retinaface-batch", type=int)
    parser.add_argument("--multitask-batch", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.model_dir:
        parser.error("--model-dir is required unless ATTENTION_FACE_MODEL_DIR is set")
    run_stream(args)


if __name__ == "__main__":
    main()
