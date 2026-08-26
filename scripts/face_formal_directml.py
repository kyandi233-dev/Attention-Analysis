from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from attention_pipeline.config import load_config
from attention_pipeline.rgb.paths import RGBOutputLayout

import face_formal_dryrun_directml_v02 as optimized
import face_real_directml_pyfeat as core


SCHEMA_VERSION = "rgb-face-formal-pyfeat-dml-v1.0"


def _load_inputs(config_path: str, subject: str) -> tuple[Any, pd.DataFrame, Path, Path, Path]:
    config = load_config(config_path)
    layout = RGBOutputLayout.from_config(config)
    frames_csv = layout.subject_file(subject, "face_frames.csv")
    prepare_manifest = layout.subject_file(subject, "face_prepare_manifest.json")
    if not frames_csv.is_file():
        raise FileNotFoundError(
            f"Formal Face frame manifest not found: {frames_csv}. "
            f"Run scripts/face_formal_prepare.py first."
        )
    if not prepare_manifest.is_file():
        raise FileNotFoundError(prepare_manifest)

    frames = pd.read_csv(frames_csv)
    required = {
        "subject", "benchmark_index", "video_frame_position", "capture_frame_idx",
        "unix_ms", "target_unix_ms", "sample_error_ms", "phase",
    }
    if frames.empty or not required.issubset(frames.columns):
        missing = sorted(required - set(frames.columns))
        raise ValueError(f"Invalid formal Face frame manifest {frames_csv}; missing={missing}")
    if set(frames["subject"].astype(str).unique()) != {subject}:
        raise ValueError(f"Frame manifest contains a different subject: {frames_csv}")
    frames = frames.sort_values("benchmark_index").reset_index(drop=True)

    meta = json.loads(prepare_manifest.read_text(encoding="utf-8"))
    source_video = Path(str(meta["source_video"])).expanduser().resolve()
    if not source_video.is_file():
        raise FileNotFoundError(source_video)
    return config, frames, source_video, frames_csv, prepare_manifest


def _existing_complete(raw_path: Path, manifest_path: Path) -> dict[str, Any] | None:
    if raw_path.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if manifest.get("completion_status") == "complete":
            return manifest
    return None


def run_formal(args: argparse.Namespace) -> dict[str, Any]:
    config, frames, source_video, frames_csv, prepare_manifest = _load_inputs(
        args.config, args.subject
    )
    layout = RGBOutputLayout.from_config(config)
    out_path = layout.subject_file(args.subject, "face_raw.parquet")
    manifest_path = layout.subject_file(args.subject, "face_raw_manifest.json")

    if not args.force:
        existing = _existing_complete(out_path, manifest_path)
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
                "Partial formal Face output exists. Inspect it first or rerun with --force: "
                f"raw={out_path.exists()}, manifest={manifest_path.exists()}"
            )

    model_dir = Path(args.model_dir).expanduser().resolve()
    rf_model = model_dir / "pyfeat211_retinaface_r34.onnx"
    mt_model = model_dir / "pyfeat211_multitask_scientific_core.onnx"
    for model in (rf_model, mt_model):
        if not model.is_file():
            raise FileNotFoundError(model)

    face_cfg = config.section("face")
    retinaface_batch = int(args.retinaface_batch or face_cfg.get("retinaface_batch", 8))
    multitask_batch = int(args.multitask_batch or face_cfg.get("multitask_batch", 16))
    if retinaface_batch <= 0 or multitask_batch <= 0:
        raise ValueError("Face batch sizes must be positive")

    rf_sess, mt_sess = core._session(rf_model), core._session(mt_model)
    rf_input_name = rf_sess.get_inputs()[0].name
    mt_input_name = mt_sess.get_inputs()[0].name

    stage = {
        "decode_preprocess_cpu_sec": 0.0,
        "reader_thread_wall_sec": 0.0,
        "retinaface_dml_sec": 0.0,
        "decode_nms_crop_cpu_sec": 0.0,
        "multitask_preprocess_cpu_sec": 0.0,
        "multitask_dml_sec": 0.0,
        "postprocess_cpu_sec": 0.0,
        "parquet_write_sec": 0.0,
    }
    counters = {
        "retinaface_calls": 0,
        "multitask_full_batch_calls": 0,
        "multitask_partial_batch_calls": 0,
        "faces_sent_to_multitask": 0,
    }
    rows: list[dict[str, Any]] = []
    pending_chips: list[np.ndarray] = []
    pending_meta: list[dict[str, Any]] = []
    priors: np.ndarray | None = None

    def flush_multitask(n: int) -> None:
        if n <= 0:
            return
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

    q: queue.Queue = queue.Queue(maxsize=max(1, int(args.prefetch_batches)))
    stop_event = threading.Event()
    reader = threading.Thread(
        target=optimized._reader_worker,
        args=(frames, source_video, retinaface_batch, q, stage, stop_event),
        kwargs={"seek_threshold_frames": int(args.seek_threshold_frames)},
        daemon=True,
    )

    wall_started = time.perf_counter()
    reader.start()
    try:
        while True:
            payload = q.get()
            if payload is None:
                break
            if isinstance(payload, BaseException):
                raise payload
            batch_df, rgb_list, rf_in = payload
            h, w = rgb_list[0].shape[:2]
            if priors is None:
                priors = core._generate_priors(h, w)

            t0 = time.perf_counter()
            loc, conf, landm = rf_sess.run(None, {rf_input_name: rf_in})
            stage["retinaface_dml_sec"] += time.perf_counter() - t0
            counters["retinaface_calls"] += 1

            t0 = time.perf_counter()
            boxes_all = core._decode_boxes(np.asarray(loc), priors)
            landmarks_all = core._decode_landmarks(np.asarray(landm), priors)
            boxes_all *= np.array([w, h, w, h], dtype=np.float32)[None, None, :]
            landmarks_all *= np.tile(np.array([w, h], dtype=np.float32), 5)[None, None, :]

            for bi, row in enumerate(batch_df.itertuples(index=False)):
                scores = np.asarray(conf)[bi, :, 1]
                mask = scores > 0.02
                boxes, sc, lms5 = boxes_all[bi, mask], scores[mask], landmarks_all[bi, mask]
                if len(sc) > 5000:
                    order = np.argsort(sc)[::-1][:5000]
                    boxes, sc, lms5 = boxes[order], sc[order], lms5[order]
                keep = core._nms(boxes, sc, 0.4)
                boxes, sc, lms5 = boxes[keep], sc[keep], lms5[keep]
                keep2 = sc >= 0.5
                boxes, sc, lms5 = boxes[keep2][:750], sc[keep2][:750], lms5[keep2][:750]

                if len(sc) == 0:
                    rows.append(
                        optimized._base_context(
                            row,
                            detected=False,
                            face_rank=0,
                            frame_h=h,
                            frame_w=w,
                        )
                    )
                    continue

                for rank, (box, score, lmk5) in enumerate(zip(boxes, sc, lms5)):
                    chip, crop = core._square_reflect_crop(rgb_list[bi], box)
                    pending_chips.append(chip)
                    pending_meta.append(
                        {
                            "base": optimized._base_context(
                                row,
                                detected=True,
                                face_rank=rank,
                                frame_h=h,
                                frame_w=w,
                            ),
                            "score": float(score),
                            "crop": crop,
                            "rf_box": box.astype(np.float32),
                            "rf_landmarks5": lmk5.astype(np.float32),
                        }
                    )
            stage["decode_nms_crop_cpu_sec"] += time.perf_counter() - t0

            while len(pending_chips) >= multitask_batch:
                flush_multitask(multitask_batch)
    finally:
        stop_event.set()
        reader.join(timeout=10.0)
        if reader.is_alive():
            raise RuntimeError("Face prefetch reader did not stop cleanly")

    if pending_chips:
        flush_multitask(len(pending_chips))

    pipeline_wall_sec = time.perf_counter() - wall_started
    raw = pd.DataFrame(rows)
    if not raw.empty and {"benchmark_index", "face_rank"}.issubset(raw.columns):
        raw = raw.sort_values(["benchmark_index", "face_rank"]).reset_index(drop=True)

    t0 = time.perf_counter()
    raw.to_parquet(out_path, index=False, engine="pyarrow", compression="zstd")
    stage["parquet_write_sec"] = time.perf_counter() - t0
    total_with_write_sec = time.perf_counter() - wall_started

    detected = raw.get("detected", pd.Series(dtype=bool)).fillna(False).astype(bool)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "face-formal-directml",
        "output_mode": "formal",
        "completion_status": "complete",
        "subject": args.subject,
        "candidate": "pyfeat_detectorv2_scientific_core",
        "source_video": str(source_video),
        "source_frame_manifest": str(frames_csv),
        "source_prepare_manifest": str(prepare_manifest),
        "expected_input_frames": int(len(frames)),
        "unique_output_frames": int(raw["benchmark_index"].nunique()) if not raw.empty else 0,
        "output_rows": int(len(raw)),
        "detected_rows": int(detected.sum()),
        "retinaface_batch": retinaface_batch,
        "multitask_batch": multitask_batch,
        "prefetch_batches": int(args.prefetch_batches),
        "seek_threshold_frames": int(args.seek_threshold_frames),
        "requested_inference_fps": float(face_cfg.get("inference_fps", 15.0)),
        "execution_provider": str(face_cfg.get("execution_provider", "DmlExecutionProvider")),
        "config_path": str(config.path),
        "config_digest": config.digest,
        "timing_sec": {
            **stage,
            "pipeline_wall_before_parquet_write": pipeline_wall_sec,
            "total_wall_with_parquet_write": total_with_write_sec,
        },
        "input_frames_per_sec_pipeline": (
            len(frames) / pipeline_wall_sec if pipeline_wall_sec > 0 else None
        ),
        "input_frames_per_sec_including_write": (
            len(frames) / total_with_write_sec if total_with_write_sec > 0 else None
        ),
        "counters": counters,
        "models": {
            rf_model.stem: {"path": str(rf_model), "sha256": core._sha256(rf_model)},
            mt_model.stem: {"path": str(mt_model), "sha256": core._sha256(mt_model)},
        },
        "raw_output": str(out_path),
        "notes": [
            "Directly decodes timestamp-selected frames from the original AVI; no JPEG round-trip.",
            "Uses the already validated first-tier optimization: reader prefetch, RetinaFace B8 and pooled multitask B16.",
            "All supported scientific-core outputs and all detected faces are retained before primary-face selection.",
            "Temporal gaps remain represented in the frame manifest/QC columns and are not used as a subject exclusion rule here.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-span Py-Feat DirectML formal Face runner")
    parser.add_argument("--config", default="configs/rgb_analysis.yaml")
    parser.add_argument("--subject", required=True)
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("ATTENTION_FACE_MODEL_DIR"),
        help="Directory containing pyfeat211_retinaface_r34.onnx and pyfeat211_multitask_scientific_core.onnx",
    )
    parser.add_argument("--retinaface-batch", type=int)
    parser.add_argument("--multitask-batch", type=int)
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--seek-threshold-frames", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.model_dir:
        parser.error("--model-dir is required unless ATTENTION_FACE_MODEL_DIR is set")
    if args.prefetch_batches <= 0:
        parser.error("--prefetch-batches must be positive")
    run_formal(args)


if __name__ == "__main__":
    main()
