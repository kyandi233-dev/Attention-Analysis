from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from attention_pipeline.config import load_config
from attention_pipeline.rgb.paths import RGBOutputLayout

import face_formal_dryrun_directml_v02 as optimized
import face_real_directml_pyfeat as core


SCHEMA_VERSION = "rgb-face-formal-pyfeat-dml-v1.1"


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


def _postprocess_retina_batch(
    batch_df: pd.DataFrame,
    rgb_list: list[np.ndarray],
    loc: np.ndarray,
    conf: np.ndarray,
    landm: np.ndarray,
    priors: np.ndarray,
    *,
    face_threshold: float,
    nms_threshold: float,
    max_candidates: int,
    max_faces_per_frame: int,
) -> dict[str, Any]:
    """CPU RetinaFace postprocess with final threshold applied before decode/NMS.

    This is mathematically equivalent to applying the same final score threshold
    after NMS: lower-scoring candidates cannot suppress a higher-scoring candidate
    because greedy NMS processes boxes in descending score order.
    """
    started = time.perf_counter()
    loc_arr = np.asarray(loc)
    conf_arr = np.asarray(conf)
    landm_arr = np.asarray(landm)
    h, w = rgb_list[0].shape[:2]

    no_face_rows: list[dict[str, Any]] = []
    chips: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []
    candidate_count = 0
    nms_input_count = 0
    kept_face_count = 0

    for bi, row in enumerate(batch_df.itertuples(index=False)):
        scores = conf_arr[bi, :, 1]
        high_idx = np.flatnonzero(scores >= face_threshold)
        candidate_count += int(len(high_idx))

        if len(high_idx) == 0:
            no_face_rows.append(
                optimized._base_context(
                    row,
                    detected=False,
                    face_rank=0,
                    frame_h=h,
                    frame_w=w,
                )
            )
            continue

        if len(high_idx) > max_candidates:
            order = np.argsort(scores[high_idx])[::-1][:max_candidates]
            high_idx = high_idx[order]

        sc = scores[high_idx]
        nms_input_count += int(len(sc))
        selected_priors = priors[high_idx]
        boxes = core._decode_boxes(
            loc_arr[bi, high_idx, :][None, ...],
            selected_priors,
        )[0]
        lms5 = core._decode_landmarks(
            landm_arr[bi, high_idx, :][None, ...],
            selected_priors,
        )[0]
        boxes *= np.array([w, h, w, h], dtype=np.float32)[None, :]
        lms5 *= np.tile(np.array([w, h], dtype=np.float32), 5)[None, :]

        keep = core._nms(boxes, sc, nms_threshold)
        boxes, sc, lms5 = boxes[keep], sc[keep], lms5[keep]
        if max_faces_per_frame > 0:
            boxes = boxes[:max_faces_per_frame]
            sc = sc[:max_faces_per_frame]
            lms5 = lms5[:max_faces_per_frame]

        if len(sc) == 0:
            no_face_rows.append(
                optimized._base_context(
                    row,
                    detected=False,
                    face_rank=0,
                    frame_h=h,
                    frame_w=w,
                )
            )
            continue

        kept_face_count += int(len(sc))
        for rank, (box, score, lmk5) in enumerate(zip(boxes, sc, lms5)):
            chip, crop = core._square_reflect_crop(rgb_list[bi], box)
            chips.append(chip)
            metas.append(
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

    return {
        "no_face_rows": no_face_rows,
        "chips": chips,
        "metas": metas,
        "elapsed_sec": time.perf_counter() - started,
        "high_score_candidates": candidate_count,
        "nms_input_candidates": nms_input_count,
        "faces_after_nms": kept_face_count,
    }


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
    retinaface_batch = int(
        args.retinaface_batch
        if args.retinaface_batch is not None
        else face_cfg.get("retinaface_batch", 16)
    )
    multitask_batch = int(
        args.multitask_batch
        if args.multitask_batch is not None
        else face_cfg.get("multitask_batch", 32)
    )
    prefetch_batches = int(
        args.prefetch_batches
        if args.prefetch_batches is not None
        else face_cfg.get("prefetch_batches", 3)
    )
    postprocess_inflight = int(
        args.postprocess_inflight
        if args.postprocess_inflight is not None
        else face_cfg.get("postprocess_inflight", 2)
    )
    seek_threshold_frames = int(
        args.seek_threshold_frames
        if args.seek_threshold_frames is not None
        else face_cfg.get("seek_threshold_frames", 120)
    )
    face_threshold = float(face_cfg.get("detection_threshold", 0.5))
    nms_threshold = float(face_cfg.get("nms_threshold", 0.4))
    max_candidates = int(face_cfg.get("max_candidates_before_nms", 5000))
    max_faces_per_frame = int(face_cfg.get("max_faces_per_frame", 750))

    if min(retinaface_batch, multitask_batch, prefetch_batches, postprocess_inflight) <= 0:
        raise ValueError("Face batch/prefetch/postprocess values must be positive")
    if not 0.0 <= face_threshold <= 1.0:
        raise ValueError("face.detection_threshold must be in [0, 1]")

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
        "retinaface_high_score_candidates": 0,
        "retinaface_nms_input_candidates": 0,
        "retinaface_faces_after_nms": 0,
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

    def consume_postprocess(future: Future) -> None:
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

    q: queue.Queue = queue.Queue(maxsize=max(1, prefetch_batches))
    stop_event = threading.Event()
    reader = threading.Thread(
        target=optimized._reader_worker,
        args=(frames, source_video, retinaface_batch, q, stage, stop_event),
        kwargs={"seek_threshold_frames": seek_threshold_frames},
        daemon=True,
    )

    wall_started = time.perf_counter()
    reader.start()
    post_futures: deque[Future] = deque()
    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="retina-post") as executor:
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

                post_futures.append(
                    executor.submit(
                        _postprocess_retina_batch,
                        batch_df,
                        rgb_list,
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

                if len(post_futures) >= postprocess_inflight:
                    consume_postprocess(post_futures.popleft())

            while post_futures:
                consume_postprocess(post_futures.popleft())
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
        "prefetch_batches": prefetch_batches,
        "postprocess_inflight": postprocess_inflight,
        "seek_threshold_frames": seek_threshold_frames,
        "detection_threshold": face_threshold,
        "nms_threshold": nms_threshold,
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
            (
                f"Optimized batches: RetinaFace B{retinaface_batch}, "
                f"pooled multitask B{multitask_batch}."
            ),
            (
                "The frozen final RetinaFace score threshold is applied before bbox/landmark "
                "decode and NMS; this removes low-score CPU work without changing which "
                "higher-score boxes survive greedy NMS."
            ),
            (
                "RetinaFace DirectML inference overlaps with one CPU decode/NMS/crop "
                "postprocess worker; timings are overlapping stage totals, not additive wall time."
            ),
            "All supported scientific-core outputs and all detected faces are retained before downstream primary-face selection.",
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
    parser.add_argument("--prefetch-batches", type=int)
    parser.add_argument("--postprocess-inflight", type=int)
    parser.add_argument("--seek-threshold-frames", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.model_dir:
        parser.error("--model-dir is required unless ATTENTION_FACE_MODEL_DIR is set")
    run_formal(args)


if __name__ == "__main__":
    main()
