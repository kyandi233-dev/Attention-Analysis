from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

import face_real_directml_pyfeat as core


SCHEMA_VERSION = "rgb-face-formal-dryrun-pyfeat-dml-v0.3"
_RF_MEAN = np.array([123.0, 117.0, 104.0], dtype=np.float32)


def _sample_inputs(sample_dir: Path) -> tuple[pd.DataFrame, Path, Path, Path]:
    csvs = sorted(sample_dir.glob("*_face-dryrun_frames.csv"))
    jsons = sorted(sample_dir.glob("*_face-dryrun_manifest.json"))
    if len(csvs) != 1 or len(jsons) != 1:
        raise RuntimeError(
            f"Expected one dry-run CSV and one dry-run JSON in {sample_dir}; "
            f"found csv={len(csvs)}, json={len(jsons)}"
        )
    frames = pd.read_csv(csvs[0]).sort_values("benchmark_index").reset_index(drop=True)
    required = {"benchmark_index", "video_frame_position"}
    if frames.empty or not required.issubset(frames.columns):
        raise ValueError(f"Invalid dry-run manifest: {csvs[0]}")
    meta = json.loads(jsons[0].read_text(encoding="utf-8"))
    source_video = Path(str(meta["source_video"])).expanduser().resolve()
    if not source_video.is_file():
        raise FileNotFoundError(source_video)
    return frames, source_video, csvs[0], jsons[0]


def _rf_tensor(rgb_list: list[np.ndarray]) -> np.ndarray:
    if not rgb_list:
        raise ValueError("Empty RetinaFace batch")
    h, w = rgb_list[0].shape[:2]
    if any(im.shape[:2] != (h, w) for im in rgb_list):
        raise RuntimeError("Mixed frame shapes are not supported")
    hwc = np.stack(rgb_list, axis=0).astype(np.float32, copy=False)
    hwc -= _RF_MEAN[None, None, None, :]
    return np.ascontiguousarray(np.transpose(hwc, (0, 3, 1, 2)), dtype=np.float32)


def _queue_put_interruptible(out_queue: queue.Queue, item: Any, stop_event: threading.Event) -> bool:
    while not stop_event.is_set():
        try:
            out_queue.put(item, timeout=0.2)
            return True
        except queue.Full:
            continue
    return False


def _reader_worker(
    frames: pd.DataFrame,
    source_video: Path,
    batch_size: int,
    out_queue: queue.Queue,
    timing: dict[str, float],
    stop_event: threading.Event,
    *,
    seek_threshold_frames: int = 120,
) -> None:
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        _queue_put_interruptible(
            out_queue,
            RuntimeError(f"Cannot open RGB video: {source_video}"),
            stop_event,
        )
        return
    reader_started = time.perf_counter()
    last_target: int | None = None
    try:
        for start in range(0, len(frames), batch_size):
            if stop_event.is_set():
                break
            batch_df = frames.iloc[start:start + batch_size].copy()
            t0 = time.perf_counter()
            rgb_list: list[np.ndarray] = []
            for row in batch_df.itertuples(index=False):
                target = int(row.video_frame_position)
                if (
                    last_target is None
                    or target <= last_target
                    or target - last_target > seek_threshold_frames
                ):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, float(target))
                else:
                    for _ in range(target - last_target - 1):
                        if not cap.grab():
                            raise RuntimeError(
                                f"Failed to advance RGB video before frame {target}"
                            )
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    raise RuntimeError(f"Failed to decode RGB frame {target}")
                rgb_list.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                last_target = target
            rf_in = _rf_tensor(rgb_list)
            timing["decode_preprocess_cpu_sec"] += time.perf_counter() - t0
            if not _queue_put_interruptible(out_queue, (batch_df, rgb_list, rf_in), stop_event):
                break
    except BaseException as exc:
        _queue_put_interruptible(out_queue, exc, stop_event)
    finally:
        cap.release()
        timing["reader_thread_wall_sec"] = time.perf_counter() - reader_started
        _queue_put_interruptible(out_queue, None, stop_event)


def _base_context(row: Any, *, detected: bool, face_rank: int, frame_h: int, frame_w: int) -> dict[str, Any]:
    raw = row._asdict() if hasattr(row, "_asdict") else dict(row)
    base: dict[str, Any] = {}
    for key, value in raw.items():
        if key == "image_path":
            continue
        if isinstance(value, np.generic):
            value = value.item()
        base[key] = value
    base.update(
        {
            "face_rank": int(face_rank),
            "detected": bool(detected),
            "FrameHeight": int(frame_h),
            "FrameWidth": int(frame_w),
        }
    )
    return base


def _emit_detected(meta: dict[str, Any], outputs: list[np.ndarray], out_index: int) -> dict[str, Any]:
    base = dict(meta["base"])
    au, emo, va, gaze_raw, pose_raw, mesh, bs = [o[out_index] for o in outputs]
    ox, oy, side = meta["crop"]
    rf_box = meta["rf_box"]
    lmk5 = meta["rf_landmarks5"]
    base.update(
        {
            "rf_bbox_x1": float(rf_box[0]),
            "rf_bbox_y1": float(rf_box[1]),
            "rf_bbox_x2": float(rf_box[2]),
            "rf_bbox_y2": float(rf_box[3]),
            "FaceRectX": float(ox),
            "FaceRectY": float(oy),
            "FaceRectWidth": float(side),
            "FaceRectHeight": float(side),
            "FaceScore": float(meta["score"]),
            "Pitch": -float(pose_raw[0]),
            "Roll": -float(pose_raw[2]),
            "Yaw": float(pose_raw[1]),
            "X": float(pose_raw[3]),
            "Y": float(pose_raw[4]),
            "Z": float(pose_raw[5]),
            "gaze_pitch": -float(gaze_raw[1]),
            "gaze_yaw": float(gaze_raw[0]),
            "gaze_raw_yaw": float(gaze_raw[0]),
            "gaze_raw_pitch": float(gaze_raw[1]),
            "pose_raw_0": float(pose_raw[0]),
            "pose_raw_1": float(pose_raw[1]),
            "pose_raw_2": float(pose_raw[2]),
            "pose_raw_3": float(pose_raw[3]),
            "pose_raw_4": float(pose_raw[4]),
            "pose_raw_5": float(pose_raw[5]),
        }
    )
    for j in range(5):
        base[f"rf_landmark5_x_{j}"] = float(lmk5[2 * j])
        base[f"rf_landmark5_y_{j}"] = float(lmk5[2 * j + 1])
    base["gaze_angle"] = float(
        np.arccos(
            np.clip(
                np.cos(base["gaze_pitch"]) * np.cos(base["gaze_yaw"]),
                -1.0,
                1.0,
            )
        )
    )
    for name, val in zip(core.PYFEAT_AUS, au):
        base[name] = float(val)
    for name, val in zip(core.PYFEAT_EMOTIONS, emo):
        base[name] = float(val)
    base["valence"], base["arousal"] = float(va[0]), float(va[1])

    for j in range(478):
        base[f"mesh_norm_x_{j}"] = float(mesh[j, 0])
        base[f"mesh_norm_y_{j}"] = float(mesh[j, 1])
        base[f"mesh_norm_z_{j}"] = float(mesh[j, 2])
        base[f"mesh_x_{j}"] = float(mesh[j, 0] * side + ox)
        base[f"mesh_y_{j}"] = float(mesh[j, 1] * side + oy)
        base[f"mesh_z_{j}"] = float(mesh[j, 2])
    for dlib_idx, mp_idx in enumerate(core.DLIB68_FROM_MP478):
        base[f"x_{dlib_idx}"] = float(mesh[mp_idx, 0] * side + ox)
        base[f"y_{dlib_idx}"] = float(mesh[mp_idx, 1] * side + oy)
    for name, val in zip(core.BLENDSHAPES, bs):
        base[name] = float(val)
    return base


def run_optimized(args: argparse.Namespace) -> dict[str, Any]:
    sample_dir = Path(args.sample_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, source_video, frame_manifest, sample_manifest = _sample_inputs(sample_dir)
    rf_model = model_dir / "pyfeat211_retinaface_r34.onnx"
    mt_model = model_dir / "pyfeat211_multitask_scientific_core.onnx"
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
        outputs = mt_sess.run(None, {mt_input_name: mt_inputs})
        outputs = [np.asarray(v) for v in outputs]
        stage["multitask_dml_sec"] += time.perf_counter() - t0
        counters["faces_sent_to_multitask"] += n
        if n == int(args.multitask_batch):
            counters["multitask_full_batch_calls"] += 1
        else:
            counters["multitask_partial_batch_calls"] += 1

        t0 = time.perf_counter()
        for i, meta in enumerate(metas):
            rows.append(_emit_detected(meta, outputs, i))
        stage["postprocess_cpu_sec"] += time.perf_counter() - t0
        del pending_chips[:n]
        del pending_meta[:n]

    q: queue.Queue = queue.Queue(maxsize=max(1, int(args.prefetch_batches)))
    stop_event = threading.Event()
    reader = threading.Thread(
        target=_reader_worker,
        args=(frames, source_video, int(args.retinaface_batch), q, stage, stop_event),
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
                        _base_context(
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
                            "base": _base_context(
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

            while len(pending_chips) >= int(args.multitask_batch):
                flush_multitask(int(args.multitask_batch))
    finally:
        stop_event.set()
        reader.join(timeout=5.0)
        if reader.is_alive():
            raise RuntimeError("Face prefetch reader did not stop cleanly")

    if pending_chips:
        flush_multitask(len(pending_chips))

    pipeline_wall_sec = time.perf_counter() - wall_started
    raw = pd.DataFrame(rows)
    if not raw.empty and {"benchmark_index", "face_rank"}.issubset(raw.columns):
        raw = raw.sort_values(["benchmark_index", "face_rank"]).reset_index(drop=True)

    out_path = out_dir / "pyfeat_dml_raw.parquet"
    t0 = time.perf_counter()
    raw.to_parquet(out_path, index=False, engine="pyarrow", compression="zstd")
    stage["parquet_write_sec"] = time.perf_counter() - t0
    total_with_write_sec = time.perf_counter() - wall_started

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "face-formal-dryrun-directml-optimized-v02",
        "candidate": "pyfeat_detectorv2_scientific_core",
        "optimization_scope": [
            "direct AVI decode; no JPEG round-trip",
            "reader/preprocess prefetch overlaps with main DirectML loop",
            "RetinaFace remains batch 8",
            "face chips are pooled across RetinaFace batches so multitask actually runs full batch 16 when available",
            "same RetinaFace thresholds/NMS/crop geometry/multitask model/postprocess as validated real-300 runner",
        ],
        "source_video": str(source_video),
        "dryrun_frame_manifest": str(frame_manifest),
        "dryrun_sample_manifest": str(sample_manifest),
        "expected_frames": int(len(frames)),
        "output_rows": int(len(raw)),
        "detected_rows": int(raw.get("detected", pd.Series(dtype=bool)).fillna(False).sum()),
        "retinaface_batch": int(args.retinaface_batch),
        "multitask_batch": int(args.multitask_batch),
        "prefetch_batches": int(args.prefetch_batches),
        "seek_threshold_frames": int(args.seek_threshold_frames),
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
            "Reader timing overlaps with DirectML/main-thread work and therefore must not be summed with other stage times to reconstruct wall time.",
            "This is a first-tier engineering optimization only: no detector cadence reduction, no tracking-based detector skipping, no input-resolution change.",
            "Direct AVI input avoids the JPEG-quality-95 test round-trip; small numerical drift versus JPEG dry-run output is possible and must be checked before promotion.",
            "Multi-face rows and all scientific-core outputs remain retained.",
        ],
    }
    manifest_path = out_dir / "pyfeat_dml_formal_dryrun_v02_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimized first-tier Py-Feat DirectML formal Face dry-run"
    )
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--retinaface-batch", type=int, default=8)
    parser.add_argument("--multitask-batch", type=int, default=16)
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--seek-threshold-frames", type=int, default=120)
    args = parser.parse_args()
    if args.retinaface_batch <= 0 or args.multitask_batch <= 0:
        parser.error("batch sizes must be positive")
    if args.prefetch_batches <= 0:
        parser.error("--prefetch-batches must be positive")
    run_optimized(args)


if __name__ == "__main__":
    main()
