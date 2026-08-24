from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2

from benchmark_yolo_batch import FixedBatchYoloDml
from ritnet_runtime import RitnetRuntime
from run_pipeline import PACKAGE_ROOT, expand_crop_raw, valid_box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end AMD DirectML benchmark: fixed-batch YOLO + fixed-batch RITnet"
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--frames", type=int, default=1800)
    parser.add_argument("--device", default="0")
    parser.add_argument("--confidence", type=float, default=0.40)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument(
        "--yolo-model",
        default="models/nir-eye-yolo26n-best-b8.onnx",
    )
    parser.add_argument(
        "--ritnet-model",
        default="models/ritnet-b16-fp32.onnx",
    )
    parser.add_argument("--roi-width", type=int, default=320)
    parser.add_argument("--roi-height", type=int, default=160)
    parser.add_argument("--ritnet-input-width", type=int, default=640)
    parser.add_argument("--ritnet-input-height", type=int, default=400)
    parser.add_argument("--expand-horizontal", type=float, default=0.30)
    parser.add_argument("--expand-vertical", type=float, default=0.45)
    parser.add_argument(
        "--output",
        default="outputs/amd-directml/full-pipeline-yolo-b8-ritnet-b16.json",
    )
    return parser.parse_args()


def resolve_local(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PACKAGE_ROOT / path


def frame_status(count: int) -> str:
    if count == 0:
        return "yolo_missing"
    if count == 1:
        return "single_eye"
    if count > 2:
        return "extra_boxes"
    return "two_eyes"


def flush_ritnet(
    pending: list,
    ritnet: RitnetRuntime,
    timing: dict[str, float],
    eye_status_counts: Counter,
) -> None:
    if not pending:
        return
    results = ritnet.infer_batch([item["roi"] for item in pending])
    last = dict(ritnet.last_timing)
    timing["ritnet_preprocess_ms"] += float(last.get("preprocess_ms", 0.0))
    timing["ritnet_inference_ms"] += float(last.get("gpu_and_transfer_ms", 0.0))
    timing["ritnet_postprocess_ms"] += float(last.get("postprocess_ms", 0.0))
    timing["ritnet_total_ms"] += float(last.get("total_ms", 0.0))
    timing["ritnet_calls"] += 1
    timing["ritnet_valid_rois"] += len(pending)
    for item, result in zip(pending, results):
        if result.get("found"):
            eye_status_counts["roi_clipped" if item["clipped"] else "observed"] += 1
        else:
            eye_status_counts["ritnet_missing"] += 1
    pending.clear()


def main() -> int:
    args = parse_args()
    video = Path(args.video).expanduser().resolve()
    yolo_path = resolve_local(args.yolo_model)
    ritnet_path = resolve_local(args.ritnet_model)
    output = Path(args.output).expanduser()
    output = output if output.is_absolute() else PACKAGE_ROOT / output

    yolo = FixedBatchYoloDml(yolo_path, args.device)
    if yolo.batch_size != 8:
        raise ValueError(f"This benchmark expects YOLO batch 8, got {yolo.batch_size}")

    ritnet = RitnetRuntime(
        PACKAGE_ROOT,
        ritnet_path,
        (args.ritnet_input_width, args.ritnet_input_height),
        device=args.device,
        analysis_size=(args.roi_width, args.roi_height),
        precision="fp32",
    )
    if ritnet.FIXED_BATCH_SIZE != 16:
        raise ValueError(f"This benchmark expects RITnet batch 16, got {ritnet.FIXED_BATCH_SIZE}")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start_frame))

    timing = {
        "decode_ms": 0.0,
        "yolo_preprocess_ms": 0.0,
        "yolo_inference_ms": 0.0,
        "yolo_postprocess_ms": 0.0,
        "yolo_total_ms": 0.0,
        "roi_crop_ms": 0.0,
        "ritnet_preprocess_ms": 0.0,
        "ritnet_inference_ms": 0.0,
        "ritnet_postprocess_ms": 0.0,
        "ritnet_total_ms": 0.0,
        "ritnet_calls": 0,
        "ritnet_valid_rois": 0,
    }
    frame_status_counts: Counter = Counter()
    eye_status_counts: Counter = Counter()
    pending: list[dict] = []
    processed_frames = 0
    decoded_frames = 0

    wall_started = time.perf_counter()
    try:
        while processed_frames < args.frames:
            frames = []
            for _ in range(min(yolo.batch_size, args.frames - processed_frames)):
                decode_started = time.perf_counter()
                ok, frame = cap.read()
                timing["decode_ms"] += (time.perf_counter() - decode_started) * 1000.0
                if not ok or frame is None:
                    break
                frames.append(frame)
                decoded_frames += 1
            if not frames:
                break

            detections_batch, yolo_timing = yolo.infer_batch(
                frames,
                confidence=args.confidence,
                max_det=args.max_det,
            )
            timing["yolo_preprocess_ms"] += float(yolo_timing["preprocess_ms"])
            timing["yolo_inference_ms"] += float(yolo_timing["inference_ms"])
            timing["yolo_postprocess_ms"] += float(yolo_timing["postprocess_ms"])
            timing["yolo_total_ms"] += float(yolo_timing["total_ms"])

            for frame, raw in zip(frames, detections_batch):
                detections = [
                    item
                    for item in raw
                    if item.class_id == 0 and valid_box(item.box, frame.shape)
                ]
                detections.sort(key=lambda item: item.confidence, reverse=True)
                selected = sorted(
                    detections[:2],
                    key=lambda item: (item.box[0] + item.box[2]) / 2.0,
                )
                frame_status_counts[frame_status(len(detections))] += 1

                if pending and len(pending) + len(selected) > ritnet.FIXED_BATCH_SIZE:
                    flush_ritnet(pending, ritnet, timing, eye_status_counts)

                for detection in selected:
                    crop_started = time.perf_counter()
                    roi, _, clipped = expand_crop_raw(
                        frame,
                        detection.box,
                        args.expand_horizontal,
                        args.expand_vertical,
                    )
                    timing["roi_crop_ms"] += (time.perf_counter() - crop_started) * 1000.0
                    pending.append({"roi": roi, "clipped": clipped})

                if len(pending) >= ritnet.FIXED_BATCH_SIZE:
                    flush_ritnet(pending, ritnet, timing, eye_status_counts)
                processed_frames += 1

            if len(frames) < yolo.batch_size and processed_frames < args.frames:
                break
    finally:
        cap.release()

    flush_ritnet(pending, ritnet, timing, eye_status_counts)
    wall_sec = time.perf_counter() - wall_started

    per_frame = lambda value: (float(value) / processed_frames) if processed_frames else None
    per_roi = lambda value: (
        float(value) / timing["ritnet_valid_rois"] if timing["ritnet_valid_rois"] else None
    )

    result = {
        "video": str(video),
        "start_frame": int(args.start_frame),
        "requested_frames": int(args.frames),
        "processed_frames": processed_frames,
        "decoded_frames": decoded_frames,
        "device": args.device,
        "yolo_model": str(yolo_path.resolve()),
        "yolo_batch_size": yolo.batch_size,
        "ritnet_model": str(ritnet_path.resolve()),
        "ritnet_batch_size": ritnet.FIXED_BATCH_SIZE,
        "wall_sec": wall_sec,
        "processing_fps": processed_frames / wall_sec if wall_sec else None,
        "frame_status_counts": dict(frame_status_counts),
        "eye_status_counts": dict(eye_status_counts),
        "timing": {
            **timing,
            "mean_decode_ms_per_frame": per_frame(timing["decode_ms"]),
            "mean_yolo_total_ms_per_frame": per_frame(timing["yolo_total_ms"]),
            "mean_roi_crop_ms_per_frame": per_frame(timing["roi_crop_ms"]),
            "mean_ritnet_total_ms_per_roi": per_roi(timing["ritnet_total_ms"]),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), **result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
