from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from directml_runtime import create_directml_session


@dataclass
class Detection:
    box: tuple[float, float, float, float]
    confidence: float
    class_id: int


class FixedBatchYoloDml:
    def __init__(self, model_path: Path, device: str = "0") -> None:
        self.model_path = Path(model_path)
        self.session = create_directml_session(self.model_path, device)
        inp = self.session.get_inputs()[0]
        out = self.session.get_outputs()[0]
        self.input_name = inp.name
        self.output_name = out.name
        self.input_shape = tuple(inp.shape)
        self.output_shape = tuple(out.shape)
        if len(self.input_shape) != 4 or self.input_shape[1:] != (3, 640, 640):
            raise ValueError(f"Expected fixed [B,3,640,640] input, got {self.input_shape}")
        if not isinstance(self.input_shape[0], int) or self.input_shape[0] <= 0:
            raise ValueError(f"Expected fixed batch dimension, got {self.input_shape}")
        self.batch_size = int(self.input_shape[0])
        if len(self.output_shape) != 3 or self.output_shape[0] != self.batch_size or self.output_shape[2] != 6:
            raise ValueError(f"Expected fixed [B,N,6] output, got {self.output_shape}")

    @staticmethod
    def _letterbox(frame: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
        source_h, source_w = frame.shape[:2]
        target = 640
        scale = min(target / source_w, target / source_h)
        resized_w = int(round(source_w * scale))
        resized_h = int(round(source_h * scale))
        resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        pad_w = (target - resized_w) / 2.0
        pad_h = (target - resized_h) / 2.0
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)
        tensor /= np.float32(255.0)
        return tensor, scale, (float(left), float(top))

    @staticmethod
    def _postprocess_one(
        rows: np.ndarray,
        frame_shape: tuple[int, int, int],
        scale: float,
        pad: tuple[float, float],
        confidence: float,
        max_det: int,
    ) -> list[Detection]:
        rows = np.asarray(rows, dtype=np.float32)
        rows = rows[np.isfinite(rows).all(axis=1)]
        rows = rows[rows[:, 4] >= np.float32(confidence)]
        if not len(rows):
            return []
        rows = rows[np.argsort(-rows[:, 4], kind="stable")[: int(max_det)]]
        frame_h, frame_w = frame_shape[:2]
        pad_x, pad_y = pad
        result: list[Detection] = []
        for x1, y1, x2, y2, score, class_id in rows:
            box = np.array([x1 - pad_x, y1 - pad_y, x2 - pad_x, y2 - pad_y], dtype=np.float32)
            box /= np.float32(scale)
            box[[0, 2]] = np.clip(box[[0, 2]], 0, frame_w)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, frame_h)
            result.append(
                Detection(
                    tuple(float(v) for v in box),
                    float(score),
                    int(class_id),
                )
            )
        return result

    def infer_batch(
        self,
        frames: list[np.ndarray],
        *,
        confidence: float = 0.40,
        max_det: int = 20,
    ) -> tuple[list[list[Detection]], dict[str, float]]:
        if not frames or len(frames) > self.batch_size:
            raise ValueError(f"Need 1..{self.batch_size} frames, got {len(frames)}")

        preprocess_started = time.perf_counter()
        tensors: list[np.ndarray] = []
        metadata: list[tuple[tuple[int, int, int], float, tuple[float, float]]] = []
        for frame in frames:
            tensor, scale, pad = self._letterbox(frame)
            tensors.append(tensor)
            metadata.append((frame.shape, scale, pad))
        valid = len(tensors)
        while len(tensors) < self.batch_size:
            tensors.append(tensors[-1].copy())
        batch_tensor = np.ascontiguousarray(np.stack(tensors, axis=0), dtype=np.float32)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0

        inference_started = time.perf_counter()
        output = self.session.run([self.output_name], {self.input_name: batch_tensor})[0]
        inference_ms = (time.perf_counter() - inference_started) * 1000.0

        post_started = time.perf_counter()
        results: list[list[Detection]] = []
        for index in range(valid):
            shape, scale, pad = metadata[index]
            results.append(
                self._postprocess_one(
                    output[index],
                    shape,
                    scale,
                    pad,
                    confidence,
                    max_det,
                )
            )
        postprocess_ms = (time.perf_counter() - post_started) * 1000.0
        return results, {
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "postprocess_ms": postprocess_ms,
            "total_ms": preprocess_ms + inference_ms + postprocess_ms,
            "valid_batch": valid,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark fixed-batch YOLO26n ONNX on DirectML")
    parser.add_argument("--video", required=True)
    parser.add_argument("--models", required=True, nargs="+", help="One or more ONNX model paths")
    parser.add_argument("--device", default="0")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=1800)
    parser.add_argument("--confidence", type=float, default=0.40)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--output", default="outputs/amd-directml/yolo-batch-benchmark.json")
    return parser.parse_args()


def load_frames(video: Path, start_frame: int, count: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    result: list[np.ndarray] = []
    try:
        for _ in range(count):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            result.append(frame)
    finally:
        cap.release()
    if not result:
        raise RuntimeError("No frames decoded")
    return result


def selected_pair(detections: list[Detection]) -> list[Detection]:
    chosen = [item for item in detections if item.class_id == 0][:2]
    return sorted(chosen, key=lambda item: (item.box[0] + item.box[2]) / 2.0)


def compare_to_reference(
    reference: list[list[Detection]],
    candidate: list[list[Detection]],
) -> dict[str, Any]:
    count_agree = 0
    compared_boxes = 0
    coord_abs: list[float] = []
    conf_abs: list[float] = []
    for ref_frame, cand_frame in zip(reference, candidate):
        ref_pair = selected_pair(ref_frame)
        cand_pair = selected_pair(cand_frame)
        if len(ref_pair) == len(cand_pair):
            count_agree += 1
        for ref_det, cand_det in zip(ref_pair, cand_pair):
            compared_boxes += 1
            coord_abs.extend(abs(a - b) for a, b in zip(ref_det.box, cand_det.box))
            conf_abs.append(abs(ref_det.confidence - cand_det.confidence))
    n = max(1, min(len(reference), len(candidate)))
    return {
        "frame_selected_count_agreement": count_agree / n,
        "compared_boxes": compared_boxes,
        "coord_mae_px": float(np.mean(coord_abs)) if coord_abs else None,
        "coord_max_abs_px": float(np.max(coord_abs)) if coord_abs else None,
        "confidence_mae": float(np.mean(conf_abs)) if conf_abs else None,
        "confidence_max_abs": float(np.max(conf_abs)) if conf_abs else None,
    }


def run_model(
    runtime: FixedBatchYoloDml,
    frames: list[np.ndarray],
    args: argparse.Namespace,
) -> tuple[list[list[Detection]], dict[str, Any]]:
    warmup = frames[: min(len(frames), runtime.batch_size)]
    for _ in range(max(0, args.warmup_batches)):
        runtime.infer_batch(warmup, confidence=args.confidence, max_det=args.max_det)

    all_results: list[list[Detection]] = []
    timing_total = {"preprocess_ms": 0.0, "inference_ms": 0.0, "postprocess_ms": 0.0, "total_ms": 0.0}
    wall_started = time.perf_counter()
    for start in range(0, len(frames), runtime.batch_size):
        chunk = frames[start : start + runtime.batch_size]
        results, timing = runtime.infer_batch(
            chunk,
            confidence=args.confidence,
            max_det=args.max_det,
        )
        all_results.extend(results)
        for key in timing_total:
            timing_total[key] += float(timing[key])
    wall_sec = time.perf_counter() - wall_started
    return all_results, {
        "batch_size": runtime.batch_size,
        "frames": len(frames),
        "wall_sec": wall_sec,
        "fps": len(frames) / wall_sec if wall_sec else None,
        **timing_total,
        "mean_preprocess_ms_per_frame": timing_total["preprocess_ms"] / len(frames),
        "mean_inference_ms_per_frame": timing_total["inference_ms"] / len(frames),
        "mean_postprocess_ms_per_frame": timing_total["postprocess_ms"] / len(frames),
        "mean_total_ms_per_frame": timing_total["total_ms"] / len(frames),
    }


def main() -> int:
    args = parse_args()
    video = Path(args.video).expanduser().resolve()
    frames = load_frames(video, args.start_frame, args.frames)

    result: dict[str, Any] = {
        "video": str(video),
        "start_frame": args.start_frame,
        "frames": len(frames),
        "confidence": args.confidence,
        "max_det": args.max_det,
        "device": args.device,
        "models": [],
    }
    reference_results: list[list[Detection]] | None = None
    reference_batch: int | None = None

    for model_text in args.models:
        model_path = Path(model_text).expanduser().resolve()
        runtime = FixedBatchYoloDml(model_path, args.device)
        detections, metrics = run_model(runtime, frames, args)
        entry: dict[str, Any] = {
            "model": str(model_path),
            "input_shape": list(runtime.input_shape),
            "output_shape": list(runtime.output_shape),
            **metrics,
        }
        if reference_results is None or runtime.batch_size == 1:
            reference_results = detections
            reference_batch = runtime.batch_size
            entry["parity_reference"] = True
        else:
            entry["parity_reference"] = False
            entry["parity_vs_batch"] = reference_batch
            entry["parity"] = compare_to_reference(reference_results, detections)
        result["models"].append(entry)
        print(json.dumps(entry, ensure_ascii=False, indent=2))

    output = Path(args.output)
    if not output.is_absolute():
        output = Path(__file__).resolve().parent / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
