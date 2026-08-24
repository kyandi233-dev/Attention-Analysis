from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from directml_runtime import create_directml_session


class FixedBatchRitnetDml:
    def __init__(self, model_path: Path, device: str = "0") -> None:
        self.model_path = Path(model_path)
        self.session = create_directml_session(self.model_path, device)
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 2:
            raise ValueError("RITnet ONNX must expose one input and two outputs")
        self.input_name = inputs[0].name
        self.output_names = [output.name for output in outputs]
        self.input_shape = tuple(inputs[0].shape)
        self.output_shapes = [tuple(output.shape) for output in outputs]
        if len(self.input_shape) != 4 or self.input_shape[1:] != (1, 400, 640):
            raise ValueError(f"Expected fixed [B,1,400,640] input, got {self.input_shape}")
        if not isinstance(self.input_shape[0], int) or self.input_shape[0] <= 0:
            raise ValueError(f"Expected fixed batch dimension, got {self.input_shape}")
        self.batch_size = int(self.input_shape[0])
        expected = (self.batch_size, 400, 640)
        if any(shape != expected for shape in self.output_shapes):
            raise ValueError(f"Expected two fixed [B,400,640] outputs, got {self.output_shapes}")

        self.gamma_table = (255.0 * (np.linspace(0, 1, 256) ** 0.8)).astype(np.uint8)
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

    def _preprocess_one(self, roi_gray: np.ndarray) -> np.ndarray:
        if roi_gray is None or roi_gray.size == 0:
            raise ValueError("Empty RITnet ROI")
        if roi_gray.ndim == 3:
            roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(roi_gray, (640, 400), interpolation=cv2.INTER_LINEAR)
        image = cv2.LUT(image, self.gamma_table)
        image = self.clahe.apply(image)
        return np.ascontiguousarray(image)

    @staticmethod
    def _postprocess_one(pred: np.ndarray, pupil_prob: np.ndarray) -> dict[str, Any]:
        pupil = (pred == 3).astype(np.uint8)
        mask = cv2.resize(pupil, (320, 160), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"found": False}
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 5 or cv2.contourArea(contour) < 5:
            return {"found": False}
        (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
        area = float(cv2.contourArea(contour))
        return {
            "found": True,
            "center_x": float(cx),
            "center_y": float(cy),
            "axis_a": float(axis_a),
            "axis_b": float(axis_b),
            "angle_deg": float(angle),
            "mask_area": area,
            "equiv_diameter": float(2 * np.sqrt(area / np.pi)),
            "pupil_confidence": float(pupil_prob[pupil.astype(bool)].mean()) if pupil.any() else 0.0,
        }

    def infer_batch(self, rois: list[np.ndarray]) -> tuple[list[dict[str, Any]], dict[str, float]]:
        if not rois or len(rois) > self.batch_size:
            raise ValueError(f"Need 1..{self.batch_size} ROIs, got {len(rois)}")

        preprocess_started = time.perf_counter()
        images = [self._preprocess_one(roi) for roi in rois]
        valid = len(images)
        while len(images) < self.batch_size:
            images.append(images[-1])
        tensor = np.stack(images, axis=0).astype(np.float32, copy=False)
        tensor = ((tensor / np.float32(255.0) - np.float32(0.5)) / np.float32(0.5))[:, None, :, :]
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0

        inference_started = time.perf_counter()
        first, second = self.session.run(self.output_names, {self.input_name: tensor})
        inference_ms = (time.perf_counter() - inference_started) * 1000.0

        # Identify outputs by dtype so exporter/runtime output naming is irrelevant.
        if np.asarray(first).dtype == np.uint8:
            pred_batch = np.asarray(first[:valid], dtype=np.uint8)
            pupil_prob_batch = np.asarray(second[:valid], dtype=np.float32)
        else:
            pred_batch = np.asarray(second[:valid], dtype=np.uint8)
            pupil_prob_batch = np.asarray(first[:valid], dtype=np.float32)

        post_started = time.perf_counter()
        results = [
            self._postprocess_one(pred, pupil_prob)
            for pred, pupil_prob in zip(pred_batch, pupil_prob_batch)
        ]
        postprocess_ms = (time.perf_counter() - post_started) * 1000.0
        return results, {
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "postprocess_ms": postprocess_ms,
            "total_ms": preprocess_ms + inference_ms + postprocess_ms,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark fixed-batch RITnet ONNX on DirectML")
    parser.add_argument("--video", required=True)
    parser.add_argument("--eyes-csv", required=True)
    parser.add_argument("--models", required=True, nargs="+")
    parser.add_argument("--device", default="0")
    parser.add_argument("--phase", default="block1")
    parser.add_argument("--rois", type=int, default=1680)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--output", default="outputs/amd-directml/ritnet-batch-benchmark.json")
    return parser.parse_args()


def load_roi_specs(eyes_csv: Path, phase: str, count: int) -> list[dict[str, int]]:
    specs: list[dict[str, int]] = []
    with eyes_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if phase and str(row.get("phase", "")).strip().lower() != phase.lower():
                continue
            try:
                spec = {
                    "frame_idx": int(row["frame_idx"]),
                    "x1": int(float(row["roi_x1"])),
                    "y1": int(float(row["roi_y1"])),
                    "x2": int(float(row["roi_x2"])),
                    "y2": int(float(row["roi_y2"])),
                }
            except (KeyError, TypeError, ValueError):
                continue
            if spec["x2"] <= spec["x1"] or spec["y2"] <= spec["y1"]:
                continue
            specs.append(spec)
            if len(specs) >= count:
                break
    if len(specs) < count:
        raise RuntimeError(f"Requested {count} ROI rows but found only {len(specs)} for phase={phase!r}")
    return specs


def load_rois(video: Path, specs: list[dict[str, int]]) -> list[np.ndarray]:
    grouped: dict[int, list[tuple[int, dict[str, int]]]] = defaultdict(list)
    for index, spec in enumerate(specs):
        grouped[int(spec["frame_idx"])].append((index, spec))

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    rois: list[np.ndarray | None] = [None] * len(specs)
    try:
        for frame_idx in sorted(grouped):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Cannot decode frame {frame_idx}")
            height, width = frame.shape[:2]
            for index, spec in grouped[frame_idx]:
                x1 = max(0, min(width, spec["x1"]))
                x2 = max(0, min(width, spec["x2"]))
                y1 = max(0, min(height, spec["y1"]))
                y2 = max(0, min(height, spec["y2"]))
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    raise RuntimeError(f"Empty crop at frame {frame_idx}: {spec}")
                rois[index] = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    finally:
        cap.release()

    if any(roi is None for roi in rois):
        raise RuntimeError("Failed to materialize all ROI crops")
    return [roi for roi in rois if roi is not None]


def compare_results(reference: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    n = min(len(reference), len(candidate))
    found_agree = 0
    center_dist: list[float] = []
    diameter_abs: list[float] = []
    axis_a_abs: list[float] = []
    axis_b_abs: list[float] = []
    area_abs: list[float] = []
    confidence_abs: list[float] = []
    compared_found = 0

    for ref, cand in zip(reference[:n], candidate[:n]):
        if bool(ref.get("found")) == bool(cand.get("found")):
            found_agree += 1
        if not ref.get("found") or not cand.get("found"):
            continue
        compared_found += 1
        dx = float(ref["center_x"]) - float(cand["center_x"])
        dy = float(ref["center_y"]) - float(cand["center_y"])
        center_dist.append(float(np.hypot(dx, dy)))
        diameter_abs.append(abs(float(ref["equiv_diameter"]) - float(cand["equiv_diameter"])))
        axis_a_abs.append(abs(float(ref["axis_a"]) - float(cand["axis_a"])))
        axis_b_abs.append(abs(float(ref["axis_b"]) - float(cand["axis_b"])))
        area_abs.append(abs(float(ref["mask_area"]) - float(cand["mask_area"])))
        confidence_abs.append(abs(float(ref["pupil_confidence"]) - float(cand["pupil_confidence"])))

    def stats(values: list[float], prefix: str) -> dict[str, float | None]:
        return {
            f"{prefix}_mae": float(np.mean(values)) if values else None,
            f"{prefix}_max_abs": float(np.max(values)) if values else None,
        }

    return {
        "found_agreement": found_agree / max(1, n),
        "compared_found": compared_found,
        **stats(center_dist, "center_distance_px"),
        **stats(diameter_abs, "equiv_diameter_px"),
        **stats(axis_a_abs, "axis_a_px"),
        **stats(axis_b_abs, "axis_b_px"),
        **stats(area_abs, "mask_area_px"),
        **stats(confidence_abs, "pupil_confidence"),
    }


def run_model(runtime: FixedBatchRitnetDml, rois: list[np.ndarray], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    warmup = rois[: runtime.batch_size]
    for _ in range(max(0, args.warmup_batches)):
        runtime.infer_batch(warmup)

    timing_total = {"preprocess_ms": 0.0, "inference_ms": 0.0, "postprocess_ms": 0.0, "total_ms": 0.0}
    all_results: list[dict[str, Any]] = []
    wall_started = time.perf_counter()
    for start in range(0, len(rois), runtime.batch_size):
        chunk = rois[start : start + runtime.batch_size]
        results, timing = runtime.infer_batch(chunk)
        all_results.extend(results)
        for key in timing_total:
            timing_total[key] += float(timing[key])
    wall_sec = time.perf_counter() - wall_started
    n = len(rois)
    return all_results, {
        "batch_size": runtime.batch_size,
        "rois": n,
        "wall_sec": wall_sec,
        "roi_per_sec": n / wall_sec if wall_sec else None,
        **timing_total,
        "mean_preprocess_ms_per_roi": timing_total["preprocess_ms"] / n,
        "mean_inference_ms_per_roi": timing_total["inference_ms"] / n,
        "mean_postprocess_ms_per_roi": timing_total["postprocess_ms"] / n,
        "mean_total_ms_per_roi": timing_total["total_ms"] / n,
    }


def main() -> int:
    args = parse_args()
    video = Path(args.video).expanduser().resolve()
    eyes_csv = Path(args.eyes_csv).expanduser().resolve()
    specs = load_roi_specs(eyes_csv, args.phase, args.rois)
    rois = load_rois(video, specs)

    result: dict[str, Any] = {
        "video": str(video),
        "eyes_csv": str(eyes_csv),
        "phase": args.phase,
        "rois": len(rois),
        "device": args.device,
        "models": [],
    }
    reference_results: list[dict[str, Any]] | None = None
    reference_batch: int | None = None

    for model_text in args.models:
        model_path = Path(model_text).expanduser().resolve()
        runtime = FixedBatchRitnetDml(model_path, args.device)
        model_results, timing = run_model(runtime, rois, args)
        row: dict[str, Any] = {
            "model": str(model_path),
            "input_shape": list(runtime.input_shape),
            "output_shapes": [list(shape) for shape in runtime.output_shapes],
            **timing,
        }
        if reference_results is None:
            reference_results = model_results
            reference_batch = runtime.batch_size
            row["parity_reference"] = True
        else:
            row["parity_reference"] = False
            row["parity_vs_batch"] = reference_batch
            row["parity"] = compare_results(reference_results, model_results)
        result["models"].append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Benchmark JSON -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
