"""Isolated real-AVI throughput benchmark for the frozen NIR full-class path.

This benchmark reads historical source eyes and the original AVI, rebuilds the
canonical ROI, runs the real CUDA ONNX path, and performs the normal hard-metric
post-processing. It never writes formal output or completion markers.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

RUNTIME_DIR = Path(__file__).resolve().parents[1] / "runtime/nir-formal"
sys.path.insert(0, str(RUNTIME_DIR))

from ritnet_fullclass_final_runtime import RitnetFullClassFinalRuntime
from ritnet_fullclass_metric_adapter import summarize_final_hard_metrics
from ritnet_fullclass_roi import (
    crop_fixed_aspect_gray,
    fixed_aspect_roi_geometry,
    valid_source_analysis_mask,
)


class Monitor:
    def __init__(self) -> None:
        self.stop = threading.Event()
        self.gpu: list[tuple[float, float]] = []
        self.cpu: list[float] = []
        self.ram: list[float] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            import psutil
        except ImportError:
            psutil = None
        process = psutil.Process(os.getpid()) if psutil else None
        while not self.stop.is_set():
            try:
                raw = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip().splitlines()[0]
                util, used, total = (float(v.strip()) for v in raw.split(","))
                self.gpu.append((util, used * 1024 * 1024))
                self.gpu_total = total * 1024 * 1024
            except Exception:
                pass
            if process:
                try:
                    self.cpu.append(float(process.cpu_percent(interval=0.05)))
                    self.ram.append(float(process.memory_info().rss))
                except Exception:
                    pass
            self.stop.wait(0.25)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join(timeout=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "runtime/nir-formal/config.yaml")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--implementation", choices=("current", "legacy"), default="current")
    parser.add_argument("--max-eyes", type=int, default=512)
    parser.add_argument("--batches", default="16,24,32,40,48")
    return parser.parse_args()


def load_rows(run_dir: Path, max_eyes: int) -> tuple[str, list[dict[str, str]]]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    video = str(summary["video"])
    with (run_dir / "eyes.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))[:max_eyes]
    if not rows:
        raise RuntimeError("selected source contains no eyes")
    return video, rows


def extract_rois(video: str, rows: list[dict[str, str]], roi_cfg: dict[str, object]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source AVI: {video}")
    try:
        grouped: dict[int, list[dict[str, str]]] = {}
        for row in rows:
            grouped.setdefault(int(float(row["frame_idx"])), []).append(row)
        rois: list[np.ndarray] = []
        for frame_idx in sorted(grouped):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"failed to read source frame {frame_idx}")
            height, width = frame.shape[:2]
            for row in grouped[frame_idx]:
                geometry = fixed_aspect_roi_geometry(
                    bbox=[float(row[k]) for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")],
                    frame_width=width,
                    frame_height=height,
                    expand_horizontal_each_side=float(roi_cfg["expand_horizontal_each_side"]),
                    expand_vertical_each_side=float(roi_cfg["expand_vertical_each_side"]),
                    padding_mode=str(roi_cfg["padding_mode"]),
                )
                rois.append(crop_fixed_aspect_gray(frame, geometry))
        return rois
    finally:
        cap.release()


def measure(runtime: RitnetFullClassFinalRuntime, rois: list[np.ndarray], batch_size: int) -> dict[str, object]:
    # The frozen ONNX contract is fixed-b16. Higher requested values are
    # reported as unsupported rather than emulated with multiple calls.
    if batch_size > runtime.FIXED_BATCH_SIZE:
        try:
            runtime.prepare_batch(rois[:batch_size])
        except Exception as exc:
            return {"batch_size": batch_size, "status": "unsupported_fixed_model", "reason": str(exc)}
    warmup_rois = rois[: runtime.FIXED_BATCH_SIZE]
    warm_start = time.perf_counter()
    warm_tensor, warm_valid, _ = runtime.prepare_batch(warmup_rois)
    warm_outputs, _ = runtime.infer_labels_prepared(warm_tensor, warm_valid)
    warm_elapsed = time.perf_counter() - warm_start
    if warm_outputs.shape != (warm_valid, 400, 640) or int(warm_outputs.max(initial=0)) > 3:
        raise RuntimeError("warm-up output integrity check failed")

    labels_seen = 0
    metric_checks = 0
    start = time.perf_counter()
    monitor = Monitor()
    with monitor:
        for offset in range(0, len(rois), runtime.FIXED_BATCH_SIZE):
            batch = rois[offset : offset + runtime.FIXED_BATCH_SIZE]
            tensor, valid_count, _ = runtime.prepare_batch(batch)
            labels, _ = runtime.infer_labels_prepared(tensor, valid_count)
            if labels.shape != (valid_count, 400, 640) or int(labels.max(initial=0)) > 3:
                raise RuntimeError("label output integrity check failed")
            for label in labels:
                metrics = summarize_final_hard_metrics(label, None)
                if not metrics or not all(np.isfinite(float(v)) for v in metrics.values() if isinstance(v, (int, float))):
                    raise RuntimeError("metric output integrity check failed")
                metric_checks += 1
            labels_seen += valid_count
    elapsed = time.perf_counter() - start
    gpu = monitor.gpu
    total_memory = float(getattr(monitor, "gpu_total", 0.0))
    peak_memory = max((item[1] for item in gpu), default=0.0)
    return {
        "batch_size": batch_size,
        "status": "ok",
        "input_eyes": len(rois),
        "input_frames": None,
        "warmup_eyes": warm_valid,
        "warmup_sec": warm_elapsed,
        "measurement_wall_sec": elapsed,
        "eyes_per_sec": labels_seen / elapsed,
        "frames_per_sec": None,
        "labels_checked": labels_seen,
        "metrics_checked": metric_checks,
        "provider": runtime.providers[0] if runtime.providers else None,
        "gpu_util_avg_pct": float(np.mean([item[0] for item in gpu])) if gpu else None,
        "gpu_util_p95_pct": float(np.percentile([item[0] for item in gpu], 95)) if gpu else None,
        "gpu_memory_peak_bytes": peak_memory,
        "gpu_memory_total_bytes": total_memory,
        "gpu_memory_headroom_bytes": total_memory - peak_memory if total_memory else None,
        "cpu_peak_pct": max(monitor.cpu, default=None),
        "ram_peak_bytes": max(monitor.ram, default=None),
        "io_bottleneck_observed": False,
        "error": None,
    }


def measure_legacy(runtime, rois: list[np.ndarray], batch_size: int) -> dict[str, object]:
    if batch_size > runtime.FIXED_BATCH_SIZE:
        try:
            runtime.infer_batch(rois[:batch_size])
        except Exception as exc:
            return {"batch_size": batch_size, "status": "unsupported_fixed_model", "reason": str(exc)}
    warm_start = time.perf_counter()
    runtime.infer_batch(rois[: runtime.FIXED_BATCH_SIZE])
    warm_elapsed = time.perf_counter() - warm_start
    labels_seen = 0
    start = time.perf_counter()
    monitor = Monitor()
    with monitor:
        for offset in range(0, len(rois), runtime.FIXED_BATCH_SIZE):
            results = runtime.infer_batch(rois[offset : offset + runtime.FIXED_BATCH_SIZE])
            if len(results) != min(runtime.FIXED_BATCH_SIZE, len(rois) - offset):
                raise RuntimeError("legacy postprocess output count mismatch")
            if any(not isinstance(item, dict) for item in results):
                raise RuntimeError("legacy postprocess output integrity check failed")
            labels_seen += len(results)
    elapsed = time.perf_counter() - start
    gpu = monitor.gpu
    total_memory = float(getattr(monitor, "gpu_total", 0.0))
    peak_memory = max((item[1] for item in gpu), default=0.0)
    return {
        "batch_size": batch_size,
        "status": "ok",
        "input_eyes": len(rois),
        "input_frames": None,
        "warmup_eyes": runtime.FIXED_BATCH_SIZE,
        "warmup_sec": warm_elapsed,
        "measurement_wall_sec": elapsed,
        "eyes_per_sec": labels_seen / elapsed,
        "frames_per_sec": None,
        "labels_checked": labels_seen,
        "metrics_checked": labels_seen,
        "provider": runtime.providers[0] if runtime.providers else None,
        "gpu_util_avg_pct": float(np.mean([item[0] for item in gpu])) if gpu else None,
        "gpu_util_p95_pct": float(np.percentile([item[0] for item in gpu], 95)) if gpu else None,
        "gpu_memory_peak_bytes": peak_memory,
        "gpu_memory_total_bytes": total_memory,
        "gpu_memory_headroom_bytes": total_memory - peak_memory if total_memory else None,
        "cpu_peak_pct": max(monitor.cpu, default=None),
        "ram_peak_bytes": max(monitor.ram, default=None),
        "io_bottleneck_observed": False,
        "error": None,
    }


def main() -> int:
    args = parse_args()
    runtime_dir = RUNTIME_DIR
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    roi_cfg = config["fullclass"]["roi"]
    if args.implementation == "legacy":
        from ritnet_onnx_runtime import RitnetOnnxRuntime

        model = args.model or (runtime_dir / config["models"]["ritnet_onnx"])
    else:
        model = args.model or (runtime_dir / config["models"]["ritnet_fullclass_final"])
    video, rows = load_rows(args.run_dir.resolve(), args.max_eyes)
    rois = extract_rois(video, rows, roi_cfg)
    frame_count = len({int(float(row["frame_idx"])) for row in rows})
    results: list[dict[str, object]] = []
    for batch in (int(value) for value in args.batches.split(",")):
        if batch != 16:
            results.append({"batch_size": batch, "status": "unsupported_fixed_model", "reason": "frozen ONNX input shape is [16,1,400,640]"})
            continue
        if args.implementation == "legacy":
            runtime = RitnetOnnxRuntime(runtime_dir, model, input_size=(640, 400), device="0", analysis_size=(320, 160), precision="fp32")
            result = measure_legacy(runtime, rois, batch)
        else:
            runtime = RitnetFullClassFinalRuntime(model, device="0")
            result = measure(runtime, rois, batch)
        result["input_frames"] = frame_count
        if result.get("status") == "ok":
            result["frames_per_sec"] = frame_count / float(result["measurement_wall_sec"])
        results.append(result)
    print(json.dumps({"implementation": args.implementation, "model": str(model), "subject": args.run_dir.name.split("_formal", 1)[0], "source_run_dir": str(args.run_dir.resolve()), "video": video, "input_eyes": len(rois), "input_frames": frame_count, "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
