"""Stage-level profile of old/current RITnet paths on identical real input."""
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
from ritnet_fullclass_final_runtime import OUTPUT_NAMES, RitnetFullClassFinalRuntime
from ritnet_fullclass_metric_adapter import summarize_final_hard_metrics
from ritnet_fullclass_roi import crop_fixed_aspect_gray, fixed_aspect_roi_geometry, valid_source_analysis_mask
from ritnet_fullclass_uncertainty import summarize_uncertainty
from ritnet_onnx_runtime import RitnetOnnxRuntime


class Monitor:
    def __init__(self) -> None:
        self.stop = threading.Event()
        self.gpu: list[tuple[float, float]] = []
        self.cpu: list[float] = []
        self.ram: list[float] = []
        self.gpu_total = 0.0
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
                    text=True, stderr=subprocess.DEVNULL,
                ).strip().splitlines()[0]
                util, used, total = (float(value.strip()) for value in raw.split(","))
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
    parser.add_argument("--config", type=Path, default=RUNTIME_DIR / "config.yaml")
    parser.add_argument("--max-eyes", type=int, default=1024)
    return parser.parse_args()


def load_rows(run_dir: Path, max_eyes: int) -> tuple[str, list[dict[str, str]]]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    with (run_dir / "eyes.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))[:max_eyes]
    return str(summary["video"]), rows


def read_rois_timed(video: str, rows: list[dict[str, str]], roi_cfg: dict[str, object]) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, float], int]:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open AVI: {video}")
    times = {key: 0.0 for key in ("video_decode", "source_row_roi_resolution", "crop_resize_roi", "temporal_source_mask")}
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(int(float(row["frame_idx"])), []).append(row)
    rois: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    try:
        for frame_idx in sorted(grouped):
            started = time.perf_counter()
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            times["video_decode"] += time.perf_counter() - started
            if not ok or frame is None:
                raise RuntimeError(f"failed to read frame {frame_idx}")
            height, width = frame.shape[:2]
            for row in grouped[frame_idx]:
                started = time.perf_counter()
                bbox = [float(row[key]) for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")]
                geometry = fixed_aspect_roi_geometry(
                    bbox=bbox, frame_width=width, frame_height=height,
                    expand_horizontal_each_side=float(roi_cfg["expand_horizontal_each_side"]),
                    expand_vertical_each_side=float(roi_cfg["expand_vertical_each_side"]),
                    padding_mode=str(roi_cfg["padding_mode"]),
                )
                times["source_row_roi_resolution"] += time.perf_counter() - started
                started = time.perf_counter()
                rois.append(crop_fixed_aspect_gray(frame, geometry))
                times["crop_resize_roi"] += time.perf_counter() - started
                started = time.perf_counter()
                masks.append(valid_source_analysis_mask(geometry))
                times["temporal_source_mask"] += time.perf_counter() - started
    finally:
        cap.release()
    return rois, masks, times, len(grouped)


def _summary(times: dict[str, float], wall: float, monitor: Monitor, eyes: int, frames: int, provider: str) -> dict[str, object]:
    gpu = monitor.gpu
    peak = max((value[1] for value in gpu), default=0.0)
    total = monitor.gpu_total
    return {
        "eyes": eyes, "frames": frames, "measurement_wall_sec": wall,
        "eyes_per_sec": eyes / wall, "frames_per_sec": frames / wall,
        "stage_sec": times, "stage_sum_sec": sum(times.values()),
        "provider": provider,
        "gpu_util_avg_pct": float(np.mean([value[0] for value in gpu])) if gpu else None,
        "gpu_util_p95_pct": float(np.percentile([value[0] for value in gpu], 95)) if gpu else None,
        "gpu_memory_peak_bytes": peak, "gpu_memory_total_bytes": total,
        "gpu_memory_headroom_bytes": total - peak if total else None,
        "cpu_peak_pct": max(monitor.cpu, default=None), "ram_peak_bytes": max(monitor.ram, default=None),
    }


def profile_current(rois: list[np.ndarray], masks: list[np.ndarray], model: Path) -> dict[str, object]:
    runtime = RitnetFullClassFinalRuntime(model, device="0")
    warm_tensor, warm_valid, _ = runtime.prepare_batch(rois[:16])
    runtime.infer_prepared(warm_tensor, warm_valid)
    times = {key: 0.0 for key in ("preprocess", "inference", "probability_uncertainty", "metrics", "orchestration")}
    labels_seen = 0
    with Monitor() as monitor:
        started_all = time.perf_counter()
        for offset in range(0, len(rois), 16):
            batch = rois[offset : offset + 16]
            valid = masks[offset : offset + 16]
            started = time.perf_counter()
            tensor, valid_count, _ = runtime.prepare_batch(batch)
            times["preprocess"] += time.perf_counter() - started
            started = time.perf_counter()
            raw = runtime.session.run(list(OUTPUT_NAMES), {runtime.input_name: tensor})
            times["inference"] += time.perf_counter() - started
            started = time.perf_counter()
            labels = runtime._validate_labels_output(raw[0], valid_count)
            class_probability = runtime._validate_class_probability(raw[1])
            max_probability = runtime._validate_float_map("max_probability", raw[2], lower=0.0, upper=1.0)
            margin = runtime._validate_float_map("top1_top2_margin", raw[3], lower=0.0, upper=1.0)
            entropy = runtime._validate_float_map("entropy", raw[4], lower=0.0, upper=np.log(4.0))
            times["probability_uncertainty"] += time.perf_counter() - started
            started = time.perf_counter()
            for index in range(valid_count):
                hard = summarize_final_hard_metrics(labels[index], valid[index])
                if not hard:
                    raise RuntimeError("current metric integrity check failed")
            times["metrics"] += time.perf_counter() - started
            started = time.perf_counter()
            for index in range(valid_count):
                summarize_uncertainty(
                    labels=labels[index], valid_source_mask=valid[index],
                    class_probability=class_probability[index], max_probability=max_probability[index],
                    top1_top2_margin=margin[index], entropy=entropy[index],
                    boundary_band_px=5, low_max_probability_threshold=None,
                )
            times["probability_uncertainty"] += time.perf_counter() - started
            labels_seen += valid_count
            times["orchestration"] += max(0.0, time.perf_counter() - started_all - sum(times.values()))
        wall = time.perf_counter() - started_all
    return _summary(times, wall, monitor, labels_seen, len({id(mask) for mask in masks}), runtime.providers[0])


def profile_legacy(rois: list[np.ndarray], model: Path) -> dict[str, object]:
    runtime = RitnetOnnxRuntime(RUNTIME_DIR, model, input_size=(640, 400), device="0", analysis_size=(320, 160), precision="fp32")
    runtime.infer_batch(rois[:16])
    times = {key: 0.0 for key in ("preprocess", "inference", "probability_uncertainty", "metrics", "orchestration")}
    seen = 0
    with Monitor() as monitor:
        started_all = time.perf_counter()
        for offset in range(0, len(rois), 16):
            batch = rois[offset : offset + 16]
            started = time.perf_counter()
            images = [runtime._preprocess_one(roi) for roi in batch]
            images.extend([images[-1]] * (16 - len(images)))
            tensor = np.ascontiguousarray((((np.stack(images).astype(np.float32) / 255.0) - 0.5) / 0.5)[:, None])
            times["preprocess"] += time.perf_counter() - started
            started = time.perf_counter()
            labels, probabilities = runtime.session.run(runtime.output_names, {runtime.input_name: tensor})
            times["inference"] += time.perf_counter() - started
            started = time.perf_counter()
            results = [runtime._postprocess_one(label, probability) for label, probability in zip(labels[: len(batch)], probabilities[: len(batch)])]
            times["metrics"] += time.perf_counter() - started
            if len(results) != len(batch):
                raise RuntimeError("legacy metric integrity check failed")
            seen += len(results)
            times["orchestration"] += max(0.0, time.perf_counter() - started_all - sum(times.values()))
        wall = time.perf_counter() - started_all
    return _summary(times, wall, monitor, seen, 0, runtime.providers[0])


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    video, rows = load_rows(args.run_dir.resolve(), args.max_eyes)
    rois, masks, io_times, frame_count = read_rois_timed(video, rows, config["fullclass"]["roi"])
    current_model = RUNTIME_DIR / config["models"]["ritnet_fullclass_final"]
    old_model = RUNTIME_DIR / config["models"]["ritnet_onnx"]
    # Warm-up uses the same first 16 real ROIs and is excluded from both reports.
    current = profile_current(rois, masks, current_model)
    legacy = profile_legacy(rois, old_model)
    for report in (current, legacy):
        report["stage_sec"]["video_decode"] = io_times["video_decode"]
        report["stage_sec"]["source_row_roi_resolution"] = io_times["source_row_roi_resolution"]
        report["stage_sec"]["crop_canonical_roi"] = io_times["crop_resize_roi"]
        report["stage_sec"]["temporal_source_mask"] = io_times["temporal_source_mask"]
        report["pipeline_wall_sec"] = float(report["measurement_wall_sec"]) + sum(io_times.values())
        report["pipeline_eyes_per_sec"] = len(rois) / report["pipeline_wall_sec"]
        report["stage_sum_sec"] = sum(report["stage_sec"].values())
        report["frames"] = frame_count
        report["frames_per_sec"] = frame_count / float(report["measurement_wall_sec"])
    print(json.dumps({"video": video, "input_eyes": len(rois), "input_frames": frame_count, "current": current, "legacy": legacy}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
