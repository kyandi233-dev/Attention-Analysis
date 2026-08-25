"""Portable NIR eye pipeline: diagnostic runner + optimized formal analysis.

``run`` preserves the original short-video diagnostic behavior, including optional
CSRT/KCF reproduction. ``formal`` is the production-candidate path for FocusWave
v3.1.3 subjects (sub-031 and later): per-frame YOLO, phase-aware frame selection,
raw expanded crops, batched RITnet, optional CUDA FP16, and sparse QC overlays.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from formal_completion import (
    REQUIRED_ARTIFACTS,
    SCHEMA_VERSION as COMPLETION_SCHEMA_VERSION,
    expected_frame_keys,
    validate_completion,
    write_completion,
)
from onnx_cuda_runtime import YoloCudaRuntime
from phase_windows import PhaseWindow, resolve_phase_windows
from ritnet_onnx_runtime import RitnetOnnxRuntime
from ritnet_runtime import RitnetRuntime


PACKAGE_ROOT = Path(__file__).resolve().parent
_SUBJECT_NUMBER_RE = re.compile(r"^sub-(\d+)$")


@dataclass
class Detection:
    box: tuple[float, float, float, float]
    confidence: float


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_package_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PACKAGE_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_timestamp_map(video: Path) -> tuple[Path | None, dict[int, int]]:
    timestamp_path = video.with_name(f"{video.stem}_timestamps.csv")
    if not timestamp_path.exists():
        return None, {}
    result: dict[int, int] = {}
    with timestamp_path.open(newline="", encoding="utf-8-sig") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row or len(row) < 2:
                continue
            try:
                frame_idx, unix_ms = int(float(row[0])), int(float(row[1]))
            except ValueError as exc:
                raise ValueError(f"Invalid timestamp row {line_number}: {timestamp_path}") from exc
            if frame_idx in result:
                raise ValueError(f"Duplicate timestamp frame {frame_idx}: {timestamp_path}")
            result[frame_idx] = unix_ms
    return timestamp_path, result


def normalize_subject(subject: str) -> tuple[str, str]:
    stem = subject.strip().rstrip("_")
    if not stem.startswith("sub-"):
        raise ValueError(f"Invalid subject: {subject}")
    return f"{stem}_", stem


def subject_number(subject: str) -> int:
    match = _SUBJECT_NUMBER_RE.fullmatch(subject.strip().rstrip("_"))
    if not match:
        raise ValueError(f"Subject must use numeric form sub-XXX: {subject}")
    return int(match.group(1))


def discover_videos(roots: list[str], min_subject_number: int | None = None) -> list[dict[str, str]]:
    rows = []
    for root_text in roots:
        root = Path(root_text)
        if not root.exists():
            continue
        for video in sorted(root.glob("sub-*_/nir/*_nir.avi")):
            subject = video.parents[1].name.rstrip("_")
            if min_subject_number is not None:
                try:
                    if subject_number(subject) < int(min_subject_number):
                        continue
                except ValueError:
                    continue
            rows.append({"root": str(root), "subject": subject, "video": str(video)})
    return rows


def resolve_video(
    config: dict[str, Any],
    subject: str | None,
    root: str | None,
    video: str | None,
) -> tuple[str, Path]:
    if video:
        path = Path(video)
        if not path.exists():
            raise FileNotFoundError(path)
        match = path.stem.removesuffix("_nir")
        return match, path
    if not subject:
        raise ValueError("Provide --subject or --video")
    directory, stem = normalize_subject(subject)
    roots = [root] if root else config["data"]["roots"]
    candidates = [Path(item) / directory / "nir" / f"{stem}_nir.avi" for item in roots]
    found = [path for path in candidates if path.exists()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected exactly one video for {stem}; found={found}; checked={candidates}")
    return stem, found[0]


def tracker_factory(method: str):
    method = method.lower()
    if method == "none":
        return None
    names = {"csrt": "TrackerCSRT_create", "kcf": "TrackerKCF_create"}
    if method not in names:
        raise ValueError(f"Unsupported tracker: {method}")
    creator = getattr(cv2, names[method], None)
    if creator is None and hasattr(cv2, "legacy"):
        creator = getattr(cv2.legacy, names[method], None)
    if creator is None:
        raise RuntimeError(f"{method.upper()} unavailable; install opencv-contrib-python")
    return creator


def valid_box(box: tuple[float, float, float, float], frame_shape: tuple[int, ...]) -> bool:
    x1, y1, x2, y2 = box
    height, width = frame_shape[:2]
    return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height and (x2 - x1) >= 4 and (y2 - y1) >= 4


def xyxy_to_xywh(box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return tuple(int(round(value)) for value in (x1, y1, x2 - x1, y2 - y1))


def xywh_to_xyxy(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, width, height = box
    return x, y, x + width, y + height


def center_jump_ok(
    previous: tuple[float, float, float, float],
    current: tuple[float, float, float, float],
    fraction: float,
) -> bool:
    px1, py1, px2, py2 = previous
    cx1, cy1, cx2, cy2 = current
    previous_center = np.array([(px1 + px2) / 2, (py1 + py2) / 2])
    current_center = np.array([(cx1 + cx2) / 2, (cy1 + cy2) / 2])
    return float(np.linalg.norm(current_center - previous_center)) <= max(1.0, px2 - px1) * fraction


def _expanded_crop_box(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
    horizontal: float,
    vertical: float,
) -> tuple[tuple[int, int, int, int], bool]:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    raw = (
        x1 - horizontal * width,
        y1 - vertical * height,
        x2 + horizontal * width,
        y2 + vertical * height,
    )
    frame_h, frame_w = frame.shape[:2]
    clipped = raw[0] < 0 or raw[1] < 0 or raw[2] > frame_w or raw[3] > frame_h
    crop_box = (
        max(0, int(np.floor(raw[0]))),
        max(0, int(np.floor(raw[1]))),
        min(frame_w, int(np.ceil(raw[2]))),
        min(frame_h, int(np.ceil(raw[3]))),
    )
    rx1, ry1, rx2, ry2 = crop_box
    if rx2 <= rx1 or ry2 <= ry1:
        raise ValueError("Empty expanded ROI")
    return crop_box, clipped


def expand_crop(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
    horizontal: float,
    vertical: float,
    output_size: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int, int, int], bool]:
    """Compatibility crop for the original diagnostic runner."""
    crop_box, clipped = _expanded_crop_box(frame, box, horizontal, vertical)
    rx1, ry1, rx2, ry2 = crop_box
    gray = cv2.cvtColor(frame[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, output_size), crop_box, clipped


def expand_crop_raw(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
    horizontal: float,
    vertical: float,
) -> tuple[np.ndarray, tuple[int, int, int, int], bool]:
    """Return the native expanded crop; formal RITnet resizes it only once."""
    crop_box, clipped = _expanded_crop_box(frame, box, horizontal, vertical)
    rx1, ry1, rx2, ry2 = crop_box
    gray = cv2.cvtColor(frame[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2GRAY)
    return np.ascontiguousarray(gray), crop_box, clipped


def yolo_detect(model: Any, frame: np.ndarray, config: dict[str, Any], device: str) -> list[Detection]:
    cfg = config["yolo"]
    if hasattr(model, "detect"):
        rows = model.detect(
            frame,
            confidence=float(cfg["confidence"]),
            max_det=int(cfg["max_det"]),
        )
        return [Detection(box, confidence) for box, confidence, class_id in rows if class_id == 0]
    result = model.predict(
        frame,
        conf=float(cfg["confidence"]),
        imgsz=int(cfg["imgsz"]),
        iou=float(cfg["nms_iou"]),
        max_det=int(cfg["max_det"]),
        device=device,
        verbose=False,
    )[0]
    if result.boxes is None:
        return []
    detections = [
        Detection(tuple(map(float, box)), float(conf))
        for box, conf, cls in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
        )
        if int(cls) == 0
    ]
    return sorted(detections, key=lambda item: item.confidence, reverse=True)


def yolo_detect_batch(
    model: Any,
    frames: list[np.ndarray],
    config: dict[str, Any],
    device: str,
) -> list[list[Detection]]:
    """Run the same YOLO settings on a bounded batch of frames.

    This is an experimental validation path. It keeps the downstream formal
    processing frame-by-frame and only changes YOLO scheduling.
    """
    if not frames:
        return []
    cfg = config["yolo"]
    results = model.predict(
        source=frames,
        conf=float(cfg["confidence"]),
        imgsz=int(cfg["imgsz"]),
        iou=float(cfg["nms_iou"]),
        max_det=int(cfg["max_det"]),
        device=device,
        batch=len(frames),
        verbose=False,
    )
    output: list[list[Detection]] = []
    for result in results:
        if result.boxes is None:
            output.append([])
            continue
        detections = [
            Detection(tuple(map(float, box)), float(conf))
            for box, conf, cls in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
            )
            if int(cls) == 0
        ]
        output.append(sorted(detections, key=lambda item: item.confidence, reverse=True))
    if len(output) != len(frames):
        raise RuntimeError(f"YOLO batch returned {len(output)} results for {len(frames)} frames")
    return output


def init_trackers(frame: np.ndarray, boxes: list[Detection], creator) -> list[Any]:
    if creator is None or len(boxes) < 2:
        return []
    trackers = []
    for detection in boxes[:2]:
        tracker = creator()
        tracker.init(frame, xyxy_to_xywh(detection.box))
        trackers.append(tracker)
    return trackers


def update_trackers(
    frame: np.ndarray,
    trackers: list[Any],
    previous: list[Detection],
    jump_fraction: float,
) -> list[Detection] | None:
    if len(trackers) != 2 or len(previous) != 2:
        return None
    current = []
    for tracker, old in zip(trackers, previous):
        ok, raw = tracker.update(frame)
        if not ok:
            return None
        box = xywh_to_xyxy(tuple(map(float, raw)))
        if not valid_box(box, frame.shape) or not center_jump_ok(old.box, box, jump_fraction):
            return None
        current.append(Detection(box, old.confidence))
    return current


def draw_overlay(frame: np.ndarray, rows: list[dict[str, Any]], frame_status: str) -> np.ndarray:
    output = frame.copy()
    for row in rows:
        x1, y1, x2, y2 = [
            int(round(row[key])) for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
        ]
        color = (0, 220, 0) if row["status"] in {"observed", "roi_clipped"} else (0, 180, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output,
            f"{row['eye']} {row['source']} {row['status']}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        output,
        frame_status,
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 180, 0),
        2,
    )
    return output


def _make_ritnet(
    config: dict[str, Any],
    device: str,
    *,
    precision: str | None = None,
    backend: str = "pytorch-cuda",
) -> Any:
    is_ort = backend == "ort-cuda"
    ritnet_path = resolve_package_path(
        config["models"]["ritnet_onnx"] if is_ort else config["models"]["ritnet"]
    )
    runtime_class = RitnetOnnxRuntime if is_ort else RitnetRuntime
    return runtime_class(
        PACKAGE_ROOT,
        ritnet_path,
        (int(config["ritnet"]["input_width"]), int(config["ritnet"]["input_height"])),
        device=device,
        analysis_size=(int(config["roi"]["width"]), int(config["roi"]["height"])),
        precision=precision or str(config["ritnet"].get("precision", "fp32")),
    )


def _require_pytorch_cuda(device: str) -> None:
    import torch

    requested = str(device).strip().lower()
    if requested == "cpu" or not torch.cuda.is_available():
        raise RuntimeError(
            "Formal PyTorch inference requires CUDA; refusing silent CPU fallback"
        )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def run(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Original diagnostic mode; optional tracking and scalar RITnet are preserved."""
    subject, video = resolve_video(config, args.subject, args.root, args.video)
    model_path = resolve_package_path(config["models"]["yolo"])
    ritnet_path = resolve_package_path(config["models"]["ritnet"])
    device = args.device
    model = YOLO(str(model_path))
    if model.names != {0: "eye"}:
        raise ValueError(f"Unexpected YOLO classes: {model.names}")
    use_ritnet = bool(config["ritnet"]["enabled"]) and not args.skip_ritnet
    precision = args.ritnet_precision or str(config["ritnet"].get("precision", "fp32"))
    ritnet = _make_ritnet(config, device, precision=precision) if use_ritnet else None

    tracker_method = args.tracker or config["tracking"]["method"]
    creator = tracker_factory(tracker_method)
    redetect_interval = args.redetect_interval or int(config["tracking"]["redetect_interval"])
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    timestamp_path, unix_by_frame = load_timestamp_map(video)
    start_frame = max(0, int(round(args.start_sec * fps)))
    if args.full_video:
        end_frame = total_frames
    else:
        duration = args.duration_sec or float(config["data"]["default_duration_sec"])
        end_frame = min(total_frames, start_frame + int(round(duration * fps)))
    if start_frame >= total_frames or end_frame <= start_frame:
        raise ValueError(f"Invalid frame window {start_frame}:{end_frame} / {total_frames}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_root = Path(args.output) if args.output else resolve_package_path(config["output"]["root"])
    run_name = f"{subject}_{start_frame:08d}_{end_frame:08d}_{tracker_method}_r{redetect_interval}"
    out = output_root / run_name
    overlays = out / "overlays"
    rois = out / "rois"
    overlays.mkdir(parents=True, exist_ok=True)
    if args.save_rois or config["output"]["save_rois"]:
        rois.mkdir(parents=True, exist_ok=True)

    eye_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    trackers: list[Any] = []
    selected: list[Detection] = []
    wall_start = time.perf_counter()
    frame_idx = start_frame
    overlay_stride = int(config["output"]["overlay_stride"])

    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            frame_rows.append({"frame_idx": frame_idx, "status": "video_read_failed"})
            break
        frame_start = time.perf_counter()
        force_detect = creator is None or not selected or ((frame_idx - start_frame) % redetect_interval == 0)
        if creator is None:
            redetect_reason = "tracker_disabled"
        elif not selected:
            redetect_reason = "initial_or_previous_detection_missing"
        elif (frame_idx - start_frame) % redetect_interval == 0:
            redetect_reason = "scheduled"
        else:
            redetect_reason = "none"

        source = "yolo"
        detections: list[Detection] = []
        if not force_detect:
            tracked = update_trackers(
                frame,
                trackers,
                selected,
                float(config["tracking"]["center_jump_width_fraction"]),
            )
            if tracked is None:
                force_detect = True
                redetect_reason = "tracker_failure"
            else:
                selected = tracked
                detections = tracked
                source = "tracker"
        if force_detect:
            detections = yolo_detect(model, frame, config, device)
            selected = sorted(detections[:2], key=lambda item: (item.box[0] + item.box[2]) / 2)
            trackers = init_trackers(frame, selected, creator)
            source = "yolo"

        if len(detections) == 0:
            frame_status = "yolo_missing"
        elif len(detections) == 1:
            frame_status = "single_eye"
        elif len(detections) > 2 and source == "yolo":
            frame_status = "extra_boxes"
        else:
            frame_status = "two_eyes"

        current_rows = []
        for eye, detection in zip(("frame_left", "frame_right"), selected):
            roi_start = time.perf_counter()
            roi, crop_box, clipped = expand_crop(
                frame,
                detection.box,
                float(config["roi"]["expand_horizontal_each_side"]),
                float(config["roi"]["expand_vertical_each_side"]),
                (int(config["roi"]["width"]), int(config["roi"]["height"])),
            )
            pupil = ritnet.infer(roi) if ritnet else {"found": False}
            status = "observed" if pupil.get("found") else ("ritnet_missing" if ritnet else "roi_only")
            if clipped:
                status = "roi_clipped" if status == "observed" else status
            x1, y1, x2, y2 = detection.box
            rx1, ry1, rx2, ry2 = crop_box
            row = {
                "subject": subject,
                "video": str(video),
                "frame_idx": frame_idx,
                "video_time_ms": float(cap.get(cv2.CAP_PROP_POS_MSEC)),
                "unix_ms": unix_by_frame.get(frame_idx),
                "eye": eye,
                "source": source,
                "redetect_reason": redetect_reason,
                "frame_status": frame_status,
                "status": status,
                "anchor_yolo_confidence": detection.confidence,
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "roi_x1": rx1,
                "roi_y1": ry1,
                "roi_x2": rx2,
                "roi_y2": ry2,
                "roi_clipped": clipped,
                "ritnet_found": bool(pupil.get("found")),
                "ritnet_device": str(ritnet.device) if ritnet else "disabled",
                "ritnet_precision": ritnet.precision if ritnet else "disabled",
                "pupil_center_x": pupil.get("center_x"),
                "pupil_center_y": pupil.get("center_y"),
                "pupil_axis_a": pupil.get("axis_a"),
                "pupil_axis_b": pupil.get("axis_b"),
                "pupil_angle_deg": pupil.get("angle_deg"),
                "pupil_equiv_diameter": pupil.get("equiv_diameter"),
                "pupil_confidence": pupil.get("pupil_confidence"),
                "eye_processing_ms": (time.perf_counter() - roi_start) * 1000,
            }
            eye_rows.append(row)
            current_rows.append(row)
            if rois.exists():
                cv2.imwrite(str(rois / f"f{frame_idx:08d}_{eye}.png"), roi)

        frame_rows.append(
            {
                "subject": subject,
                "video": str(video),
                "frame_idx": frame_idx,
                "video_time_ms": float(cap.get(cv2.CAP_PROP_POS_MSEC)),
                "unix_ms": unix_by_frame.get(frame_idx),
                "source": source,
                "redetect_reason": redetect_reason,
                "status": frame_status,
                "raw_detection_count": len(detections),
                "selected_eye_count": len(selected),
                "frame_processing_ms": (time.perf_counter() - frame_start) * 1000,
            }
        )
        if overlay_stride > 0 and (frame_idx - start_frame) % overlay_stride == 0:
            cv2.imwrite(
                str(overlays / f"f{frame_idx:08d}.jpg"),
                draw_overlay(frame, current_rows, frame_status),
            )
        frame_idx += 1

    cap.release()
    elapsed = time.perf_counter() - wall_start
    _write_csv(out / "eyes.csv", eye_rows)
    _write_csv(out / "frames.csv", frame_rows)
    summary = {
        "subject": subject,
        "video": str(video),
        "start_frame": start_frame,
        "requested_end_frame": end_frame,
        "processed_frames": len(frame_rows),
        "fps_source": fps,
        "elapsed_sec": elapsed,
        "processing_fps": len(frame_rows) / elapsed if elapsed else None,
        "tracker": tracker_method,
        "redetect_interval": redetect_interval,
        "ritnet_enabled": bool(ritnet),
        "ritnet_device": str(ritnet.device) if ritnet else "disabled",
        "ritnet_precision": ritnet.precision if ritnet else "disabled",
        "timestamp_file": str(timestamp_path) if timestamp_path else None,
        "frame_status_counts": _counts(row.get("status") for row in frame_rows),
        "eye_status_counts": _counts(row.get("status") for row in eye_rows),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "run_manifest.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "command": " ".join(sys.argv),
                "package": config["package"],
                "config": config,
                "effective_parameters": {
                    "mode": "diagnostic",
                    "tracker": tracker_method,
                    "redetect_interval": redetect_interval,
                    "ritnet_precision": ritnet.precision if ritnet else "disabled",
                    "ritnet_batch_size": 1 if ritnet else 0,
                    "device": device,
                },
                "python": sys.version,
                "platform": platform.platform(),
                "opencv": cv2.__version__,
                "yolo_sha256": sha256(model_path),
                "ritnet_sha256": sha256(ritnet_path),
                "cuda_available": _cuda_available(),
                "timestamp_file": str(timestamp_path) if timestamp_path else None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(out.resolve()), **summary}, ensure_ascii=False, indent=2))
    return 0


def _phase_names(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    if getattr(args, "phases", None):
        values = [part.strip().lower() for part in args.phases.split(",") if part.strip()]
    else:
        values = [str(value).strip().lower() for value in config["formal"]["phases"]]
    if not values:
        raise ValueError("At least one formal phase is required")
    return values


def _frame_status(detections: list[Detection]) -> str:
    if len(detections) == 0:
        return "yolo_missing"
    if len(detections) == 1:
        return "single_eye"
    if len(detections) > 2:
        return "extra_boxes"
    return "two_eyes"


def _flush_formal_batch(
    pending: list[dict[str, Any]],
    *,
    ritnet: RitnetRuntime | None,
    eye_rows: list[dict[str, Any]],
    frame_lookup: dict[tuple[str, int, int], dict[str, Any]],
    overlay_pending: dict[tuple[str, int, int], tuple[np.ndarray, str]],
    overlays: Path,
    rois: Path | None,
    analysis_size: tuple[int, int],
) -> None:
    if not pending:
        return

    if ritnet:
        pupils = ritnet.infer_batch([item["roi"] for item in pending])
        timing = dict(ritnet.last_timing)
        share_ms = float(timing.get("total_ms", 0.0)) / max(1, len(pending))
        batch_size = int(timing.get("batch_size", len(pending)))
    else:
        pupils = [{"found": False} for _ in pending]
        timing = {"batch_size": 0, "total_ms": 0.0}
        share_ms = 0.0
        batch_size = 0

    rows_by_frame: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for item, pupil in zip(pending, pupils):
        row = item["row"]
        clipped = bool(row["roi_clipped"])
        status = "observed" if pupil.get("found") else ("ritnet_missing" if ritnet else "roi_only")
        if clipped and status == "observed":
            status = "roi_clipped"
        row.update(
            {
                "status": status,
                "ritnet_found": bool(pupil.get("found")),
                "ritnet_device": str(ritnet.device) if ritnet else "disabled",
                "ritnet_precision": ritnet.precision if ritnet else "disabled",
                "ritnet_batch_size": batch_size,
                "pupil_center_x": pupil.get("center_x"),
                "pupil_center_y": pupil.get("center_y"),
                "pupil_axis_a": pupil.get("axis_a"),
                "pupil_axis_b": pupil.get("axis_b"),
                "pupil_angle_deg": pupil.get("angle_deg"),
                "pupil_mask_area": pupil.get("mask_area"),
                "pupil_equiv_diameter": pupil.get("equiv_diameter"),
                "pupil_confidence": pupil.get("pupil_confidence"),
                "eye_processing_ms": float(item["roi_crop_ms"]) + share_ms,
                "ritnet_attributed_ms": share_ms,
            }
        )
        eye_rows.append(row)

        key = item["frame_key"]
        rows_by_frame[key].append(row)
        frame_row = frame_lookup[key]
        frame_row["ritnet_attributed_ms"] += share_ms
        frame_row["ritnet_batch_size_max"] = max(int(frame_row["ritnet_batch_size_max"]), batch_size)

        if rois is not None:
            analysis_w, analysis_h = analysis_size
            saved = cv2.resize(item["roi"], (analysis_w, analysis_h), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(
                str(rois / f"{row['phase']}_s{row['phase_segment']}_f{row['frame_idx']:08d}_{row['eye']}.png"),
                saved,
            )

    # All eye ROIs from a frame enter pending together; therefore an overlay frame
    # that appears in this flush is complete and can be written now.
    for key, rows in rows_by_frame.items():
        if key not in overlay_pending:
            continue
        frame, frame_status = overlay_pending.pop(key)
        overlay_started = time.perf_counter()
        phase, segment, frame_idx = key
        cv2.imwrite(
            str(overlays / f"{phase}_s{segment}_f{frame_idx:08d}.jpg"),
            draw_overlay(frame, rows, frame_status),
        )
        frame_lookup[key]["overlay_write_ms"] += (time.perf_counter() - overlay_started) * 1000.0

    pending.clear()


def formal(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Optimized FocusWave v3.1.3 formal analysis path."""
    subject, video = resolve_video(config, args.subject, args.root, args.video)
    minimum = int(config["formal"].get("min_subject_number", 31))
    number = subject_number(subject)
    if number < minimum:
        raise ValueError(
            f"Formal v3.1.3 analysis is configured for sub-{minimum:03d} and later; got {subject}. "
            "Earlier three-block recordings are intentionally excluded for now."
        )

    device = args.device
    backend = str(args.backend or config.get("inference", {}).get("backend", "pytorch-cuda"))
    if backend not in {"pytorch-cuda", "ort-cuda"}:
        raise ValueError(f"Unsupported formal inference backend: {backend}")
    is_ort = backend == "ort-cuda"
    if not is_ort:
        _require_pytorch_cuda(device)
    model_path = resolve_package_path(
        config["models"]["yolo_onnx"] if is_ort else config["models"]["yolo"]
    )
    ritnet_path = resolve_package_path(
        config["models"]["ritnet_onnx"] if is_ort else config["models"]["ritnet"]
    )
    model = YoloCudaRuntime(model_path, device=device) if is_ort else YOLO(str(model_path))
    if model.names != {0: "eye"}:
        raise ValueError(f"Unexpected YOLO classes: {model.names}")

    use_ritnet = bool(config["ritnet"]["enabled"]) and not args.skip_ritnet
    precision = args.ritnet_precision or str(config["ritnet"].get("precision", "fp32"))
    batch_size = args.ritnet_batch_size or int(config["ritnet"].get("batch_size", 16))
    if batch_size <= 0:
        raise ValueError("RITnet batch size must be positive")
    if is_ort and (precision != "fp32" or batch_size != 16):
        raise ValueError("ORT CUDA profile is frozen to FP32 and RITnet batch=16")
    ritnet = (
        _make_ritnet(config, device, precision=precision, backend=backend)
        if use_ritnet
        else None
    )
    if ritnet and not is_ort and getattr(ritnet.device, "type", None) != "cuda":
        raise RuntimeError("RITnet did not initialize on CUDA; refusing CPU fallback")

    timestamp_path, unix_by_frame = load_timestamp_map(video)
    if timestamp_path is None or not unix_by_frame:
        raise FileNotFoundError(
            f"Formal analysis requires the NIR timestamp CSV beside the video: {video}"
        )

    phases = _phase_names(args, config)
    configured_phases = [str(value) for value in config["formal"].get("phases", [])]
    is_full_phase_run = phases == configured_phases
    windows = resolve_phase_windows(
        video,
        unix_by_frame,
        phases,
        baseline_duration_sec=float(config["formal"].get("baseline_duration_sec", 180)),
        practice_trial_duration_ms=int(config["formal"].get("practice_trial_duration_ms", 1150)),
        timeline_path=Path(args.recovery_timeline).resolve()
        if getattr(args, "recovery_timeline", None)
        else None,
        timeline_source=(
            "reconstructed_task_timeline"
            if getattr(args, "recovery_timeline", None)
            else None
        ),
    )

    recovery_mode = bool(getattr(args, "recovery_timeline", None))

    yolo_batch_size = int(
        getattr(args, "yolo_batch_size", None)
        or config.get("inference", {}).get("yolo_batch_size", 1)
    )
    if yolo_batch_size < 1:
        raise ValueError("YOLO batch size must be positive")
    if yolo_batch_size > 1 and is_ort:
        raise ValueError("Experimental YOLO batching is available only for pytorch-cuda")

    expected_blocks = int(config["formal"].get("expected_formal_blocks", 2))
    requested_blocks = sorted(
        int(phase[5:]) for phase in phases if phase.startswith("block") and phase[5:].isdigit()
    )
    if requested_blocks and max(requested_blocks) > expected_blocks:
        raise ValueError(
            f"Configured FocusWave release expects {expected_blocks} formal blocks, "
            f"but phases request block{max(requested_blocks)}"
        )

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Validation-only path: precompute bounded YOLO batches, then let the
    # existing formal frame/ROI/RITnet/output path consume those detections.
    # This deliberately uses a second video read so downstream semantics remain
    # identical and the experimental results stay separate from formal output.
    batched_detections: dict[int, list[Detection]] | None = None
    if yolo_batch_size > 1:
        batched_detections = {}
        batch_cap = cv2.VideoCapture(str(video))
        if not batch_cap.isOpened():
            raise RuntimeError(f"Cannot open video for YOLO batch prepass: {video}")
        try:
            for window in windows:
                batch_frames: list[np.ndarray] = []
                batch_indices: list[int] = []
                batch_cap.set(cv2.CAP_PROP_POS_FRAMES, window.start_frame_idx)
                for frame_idx in range(window.start_frame_idx, window.end_frame_idx + 1):
                    ok, frame = batch_cap.read()
                    if not ok or frame is None:
                        raise RuntimeError(f"Video read failed during YOLO batch prepass at frame {frame_idx}")
                    batch_frames.append(frame)
                    batch_indices.append(frame_idx)
                    if len(batch_frames) >= yolo_batch_size:
                        rows = yolo_detect_batch(model, batch_frames, config, device)
                        batched_detections.update(zip(batch_indices, rows))
                        batch_frames.clear()
                        batch_indices.clear()
                if batch_frames:
                    rows = yolo_detect_batch(model, batch_frames, config, device)
                    batched_detections.update(zip(batch_indices, rows))
        finally:
            batch_cap.release()

    release = str(config["formal"].get("focuswave_release", "v3.1.3"))
    output_root = Path(args.output) if args.output else resolve_package_path(config["output"]["root"])
    suffixes: list[str] = []
    if is_ort:
        suffixes.append("ort-cuda")
    if not is_full_phase_run:
        suffixes.append("partial-" + "-".join(phases))
    if recovery_mode:
        suffixes.append("recovery")
    if args.max_frames is not None:
        suffixes.append(f"smoke{args.max_frames}")
    run_suffix = "_" + "_".join(suffixes) if suffixes else ""
    yolo_suffix = f"_yolo{yolo_batch_size}" if yolo_batch_size > 1 else ""
    run_name = f"{subject}_formal_{release}{yolo_suffix}_b{batch_size}_{precision}{run_suffix}"
    out = output_root / run_name
    overlays = out / "overlays"
    overlays.mkdir(parents=True, exist_ok=True)
    rois_path: Path | None = None
    if args.save_rois or config["output"]["save_rois"]:
        rois_path = out / "rois"
        rois_path.mkdir(parents=True, exist_ok=True)

    window_dicts = [window.to_dict() for window in windows]
    expected_keys = expected_frame_keys(window_dicts)
    started_at_utc = datetime.now(timezone.utc).isoformat()
    video_identity = str(video.resolve())
    run_identity = {
        "subject": subject,
        "video": video_identity,
        "package_version": str(config["package"]["version"]),
        "focuswave_release": release,
        "inference_backend": backend,
        "phases": phases,
        "ritnet_enabled": bool(ritnet),
        "ritnet_precision": ritnet.precision if ritnet else "disabled",
        "ritnet_batch_size": batch_size if ritnet else 0,
        "yolo_batch_size": yolo_batch_size,
        "max_frames": args.max_frames,
        "recovery_mode": recovery_mode,
        "timeline_file": (
            str(Path(args.recovery_timeline).resolve())
            if recovery_mode
            else str((video.parent.parent / "beh" / "master_timeline.csv").resolve())
        ),
    }
    run_id = hashlib.sha256(
        json.dumps(
            {"identity": run_identity, "phase_windows": window_dicts},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    completion_base = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "run_id": run_id,
        **run_identity,
        "expected_frames": len(expected_keys),
        "processed_frames": 0,
        "decoded_frames": 0,
        "video_read_failure_count": 0,
        "missing_expected_frame_count": len(expected_keys),
        "unexpected_frame_count": 0,
        "truncated_for_smoke_test": False,
        "partial_phase_selection": not is_full_phase_run,
        "recovery_mode": recovery_mode,
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "started_at_utc": started_at_utc,
        "finished_at_utc": None,
    }
    write_completion(out, {**completion_base, "status": "running"})

    (out / "phase_windows.json").write_text(
        json.dumps(window_dicts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    eye_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    frame_lookup: dict[tuple[str, int, int], dict[str, Any]] = {}
    overlay_pending: dict[tuple[str, int, int], tuple[np.ndarray, str]] = {}
    pending: list[dict[str, Any]] = []
    overlay_stride = int(config["output"]["overlay_stride"])
    analysis_size = (int(config["roi"]["width"]), int(config["roi"]["height"]))

    wall_started = time.perf_counter()
    read_failed = False
    stop_requested = False
    decoded_frames = 0

    try:
        for window in windows:
            if window.start_frame_idx < 0 or window.end_frame_idx >= total_frames:
                raise ValueError(
                    f"{window.phase} frame range {window.start_frame_idx}:{window.end_frame_idx} "
                    f"is outside video length {total_frames}"
                )

            cap.set(cv2.CAP_PROP_POS_FRAMES, window.start_frame_idx)
            for frame_idx in range(window.start_frame_idx, window.end_frame_idx + 1):
                decode_started = time.perf_counter()
                ok, frame = cap.read()
                decode_ms = (time.perf_counter() - decode_started) * 1000.0
                frame_key = (window.phase, window.segment, frame_idx)
                unix_ms = unix_by_frame.get(frame_idx)

                if not ok or frame is None:
                    frame_row = {
                        "subject": subject,
                        "video": str(video),
                        "phase": window.phase,
                        "phase_segment": window.segment,
                        "frame_idx": frame_idx,
                        "video_time_ms": float(cap.get(cv2.CAP_PROP_POS_MSEC)),
                        "unix_ms": unix_ms,
                        "phase_time_ms": (unix_ms - window.start_unix_ms) if unix_ms is not None else None,
                        "source": "video",
                        "redetect_reason": "not_applicable",
                        "status": "video_read_failed",
                        "raw_detection_count": 0,
                        "selected_eye_count": 0,
                        "decode_ms": decode_ms,
                        "yolo_ms": 0.0,
                        "roi_crop_ms": 0.0,
                        "ritnet_attributed_ms": 0.0,
                        "ritnet_batch_size_max": 0,
                        "overlay_write_ms": 0.0,
                    }
                    frame_rows.append(frame_row)
                    frame_lookup[frame_key] = frame_row
                    read_failed = True
                    break

                decoded_frames += 1

                yolo_started = time.perf_counter()
                detections = (
                    batched_detections[frame_idx]
                    if batched_detections is not None
                    else yolo_detect(model, frame, config, device)
                )
                yolo_ms = (time.perf_counter() - yolo_started) * 1000.0
                selected = sorted(
                    detections[:2],
                    key=lambda item: (item.box[0] + item.box[2]) / 2,
                )
                frame_status = _frame_status(detections)

                frame_row = {
                    "subject": subject,
                    "video": str(video),
                    "phase": window.phase,
                    "phase_segment": window.segment,
                    "frame_idx": frame_idx,
                    "video_time_ms": float(cap.get(cv2.CAP_PROP_POS_MSEC)),
                    "unix_ms": unix_ms,
                    "phase_time_ms": (unix_ms - window.start_unix_ms) if unix_ms is not None else None,
                    "source": "yolo",
                    "redetect_reason": "tracker_disabled",
                    "status": frame_status,
                    "raw_detection_count": len(detections),
                    "selected_eye_count": len(selected),
                    "decode_ms": decode_ms,
                    "yolo_ms": yolo_ms,
                    "roi_crop_ms": 0.0,
                    "ritnet_attributed_ms": 0.0,
                    "ritnet_batch_size_max": 0,
                    "overlay_write_ms": 0.0,
                }
                frame_rows.append(frame_row)
                frame_lookup[frame_key] = frame_row

                save_overlay = (
                    overlay_stride > 0
                    and (frame_idx - window.start_frame_idx) % overlay_stride == 0
                )
                if save_overlay:
                    overlay_pending[frame_key] = (frame.copy(), frame_status)

                # Keep the configured batch size as a real upper target. Flush the
                # previous complete frames before adding this frame if necessary.
                if pending and len(pending) + len(selected) > batch_size:
                    _flush_formal_batch(
                        pending,
                        ritnet=ritnet,
                        eye_rows=eye_rows,
                        frame_lookup=frame_lookup,
                        overlay_pending=overlay_pending,
                        overlays=overlays,
                        rois=rois_path,
                        analysis_size=analysis_size,
                    )

                for eye, detection in zip(("frame_left", "frame_right"), selected):
                    roi_started = time.perf_counter()
                    roi, crop_box, clipped = expand_crop_raw(
                        frame,
                        detection.box,
                        float(config["roi"]["expand_horizontal_each_side"]),
                        float(config["roi"]["expand_vertical_each_side"]),
                    )
                    roi_crop_ms = (time.perf_counter() - roi_started) * 1000.0
                    frame_row["roi_crop_ms"] += roi_crop_ms

                    x1, y1, x2, y2 = detection.box
                    rx1, ry1, rx2, ry2 = crop_box
                    base_row = {
                        "subject": subject,
                        "video": str(video),
                        "phase": window.phase,
                        "phase_segment": window.segment,
                        "frame_idx": frame_idx,
                        "video_time_ms": frame_row["video_time_ms"],
                        "unix_ms": unix_ms,
                        "phase_time_ms": frame_row["phase_time_ms"],
                        "eye": eye,
                        "source": "yolo",
                        "redetect_reason": "tracker_disabled",
                        "frame_status": frame_status,
                        "status": "pending",
                        "anchor_yolo_confidence": detection.confidence,
                        "bbox_x1": x1,
                        "bbox_y1": y1,
                        "bbox_x2": x2,
                        "bbox_y2": y2,
                        "roi_x1": rx1,
                        "roi_y1": ry1,
                        "roi_x2": rx2,
                        "roi_y2": ry2,
                        "roi_clipped": clipped,
                    }
                    pending.append(
                        {
                            "roi": roi,
                            "row": base_row,
                            "frame_key": frame_key,
                            "roi_crop_ms": roi_crop_ms,
                        }
                    )

                if not selected and frame_key in overlay_pending:
                    overlay_frame, overlay_status = overlay_pending.pop(frame_key)
                    overlay_started = time.perf_counter()
                    cv2.imwrite(
                        str(
                            overlays
                            / f"{window.phase}_s{window.segment}_f{frame_idx:08d}.jpg"
                        ),
                        draw_overlay(overlay_frame, [], overlay_status),
                    )
                    frame_row["overlay_write_ms"] += (
                        time.perf_counter() - overlay_started
                    ) * 1000.0

                # Flush after a complete frame so the two eyes of one frame are never
                # split across batches. With batch=16 this is normally 8 two-eye frames.
                if len(pending) >= batch_size:
                    _flush_formal_batch(
                        pending,
                        ritnet=ritnet,
                        eye_rows=eye_rows,
                        frame_lookup=frame_lookup,
                        overlay_pending=overlay_pending,
                        overlays=overlays,
                        rois=rois_path,
                        analysis_size=analysis_size,
                    )

                if args.max_frames and len(frame_rows) >= args.max_frames:
                    stop_requested = True
                    break

            _flush_formal_batch(
                pending,
                ritnet=ritnet,
                eye_rows=eye_rows,
                frame_lookup=frame_lookup,
                overlay_pending=overlay_pending,
                overlays=overlays,
                rois=rois_path,
                analysis_size=analysis_size,
            )
            if read_failed or stop_requested:
                break
    finally:
        cap.release()

    # Cost attribution across batched inference. This is not queue latency; overall
    # throughput is reported separately from wall-clock elapsed time.
    for row in frame_rows:
        row["frame_processing_ms"] = sum(
            float(row.get(name, 0.0) or 0.0)
            for name in (
                "decode_ms",
                "yolo_ms",
                "roi_crop_ms",
                "ritnet_attributed_ms",
                "overlay_write_ms",
            )
        )

    elapsed = time.perf_counter() - wall_started
    _write_csv(out / "eyes.csv", eye_rows)
    _write_csv(out / "frames.csv", frame_rows)

    phase_summary: dict[str, dict[str, Any]] = {}
    for phase in sorted({window.phase for window in windows}, key=phases.index):
        phase_frames = [row for row in frame_rows if row.get("phase") == phase]
        phase_eyes = [row for row in eye_rows if row.get("phase") == phase]
        phase_summary[phase] = {
            "processed_frames": len(phase_frames),
            "frame_status_counts": _counts(row.get("status") for row in phase_frames),
            "eye_status_counts": _counts(row.get("status") for row in phase_eyes),
        }

    summary = {
        "subject": subject,
        "video": video_identity,
        "mode": "recovery" if recovery_mode else "formal",
        "inference_backend": backend,
        "focuswave_release": release,
        "phases": phases,
        "processed_frames": len(frame_rows),
        "fps_source": fps,
        "elapsed_sec": elapsed,
        "processing_fps": len(frame_rows) / elapsed if elapsed else None,
        "tracker": "none",
        "ritnet_enabled": bool(ritnet),
        "ritnet_device": str(ritnet.device) if ritnet else "disabled",
        "ritnet_precision": ritnet.precision if ritnet else "disabled",
        "ritnet_batch_size": batch_size if ritnet else 0,
        "timestamp_file": str(timestamp_path),
        "frame_status_counts": _counts(row.get("status") for row in frame_rows),
        "eye_status_counts": _counts(row.get("status") for row in eye_rows),
        "phase_summary": phase_summary,
        "truncated_for_smoke_test": bool(args.max_frames is not None),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "run_manifest.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "command": " ".join(sys.argv),
                "package": config["package"],
                "config": config,
                "effective_parameters": {
                    "mode": "recovery" if recovery_mode else "formal",
                    "inference_backend": backend,
                    "focuswave_release": release,
                    "min_subject_number": minimum,
                    "phases": phases,
                    "tracker": "none",
                    "yolo_every_frame": True,
                    "ritnet_batch_size": batch_size if ritnet else 0,
                    "ritnet_precision": ritnet.precision if ritnet else "disabled",
                    "ritnet_analysis_size": list(analysis_size),
                    "ritnet_input_size": [
                        int(config["ritnet"]["input_width"]),
                        int(config["ritnet"]["input_height"]),
                    ],
                    "overlay_stride": overlay_stride,
                    "device": device,
                    "max_frames": args.max_frames,
                    "recovery_mode": recovery_mode,
                    "timeline_file": run_identity["timeline_file"],
                },
                "phase_windows": window_dicts,
                "python": sys.version,
                "platform": platform.platform(),
                "opencv": cv2.__version__,
                "yolo_sha256": sha256(model_path),
                "ritnet_sha256": sha256(ritnet_path),
                "ritnet_external_data_sha256": (
                    sha256(resolve_package_path(config["models"]["ritnet_onnx_external_data"]))
                    if is_ort
                    else None
                ),
                "cuda_available": _cuda_available(),
                "timestamp_file": str(timestamp_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    actual_keys = {
        (str(row["phase"]), int(row["phase_segment"]), int(row["frame_idx"]))
        for row in frame_rows
    }
    missing_expected = expected_keys - actual_keys
    unexpected = actual_keys - expected_keys
    failure_count = sum(row.get("status") == "video_read_failed" for row in frame_rows)
    artifact_complete = (
        not read_failed
        and not stop_requested
        and (is_full_phase_run or recovery_mode)
        and decoded_frames == len(expected_keys)
        and len(frame_rows) == len(expected_keys)
        and not missing_expected
        and not unexpected
        and failure_count == 0
    )
    final_payload = {
        **completion_base,
        "status": "running",
        "processed_frames": len(frame_rows),
        "decoded_frames": decoded_frames,
        "video_read_failure_count": failure_count,
        "missing_expected_frame_count": len(missing_expected),
        "unexpected_frame_count": len(unexpected),
        "truncated_for_smoke_test": bool(args.max_frames is not None),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    if read_failed:
        final_payload["status"] = "failed"
        write_completion(out, final_payload)
        print(json.dumps({"output": str(out.resolve()), **summary}, ensure_ascii=False, indent=2))
        return 3

    if args.max_frames is not None or (not is_full_phase_run and not recovery_mode):
        final_payload["status"] = "smoke_complete"
        write_completion(out, final_payload)
        print(json.dumps({"output": str(out.resolve()), **summary}, ensure_ascii=False, indent=2))
        return 0

    if not artifact_complete:
        final_payload["status"] = "failed"
        write_completion(out, final_payload)
        print(json.dumps({"output": str(out.resolve()), **summary}, ensure_ascii=False, indent=2))
        return 4

    write_completion(out, final_payload)
    preflight = validate_completion(out, run_identity, accepted_statuses=("running",))
    if not preflight.valid:
        final_payload["status"] = "failed"
        final_payload["validation_error"] = preflight.reason
        write_completion(out, final_payload)
        print(json.dumps({"output": str(out.resolve()), **summary}, ensure_ascii=False, indent=2))
        return 4

    final_payload["status"] = "recovery_complete" if recovery_mode else "complete"
    write_completion(out, final_payload)
    published = validate_completion(
        out,
        run_identity,
        accepted_statuses=("recovery_complete",) if recovery_mode else ("complete",),
    )
    if not published.valid:
        final_payload["status"] = "failed"
        final_payload["validation_error"] = published.reason
        write_completion(out, final_payload)
        print(json.dumps({"output": str(out.resolve()), **summary}, ensure_ascii=False, indent=2))
        return 4
    print(json.dumps({"output": str(out.resolve()), **summary}, ensure_ascii=False, indent=2))
    return 0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _counts(values) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key = str(value)
        result[key] = result.get(key, 0) + 1
    return result


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def check_environment(config: dict[str, Any]) -> int:
    import torch

    checks = {
        "python": sys.version,
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "yolo_model": str(resolve_package_path(config["models"]["yolo"])),
        "ritnet_model": str(resolve_package_path(config["models"]["ritnet"])),
        "yolo_onnx_model": str(resolve_package_path(config["models"]["yolo_onnx"])),
        "ritnet_onnx_model": str(resolve_package_path(config["models"]["ritnet_onnx"])),
        "formal_min_subject_number": int(config.get("formal", {}).get("min_subject_number", 31)),
        "formal_phases": config.get("formal", {}).get("phases", []),
        "ritnet_default_precision": config.get("ritnet", {}).get("precision", "fp32"),
        "ritnet_default_batch_size": int(config.get("ritnet", {}).get("batch_size", 16)),
    }
    for method in ("csrt", "kcf"):
        try:
            tracker_factory(method)
            checks[f"tracker_{method}"] = True
        except Exception as exc:
            checks[f"tracker_{method}"] = str(exc)
    checks["models_exist"] = all(
        resolve_package_path(config["models"][key]).exists() for key in ("yolo", "ritnet")
    )
    checks["onnx_models_exist"] = all(
        resolve_package_path(config["models"][key]).exists()
        for key in ("yolo_onnx", "ritnet_onnx", "ritnet_onnx_external_data")
    )
    try:
        from onnx_cuda_runtime import _import_onnxruntime

        ort = _import_onnxruntime()
        checks["onnxruntime"] = ort.__version__
        checks["onnxruntime_providers"] = list(ort.get_available_providers())
    except Exception as exc:
        checks["onnxruntime"] = str(exc)
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if not checks["models_exist"]:
        return 2
    return 0 if checks["cuda_available"] else 3


def _add_common_target_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--subject")
    parser.add_argument("--root")
    parser.add_argument("--video")
    parser.add_argument("--device", default="0")
    parser.add_argument("--skip-ritnet", action="store_true")
    parser.add_argument("--save-rois", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--ritnet-precision", choices=("fp32", "fp16"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-env")
    discover_parser = sub.add_parser("discover")
    discover_parser.add_argument("--formal-only", action="store_true")

    run_parser = sub.add_parser("run")
    _add_common_target_args(run_parser)
    run_parser.add_argument("--start-sec", type=float, default=0.0)
    run_parser.add_argument("--duration-sec", type=float)
    run_parser.add_argument("--full-video", action="store_true")
    run_parser.add_argument("--tracker", choices=("none", "csrt", "kcf"))
    run_parser.add_argument("--redetect-interval", type=int)

    formal_parser = sub.add_parser("formal")
    _add_common_target_args(formal_parser)
    formal_parser.add_argument("--backend", choices=("pytorch-cuda", "ort-cuda"))
    formal_parser.add_argument("--ritnet-batch-size", type=int)
    formal_parser.add_argument(
        "--yolo-batch-size",
        type=int,
        default=None,
        help="YOLO prepass batch size; defaults to config.inference.yolo_batch_size",
    )
    formal_parser.add_argument(
        "--max-frames",
        type=int,
        help="Smoke-test limit; output is never accepted as a complete formal run",
    )
    formal_parser.add_argument(
        "--phases",
        help="Comma-separated phases; default comes from config.yaml",
    )
    formal_parser.add_argument(
        "--recovery-timeline",
        type=Path,
        help=(
            "External reconstructed task timeline for an isolated recovery run. "
            "The result is marked recovery_complete and is never treated as formal complete."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.command == "check-env":
        return check_environment(config)
    if args.command == "discover":
        minimum = (
            int(config["formal"].get("min_subject_number", 31))
            if args.formal_only
            else None
        )
        rows = discover_videos(config["data"]["roots"], minimum)
        print(json.dumps({"count": len(rows), "videos": rows}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "formal":
        return formal(args, config)
    return run(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
