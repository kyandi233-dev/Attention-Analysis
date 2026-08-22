"""Portable YOLO -> tracking -> eye ROI -> RITnet short-video trial.

This is an admission/diagnostic runner. It deliberately defaults to 60 seconds
and requires ``--full-video`` for an unrestricted run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from ritnet_runtime import RitnetRuntime


PACKAGE_ROOT = Path(__file__).resolve().parent


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


def discover_videos(roots: list[str]) -> list[dict[str, str]]:
    rows = []
    for root_text in roots:
        root = Path(root_text)
        if not root.exists():
            continue
        for video in sorted(root.glob("sub-*_/nir/*_nir.avi")):
            rows.append({"root": str(root), "subject": video.parents[1].name.rstrip("_"), "video": str(video)})
    return rows


def resolve_video(config: dict[str, Any], subject: str | None, root: str | None, video: str | None) -> tuple[str, Path]:
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


def center_jump_ok(previous: tuple[float, float, float, float], current: tuple[float, float, float, float], fraction: float) -> bool:
    px1, py1, px2, py2 = previous
    cx1, cy1, cx2, cy2 = current
    previous_center = np.array([(px1 + px2) / 2, (py1 + py2) / 2])
    current_center = np.array([(cx1 + cx2) / 2, (cy1 + cy2) / 2])
    return float(np.linalg.norm(current_center - previous_center)) <= max(1.0, px2 - px1) * fraction


def expand_crop(frame: np.ndarray, box: tuple[float, float, float, float], horizontal: float, vertical: float,
                output_size: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int, int, int], bool]:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    raw = (x1 - horizontal * width, y1 - vertical * height, x2 + horizontal * width, y2 + vertical * height)
    frame_h, frame_w = frame.shape[:2]
    clipped = raw[0] < 0 or raw[1] < 0 or raw[2] > frame_w or raw[3] > frame_h
    crop_box = (max(0, int(np.floor(raw[0]))), max(0, int(np.floor(raw[1]))),
                min(frame_w, int(np.ceil(raw[2]))), min(frame_h, int(np.ceil(raw[3]))))
    rx1, ry1, rx2, ry2 = crop_box
    if rx2 <= rx1 or ry2 <= ry1:
        raise ValueError("Empty expanded ROI")
    gray = cv2.cvtColor(frame[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, output_size), crop_box, clipped


def yolo_detect(model: YOLO, frame: np.ndarray, config: dict[str, Any], device: str) -> list[Detection]:
    cfg = config["yolo"]
    result = model.predict(frame, conf=float(cfg["confidence"]), imgsz=int(cfg["imgsz"]),
                           iou=float(cfg["nms_iou"]), max_det=int(cfg["max_det"]),
                           device=device, verbose=False)[0]
    if result.boxes is None:
        return []
    detections = [Detection(tuple(map(float, box)), float(conf))
                  for box, conf, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                                            result.boxes.cls.cpu().numpy()) if int(cls) == 0]
    return sorted(detections, key=lambda item: item.confidence, reverse=True)


def init_trackers(frame: np.ndarray, boxes: list[Detection], creator) -> list[Any]:
    if creator is None or len(boxes) < 2:
        return []
    trackers = []
    for detection in boxes[:2]:
        tracker = creator()
        tracker.init(frame, xyxy_to_xywh(detection.box))
        trackers.append(tracker)
    return trackers


def update_trackers(frame: np.ndarray, trackers: list[Any], previous: list[Detection], jump_fraction: float) -> list[Detection] | None:
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
        x1, y1, x2, y2 = [int(round(row[key])) for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")]
        color = (0, 220, 0) if row["status"] == "observed" else (0, 180, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output, f"{row['eye']} {row['source']} {row['status']}", (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    cv2.putText(output, frame_status, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 0), 2)
    return output


def run(args: argparse.Namespace, config: dict[str, Any]) -> int:
    subject, video = resolve_video(config, args.subject, args.root, args.video)
    model_path = resolve_package_path(config["models"]["yolo"])
    ritnet_path = resolve_package_path(config["models"]["ritnet"])
    device = args.device
    model = YOLO(str(model_path))
    if model.names != {0: "eye"}:
        raise ValueError(f"Unexpected YOLO classes: {model.names}")
    use_ritnet = bool(config["ritnet"]["enabled"]) and not args.skip_ritnet
    ritnet = RitnetRuntime(PACKAGE_ROOT, ritnet_path,
                           (int(config["ritnet"]["input_width"]), int(config["ritnet"]["input_height"])),
                           device=device) if use_ritnet else None
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
            tracked = update_trackers(frame, trackers, selected,
                                      float(config["tracking"]["center_jump_width_fraction"]))
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
                frame, detection.box,
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
                "subject": subject, "video": str(video), "frame_idx": frame_idx,
                "video_time_ms": float(cap.get(cv2.CAP_PROP_POS_MSEC)), "unix_ms": unix_by_frame.get(frame_idx),
                "eye": eye, "source": source, "redetect_reason": redetect_reason,
                "frame_status": frame_status, "status": status, "anchor_yolo_confidence": detection.confidence,
                "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                "roi_x1": rx1, "roi_y1": ry1, "roi_x2": rx2, "roi_y2": ry2, "roi_clipped": clipped,
                "ritnet_found": bool(pupil.get("found")), "ritnet_device": str(ritnet.device) if ritnet else "disabled",
                "pupil_center_x": pupil.get("center_x"), "pupil_center_y": pupil.get("center_y"),
                "pupil_axis_a": pupil.get("axis_a"), "pupil_axis_b": pupil.get("axis_b"),
                "pupil_angle_deg": pupil.get("angle_deg"), "pupil_equiv_diameter": pupil.get("equiv_diameter"),
                "pupil_confidence": pupil.get("pupil_confidence"),
                "eye_processing_ms": (time.perf_counter() - roi_start) * 1000,
            }
            eye_rows.append(row)
            current_rows.append(row)
            if rois.exists():
                cv2.imwrite(str(rois / f"f{frame_idx:08d}_{eye}.png"), roi)
        frame_rows.append({"subject": subject, "video": str(video), "frame_idx": frame_idx,
                           "video_time_ms": float(cap.get(cv2.CAP_PROP_POS_MSEC)), "unix_ms": unix_by_frame.get(frame_idx),
                           "source": source, "redetect_reason": redetect_reason,
                           "status": frame_status, "raw_detection_count": len(detections),
                           "selected_eye_count": len(selected), "frame_processing_ms": (time.perf_counter() - frame_start) * 1000})
        if overlay_stride > 0 and (frame_idx - start_frame) % overlay_stride == 0:
            cv2.imwrite(str(overlays / f"f{frame_idx:08d}.jpg"), draw_overlay(frame, current_rows, frame_status))
        frame_idx += 1
    cap.release()
    elapsed = time.perf_counter() - wall_start
    _write_csv(out / "eyes.csv", eye_rows)
    _write_csv(out / "frames.csv", frame_rows)
    summary = {
        "subject": subject, "video": str(video), "start_frame": start_frame, "requested_end_frame": end_frame,
        "processed_frames": len(frame_rows), "fps_source": fps, "elapsed_sec": elapsed,
        "processing_fps": len(frame_rows) / elapsed if elapsed else None, "tracker": tracker_method,
        "redetect_interval": redetect_interval, "ritnet_enabled": bool(ritnet),
        "ritnet_device": str(ritnet.device) if ritnet else "disabled",
        "timestamp_file": str(timestamp_path) if timestamp_path else None,
        "frame_status_counts": _counts(row.get("status") for row in frame_rows),
        "eye_status_counts": _counts(row.get("status") for row in eye_rows),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "run_manifest.json").write_text(json.dumps({
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "command": " ".join(sys.argv),
        "package": config["package"], "config": config, "python": sys.version, "platform": platform.platform(),
        "opencv": cv2.__version__, "yolo_sha256": sha256(model_path),
        "ritnet_sha256": sha256(ritnet_path), "cuda_available": _cuda_available(),
        "timestamp_file": str(timestamp_path) if timestamp_path else None,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
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
        "python": sys.version, "platform": platform.platform(), "opencv": cv2.__version__,
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "yolo_model": str(resolve_package_path(config["models"]["yolo"])),
        "ritnet_model": str(resolve_package_path(config["models"]["ritnet"])),
    }
    for method in ("csrt", "kcf"):
        try:
            tracker_factory(method)
            checks[f"tracker_{method}"] = True
        except Exception as exc:
            checks[f"tracker_{method}"] = str(exc)
    checks["models_exist"] = all(resolve_package_path(config["models"][key]).exists() for key in ("yolo", "ritnet"))
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 0 if checks["models_exist"] else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-env")
    sub.add_parser("discover")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--subject")
    run_parser.add_argument("--root")
    run_parser.add_argument("--video")
    run_parser.add_argument("--start-sec", type=float, default=0.0)
    run_parser.add_argument("--duration-sec", type=float)
    run_parser.add_argument("--full-video", action="store_true")
    run_parser.add_argument("--tracker", choices=("none", "csrt", "kcf"))
    run_parser.add_argument("--redetect-interval", type=int)
    run_parser.add_argument("--device", default="0")
    run_parser.add_argument("--skip-ritnet", action="store_true")
    run_parser.add_argument("--save-rois", action="store_true")
    run_parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.command == "check-env":
        return check_environment(config)
    if args.command == "discover":
        rows = discover_videos(config["data"]["roots"])
        print(json.dumps({"count": len(rows), "videos": rows}, ensure_ascii=False, indent=2))
        return 0
    return run(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
