"""Production AMD/DirectML formal runner with batched YOLO.

The validated production combination is YOLO fixed batch 8 plus RITnet fixed
batch 16. This runner reuses the stable phase, ROI, QC, output, and completion
contracts from ``run_pipeline.py`` while batching YOLO frames explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from directml_runtime import YoloDirectMLRuntime
from formal_completion import (
    REQUIRED_ARTIFACTS,
    SCHEMA_VERSION as COMPLETION_SCHEMA_VERSION,
    expected_frame_keys,
    validate_completion,
    write_completion,
)
from phase_windows import resolve_phase_windows
from run_pipeline import (
    PACKAGE_ROOT,
    Detection,
    _counts,
    _flush_formal_batch,
    _frame_status,
    _make_ritnet,
    _phase_names,
    _validate_amd_settings,
    _write_csv,
    draw_overlay,
    ensure_amd_output_root,
    expand_crop_raw,
    load_config,
    load_timestamp_map,
    resolve_package_path,
    resolve_video,
    sha256,
    subject_number,
    valid_box,
)


def _make_formal_yolo(config: dict[str, Any], device: str) -> YoloDirectMLRuntime:
    value = config.get("models", {}).get("yolo_formal")
    if not value:
        raise KeyError("config models.yolo_formal is required for batched formal analysis")
    runtime = YoloDirectMLRuntime(resolve_package_path(value), device=device)
    configured = int(config.get("yolo", {}).get("batch_size", 8))
    if runtime.batch_size != configured:
        raise ValueError(
            f"YOLO formal model batch={runtime.batch_size} but config yolo.batch_size={configured}"
        )
    return runtime


def _convert_detections(
    raw: list[tuple[tuple[float, float, float, float], float, int]],
    frame,
) -> list[Detection]:
    detections = [
        Detection(box, confidence)
        for box, confidence, class_id in raw
        if class_id == 0 and valid_box(box, frame.shape)
    ]
    return sorted(detections, key=lambda item: item.confidence, reverse=True)


def formal(args: argparse.Namespace, config: dict[str, Any]) -> int:
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")

    subject, video = resolve_video(config, args.subject, args.root, args.video)
    minimum = int(config["formal"].get("min_subject_number", 31))
    if subject_number(subject) < minimum:
        raise ValueError(
            f"Formal v3.1.3 analysis is configured for sub-{minimum:03d} and later; got {subject}."
        )

    device = args.device
    yolo_path = resolve_package_path(config["models"]["yolo_formal"])
    ritnet_path = resolve_package_path(config["models"]["ritnet"])
    model = _make_formal_yolo(config, device)
    yolo_batch_size = model.batch_size

    use_ritnet = bool(config["ritnet"]["enabled"]) and not args.skip_ritnet
    precision = args.ritnet_precision or str(config["ritnet"].get("precision", "fp32"))
    ritnet_batch_size = args.ritnet_batch_size or int(config["ritnet"].get("batch_size", 16))
    _validate_amd_settings(precision, ritnet_batch_size)
    ritnet = _make_ritnet(config, device, precision=precision) if use_ritnet else None

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
    )

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

    release = str(config["formal"].get("focuswave_release", "v3.1.3"))
    output_root = ensure_amd_output_root(
        Path(args.output) if args.output else resolve_package_path(config["output"]["root"])
    )
    suffixes: list[str] = []
    if not is_full_phase_run:
        suffixes.append("partial-" + "-".join(phases))
    if args.max_frames is not None:
        suffixes.append(f"smoke{args.max_frames}")
    run_suffix = "_" + "_".join(suffixes) if suffixes else ""
    run_name = (
        f"{subject}_formal_{release}_yolo-b{yolo_batch_size}_"
        f"ritnet-b{ritnet_batch_size}_{precision}{run_suffix}"
    )
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
    yolo_hash = sha256(yolo_path)
    ritnet_hash = sha256(ritnet_path)
    run_identity = {
        "subject": subject,
        "video": video_identity,
        "package_version": str(config["package"]["version"]),
        "focuswave_release": release,
        "phases": phases,
        "yolo_batch_size": yolo_batch_size,
        "yolo_model_sha256": yolo_hash,
        "ritnet_enabled": bool(ritnet),
        "ritnet_precision": ritnet.precision if ritnet else "disabled",
        "ritnet_batch_size": ritnet_batch_size if ritnet else 0,
        "ritnet_model_sha256": ritnet_hash,
        "max_frames": args.max_frames,
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
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "started_at_utc": started_at_utc,
        "finished_at_utc": None,
    }
    write_completion(out, {**completion_base, "status": "running"})
    (out / "phase_windows.json").write_text(
        json.dumps(window_dicts, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    eye_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    frame_lookup: dict[tuple[str, int, int], dict[str, Any]] = {}
    overlay_pending: dict[tuple[str, int, int], tuple[Any, str]] = {}
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
            next_frame_idx = window.start_frame_idx
            while next_frame_idx <= window.end_frame_idx:
                if args.max_frames is not None:
                    remaining = int(args.max_frames) - len(frame_rows)
                    if remaining <= 0:
                        stop_requested = True
                        break
                    target = min(yolo_batch_size, remaining)
                else:
                    target = yolo_batch_size
                target = min(target, window.end_frame_idx - next_frame_idx + 1)

                batch_items: list[dict[str, Any]] = []
                for _ in range(target):
                    frame_idx = next_frame_idx
                    decode_started = time.perf_counter()
                    ok, frame = cap.read()
                    decode_ms = (time.perf_counter() - decode_started) * 1000.0
                    unix_ms = unix_by_frame.get(frame_idx)
                    video_time_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                    next_frame_idx += 1

                    if not ok or frame is None:
                        frame_key = (window.phase, window.segment, frame_idx)
                        frame_row = {
                            "subject": subject,
                            "video": str(video),
                            "phase": window.phase,
                            "phase_segment": window.segment,
                            "frame_idx": frame_idx,
                            "video_time_ms": video_time_ms,
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
                    batch_items.append(
                        {
                            "frame_idx": frame_idx,
                            "frame": frame,
                            "decode_ms": decode_ms,
                            "unix_ms": unix_ms,
                            "video_time_ms": video_time_ms,
                        }
                    )

                if read_failed:
                    break
                if not batch_items:
                    break

                yolo_started = time.perf_counter()
                raw_batches = model.detect_batch(
                    [item["frame"] for item in batch_items],
                    confidence=float(config["yolo"]["confidence"]),
                    max_det=int(config["yolo"]["max_det"]),
                )
                yolo_elapsed_ms = (time.perf_counter() - yolo_started) * 1000.0
                yolo_share_ms = yolo_elapsed_ms / len(batch_items)

                for item, raw in zip(batch_items, raw_batches):
                    frame_idx = int(item["frame_idx"])
                    frame = item["frame"]
                    unix_ms = item["unix_ms"]
                    frame_key = (window.phase, window.segment, frame_idx)
                    detections = _convert_detections(raw, frame)
                    selected = sorted(
                        detections[:2], key=lambda det: (det.box[0] + det.box[2]) / 2
                    )
                    frame_status = _frame_status(detections)
                    frame_row = {
                        "subject": subject,
                        "video": str(video),
                        "phase": window.phase,
                        "phase_segment": window.segment,
                        "frame_idx": frame_idx,
                        "video_time_ms": item["video_time_ms"],
                        "unix_ms": unix_ms,
                        "phase_time_ms": (unix_ms - window.start_unix_ms) if unix_ms is not None else None,
                        "source": "yolo",
                        "redetect_reason": "tracker_disabled",
                        "status": frame_status,
                        "raw_detection_count": len(detections),
                        "selected_eye_count": len(selected),
                        "decode_ms": float(item["decode_ms"]),
                        "yolo_ms": yolo_share_ms,
                        "yolo_batch_size": yolo_batch_size,
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

                    if pending and len(pending) + len(selected) > ritnet_batch_size:
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
                            "yolo_batch_size": yolo_batch_size,
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
                            str(overlays / f"{window.phase}_s{window.segment}_f{frame_idx:08d}.jpg"),
                            draw_overlay(overlay_frame, [], overlay_status),
                        )
                        frame_row["overlay_write_ms"] += (
                            time.perf_counter() - overlay_started
                        ) * 1000.0

                    if len(pending) >= ritnet_batch_size:
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
        "mode": "formal",
        "focuswave_release": release,
        "phases": phases,
        "processed_frames": len(frame_rows),
        "fps_source": fps,
        "elapsed_sec": elapsed,
        "processing_fps": len(frame_rows) / elapsed if elapsed else None,
        "tracker": "none",
        "yolo_batch_size": yolo_batch_size,
        "ritnet_enabled": bool(ritnet),
        "ritnet_device": str(ritnet.device) if ritnet else "disabled",
        "ritnet_precision": ritnet.precision if ritnet else "disabled",
        "ritnet_batch_size": ritnet_batch_size if ritnet else 0,
        "timestamp_file": str(timestamp_path),
        "frame_status_counts": _counts(row.get("status") for row in frame_rows),
        "eye_status_counts": _counts(row.get("status") for row in eye_rows),
        "phase_summary": phase_summary,
        "truncated_for_smoke_test": bool(args.max_frames is not None),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "run_manifest.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "command": " ".join(sys.argv),
                "package": config["package"],
                "config": config,
                "effective_parameters": {
                    "mode": "formal",
                    "focuswave_release": release,
                    "min_subject_number": minimum,
                    "phases": phases,
                    "tracker": "none",
                    "yolo_every_frame": True,
                    "yolo_batch_size": yolo_batch_size,
                    "yolo_model": str(yolo_path),
                    "ritnet_batch_size": ritnet_batch_size if ritnet else 0,
                    "ritnet_precision": ritnet.precision if ritnet else "disabled",
                    "ritnet_analysis_size": list(analysis_size),
                    "ritnet_input_size": [
                        int(config["ritnet"]["input_width"]),
                        int(config["ritnet"]["input_height"]),
                    ],
                    "overlay_stride": overlay_stride,
                    "device": device,
                    "max_frames": args.max_frames,
                },
                "phase_windows": window_dicts,
                "python": sys.version,
                "platform": platform.platform(),
                "opencv": cv2.__version__,
                "yolo_sha256": yolo_hash,
                "ritnet_sha256": ritnet_hash,
                "ritnet_external_data_sha256": sha256(
                    resolve_package_path(config["models"]["ritnet_external_data"])
                ),
                "directml_providers": model.providers,
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
        and is_full_phase_run
        and decoded_frames == len(expected_keys)
        and len(frame_rows) == len(expected_keys)
        and not missing_expected
        and not unexpected
        and failure_count == 0
    )
    finished_at_utc = datetime.now(timezone.utc).isoformat()
    final_payload = {
        **completion_base,
        "status": "running",
        "processed_frames": len(frame_rows),
        "decoded_frames": decoded_frames,
        "video_read_failure_count": failure_count,
        "missing_expected_frame_count": len(missing_expected),
        "unexpected_frame_count": len(unexpected),
        "truncated_for_smoke_test": bool(args.max_frames is not None),
        "finished_at_utc": finished_at_utc,
    }

    if read_failed:
        final_payload["status"] = "failed"
        write_completion(out, final_payload)
        print(json.dumps({"output": str(out.resolve()), **summary}, ensure_ascii=False, indent=2))
        return 3

    if args.max_frames is not None or not is_full_phase_run:
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

    final_payload["status"] = "complete"
    write_completion(out, final_payload)
    published = validate_completion(out, run_identity)
    if not published.valid:
        final_payload["status"] = "failed"
        final_payload["validation_error"] = published.reason
        write_completion(out, final_payload)
        print(json.dumps({"output": str(out.resolve()), **summary}, ensure_ascii=False, indent=2))
        return 4

    print(json.dumps({"output": str(out.resolve()), **summary}, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AMD DirectML batched formal NIR runner")
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config.yaml")
    parser.add_argument("--subject")
    parser.add_argument("--root")
    parser.add_argument("--video")
    parser.add_argument("--device", default="0")
    parser.add_argument("--skip-ritnet", action="store_true")
    parser.add_argument("--save-rois", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--ritnet-precision", choices=("fp32",))
    parser.add_argument("--ritnet-batch-size", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--phases")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return formal(args, load_config(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
