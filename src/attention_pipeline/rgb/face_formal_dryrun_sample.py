from __future__ import annotations

import json
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from attention_pipeline.config import Config
from attention_pipeline.rgb.audit import read_rgb_timestamps, video_metadata
from attention_pipeline.rgb.behavior import BehaviorIndex, empty_behavior_context
from attention_pipeline.rgb.face_benchmark import _configured_exclusion, _find_subject
from attention_pipeline.rgb.face_continuous import _nearest_position
from attention_pipeline.rgb.paths import RGBOutputLayout
from attention_pipeline.rgb.timeline import detailed_rgb_intervals


FACE_FORMAL_DRYRUN_SAMPLE_SCHEMA = "rgb-face-formal-dryrun-sample-v0.1"


def _dryrun_dir(layout: RGBOutputLayout, subject: str) -> Path:
    path = layout.test_dir() / "face-formal-dryrun" / subject
    path.mkdir(parents=True, exist_ok=True)
    return path


def _block_from_phase(phase: str) -> int | None:
    if phase.startswith("block") and phase[5:].isdigit():
        return int(phase[5:])
    return None


def _window_bounds(interval, duration_sec: float, anchor: str) -> tuple[int, int]:
    duration_ms = int(round(duration_sec * 1000.0))
    available = int(interval.end_unix_ms - interval.start_unix_ms)
    if duration_ms > available:
        raise ValueError(
            f"Dry-run window {duration_sec}s exceeds phase {interval.phase} duration {available / 1000:.3f}s"
        )
    if anchor == "start":
        start = int(interval.start_unix_ms)
    elif anchor == "end":
        start = int(interval.end_unix_ms - duration_ms)
    elif anchor == "middle":
        start = int(interval.start_unix_ms + (available - duration_ms) // 2)
    else:
        raise ValueError(f"Unsupported dry-run anchor: {anchor}")
    return start, int(start + duration_ms)


def run_face_formal_dryrun_sample(config: Config, subject: str) -> dict[str, object]:
    """Extract continuous 15 Hz representative windows without Face inference.

    The dry-run intentionally stresses both stable task periods and phases where
    another person may enter the camera view. It is test-only and never replaces
    the future direct-AVI formal runner.
    """
    excluded, reason = _configured_exclusion(config, subject)
    if excluded:
        raise ValueError(f"Subject {subject} is excluded from RGB analysis: {reason}")

    files = _find_subject(config, subject)
    timestamps = read_rgb_timestamps(files.timestamps)
    metadata = video_metadata(files.video)
    if not metadata["video_open_ok"]:
        raise RuntimeError(f"RGB video cannot be opened: {files.video}")
    if int(metadata["video_frame_count_nominal"]) != len(timestamps):
        raise ValueError(f"AVI/timestamp row mismatch for {subject}")

    face_cfg = config.section("face")
    dry_cfg = face_cfg.get("formal_dryrun", {})
    if not isinstance(dry_cfg, dict):
        dry_cfg = {}
    inference_fps = float(dry_cfg.get("inference_fps", face_cfg.get("inference_fps", 15.0) or 15.0))
    jpeg_quality = int(dry_cfg.get("jpeg_quality", 95))
    windows = dry_cfg.get(
        "windows",
        [
            {"name": "baseline_start", "phase": "baseline", "anchor": "start", "duration_sec": 30},
            {"name": "baseline_end", "phase": "baseline", "anchor": "end", "duration_sec": 30},
            {"name": "block1_middle", "phase": "block1", "anchor": "middle", "duration_sec": 60},
            {"name": "interblock_middle", "phase": "interblock_transition", "anchor": "middle", "duration_sec": 60},
            {"name": "block2_middle", "phase": "block2", "anchor": "middle", "duration_sec": 60},
        ],
    )
    if inference_fps <= 0:
        raise ValueError("face.formal_dryrun.inference_fps must be > 0")
    if float(metadata.get("video_fps_nominal") or 0) + 1e-9 < inference_fps:
        raise ValueError(
            f"Requested Face dry-run {inference_fps} Hz exceeds nominal source fps {metadata.get('video_fps_nominal')}"
        )
    if not isinstance(windows, list) or not windows:
        raise ValueError("face.formal_dryrun.windows must be a non-empty list")

    focuswave = config.section("focuswave")
    baseline_duration_sec = float(focuswave.get("baseline_duration_sec", 180))
    expected_blocks = int(focuswave.get("expected_blocks", 2))
    trial_duration_ms = int(focuswave.get("trial_duration_ms", 1150))
    intervals = detailed_rgb_intervals(
        files.master_timeline,
        baseline_duration_sec=baseline_duration_sec,
        expected_blocks=expected_blocks,
    )
    by_phase = {interval.phase: interval for interval in intervals}
    all_times = [int(row[1]) for row in timestamps]
    step_ms = 1000.0 / inference_fps

    selected: dict[int, dict[str, object]] = {}
    window_summaries: list[dict[str, object]] = []
    for spec in windows:
        if not isinstance(spec, dict):
            raise ValueError(f"Invalid dry-run window spec: {spec!r}")
        name = str(spec.get("name") or spec.get("phase"))
        phase = str(spec["phase"])
        anchor = str(spec.get("anchor", "middle"))
        duration_sec = float(spec.get("duration_sec", 60.0))
        if phase not in by_phase:
            raise RuntimeError(f"Dry-run phase {phase} not found for {subject}")
        interval = by_phase[phase]
        window_start, window_end = _window_bounds(interval, duration_sec, anchor)
        lo = bisect_left(all_times, int(interval.start_unix_ms))
        hi = bisect_left(all_times, int(interval.end_unix_ms)) - 1
        targets = [
            int(round(window_start + i * step_ms))
            for i in range(int(round(duration_sec * inference_fps)))
        ]
        positions: list[int] = []
        for target in targets:
            pos = _nearest_position(all_times, target, lo, hi)
            positions.append(pos)
            existing = selected.get(pos)
            payload = {
                "target_unix_ms": target,
                "dryrun_window": name,
                "dryrun_phase": phase,
                "dryrun_anchor": anchor,
            }
            if existing is None or abs(all_times[pos] - target) < abs(all_times[pos] - int(existing["target_unix_ms"])):
                selected[pos] = payload
        window_summaries.append(
            {
                "name": name,
                "phase": phase,
                "anchor": anchor,
                "duration_sec": duration_sec,
                "window_start_unix_ms": window_start,
                "window_end_unix_ms": window_end,
                "target_frames": len(targets),
                "unique_source_positions": len(set(positions)),
            }
        )

    selected_positions = sorted(selected)
    if not selected_positions:
        raise RuntimeError("No Face formal dry-run frames selected")

    behavior_indexes = {
        1: BehaviorIndex.from_csv(files.block1_behavior),
        2: BehaviorIndex.from_csv(files.block2_behavior),
    }
    layout = RGBOutputLayout.from_config(config)
    root = _dryrun_dir(layout, subject)
    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = root / f"{subject}_face-dryrun_frames.csv"
    manifest_json = root / f"{subject}_face-dryrun_manifest.json"

    records: list[dict[str, object]] = []
    selected_set = set(selected_positions)
    cap = cv2.VideoCapture(str(files.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open RGB video: {files.video}")
    first_pos, last_pos = selected_positions[0], selected_positions[-1]
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(first_pos))
    try:
        for pos in range(first_pos, last_pos + 1):
            if not cap.grab():
                raise RuntimeError(f"Failed to advance RGB video at frame {pos}")
            if pos not in selected_set:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                raise RuntimeError(f"Failed to decode selected RGB frame {pos}")
            capture_idx, unix_ms = timestamps[pos]
            meta = selected[pos]
            phase = str(meta["dryrun_phase"])
            block = _block_from_phase(phase)
            behavior = empty_behavior_context()
            if block is not None:
                behavior = behavior_indexes[block].context_at(int(unix_ms), trial_duration_ms=trial_duration_ms)
            filename = f"{subject}_f{pos:08d}_t{int(unix_ms)}.jpg"
            image_path = frames_dir / filename
            # OpenCV's Windows imwrite may reject non-ASCII paths. Encode in memory
            # and write bytes so the formal output root can remain the project path.
            encoded_ok, encoded = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            )
            if not encoded_ok:
                raise RuntimeError(f"Failed to write dry-run frame: {image_path}")
            image_path.write_bytes(encoded.tobytes())
            row: dict[str, object] = {
                "schema_version": FACE_FORMAL_DRYRUN_SAMPLE_SCHEMA,
                "subject": subject,
                "video_frame_position": int(pos),
                "capture_frame_idx": int(capture_idx),
                "unix_ms": int(unix_ms),
                "target_unix_ms": int(meta["target_unix_ms"]),
                "sample_error_ms": int(unix_ms) - int(meta["target_unix_ms"]),
                "phase": phase,
                "block": block,
                "dryrun_window": str(meta["dryrun_window"]),
                "image_path": str(image_path),
                "jpeg_quality": jpeg_quality,
            }
            row.update(behavior)
            records.append(row)
    finally:
        cap.release()

    table = pd.DataFrame(records).sort_values("unix_ms").reset_index(drop=True)
    table.insert(0, "benchmark_index", np.arange(len(table), dtype=int))
    table["dt_ms"] = pd.to_numeric(table["unix_ms"], errors="coerce").diff()
    table["capture_gap_before"] = pd.to_numeric(table["capture_frame_idx"], errors="coerce").diff().fillna(1) > 3
    table["temporal_gap"] = table["dt_ms"] > max(250.0, step_ms * 2.5)
    table.to_csv(manifest_csv, index=False, encoding="utf-8-sig")

    summary = {
        "schema_version": FACE_FORMAL_DRYRUN_SAMPLE_SCHEMA,
        "stage": "face-formal-dryrun-sample",
        "subject": subject,
        "purpose": "15 Hz representative continuous windows for formal Face tracking/eyelid/schema dry-run",
        "requested_inference_fps": inference_fps,
        "source_video_fps_nominal": metadata.get("video_fps_nominal"),
        "selected_frames": int(len(table)),
        "median_dt_ms": float(table["dt_ms"].dropna().median()) if table["dt_ms"].notna().any() else None,
        "max_dt_ms": float(table["dt_ms"].dropna().max()) if table["dt_ms"].notna().any() else None,
        "temporal_gap_rows": int(table["temporal_gap"].fillna(False).sum()),
        "capture_gap_rows": int(table["capture_gap_before"].fillna(False).sum()),
        "max_abs_sample_error_ms": int(pd.to_numeric(table["sample_error_ms"], errors="coerce").abs().max()),
        "windows": window_summaries,
        "jpeg_quality": jpeg_quality,
        "source_video": str(files.video),
        "source_timestamps": str(files.timestamps),
        "frames_dir": str(frames_dir),
        "frames_csv": str(manifest_csv),
        "video_metadata": metadata,
        "notes": [
            "Test-only JPEG extraction; formal full-cohort runner will decode selected frames directly from AVI.",
            "Sampling is timestamp-driven rather than frame-modulo driven so capture gaps remain explicit.",
        ],
    }
    manifest_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["manifest"] = str(manifest_json)
    return summary
