from __future__ import annotations

import json
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from attention_pipeline.config import Config
from attention_pipeline.rgb.audit import read_rgb_timestamps, video_metadata
from attention_pipeline.rgb.behavior import BehaviorIndex
from attention_pipeline.rgb.face_benchmark import _configured_exclusion, _find_subject
from attention_pipeline.rgb.paths import RGBOutputLayout
from attention_pipeline.rgb.timeline import detailed_rgb_intervals


FACE_CONTINUOUS_SAMPLE_SCHEMA = "rgb-face-continuous-sample-v0.1"
DEFAULT_DURATION_SEC = 30.0
DEFAULT_INFERENCE_FPS = 10.0
DEFAULT_PHASE = "block1"


def _continuous_dir(layout: RGBOutputLayout, subject: str) -> Path:
    path = layout.test_dir() / "face-continuous" / subject
    path.mkdir(parents=True, exist_ok=True)
    return path


def _nearest_position(times: list[int], target: int, lo: int, hi: int) -> int:
    pos = bisect_left(times, target, lo, hi + 1)
    candidates: list[int] = []
    if lo <= pos <= hi:
        candidates.append(pos)
    if lo <= pos - 1 <= hi:
        candidates.append(pos - 1)
    if not candidates:
        raise RuntimeError(f"No timestamp position available near target {target}")
    return min(candidates, key=lambda i: abs(int(times[i]) - int(target)))


def run_face_continuous_sample(config: Config, subject: str) -> dict[str, object]:
    """Extract a deterministic contiguous Block1 window at real-time 10 fps.

    This stage performs no Face inference. It creates one shared frame set so
    Py-Feat and LibreFace can be compared on temporal stability and true
    end-to-end runtime using identical source frames.
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

    focuswave = config.section("focuswave")
    face_cfg = config.section("face")
    benchmark_cfg = face_cfg.get("continuous_benchmark", {})
    if not isinstance(benchmark_cfg, dict):
        benchmark_cfg = {}

    duration_sec = float(benchmark_cfg.get("duration_sec", DEFAULT_DURATION_SEC))
    inference_fps = float(benchmark_cfg.get("inference_fps", DEFAULT_INFERENCE_FPS))
    phase_name = str(benchmark_cfg.get("phase", DEFAULT_PHASE))
    jpeg_quality = int(benchmark_cfg.get("jpeg_quality", face_cfg.get("benchmark", {}).get("jpeg_quality", 95)))
    if duration_sec <= 0 or inference_fps <= 0:
        raise ValueError("continuous benchmark duration_sec and inference_fps must be positive")

    baseline_duration_sec = float(focuswave.get("baseline_duration_sec", 180))
    expected_blocks = int(focuswave.get("expected_blocks", 2))
    trial_duration_ms = int(focuswave.get("trial_duration_ms", 1150))
    intervals = detailed_rgb_intervals(
        files.master_timeline,
        baseline_duration_sec=baseline_duration_sec,
        expected_blocks=expected_blocks,
    )
    matches = [interval for interval in intervals if interval.phase == phase_name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {phase_name} interval, found {len(matches)}")
    interval = matches[0]

    duration_ms = int(round(duration_sec * 1000.0))
    interval_duration_ms = int(interval.end_unix_ms - interval.start_unix_ms)
    if duration_ms > interval_duration_ms:
        raise ValueError(
            f"Requested continuous window {duration_ms} ms exceeds {phase_name} duration {interval_duration_ms} ms"
        )

    window_start = int(interval.start_unix_ms + (interval_duration_ms - duration_ms) // 2)
    window_end = int(window_start + duration_ms)

    all_times = [int(row[1]) for row in timestamps]
    lo = bisect_left(all_times, int(interval.start_unix_ms))
    hi = bisect_left(all_times, int(interval.end_unix_ms)) - 1
    if hi < lo:
        raise RuntimeError(f"No RGB frames found inside {phase_name}")

    step_ms = 1000.0 / inference_fps
    target_times = [int(round(window_start + i * step_ms)) for i in range(int(round(duration_sec * inference_fps)))]
    selected: list[tuple[int, int]] = []
    seen: set[int] = set()
    for target in target_times:
        pos = _nearest_position(all_times, target, lo, hi)
        if pos in seen:
            continue
        seen.add(pos)
        selected.append((pos, target))
    selected.sort(key=lambda item: item[0])
    if not selected:
        raise RuntimeError("No continuous benchmark frames selected")

    behavior = BehaviorIndex.from_csv(files.block1_behavior)
    layout = RGBOutputLayout.from_config(config)
    root = _continuous_dir(layout, subject)
    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = root / f"{subject}_face-continuous_frames.csv"
    manifest_json = root / f"{subject}_face-continuous_manifest.json"

    selected_targets = {pos: target for pos, target in selected}
    selected_positions = set(selected_targets)
    records: list[dict[str, object]] = []

    cap = cv2.VideoCapture(str(files.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open RGB video: {files.video}")
    first_pos = selected[0][0]
    last_pos = selected[-1][0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(first_pos))
    try:
        for pos in range(first_pos, last_pos + 1):
            if not cap.grab():
                raise RuntimeError(f"Failed to advance RGB video at frame {pos}")
            if pos not in selected_positions:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                raise RuntimeError(f"Failed to decode selected RGB frame {pos}")

            capture_idx, unix_ms = timestamps[pos]
            target = selected_targets[pos]
            filename = f"{subject}_f{pos:08d}_t{int(unix_ms)}.jpg"
            image_path = frames_dir / filename
            if not cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]):
                raise RuntimeError(f"Failed to write continuous benchmark frame: {image_path}")

            row: dict[str, object] = {
                "schema_version": FACE_CONTINUOUS_SAMPLE_SCHEMA,
                "subject": subject,
                "video_frame_position": int(pos),
                "capture_frame_idx": int(capture_idx),
                "unix_ms": int(unix_ms),
                "target_unix_ms": int(target),
                "sample_error_ms": int(unix_ms) - int(target),
                "phase": phase_name,
                "block": 1,
                "image_path": str(image_path),
                "jpeg_quality": jpeg_quality,
            }
            row.update(behavior.context_at(int(unix_ms), trial_duration_ms=trial_duration_ms))
            records.append(row)
    finally:
        cap.release()

    table = pd.DataFrame(records).sort_values("unix_ms").reset_index(drop=True)
    table.insert(0, "benchmark_index", np.arange(len(table), dtype=int))
    table["dt_ms"] = pd.to_numeric(table["unix_ms"], errors="coerce").diff()
    gap_threshold_ms = max(250.0, step_ms * 2.5)
    table["temporal_gap"] = table["dt_ms"] > gap_threshold_ms
    table.to_csv(manifest_csv, index=False, encoding="utf-8-sig")

    summary = {
        "schema_version": FACE_CONTINUOUS_SAMPLE_SCHEMA,
        "stage": "face-continuous-sample",
        "subject": subject,
        "purpose": "shared contiguous window for temporal stability and end-to-end Face benchmark",
        "phase": phase_name,
        "requested_duration_sec": duration_sec,
        "requested_inference_fps": inference_fps,
        "expected_target_frames": int(round(duration_sec * inference_fps)),
        "selected_frames": int(len(table)),
        "window_start_unix_ms": window_start,
        "window_end_unix_ms": window_end,
        "first_sample_unix_ms": int(table["unix_ms"].iloc[0]),
        "last_sample_unix_ms": int(table["unix_ms"].iloc[-1]),
        "median_dt_ms": float(table["dt_ms"].dropna().median()) if table["dt_ms"].notna().any() else None,
        "max_dt_ms": float(table["dt_ms"].dropna().max()) if table["dt_ms"].notna().any() else None,
        "temporal_gap_rows": int(table["temporal_gap"].fillna(False).sum()),
        "max_abs_sample_error_ms": int(pd.to_numeric(table["sample_error_ms"], errors="coerce").abs().max()),
        "jpeg_quality": jpeg_quality,
        "source_video": str(files.video),
        "source_timestamps": str(files.timestamps),
        "frames_dir": str(frames_dir),
        "frames_csv": str(manifest_csv),
        "video_metadata": metadata,
    }
    manifest_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["manifest"] = str(manifest_json)
    return summary
