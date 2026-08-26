from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from attention_pipeline.config import Config
from attention_pipeline.rgb.audit import read_rgb_timestamps, video_metadata
from attention_pipeline.rgb.behavior import BehaviorIndex, empty_behavior_context
from attention_pipeline.rgb.discover import RGBSubjectFiles, discover_rgb_subjects
from attention_pipeline.rgb.paths import RGBOutputLayout
from attention_pipeline.rgb.timeline import detailed_rgb_intervals, formal_analysis_span


FACE_BENCHMARK_SAMPLE_SCHEMA = "rgb-face-benchmark-sample-v0.1"
DEFAULT_PHASE_SAMPLES = {
    "baseline": 50,
    "instructions": 25,
    "practice": 35,
    "transition": 15,
    "block1": 90,
    "interblock_transition": 45,
    "block2": 90,
}


def _find_subject(config: Config, subject: str) -> RGBSubjectFiles:
    records, duplicates = discover_rgb_subjects(config)
    if subject in duplicates:
        raise RuntimeError(f"Subject {subject} is duplicated across data roots: {duplicates[subject]}")
    for record in records:
        if record.subject == subject:
            return record
    raise FileNotFoundError(f"RGB subject not discovered: {subject}")


def _configured_exclusion(config: Config, subject: str) -> tuple[bool, str]:
    raw = config.section("data").get("exclude", {})
    if isinstance(raw, dict) and subject in raw:
        return True, str(raw[subject])
    if isinstance(raw, list) and subject in {str(v) for v in raw}:
        return True, "configured exclusion"
    return False, ""


def _phase_at(unix_ms: int, intervals) -> str:
    for interval in intervals:
        if interval.start_unix_ms <= unix_ms < interval.end_unix_ms:
            return interval.phase
    if intervals and unix_ms == intervals[-1].end_unix_ms:
        return intervals[-1].phase
    return "outside_analysis_span"


def _block_from_phase(phase: str) -> int | None:
    if phase.startswith("block") and phase[5:].isdigit():
        return int(phase[5:])
    return None


def _evenly_spaced_positions(positions: list[int], count: int) -> list[int]:
    if count <= 0 or not positions:
        return []
    if len(positions) <= count:
        return positions.copy()
    idx = np.linspace(0, len(positions) - 1, num=count)
    chosen = [positions[int(round(i))] for i in idx]
    return list(dict.fromkeys(chosen))


def _benchmark_dir(layout: RGBOutputLayout, subject: str) -> Path:
    path = layout.test_dir() / "face-benchmark" / subject
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_face_benchmark_sample(config: Config, subject: str) -> dict[str, object]:
    """Extract one deterministic, phase-stratified frame set for all Face candidates.

    This stage runs no Face model. It exists so Py-Feat and LibreFace receive the
    exact same source frames and can be compared fairly without repeatedly
    sampling the large source AVI in different ways.
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
    benchmark_cfg = face_cfg.get("benchmark", {}) if isinstance(face_cfg.get("benchmark", {}), dict) else {}
    phase_samples_raw = benchmark_cfg.get("phase_samples", DEFAULT_PHASE_SAMPLES)
    phase_samples = {
        str(k): int(v) for k, v in phase_samples_raw.items()
    } if isinstance(phase_samples_raw, dict) else DEFAULT_PHASE_SAMPLES.copy()
    jpeg_quality = int(benchmark_cfg.get("jpeg_quality", 95))

    baseline_duration_sec = float(focuswave.get("baseline_duration_sec", 180))
    expected_blocks = int(focuswave.get("expected_blocks", 2))
    trial_duration_ms = int(focuswave.get("trial_duration_ms", 1150))
    intervals = detailed_rgb_intervals(
        files.master_timeline,
        baseline_duration_sec=baseline_duration_sec,
        expected_blocks=expected_blocks,
    )
    analysis_start, analysis_end = formal_analysis_span(
        files.master_timeline,
        baseline_duration_sec=baseline_duration_sec,
        expected_blocks=expected_blocks,
    )

    all_times = [row[1] for row in timestamps]
    start_position = bisect_left(all_times, analysis_start)
    end_position = bisect_right(all_times, analysis_end) - 1
    if end_position < start_position:
        raise ValueError(f"No RGB frames inside formal analysis span for {subject}")

    by_phase: dict[str, list[int]] = {phase: [] for phase in phase_samples}
    phase_by_position: dict[int, str] = {}
    for pos in range(start_position, end_position + 1):
        unix_ms = int(timestamps[pos][1])
        phase = _phase_at(unix_ms, intervals)
        phase_by_position[pos] = phase
        if phase in by_phase:
            by_phase[phase].append(pos)

    selected: list[int] = []
    selected_phase_counts: dict[str, int] = {}
    for phase, requested in phase_samples.items():
        picks = _evenly_spaced_positions(by_phase.get(phase, []), requested)
        selected.extend(picks)
        selected_phase_counts[phase] = len(picks)
    selected = sorted(set(selected))
    if not selected:
        raise RuntimeError(f"No benchmark frames selected for {subject}")

    behavior_indexes = {
        1: BehaviorIndex.from_csv(files.block1_behavior),
        2: BehaviorIndex.from_csv(files.block2_behavior),
    }

    layout = RGBOutputLayout.from_config(config)
    root = _benchmark_dir(layout, subject)
    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = root / f"{subject}_face-benchmark_frames.csv"
    manifest_json = root / f"{subject}_face-benchmark_manifest.json"

    selected_set = set(selected)
    records: list[dict[str, object]] = []
    cap = cv2.VideoCapture(str(files.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open RGB video: {files.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(selected[0]))
    try:
        for pos in range(selected[0], selected[-1] + 1):
            if not cap.grab():
                raise RuntimeError(f"Failed to advance RGB video at frame {pos}")
            if pos not in selected_set:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                raise RuntimeError(f"Failed to decode selected RGB frame {pos}")

            capture_idx, unix_ms = timestamps[pos]
            phase = phase_by_position[pos]
            block = _block_from_phase(phase)
            behavior = empty_behavior_context()
            if block is not None:
                behavior = behavior_indexes[block].context_at(
                    int(unix_ms), trial_duration_ms=trial_duration_ms
                )

            filename = f"{subject}_f{pos:08d}_t{int(unix_ms)}.jpg"
            image_path = frames_dir / filename
            if not cv2.imwrite(
                str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            ):
                raise RuntimeError(f"Failed to write benchmark frame: {image_path}")

            row: dict[str, object] = {
                "schema_version": FACE_BENCHMARK_SAMPLE_SCHEMA,
                "subject": subject,
                "video_frame_position": int(pos),
                "capture_frame_idx": int(capture_idx),
                "unix_ms": int(unix_ms),
                "phase": phase,
                "block": block,
                "image_path": str(image_path),
                "jpeg_quality": jpeg_quality,
            }
            row.update(behavior)
            records.append(row)
    finally:
        cap.release()

    table = pd.DataFrame(records).sort_values("video_frame_position").reset_index(drop=True)
    table.insert(0, "benchmark_index", np.arange(len(table), dtype=int))
    table.to_csv(manifest_csv, index=False, encoding="utf-8-sig")

    summary = {
        "schema_version": FACE_BENCHMARK_SAMPLE_SCHEMA,
        "stage": "face-sample",
        "subject": subject,
        "purpose": "shared deterministic input set for Py-Feat vs LibreFace benchmark; no Face inference performed",
        "source_video": str(files.video),
        "source_timestamps": str(files.timestamps),
        "analysis_start_unix_ms": int(analysis_start),
        "analysis_end_unix_ms": int(analysis_end),
        "requested_phase_samples": phase_samples,
        "selected_phase_samples": selected_phase_counts,
        "selected_frames": int(len(table)),
        "jpeg_quality": jpeg_quality,
        "frames_dir": str(frames_dir),
        "frames_csv": str(manifest_csv),
        "video_metadata": metadata,
    }
    manifest_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["manifest"] = str(manifest_json)
    return summary
