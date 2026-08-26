from __future__ import annotations

import json
import math
from bisect import bisect_left, bisect_right
from pathlib import Path

import pandas as pd

from attention_pipeline.config import Config
from attention_pipeline.rgb.audit import read_rgb_timestamps, video_metadata
from attention_pipeline.rgb.behavior import BehaviorIndex, empty_behavior_context
from attention_pipeline.rgb.face_benchmark import _configured_exclusion, _find_subject
from attention_pipeline.rgb.face_continuous import _nearest_position
from attention_pipeline.rgb.paths import RGBOutputLayout
from attention_pipeline.rgb.timeline import detailed_rgb_intervals, formal_analysis_span


FACE_FORMAL_SAMPLE_SCHEMA = "rgb-face-formal-sample-v1.0"


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


def run_face_formal_prepare(config: Config, subject: str) -> dict[str, object]:
    """Build the full formal 15 Hz Face frame manifest without decoding JPEGs.

    This is the production counterpart of the representative dry-run sampler.
    It selects source AVI positions from the formal analysis span using Unix-ms
    timestamps and preserves temporal gaps as flags rather than excluding rows.
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
        raise ValueError(
            f"AVI/timestamp row mismatch for {subject}: "
            f"video={metadata['video_frame_count_nominal']}, timestamps={len(timestamps)}"
        )

    face_cfg = config.section("face")
    inference_fps = float(face_cfg.get("inference_fps", 15.0) or 15.0)
    if inference_fps <= 0:
        raise ValueError("face.inference_fps must be > 0")
    source_fps = float(metadata.get("video_fps_nominal") or 0.0)
    if source_fps + 1e-9 < inference_fps:
        raise ValueError(
            f"Requested Face {inference_fps} Hz exceeds nominal source fps {source_fps}"
        )

    focuswave = config.section("focuswave")
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

    all_times = [int(row[1]) for row in timestamps]
    lo = bisect_left(all_times, int(analysis_start))
    hi = bisect_right(all_times, int(analysis_end)) - 1
    if lo >= len(all_times) or hi < lo:
        raise ValueError(f"No RGB frames inside formal analysis span for {subject}")

    step_ms = 1000.0 / inference_fps
    target_count = int(math.floor((analysis_end - analysis_start) / step_ms)) + 1
    selected: dict[int, int] = {}
    for i in range(target_count):
        target = int(round(analysis_start + i * step_ms))
        if target > analysis_end:
            break
        pos = _nearest_position(all_times, target, lo, hi)
        old_target = selected.get(pos)
        if old_target is None or abs(all_times[pos] - target) < abs(all_times[pos] - old_target):
            selected[pos] = target

    if not selected:
        raise RuntimeError(f"No formal Face frames selected for {subject}")

    behavior_indexes = {
        1: BehaviorIndex.from_csv(files.block1_behavior),
        2: BehaviorIndex.from_csv(files.block2_behavior),
    }

    records: list[dict[str, object]] = []
    for sample_index, pos in enumerate(sorted(selected)):
        capture_idx, unix_ms = timestamps[pos]
        target = int(selected[pos])
        phase = _phase_at(int(unix_ms), intervals)
        block = _block_from_phase(phase)
        behavior = empty_behavior_context()
        if block is not None:
            behavior = behavior_indexes[block].context_at(
                int(unix_ms), trial_duration_ms=trial_duration_ms
            )
        row: dict[str, object] = {
            "schema_version": FACE_FORMAL_SAMPLE_SCHEMA,
            "subject": subject,
            "sample_index": int(sample_index),
            "benchmark_index": int(sample_index),
            "video_frame_position": int(pos),
            "capture_frame_idx": int(capture_idx),
            "unix_ms": int(unix_ms),
            "target_unix_ms": target,
            "sample_error_ms": int(unix_ms) - target,
            "phase": phase,
            "block": block,
        }
        row.update(behavior)
        records.append(row)

    table = pd.DataFrame(records).sort_values("unix_ms").reset_index(drop=True)
    table["dt_ms"] = pd.to_numeric(table["unix_ms"], errors="coerce").diff()
    capture_delta = pd.to_numeric(table["capture_frame_idx"], errors="coerce").diff()
    table["capture_gap_before"] = capture_delta.fillna(1) > 3
    table["temporal_gap"] = table["dt_ms"] > max(250.0, step_ms * 2.5)

    layout = RGBOutputLayout.from_config(config)
    frames_csv = layout.subject_file(subject, "face_frames.csv")
    manifest_json = layout.subject_file(subject, "face_prepare_manifest.json")
    table.to_csv(frames_csv, index=False, encoding="utf-8-sig")

    summary = {
        "schema_version": FACE_FORMAL_SAMPLE_SCHEMA,
        "stage": "face-formal-prepare",
        "output_mode": "formal",
        "subject": subject,
        "requested_inference_fps": inference_fps,
        "source_video_fps_nominal": metadata.get("video_fps_nominal"),
        "analysis_start_unix_ms": int(analysis_start),
        "analysis_end_unix_ms": int(analysis_end),
        "selected_frames": int(len(table)),
        "median_dt_ms": float(table["dt_ms"].dropna().median()) if table["dt_ms"].notna().any() else None,
        "max_dt_ms": float(table["dt_ms"].dropna().max()) if table["dt_ms"].notna().any() else None,
        "temporal_gap_rows": int(table["temporal_gap"].fillna(False).sum()),
        "capture_gap_rows": int(table["capture_gap_before"].fillna(False).sum()),
        "max_abs_sample_error_ms": int(pd.to_numeric(table["sample_error_ms"], errors="coerce").abs().max()),
        "source_video": str(files.video),
        "source_timestamps": str(files.timestamps),
        "source_master_timeline": str(files.master_timeline),
        "source_block1_behavior": str(files.block1_behavior),
        "source_block2_behavior": str(files.block2_behavior),
        "frames_csv": str(frames_csv),
        "video_metadata": metadata,
        "notes": [
            "Full formal span, not representative dry-run windows.",
            "No JPEG extraction: the DirectML runner decodes selected positions directly from the original AVI.",
            "Timestamp/capture gaps are retained as QC flags and do not exclude the subject here.",
        ],
    }
    manifest_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["manifest"] = str(manifest_json)
    return summary
