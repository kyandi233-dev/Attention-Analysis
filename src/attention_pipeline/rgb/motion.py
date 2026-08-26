from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from bisect import bisect_left, bisect_right
from datetime import datetime, timezone
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


MOTION_SCHEMA_VERSION = "rgb-motion-raw-v0.1"


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


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


def _configured_exclusion(config: Config, subject: str) -> tuple[bool, str]:
    raw = config.section("data").get("exclude", {})
    if isinstance(raw, dict):
        if subject in raw:
            return True, str(raw[subject])
        return False, ""
    if isinstance(raw, list):
        return subject in {str(value) for value in raw}, "configured exclusion"
    return False, ""


def measure_motion_pair(
    current_gray: np.ndarray,
    previous_gray: np.ndarray | None,
    *,
    dt_ms: int | None,
    median_interval_ms: float | None,
    previous_capture_idx: int | None,
    current_capture_idx: int,
    previous_gray_mean: float | None,
    gap_reset_ms: int = 100,
    irregular_dt_multiple: float = 1.5,
    pixel_diff_threshold: int = 15,
) -> dict[str, object]:
    """Measure low-cost frame statistics and gap-aware adjacent-frame motion.

    Current-frame brightness statistics are always retained. Adjacent-frame motion
    is only suppressed when temporal identity is unsafe (first analysed frame,
    non-positive time, configured timestamp gap, or an explicit capture-index gap).
    Mildly irregular dt remains measured and is flagged instead of being filtered.
    """
    gray_mean = float(current_gray.mean())
    gray_std = float(current_gray.std())
    gray_min = int(current_gray.min())
    gray_max = int(current_gray.max())

    capture_missing = 0
    if previous_capture_idx is not None:
        capture_missing = max(0, current_capture_idx - previous_capture_idx - 1)

    dt_multiple = (
        float(dt_ms) / float(median_interval_ms)
        if dt_ms is not None and median_interval_ms and median_interval_ms > 0
        else None
    )
    irregular_dt = bool(
        dt_multiple is not None and dt_multiple > float(irregular_dt_multiple)
    )

    if previous_gray is None:
        gap_before = False
        gap_reason = "analysis_start"
    elif dt_ms is None or dt_ms <= 0:
        gap_before = True
        gap_reason = "nonpositive_dt"
    else:
        by_time = dt_ms > int(gap_reset_ms)
        by_capture = capture_missing > 0
        gap_before = bool(by_time or by_capture)
        if by_time and by_capture:
            gap_reason = "timestamp_and_capture_gap"
        elif by_time:
            gap_reason = "timestamp_gap"
        elif by_capture:
            gap_reason = "capture_index_gap"
        else:
            gap_reason = ""

    motion_valid = bool(previous_gray is not None and not gap_before and dt_ms and dt_ms > 0)
    result: dict[str, object] = {
        "gray_mean": gray_mean,
        "gray_std": gray_std,
        "gray_min": gray_min,
        "gray_max": gray_max,
        "gray_mean_delta": None,
        "capture_missing_frame_indices_before": capture_missing,
        "dt_multiple_of_median": dt_multiple,
        "irregular_dt": irregular_dt,
        "gap_before": gap_before,
        "gap_duration_ms": dt_ms if gap_before else None,
        "gap_reason": gap_reason,
        "motion_valid": motion_valid,
        "mean_abs_difference": None,
        "std_abs_difference": None,
        "sum_abs_difference": None,
        "max_abs_difference": None,
        "changed_pixel_ratio": None,
        "pixel_diff_threshold": int(pixel_diff_threshold),
        "global_motion_energy": None,
        "global_motion_energy_per_sec": None,
    }
    if not motion_valid:
        return result

    diff = cv2.absdiff(current_gray, previous_gray)
    mean_abs = float(diff.mean())
    std_abs = float(diff.std())
    sum_abs = int(diff.sum(dtype=np.uint64))
    max_abs = int(diff.max())
    changed_ratio = float(np.count_nonzero(diff >= int(pixel_diff_threshold)) / diff.size)
    normalized_energy = mean_abs / 255.0
    per_sec = normalized_energy / (float(dt_ms) / 1000.0)

    result.update(
        {
            "gray_mean_delta": gray_mean - float(previous_gray_mean) if previous_gray_mean is not None else None,
            "mean_abs_difference": mean_abs,
            "std_abs_difference": std_abs,
            "sum_abs_difference": sum_abs,
            "max_abs_difference": max_abs,
            "changed_pixel_ratio": changed_ratio,
            "global_motion_energy": normalized_energy,
            "global_motion_energy_per_sec": per_sec,
        }
    )
    return result


def _open_video_at(path: Path, start_position: int) -> tuple[cv2.VideoCapture, str]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open RGB video: {path}")
    if start_position <= 0:
        return cap, "from_start"

    if cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_position)):
        reported = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        if abs(reported - start_position) <= 1:
            return cap, "opencv_seek"

    cap.release()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot reopen RGB video for sequential seek: {path}")
    for position in range(start_position):
        ok, _ = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"RGB video ended while seeking to frame {start_position}; failed at {position}")
    return cap, "sequential_fallback"


def _config_digest(config: Config) -> str:
    return hashlib.sha256(config.path.read_bytes()).hexdigest()


def _git_commit(config: Config) -> str | None:
    repo_root = config.path.parent.parent
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def _file_info(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    info: dict[str, object] = {"path": str(path), "exists": path.exists()}
    if path.exists():
        stat = path.stat()
        info.update({"size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    return info


def _find_subject(config: Config, subject: str) -> RGBSubjectFiles:
    records, duplicates = discover_rgb_subjects(config)
    if subject in duplicates:
        raise RuntimeError(f"Subject {subject} is duplicated across data roots: {duplicates[subject]}")
    for record in records:
        if record.subject == subject:
            return record
    raise FileNotFoundError(f"RGB subject not discovered: {subject}")


def run_motion_test(config: Config, subject: str) -> dict[str, object]:
    """Run one-subject full-FPS Motion Energy pilot into Beijing-RGB/_test."""
    excluded, exclusion_reason = _configured_exclusion(config, subject)
    if excluded:
        raise ValueError(f"Subject {subject} is excluded from RGB analysis: {exclusion_reason}")

    files = _find_subject(config, subject)
    timestamps = read_rgb_timestamps(files.timestamps)
    if not timestamps:
        raise ValueError(f"No RGB timestamps: {files.timestamps}")

    metadata = video_metadata(files.video)
    if not metadata["video_open_ok"]:
        raise RuntimeError(f"RGB video cannot be opened: {files.video}")
    nominal_count = int(metadata["video_frame_count_nominal"])
    if nominal_count != len(timestamps):
        raise ValueError(
            f"AVI/timestamp row mismatch for {subject}: video={nominal_count}, timestamps={len(timestamps)}"
        )

    focuswave = config.section("focuswave")
    motion_cfg = config.section("motion")
    baseline_duration_sec = float(focuswave.get("baseline_duration_sec", 180))
    expected_blocks = int(focuswave.get("expected_blocks", 2))
    trial_duration_ms = int(focuswave.get("trial_duration_ms", 1150))
    gap_reset_ms = int(motion_cfg.get("gap_reset_ms", 100))
    irregular_dt_multiple = float(motion_cfg.get("irregular_dt_multiple", 1.5))
    pixel_diff_threshold = int(motion_cfg.get("pixel_diff_threshold", 15))

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
    if start_position >= len(timestamps) or end_position < start_position:
        raise ValueError(f"No RGB frames inside formal analysis span for {subject}")

    analysis_rows = timestamps[start_position : end_position + 1]
    positive_dt = [b[1] - a[1] for a, b in zip(analysis_rows, analysis_rows[1:]) if b[1] > a[1]]
    median_interval_ms = _median(positive_dt)

    behavior_indexes = {
        1: BehaviorIndex.from_csv(files.block1_behavior),
        2: BehaviorIndex.from_csv(files.block2_behavior),
    }

    layout = RGBOutputLayout.from_config(config)
    output_path = layout.test_file(f"{subject}_motion-test.parquet")
    manifest_path = layout.test_file(f"{subject}_motion-test_manifest.json")

    started_utc = datetime.now(timezone.utc).isoformat()
    started_clock = time.perf_counter()
    cap, seek_mode = _open_video_at(files.video, start_position)
    rows: list[dict[str, object]] = []
    previous_gray: np.ndarray | None = None
    previous_gray_mean: float | None = None
    previous_capture_idx: int | None = None
    previous_unix_ms: int | None = None

    try:
        for video_position in range(start_position, end_position + 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Failed to read {subject} RGB frame at video position {video_position}")

            capture_idx, unix_ms = timestamps[video_position]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dt_ms = unix_ms - previous_unix_ms if previous_unix_ms is not None else None
            measurement = measure_motion_pair(
                gray,
                previous_gray,
                dt_ms=dt_ms,
                median_interval_ms=median_interval_ms,
                previous_capture_idx=previous_capture_idx,
                current_capture_idx=capture_idx,
                previous_gray_mean=previous_gray_mean,
                gap_reset_ms=gap_reset_ms,
                irregular_dt_multiple=irregular_dt_multiple,
                pixel_diff_threshold=pixel_diff_threshold,
            )

            phase = _phase_at(unix_ms, intervals)
            block = _block_from_phase(phase)
            behavior = empty_behavior_context()
            if block is not None and behavior_indexes.get(block) is not None:
                behavior = behavior_indexes[block].context_at(
                    unix_ms, trial_duration_ms=trial_duration_ms
                )

            row: dict[str, object] = {
                "subject": subject,
                "video_frame_position": video_position,
                "capture_frame_idx": capture_idx,
                "unix_ms": unix_ms,
                "dt_ms": dt_ms,
                "phase": phase,
                "block": block,
            }
            row.update(behavior)
            row.update(measurement)
            rows.append(row)

            previous_gray = gray
            previous_gray_mean = float(measurement["gray_mean"])
            previous_capture_idx = capture_idx
            previous_unix_ms = unix_ms
    finally:
        cap.release()

    table = pd.DataFrame(rows)
    table.to_parquet(output_path, index=False, engine="pyarrow", compression="zstd")
    elapsed_sec = time.perf_counter() - started_clock
    finished_utc = datetime.now(timezone.utc).isoformat()

    valid_motion = int(table["motion_valid"].astype(bool).sum()) if not table.empty else 0
    gap_rows = int(table["gap_before"].astype(bool).sum()) if not table.empty else 0
    irregular_rows = int(table["irregular_dt"].astype(bool).sum()) if not table.empty else 0
    phase_counts = {str(key): int(value) for key, value in table["phase"].value_counts().to_dict().items()}

    manifest = {
        "schema_version": str(motion_cfg.get("schema_version", MOTION_SCHEMA_VERSION)),
        "stage": "motion-test",
        "subject": subject,
        "output_mode": "test",
        "run_started_utc": started_utc,
        "run_finished_utc": finished_utc,
        "elapsed_sec": elapsed_sec,
        "processing_fps": (len(table) / elapsed_sec) if elapsed_sec > 0 else None,
        "attention_analysis_git_commit": _git_commit(config),
        "config_path": str(config.path),
        "config_sha256": _config_digest(config),
        "focuswave_provenance": {
            "repository": focuswave.get("repository"),
            "branch": focuswave.get("branch"),
            "accepted_formal_versions": focuswave.get("accepted_formal_versions"),
            "formal_structure": focuswave.get("formal_structure"),
        },
        "source": {
            "video": _file_info(files.video),
            "timestamps": _file_info(files.timestamps),
            "master_timeline": _file_info(files.master_timeline),
            "block1_behavior": _file_info(files.block1_behavior),
            "block2_behavior": _file_info(files.block2_behavior),
        },
        "video_metadata": metadata,
        "analysis_span": {
            "requested_start_unix_ms": analysis_start,
            "requested_end_unix_ms": analysis_end,
            "first_output_unix_ms": int(table["unix_ms"].iloc[0]) if not table.empty else None,
            "last_output_unix_ms": int(table["unix_ms"].iloc[-1]) if not table.empty else None,
            "first_video_frame_position": start_position,
            "last_video_frame_position": end_position,
            "median_interval_ms": median_interval_ms,
        },
        "parameters": {
            "process_full_fps": bool(motion_cfg.get("process_full_fps", True)),
            "gap_reset_ms": gap_reset_ms,
            "irregular_dt_multiple": irregular_dt_multiple,
            "pixel_diff_threshold": pixel_diff_threshold,
            "trial_duration_ms": trial_duration_ms,
            "motion_definition": "mean absolute grayscale frame difference / 255",
            "motion_rate_definition": "global_motion_energy / dt_seconds",
            "gap_policy": "retain current frame; adjacent-frame metrics missing after timestamp/capture gap",
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "video_seek_mode": seek_mode,
            "parquet_engine": "pyarrow",
            "parquet_compression": "zstd",
        },
        "output": {
            "parquet": str(output_path),
            "manifest": str(manifest_path),
            "rows": int(len(table)),
            "motion_valid_rows": valid_motion,
            "gap_reset_rows": gap_rows,
            "irregular_dt_rows": irregular_rows,
            "phase_rows": phase_counts,
            "parquet_size_bytes": int(output_path.stat().st_size),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
