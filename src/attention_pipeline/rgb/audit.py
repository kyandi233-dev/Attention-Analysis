from __future__ import annotations

import csv
from pathlib import Path

import cv2
import pandas as pd

from attention_pipeline.config import Config
from attention_pipeline.rgb.discover import RGBSubjectFiles, discover_rgb_subjects
from attention_pipeline.rgb.timeline import formal_analysis_span


def read_rgb_timestamps(path: Path) -> list[tuple[int, int]]:
    if not path.exists():
        return []
    rows: list[tuple[int, int]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row or len(row) < 2:
                continue
            try:
                frame_idx = int(float(row[0]))
                unix_ms = int(float(row[1]))
            except ValueError as exc:
                raise ValueError(f"Invalid RGB timestamp row {line_number}: {path}") from exc
            rows.append((frame_idx, unix_ms))
    return rows


def video_metadata(path: Path) -> dict[str, float | int | bool]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {
            "video_open_ok": False,
            "video_width": 0,
            "video_height": 0,
            "video_fps_nominal": 0.0,
            "video_frame_count_nominal": 0,
            "video_duration_sec_nominal": 0.0,
        }
    try:
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return {
        "video_open_ok": True,
        "video_width": width,
        "video_height": height,
        "video_fps_nominal": fps,
        "video_frame_count_nominal": frame_count,
        "video_duration_sec_nominal": duration,
    }


def _timestamp_metrics(rows: list[tuple[int, int]]) -> dict[str, object]:
    if not rows:
        return {
            "timestamp_rows": 0,
            "timestamp_first_frame": None,
            "timestamp_last_frame": None,
            "timestamp_frame_index_contiguous": False,
            "timestamp_missing_frame_indices": None,
            "timestamp_start_unix_ms": None,
            "timestamp_end_unix_ms": None,
            "timestamp_unix_monotonic": False,
            "timestamp_median_interval_ms": None,
            "timestamp_max_interval_ms": None,
        }
    frame_ids = [item[0] for item in rows]
    times = [item[1] for item in rows]
    frame_contiguous = all(b - a == 1 for a, b in zip(frame_ids, frame_ids[1:]))
    expected_span_count = frame_ids[-1] - frame_ids[0] + 1
    missing_frame_indices = max(0, expected_span_count - len(frame_ids))
    time_monotonic = all(b > a for a, b in zip(times, times[1:]))
    intervals = [b - a for a, b in zip(times, times[1:])]
    if intervals:
        ordered = sorted(intervals)
        mid = len(ordered) // 2
        median = (
            float(ordered[mid])
            if len(ordered) % 2
            else (ordered[mid - 1] + ordered[mid]) / 2.0
        )
        max_interval = max(intervals)
    else:
        median = None
        max_interval = None
    return {
        "timestamp_rows": len(rows),
        "timestamp_first_frame": frame_ids[0],
        "timestamp_last_frame": frame_ids[-1],
        "timestamp_frame_index_contiguous": frame_contiguous,
        "timestamp_missing_frame_indices": missing_frame_indices,
        "timestamp_start_unix_ms": times[0],
        "timestamp_end_unix_ms": times[-1],
        "timestamp_unix_monotonic": time_monotonic,
        "timestamp_median_interval_ms": median,
        "timestamp_max_interval_ms": max_interval,
    }


def _exclusion(config: Config, subject: str) -> tuple[bool, str]:
    raw = config.section("data").get("exclude", {})
    if isinstance(raw, dict):
        if subject in raw:
            return True, str(raw[subject])
        return False, ""
    if isinstance(raw, list):
        return subject in {str(value) for value in raw}, "configured exclusion"
    return False, ""


def audit_subject(files: RGBSubjectFiles, config: Config) -> dict[str, object]:
    focuswave = config.section("focuswave")
    timestamps = read_rgb_timestamps(files.timestamps)
    excluded, exclusion_reason = _exclusion(config, files.subject)
    row: dict[str, object] = {
        "subject": files.subject,
        "root": str(files.root),
        "analysis_excluded": excluded,
        "exclusion_reason": exclusion_reason,
        "video": str(files.video),
        "timestamps": str(files.timestamps),
        "master_timeline": str(files.master_timeline),
        "block1_behavior": str(files.block1_behavior) if files.block1_behavior else "",
        "block2_behavior": str(files.block2_behavior) if files.block2_behavior else "",
        "video_exists": files.video.exists(),
        "timestamps_exists": files.timestamps.exists(),
        "master_timeline_exists": files.master_timeline.exists(),
        "block1_behavior_exists": bool(files.block1_behavior and files.block1_behavior.exists()),
        "block2_behavior_exists": bool(files.block2_behavior and files.block2_behavior.exists()),
    }
    row.update(video_metadata(files.video))
    row.update(_timestamp_metrics(timestamps))

    nominal_count = int(row["video_frame_count_nominal"] or 0)
    timestamp_count = int(row["timestamp_rows"] or 0)
    row["video_timestamp_count_delta"] = nominal_count - timestamp_count

    if timestamps and files.master_timeline.exists():
        try:
            analysis_start, analysis_end = formal_analysis_span(
                files.master_timeline,
                baseline_duration_sec=float(focuswave.get("baseline_duration_sec", 180)),
                expected_blocks=int(focuswave.get("expected_blocks", 2)),
            )
            ts_start = timestamps[0][1]
            ts_end = timestamps[-1][1]
            row["formal_start_unix_ms"] = analysis_start
            row["formal_end_unix_ms"] = analysis_end
            row["rgb_covers_formal_start"] = ts_start <= analysis_start <= ts_end
            row["rgb_covers_formal_end"] = ts_start <= analysis_end <= ts_end
            row["formal_timeline_parse_ok"] = True
            row["formal_timeline_error"] = ""
        except Exception as exc:
            row["formal_start_unix_ms"] = None
            row["formal_end_unix_ms"] = None
            row["rgb_covers_formal_start"] = False
            row["rgb_covers_formal_end"] = False
            row["formal_timeline_parse_ok"] = False
            row["formal_timeline_error"] = str(exc)
    else:
        row["formal_start_unix_ms"] = None
        row["formal_end_unix_ms"] = None
        row["rgb_covers_formal_start"] = False
        row["rgb_covers_formal_end"] = False
        row["formal_timeline_parse_ok"] = False
        row["formal_timeline_error"] = "missing RGB timestamps or master_timeline.csv"

    row["basic_complete"] = bool(
        row["video_open_ok"]
        and row["timestamps_exists"]
        and row["master_timeline_exists"]
        and row["block1_behavior_exists"]
        and row["block2_behavior_exists"]
        and row["timestamp_frame_index_contiguous"]
        and row["timestamp_unix_monotonic"]
        and row["formal_timeline_parse_ok"]
        and row["rgb_covers_formal_start"]
        and row["rgb_covers_formal_end"]
    )
    row["analysis_eligible"] = bool(not excluded and row["basic_complete"])
    return row


def run_audit(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    records, duplicates = discover_rgb_subjects(config)
    inventory = pd.DataFrame([audit_subject(record, config) for record in records])
    duplicate_rows = []
    for subject, paths in sorted(duplicates.items()):
        duplicate_rows.append(
            {
                "subject": subject,
                "n_locations": len(paths),
                "locations": " | ".join(str(path) for path in paths),
            }
        )
    duplicate_table = pd.DataFrame(duplicate_rows)
    return inventory, duplicate_table
