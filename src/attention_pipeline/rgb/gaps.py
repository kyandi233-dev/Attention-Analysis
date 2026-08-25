from __future__ import annotations

import pandas as pd

from attention_pipeline.config import Config
from attention_pipeline.rgb.audit import read_rgb_timestamps
from attention_pipeline.rgb.discover import RGBSubjectFiles, discover_rgb_subjects
from attention_pipeline.rgb.timeline import detailed_rgb_intervals, formal_analysis_span


def _exclusion(config: Config, subject: str) -> tuple[bool, str]:
    raw = config.section("data").get("exclude", {})
    if isinstance(raw, dict):
        if subject in raw:
            return True, str(raw[subject])
        return False, ""
    if isinstance(raw, list):
        return subject in {str(value) for value in raw}, "configured exclusion"
    return False, ""


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


def subject_timestamp_gaps(files: RGBSubjectFiles, config: Config) -> list[dict[str, object]]:
    """Return timestamp gaps above the development warning threshold.

    Existing AVI rows remain addressable by physical video position and original
    FocusWave capture frame index. Capture-index jumps are QC, not automatic
    subject failure. Downstream temporal features can therefore mark the first
    post-gap sample missing rather than create a false movement spike.
    """
    rows = read_rgb_timestamps(files.timestamps)
    if len(rows) < 2 or not files.master_timeline.exists():
        return []

    threshold_ms = int(config.section("qc").get("timestamp_gap_warning_ms", 100))
    focuswave = config.section("focuswave")
    baseline_duration_sec = float(focuswave.get("baseline_duration_sec", 180))
    expected_blocks = int(focuswave.get("expected_blocks", 2))
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

    dt_values = [current[1] - previous[1] for previous, current in zip(rows, rows[1:])]
    positive_dt = [value for value in dt_values if value > 0]
    median_interval_ms = _median(positive_dt)
    excluded, exclusion_reason = _exclusion(config, files.subject)

    output: list[dict[str, object]] = []
    for current_position in range(1, len(rows)):
        previous_position = current_position - 1
        previous_frame_idx, previous_ms = rows[previous_position]
        current_frame_idx, current_ms = rows[current_position]
        dt_ms = current_ms - previous_ms
        if dt_ms <= threshold_ms:
            continue

        midpoint_ms = previous_ms + dt_ms // 2
        phase_before = _phase_at(previous_ms, intervals)
        phase_after = _phase_at(current_ms, intervals)
        phase_midpoint = _phase_at(midpoint_ms, intervals)
        missing_capture_indices = max(0, current_frame_idx - previous_frame_idx - 1)
        estimated_missing_duration_ms = (
            max(0.0, float(dt_ms) - float(median_interval_ms))
            if median_interval_ms is not None
            else None
        )
        estimated_missing_frames_by_time = (
            max(0, int(round(float(dt_ms) / float(median_interval_ms))) - 1)
            if median_interval_ms and median_interval_ms > 0
            else None
        )
        gap_multiple = (
            float(dt_ms) / float(median_interval_ms)
            if median_interval_ms and median_interval_ms > 0
            else None
        )

        output.append(
            {
                "subject": files.subject,
                "analysis_excluded": excluded,
                "exclusion_reason": exclusion_reason,
                "source_video": str(files.video),
                "source_timestamps": str(files.timestamps),
                "previous_video_frame_position": previous_position,
                "current_video_frame_position": current_position,
                "previous_capture_frame_idx": previous_frame_idx,
                "current_capture_frame_idx": current_frame_idx,
                "missing_capture_frame_indices": missing_capture_indices,
                "previous_unix_ms": previous_ms,
                "current_unix_ms": current_ms,
                "gap_duration_ms": dt_ms,
                "median_interval_ms_subject": median_interval_ms,
                "gap_multiple_of_median": gap_multiple,
                "estimated_missing_duration_ms": estimated_missing_duration_ms,
                "estimated_missing_frames_by_time": estimated_missing_frames_by_time,
                "phase_before": phase_before,
                "phase_after": phase_after,
                "phase_midpoint": phase_midpoint,
                "phase_crossing": phase_before != phase_after,
                "block_before": _block_from_phase(phase_before),
                "block_after": _block_from_phase(phase_after),
                "inside_analysis_span": bool(previous_ms < analysis_end and current_ms > analysis_start),
                "inside_formal_block": bool(
                    _block_from_phase(phase_before) is not None
                    or _block_from_phase(phase_after) is not None
                    or _block_from_phase(phase_midpoint) is not None
                ),
                "warning_threshold_ms": threshold_ms,
            }
        )
    return output


def run_gap_audit(config: Config) -> pd.DataFrame:
    records, _ = discover_rgb_subjects(config)
    rows: list[dict[str, object]] = []
    for record in records:
        rows.extend(subject_timestamp_gaps(record, config))
    if not rows:
        return pd.DataFrame(
            columns=[
                "subject",
                "analysis_excluded",
                "gap_duration_ms",
                "phase_midpoint",
                "inside_analysis_span",
                "inside_formal_block",
            ]
        )
    table = pd.DataFrame(rows)
    return table.sort_values(["subject", "previous_unix_ms"], kind="stable").reset_index(drop=True)
