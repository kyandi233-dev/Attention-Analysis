"""Frame-level coverage reconstruction and fixed-timeline QC anchors.

Coverage is anchored to the historical formal ``frames.csv`` rather than to
successful eye rows. Therefore YOLO misses, single-eye frames and video failures
remain visible in the final output and can be sampled for QC.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from ritnet_fullclass_schema import FRAME_COVERAGE_FIELDS, FRAME_COVERAGE_SCHEMA_VERSION, project_row


ALLOWED_EYES = frozenset({"frame_left", "frame_right"})
FIXED_QC_ANCHOR_VERSION = "phase-time-interval-v1"


def _int(value: Any, name: str) -> int:
    if value is None or str(value).strip() == "":
        raise ValueError(f"missing {name}")
    return int(float(value))


def _float_or_none(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def load_source_frames(path: Path, subject: str) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "phase",
            "phase_segment",
            "frame_idx",
            "status",
            "selected_eye_count",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"frames.csv missing required columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"frames.csv contains no rows: {path}")

    seen: set[tuple[str, int, int]] = set()
    for ordinal, row in enumerate(rows):
        if row.get("subject") and str(row["subject"]).strip() != subject:
            raise ValueError(f"frames.csv subject mismatch at row {ordinal}")
        key = (
            str(row["phase"]),
            _int(row["phase_segment"], "phase_segment"),
            _int(row["frame_idx"], "frame_idx"),
        )
        if key in seen:
            raise ValueError(f"duplicate frame coverage source key: {key}")
        seen.add(key)
    return rows


def build_fixed_qc_anchor_keys(
    frame_rows: Iterable[Mapping[str, Any]],
    *,
    interval_sec: float,
) -> set[tuple[str, int, int]]:
    """Select first/middle/last plus fixed elapsed-time anchors per phase segment."""
    interval_ms = float(interval_sec) * 1000.0
    if not interval_ms > 0:
        raise ValueError("QC interval_sec must be positive")

    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        phase = str(row.get("phase") or "")
        segment = _int(row.get("phase_segment"), "phase_segment")
        groups[(phase, segment)].append(row)

    anchors: set[tuple[str, int, int]] = set()
    for (phase, segment), group in groups.items():
        ordered = sorted(group, key=lambda row: _int(row.get("frame_idx"), "frame_idx"))
        if not ordered:
            continue
        for index in (0, len(ordered) // 2, len(ordered) - 1):
            anchors.add((phase, segment, _int(ordered[index].get("frame_idx"), "frame_idx")))

        timed = [
            (row, _float_or_none(row.get("video_time_ms")))
            for row in ordered
        ]
        timed = [(row, value) for row, value in timed if value is not None]
        if timed:
            next_target = timed[0][1]
            for row, value in timed:
                if value + 1e-9 >= next_target:
                    anchors.add((phase, segment, _int(row.get("frame_idx"), "frame_idx")))
                    next_target = value + interval_ms
        else:
            raise ValueError(
                f"frame timeline for {phase}/segment{segment} lacks video_time_ms; "
                "fixed-time QC cannot be constructed deterministically"
            )
    return anchors


def build_frame_coverage(
    *,
    subject: str,
    source_frames: Iterable[Mapping[str, Any]],
    source_eye_rows: Iterable[Mapping[str, Any]],
    final_eye_rows: Iterable[Mapping[str, Any]],
    fixed_anchor_keys: set[tuple[str, int, int]],
) -> list[dict[str, Any]]:
    """Build exactly one final coverage row for every formal source frame."""
    source_eyes: dict[tuple[str, int, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in source_eye_rows:
        phase = str(row.get("phase") or "")
        segment = _int(row.get("phase_segment"), "phase_segment")
        frame = _int(row.get("frame_idx"), "frame_idx")
        eye = str(row.get("eye") or "").strip()
        if eye not in ALLOWED_EYES:
            raise ValueError(f"unsupported source eye label: {eye!r}")
        if eye in source_eyes[(phase, segment, frame)]:
            raise ValueError(f"duplicate source eye for frame: {(phase, segment, frame, eye)}")
        source_eyes[(phase, segment, frame)][eye] = row

    final_eyes: dict[tuple[str, int, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in final_eye_rows:
        phase = str(row.get("phase") or "")
        segment = _int(row.get("phase_segment"), "phase_segment")
        frame = _int(row.get("frame_idx"), "frame_idx")
        eye = str(row.get("eye") or "").strip()
        if eye not in ALLOWED_EYES:
            raise ValueError(f"unsupported final eye label: {eye!r}")
        if eye in final_eyes[(phase, segment, frame)]:
            raise ValueError(f"duplicate final eye for frame: {(phase, segment, frame, eye)}")
        final_eyes[(phase, segment, frame)][eye] = row

    output: list[dict[str, Any]] = []
    source_keys: set[tuple[str, int, int]] = set()
    for frame_row in source_frames:
        phase = str(frame_row.get("phase") or "")
        segment = _int(frame_row.get("phase_segment"), "phase_segment")
        frame = _int(frame_row.get("frame_idx"), "frame_idx")
        key = (phase, segment, frame)
        if key in source_keys:
            raise ValueError(f"duplicate source frame key: {key}")
        source_keys.add(key)
        eyes = source_eyes.get(key, {})
        finals = final_eyes.get(key, {})

        def status_for(eye: str) -> tuple[str, str | None]:
            if eye not in eyes:
                return "not_detected", None
            final = finals.get(eye)
            if final is None:
                return "missing_final_eye_row", "source_eye_present_but_final_row_missing"
            status = str(final.get("ritnet_status") or "unknown")
            reason = final.get("ritnet_failure_reason")
            return status, None if reason in (None, "") else str(reason)

        left_status, left_reason = status_for("frame_left")
        right_status, right_reason = status_for("frame_right")
        success_count = int(left_status == "success") + int(right_status == "success")
        selected_eye_count = _int(frame_row.get("selected_eye_count"), "selected_eye_count")
        source_status = str(frame_row.get("status") or "")

        if source_status == "video_read_failed":
            coverage_status = "video_read_failed"
        elif selected_eye_count == 0:
            coverage_status = "yolo_no_eye"
        elif success_count == 2:
            coverage_status = "both_eyes_success"
        elif success_count == 1:
            coverage_status = "single_eye_success"
        elif finals:
            coverage_status = "ritnet_no_success"
        else:
            coverage_status = "source_eye_without_final_result"

        row = {
            "frame_coverage_schema_version": FRAME_COVERAGE_SCHEMA_VERSION,
            "subject": subject,
            "phase": phase,
            "phase_segment": segment,
            "frame_idx": frame,
            "video_time_ms": _float_or_none(frame_row.get("video_time_ms")),
            "unix_ms": _float_or_none(frame_row.get("unix_ms")),
            "phase_time_ms": _float_or_none(frame_row.get("phase_time_ms")),
            "source_frame_status": source_status,
            "source_raw_detection_count": _int(frame_row.get("raw_detection_count") or 0, "raw_detection_count"),
            "source_selected_eye_count": selected_eye_count,
            "source_left_eye_present": "frame_left" in eyes,
            "source_right_eye_present": "frame_right" in eyes,
            "left_ritnet_status": left_status,
            "left_failure_reason": left_reason,
            "right_ritnet_status": right_status,
            "right_failure_reason": right_reason,
            "ritnet_success_eye_count": success_count,
            "coverage_status": coverage_status,
            "fixed_qc_anchor": key in fixed_anchor_keys,
        }
        output.append(project_row(row, FRAME_COVERAGE_FIELDS))

    unexpected_source_eye_keys = set(source_eyes) - source_keys
    unexpected_final_keys = set(final_eyes) - source_keys
    if unexpected_source_eye_keys:
        raise ValueError(f"source eyes reference frames absent from frames.csv: {sorted(unexpected_source_eye_keys)[:5]}")
    if unexpected_final_keys:
        raise ValueError(f"final eyes reference frames absent from frames.csv: {sorted(unexpected_final_keys)[:5]}")
    return output
