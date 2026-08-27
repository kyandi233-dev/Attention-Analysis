"""Deterministic first-order temporal QC facts for final RITnet eye metrics.

Only truly consecutive frames of the same eye within the same phase/segment are
compared. Missing-eye gaps and phase boundaries reset the delta chain so a
multi-frame gap cannot masquerade as a one-frame segmentation jump. Robust
anomaly thresholds are intentionally left for sub-031 qualification.
"""
from __future__ import annotations

from math import hypot, isfinite
from typing import Any, Iterable, Iterator, Mapping


TEMPORAL_QC_VERSION = "consecutive-eye-delta-v1"
ALLOWED_EYES = frozenset({"frame_left", "frame_right"})

DELTA_SPECS = (
    ("hard_pupil_fraction", "delta_hard_pupil_fraction"),
    ("hard_iris_outer_fraction", "delta_hard_iris_outer_fraction"),
    ("hard_ocular_fraction", "delta_hard_ocular_fraction"),
    ("pupil_to_iris_diameter_ratio", "delta_pupil_to_iris_diameter_ratio"),
    ("ocular_aperture_ratio_median", "delta_ocular_aperture_ratio_median"),
    ("ocular_max_probability_mean", "delta_ocular_max_probability_mean"),
    ("ocular_top1_top2_margin_mean", "delta_ocular_top1_top2_margin_mean"),
    ("ocular_entropy_mean", "delta_ocular_entropy_mean"),
)

TEMPORAL_OUTPUT_FIELDS = (
    "temporal_qc_version",
    "temporal_prev_frame_idx",
    "temporal_frame_gap",
    "temporal_time_gap_ms",
    "temporal_reset_reason",
    *(target for _, target in DELTA_SPECS),
    "delta_pupil_center_x",
    "delta_pupil_center_y",
    "delta_pupil_center_distance_px",
    "temporal_jump_score",
    "temporal_anomaly",
)


def _int(value: Any, name: str) -> int:
    if value is None or str(value).strip() == "":
        raise ValueError(f"missing {name}")
    return int(float(value))


def _finite_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _time_value(row: Mapping[str, Any]) -> float | None:
    unix_ms = _finite_float(row.get("unix_ms"))
    if unix_ms is not None:
        return unix_ms
    return _finite_float(row.get("video_time_ms"))


def empty_temporal() -> dict[str, Any]:
    result = {field: None for field in TEMPORAL_OUTPUT_FIELDS}
    result["temporal_qc_version"] = TEMPORAL_QC_VERSION
    return result


def _success(row: Mapping[str, Any]) -> bool:
    return str(row.get("ritnet_status") or "").strip().lower() == "success"


def iter_temporal_facts(rows: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
    """Stream copies of rows with deterministic temporal fields added."""
    previous_by_eye: dict[str, dict[str, Any]] = {}

    for source in rows:
        row = dict(source)
        temporal = empty_temporal()
        eye = str(row.get("eye") or "").strip()
        if eye not in ALLOWED_EYES:
            raise ValueError(f"unsupported eye label for temporal QC: {eye!r}")
        frame_idx = _int(row.get("frame_idx"), "frame_idx")
        phase = str(row.get("phase") or "")
        segment = _int(row.get("phase_segment"), "phase_segment")

        previous = previous_by_eye.get(eye)
        if previous is None:
            temporal["temporal_reset_reason"] = "first_observation"
        else:
            previous_frame = _int(previous.get("frame_idx"), "previous frame_idx")
            frame_gap = frame_idx - previous_frame
            if frame_gap <= 0:
                raise ValueError(
                    f"same-eye frame order must be strictly increasing: eye={eye}, "
                    f"previous={previous_frame}, current={frame_idx}"
                )
            temporal["temporal_prev_frame_idx"] = previous_frame
            temporal["temporal_frame_gap"] = frame_gap
            current_time = _time_value(row)
            previous_time = _time_value(previous)
            if current_time is not None and previous_time is not None:
                temporal["temporal_time_gap_ms"] = float(current_time - previous_time)

            previous_phase = str(previous.get("phase") or "")
            previous_segment = _int(previous.get("phase_segment"), "previous phase_segment")
            if (phase, segment) != (previous_phase, previous_segment):
                temporal["temporal_reset_reason"] = "phase_or_segment_boundary"
            elif frame_gap != 1:
                temporal["temporal_reset_reason"] = "nonconsecutive_frame_gap"
            elif not (_success(previous) and _success(row)):
                temporal["temporal_reset_reason"] = "ritnet_not_success"
            else:
                temporal["temporal_reset_reason"] = None
                for source_field, target_field in DELTA_SPECS:
                    current = _finite_float(row.get(source_field))
                    old = _finite_float(previous.get(source_field))
                    temporal[target_field] = (
                        float(current - old) if current is not None and old is not None else None
                    )

                current_x = _finite_float(row.get("pupil_center_x"))
                current_y = _finite_float(row.get("pupil_center_y"))
                old_x = _finite_float(previous.get("pupil_center_x"))
                old_y = _finite_float(previous.get("pupil_center_y"))
                if None not in (current_x, current_y, old_x, old_y):
                    dx = float(current_x - old_x)
                    dy = float(current_y - old_y)
                    temporal["delta_pupil_center_x"] = dx
                    temporal["delta_pupil_center_y"] = dy
                    temporal["delta_pupil_center_distance_px"] = float(hypot(dx, dy))

        temporal["temporal_jump_score"] = None
        temporal["temporal_anomaly"] = None
        row.update(temporal)
        previous_by_eye[eye] = row
        yield row


def add_temporal_facts(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """List helper for tests and small callers."""
    return list(iter_temporal_facts(rows))
