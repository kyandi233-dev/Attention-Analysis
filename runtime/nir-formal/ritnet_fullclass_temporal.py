"""Deterministic temporal QC facts for final RITnet eye metrics.

Only truly consecutive frames of the same eye within the same phase/segment are
compared. Missing-eye gaps, phase/segment boundaries and RITnet failures reset
the delta chain. Robust anomaly evidence uses a rolling median/MAD baseline from
previous consecutive deltas only, so the current jump cannot contaminate the
baseline that judges it. Temporal anomalies are QC facts and never delete rows.
"""
from __future__ import annotations

from collections import deque
from math import hypot, isfinite
from statistics import median
from typing import Any, Iterable, Iterator, Mapping


ROBUST_Z_SCALE = 0.6744897501960817
TEMPORAL_ANOMALY_THRESHOLD = 6.0
TEMPORAL_ANOMALY_MIN_SAMPLES = 8
TEMPORAL_ANOMALY_WINDOW = 120
TEMPORAL_JUMP_SCORE_CAP = 1_000_000.0
TEMPORAL_QC_VERSION = "consecutive-eye-rolling-mad-v2-thr6-min8-w120-cap1e6"
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

ROBUST_DELTA_FIELDS = (
    *(target for _, target in DELTA_SPECS),
    "delta_pupil_center_distance_px",
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


def _robust_z(value: float, history: deque[float]) -> float | None:
    if len(history) < TEMPORAL_ANOMALY_MIN_SAMPLES:
        return None
    center = float(median(history))
    deviations = [abs(item - center) for item in history]
    mad = float(median(deviations))
    distance = abs(float(value) - center)
    tolerance = max(1e-12, 1e-9 * max(1.0, abs(center)))
    if mad <= tolerance:
        return 0.0 if distance <= tolerance else TEMPORAL_JUMP_SCORE_CAP
    score = float(ROBUST_Z_SCALE * distance / mad)
    return min(score, TEMPORAL_JUMP_SCORE_CAP)


def _jump_score(
    temporal: Mapping[str, Any],
    history: Mapping[str, deque[float]],
) -> float | None:
    scores: list[float] = []
    for field in ROBUST_DELTA_FIELDS:
        value = _finite_float(temporal.get(field))
        if value is None:
            continue
        score = _robust_z(abs(value), history[field])
        if score is not None:
            scores.append(score)
    return max(scores) if scores else None


def _append_history(
    temporal: Mapping[str, Any],
    history: Mapping[str, deque[float]],
) -> None:
    for field in ROBUST_DELTA_FIELDS:
        value = _finite_float(temporal.get(field))
        if value is not None:
            history[field].append(abs(value))


def _new_history() -> dict[str, deque[float]]:
    return {
        field: deque(maxlen=TEMPORAL_ANOMALY_WINDOW)
        for field in ROBUST_DELTA_FIELDS
    }


def iter_temporal_facts(rows: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
    """Stream copies of rows with gap-safe deltas and robust jump evidence."""
    previous_by_eye: dict[str, dict[str, Any]] = {}
    history_by_group: dict[tuple[str, int, str], dict[str, deque[float]]] = {}

    for source in rows:
        row = dict(source)
        temporal = empty_temporal()
        eye = str(row.get("eye") or "").strip()
        if eye not in ALLOWED_EYES:
            raise ValueError(f"unsupported eye label for temporal QC: {eye!r}")
        frame_idx = _int(row.get("frame_idx"), "frame_idx")
        phase = str(row.get("phase") or "")
        segment = _int(row.get("phase_segment"), "phase_segment")
        group_key = (phase, segment, eye)
        history = history_by_group.setdefault(group_key, _new_history())

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

        if temporal["temporal_reset_reason"] is not None:
            history_by_group[group_key] = _new_history()
            history = history_by_group[group_key]
            temporal["temporal_jump_score"] = None
            temporal["temporal_anomaly"] = None
        else:
            score = _jump_score(temporal, history)
            temporal["temporal_jump_score"] = score
            temporal["temporal_anomaly"] = (
                None if score is None else bool(score >= TEMPORAL_ANOMALY_THRESHOLD)
            )
            _append_history(temporal, history)

        row.update(temporal)
        previous_by_eye[eye] = row
        yield row


def add_temporal_facts(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """List helper for tests and small callers."""
    return list(iter_temporal_facts(rows))
