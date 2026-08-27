"""Deterministic full-timeline QC selection and compact RITnet rendering.

Final QC selection is frame-centric: one selected video frame produces one
composite QC artifact even when both eyes have multiple anomaly reasons. Fixed
anchors come from ``frame_coverage`` (the complete formal frame timeline), so a
YOLO miss with zero eye rows can still be inspected. Anomaly examples are
sampled deterministically across each phase/reason rather than taking only the
first bad frames.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

from ritnet_fullclass_contract import (
    CLASS_BACKGROUND,
    CLASS_MAPPING,
    QC_OVERLAY_ALPHA,
    QC_PALETTE_BGR,
    normalize_subject,
)


QC_SELECTION_VERSION = "full-frame-timeline-even-anomaly-v1"
FRAME_FAILURE_REASONS = {
    "yolo_no_eye": "yolo_no_eye",
    "single_eye_success": "single_eye",
    "final_video_decode_failed": "final_video_decode_failed",
    "roi_invalid": "roi_invalid",
    "ritnet_no_success": "ritnet_no_success",
    "source_eye_without_final_result": "missing_final_result",
    "video_read_failed": "historical_video_read_failed",
}
EYE_BOOLEAN_REASONS = (
    ("qc_pupil_fragmented", "pupil_fragmented"),
    ("qc_iris_outer_fragmented", "iris_outer_fragmented"),
    ("qc_ocular_fragmented", "ocular_fragmented"),
    ("pupil_touches_valid_domain_edge", "pupil_real_boundary"),
    ("iris_outer_touches_valid_domain_edge", "iris_real_boundary"),
    ("ocular_touches_valid_domain_edge", "ocular_real_boundary"),
    ("temporal_anomaly", "temporal_jump"),
)
EYE_PADDING_COUNT_FIELDS = (
    "pupil_predicted_in_padding_pixels",
    "iris_outer_predicted_in_padding_pixels",
    "ocular_predicted_in_padding_pixels",
)


@dataclass(frozen=True)
class QCSelection:
    phase: str
    phase_segment: int
    frame_idx: int
    reasons: tuple[str, ...]
    eyes: tuple[str, ...]

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.phase, self.phase_segment, self.frame_idx)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return int(default)


def _to_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _frame_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("phase") or ""),
        _to_int(row.get("phase_segment"), 1),
        _to_int(row.get("frame_idx"), -1),
    )


def _even_sample_keys(
    keys: list[tuple[str, int, int]],
    limit: int,
) -> list[tuple[str, int, int]]:
    """Return at most ``limit`` keys spread deterministically over input order."""
    if limit <= 0 or not keys:
        return []
    if len(keys) <= limit:
        return list(keys)
    if limit == 1:
        return [keys[len(keys) // 2]]
    indices = [round(i * (len(keys) - 1) / (limit - 1)) for i in range(limit)]
    return [keys[index] for index in indices]


def _eye_anomaly_reasons(row: Mapping[str, Any]) -> set[str]:
    reasons: set[str] = set()
    for field, reason in EYE_BOOLEAN_REASONS:
        if _to_bool(row.get(field)) is True:
            reasons.add(reason)

    if any((_to_float(row.get(field)) or 0.0) > 0.0 for field in EYE_PADDING_COUNT_FIELDS):
        reasons.add("prediction_in_artificial_padding")

    threshold = _to_float(row.get("low_max_probability_threshold"))
    low_fraction = _to_float(row.get("ocular_low_max_probability_fraction"))
    if threshold is not None and low_fraction is not None and low_fraction > 0.0:
        reasons.add("low_model_confidence")

    if str(row.get("ritnet_status") or "").strip().lower() == "failed":
        failure = str(row.get("ritnet_failure_reason") or "")
        if failure.startswith("roi_invalid:"):
            reasons.add("roi_invalid")
        elif failure.startswith("source_video_decode_failed:"):
            reasons.add("final_video_decode_failed")
        else:
            reasons.add("ritnet_failed")
    return reasons


def build_qc_selections(
    *,
    frame_coverage_rows: Iterable[Mapping[str, Any]],
    eye_metric_rows: Iterable[Mapping[str, Any]],
    anomaly_limit_per_reason_per_phase: int,
    max_image_count: int,
) -> list[QCSelection]:
    """Select fixed-timeline and anomaly QC frames with deterministic caps.

    Fixed anchors are mandatory and are never silently dropped. If their count
    alone exceeds ``max_image_count``, configuration is invalid and the run
    fails closed. Non-fixed anomaly candidates are sampled evenly within each
    phase/reason, then merged by frame so multiple reasons do not multiply files.
    """
    anomaly_limit = int(anomaly_limit_per_reason_per_phase)
    image_limit = int(max_image_count)
    if anomaly_limit < 0:
        raise ValueError("anomaly_limit_per_reason_per_phase must be non-negative")
    if image_limit <= 0:
        raise ValueError("max_image_count must be positive")

    coverage = [dict(row) for row in frame_coverage_rows]
    order: dict[tuple[str, int, int], int] = {}
    fixed: set[tuple[str, int, int]] = set()
    candidate_reasons: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    candidate_eyes: dict[tuple[str, int, int], set[str]] = defaultdict(set)

    for ordinal, row in enumerate(coverage):
        key = _frame_key(row)
        if key[2] < 0:
            raise ValueError(f"invalid frame_coverage frame key: {key}")
        if key in order:
            raise ValueError(f"duplicate frame_coverage key: {key}")
        order[key] = ordinal
        if _to_bool(row.get("fixed_qc_anchor")) is True:
            fixed.add(key)
            candidate_reasons[key].add("fixed_anchor")
        coverage_reason = FRAME_FAILURE_REASONS.get(str(row.get("coverage_status") or ""))
        if coverage_reason:
            candidate_reasons[key].add(coverage_reason)

    for row in eye_metric_rows:
        key = _frame_key(row)
        if key not in order:
            raise ValueError(f"eye metric references frame absent from coverage: {key}")
        reasons = _eye_anomaly_reasons(row)
        if reasons:
            candidate_reasons[key].update(reasons)
            eye = str(row.get("eye") or "").strip()
            if eye:
                candidate_eyes[key].add(eye)

    if len(fixed) > image_limit:
        raise RuntimeError(
            f"fixed QC anchors alone exceed qc_image_max_count: {len(fixed)} > {image_limit}"
        )

    selected: set[tuple[str, int, int]] = set(fixed)
    selected_reasons: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    for key in fixed:
        selected_reasons[key].add("fixed_anchor")

    by_phase_reason: dict[tuple[str, str], list[tuple[str, int, int]]] = defaultdict(list)
    for key in sorted(candidate_reasons, key=lambda item: order[item]):
        for reason in sorted(candidate_reasons[key]):
            if reason == "fixed_anchor":
                continue
            by_phase_reason[(key[0], reason)].append(key)

    sampled_reason_keys: dict[tuple[str, str], list[tuple[str, int, int]]] = {}
    for phase_reason, keys in sorted(by_phase_reason.items()):
        sampled_reason_keys[phase_reason] = _even_sample_keys(keys, anomaly_limit)

    # Round-robin over phase/reason groups prevents one abundant anomaly type
    # from consuming the whole global image budget before other reasons appear.
    cursors = {group: 0 for group in sampled_reason_keys}
    groups = sorted(sampled_reason_keys)
    progress = True
    while len(selected) < image_limit and progress:
        progress = False
        for group in groups:
            keys = sampled_reason_keys[group]
            cursor = cursors[group]
            if cursor >= len(keys):
                continue
            progress = True
            key = keys[cursor]
            cursors[group] = cursor + 1
            selected_reasons[key].add(group[1])
            selected.add(key)
            if len(selected) >= image_limit:
                break

    # If a selected frame was selected for one reason, keep every other active
    # reason on that same frame in the index; this adds metadata, not images.
    for key in selected:
        selected_reasons[key].update(candidate_reasons.get(key, ()))

    result: list[QCSelection] = []
    for key in sorted(selected, key=lambda item: order[item]):
        result.append(
            QCSelection(
                phase=key[0],
                phase_segment=key[1],
                frame_idx=key[2],
                reasons=tuple(sorted(selected_reasons[key])),
                eyes=tuple(sorted(candidate_eyes.get(key, ()))),
            )
        )
    return result


def render_qc_images(
    roi_gray: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float = QC_OVERLAY_ALPHA,
) -> tuple[np.ndarray, np.ndarray]:
    """Return color labels and overlay at native RITnet label resolution."""
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    label_map = np.asarray(labels, dtype=np.uint8)
    if label_map.ndim != 2:
        raise ValueError(f"Expected 2-D label map, got shape={label_map.shape}")
    unknown = set(np.unique(label_map).tolist()) - set(CLASS_MAPPING)
    if unknown:
        raise ValueError(f"Unexpected RITnet class IDs in QC label map: {sorted(unknown)}")

    height, width = label_map.shape
    gray = np.asarray(roi_gray, dtype=np.uint8)
    if gray.ndim != 2:
        raise ValueError(f"Expected grayscale ROI, got shape={gray.shape}")
    resized_gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
    base = cv2.cvtColor(resized_gray, cv2.COLOR_GRAY2BGR)

    labels_color = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id, bgr in QC_PALETTE_BGR.items():
        labels_color[label_map == class_id] = np.asarray(bgr, dtype=np.uint8)
    blended = cv2.addWeighted(base, 1.0 - float(alpha), labels_color, float(alpha), 0.0)
    overlay = base.copy()
    ocular_mask = label_map != CLASS_BACKGROUND
    overlay[ocular_mask] = blended[ocular_mask]
    return labels_color, overlay


def _safe_token(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^0-9A-Za-z_-]+", "-", text)
    return text.strip("-") or fallback


def qc_frame_image_path(qc_dir: Path, subject: str, selection: QCSelection) -> Path:
    prefix = normalize_subject(subject)
    phase = _safe_token(selection.phase, "phase")
    stem = f"{prefix}_{phase}_s{selection.phase_segment:02d}_f{selection.frame_idx:08d}"
    return Path(qc_dir) / f"{stem}_qc.png"
