"""Deterministic QC sampling and rendering for RITnet full-class outputs."""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

from ritnet_fullclass_contract import (
    CLASS_BACKGROUND,
    CLASS_MAPPING,
    QC_ANOMALY_LIMIT_PER_REASON_PER_PHASE,
    QC_OVERLAY_ALPHA,
    QC_PALETTE_BGR,
    QC_STRIDE_FRAMES,
    normalize_subject,
)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return int(default)


def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def build_qc_anchor_frames(
    rows: Iterable[Mapping[str, Any]],
    stride_frames: int = QC_STRIDE_FRAMES,
) -> set[int]:
    """Return deterministic QC anchor frames.

    Within each phase/segment, retain first/middle/last available eye frame and
    approximately one frame every ``stride_frames`` from the first available frame.
    """
    if stride_frames <= 0:
        raise ValueError("stride_frames must be positive")

    groups: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row in rows:
        phase = str(row.get("phase", "") or "").strip() or "unknown"
        segment = _to_int(row.get("phase_segment"), 1)
        frame_idx = _to_int(row.get("frame_idx"), -1)
        if frame_idx >= 0:
            groups[(phase, segment)].add(frame_idx)

    anchors: set[int] = set()
    for frames_set in groups.values():
        frames = sorted(frames_set)
        if not frames:
            continue
        anchors.add(frames[0])
        anchors.add(frames[len(frames) // 2])
        anchors.add(frames[-1])

        next_target = frames[0]
        for frame_idx in frames:
            if frame_idx >= next_target:
                anchors.add(frame_idx)
                next_target = frame_idx + stride_frames
    return anchors


class QCSampler:
    """Select periodic anchors and a bounded number of anomaly examples."""

    def __init__(
        self,
        anchor_frames: set[int],
        anomaly_limit_per_reason_per_phase: int = QC_ANOMALY_LIMIT_PER_REASON_PER_PHASE,
    ) -> None:
        if anomaly_limit_per_reason_per_phase < 0:
            raise ValueError("anomaly_limit_per_reason_per_phase must be non-negative")
        self.anchor_frames = set(map(int, anchor_frames))
        self.anomaly_limit_per_reason_per_phase = int(anomaly_limit_per_reason_per_phase)
        self.anomaly_counts: dict[tuple[str, str], int] = defaultdict(int)
        self.reason_counts: dict[str, int] = defaultdict(int)
        self.saved_keys: set[tuple[int, str]] = set()

    def select(self, source_row: Mapping[str, Any], metrics: Mapping[str, Any]) -> list[str]:
        frame_idx = _to_int(source_row.get("frame_idx"), -1)
        eye = str(source_row.get("eye", "") or "eye")
        key = (frame_idx, eye)
        if key in self.saved_keys:
            return []

        reasons: list[str] = []
        if frame_idx in self.anchor_frames:
            reasons.append("anchor")

        candidates = {
            "roi_clipped": _to_bool(source_row.get("roi_clipped")) is True,
            "ritnet_missing": _to_bool(source_row.get("ritnet_found")) is False,
            "normalization_invalid": not bool(metrics.get("normalization_valid")),
            "ocular_fragmented": (
                _to_int(metrics.get("ocular_component_count"), 0) > 1
                and float(metrics.get("ocular_largest_component_fraction") or 1.0) < 0.90
            ),
        }
        phase = str(source_row.get("phase", "") or "").strip() or "unknown"
        for reason, active in candidates.items():
            counter_key = (phase, reason)
            if (
                active
                and self.anomaly_counts[counter_key]
                < self.anomaly_limit_per_reason_per_phase
            ):
                reasons.append(reason)
                self.anomaly_counts[counter_key] += 1

        if not reasons:
            return []

        self.saved_keys.add(key)
        for reason in reasons:
            self.reason_counts[reason] += 1
        return reasons


def render_qc_images(
    roi_gray: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float = QC_OVERLAY_ALPHA,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels_color, overlay) at native RITnet label resolution."""
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

    blended = cv2.addWeighted(
        base,
        1.0 - float(alpha),
        labels_color,
        float(alpha),
        0.0,
    )
    overlay = base.copy()
    ocular_mask = label_map != CLASS_BACKGROUND
    overlay[ocular_mask] = blended[ocular_mask]
    return labels_color, overlay


def _safe_token(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^0-9A-Za-z_-]+", "-", text)
    return text.strip("-") or fallback


def qc_image_paths(
    qc_dir: Path,
    subject: str,
    source_row: Mapping[str, Any],
) -> tuple[Path, Path]:
    prefix = normalize_subject(subject)
    phase = _safe_token(source_row.get("phase"), "phase")
    segment = _to_int(source_row.get("phase_segment"), 1)
    frame_idx = _to_int(source_row.get("frame_idx"), 0)
    eye = _safe_token(source_row.get("eye"), "eye")
    stem = f"{prefix}_{phase}_s{segment:02d}_f{frame_idx:08d}_{eye}"
    qc_dir = Path(qc_dir)
    return qc_dir / f"{stem}_labels.png", qc_dir / f"{stem}_overlay.png"


def save_qc_pair(
    qc_dir: Path,
    subject: str,
    source_row: Mapping[str, Any],
    roi_gray: np.ndarray,
    labels: np.ndarray,
) -> tuple[Path, Path]:
    qc_dir = Path(qc_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)
    labels_path, overlay_path = qc_image_paths(qc_dir, subject, source_row)
    labels_color, overlay = render_qc_images(roi_gray, labels)

    # OpenCV's Windows image writer can reject valid non-ASCII paths. Encode
    # first, then write bytes through pathlib so QC output remains portable.
    for image, path, kind in (
        (labels_color, labels_path, "labels"),
        (overlay, overlay_path, "overlay"),
    ):
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError(f"Failed to encode QC {kind} image: {path}")
        try:
            path.write_bytes(encoded.tobytes())
        except OSError as exc:
            raise RuntimeError(f"Failed to write QC {kind} image: {path}") from exc
    return labels_path, overlay_path
