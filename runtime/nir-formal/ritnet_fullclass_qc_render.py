"""Single-frame QC rendering for lean final RITnet output."""
from __future__ import annotations

from typing import Any, Mapping

import cv2
import numpy as np

from ritnet_fullclass_qc import QCSelection


QC_COMPOSITE_VERSION = "original-frame-plus-two-eye-overlays-v2-pupil-only"
PANEL_WIDTH = 640
PANEL_HEIGHT = 400
HEADER_HEIGHT = 54
EYE_ORDER = ("frame_left", "frame_right")


def _finite(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _text(canvas: np.ndarray, text: str, x: int, y: int, *, scale: float = 0.48) -> None:
    cv2.putText(
        canvas,
        str(text),
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(scale),
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )


def _clip_box(row: Mapping[str, Any], prefix: str, width: int, height: int) -> tuple[int, int, int, int] | None:
    values = [_finite(row.get(f"{prefix}_{name}")) for name in ("x1", "y1", "x2", "y2")]
    if any(value is None for value in values):
        return None
    x1, y1, x2, y2 = (int(round(value)) for value in values)
    x1 = min(max(x1, 0), width - 1)
    y1 = min(max(y1, 0), height - 1)
    x2 = min(max(x2, 0), width - 1)
    y2 = min(max(y2, 0), height - 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _draw_source_boxes(frame: np.ndarray, eye_rows: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
    canvas = np.asarray(frame).copy()
    height, width = canvas.shape[:2]
    for eye in EYE_ORDER:
        row = eye_rows.get(eye)
        if not row:
            continue
        yolo = _clip_box(row, "yolo_bbox", width, height)
        roi = _clip_box(row, "roi_source", width, height)
        if roi:
            cv2.rectangle(canvas, roi[:2], roi[2:], (0, 215, 255), 2)
        if yolo:
            cv2.rectangle(canvas, yolo[:2], yolo[2:], (0, 255, 0), 2)
            label = "L" if eye == "frame_left" else "R"
            cv2.putText(
                canvas,
                f"YOLO-{label}",
                (yolo[0], max(16, yolo[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
    return canvas


def _draw_ellipse(panel: np.ndarray, row: Mapping[str, Any], prefix: str, color: tuple[int, int, int]) -> None:
    if not _bool(row.get(f"{prefix}_fit_valid")):
        return
    cx = _finite(row.get(f"{prefix}_center_x"))
    cy = _finite(row.get(f"{prefix}_center_y"))
    short_axis = _finite(row.get(f"{prefix}_short_axis"))
    long_axis = _finite(row.get(f"{prefix}_long_axis"))
    angle = _finite(row.get(f"{prefix}_angle_deg"))
    if None in (cx, cy, short_axis, long_axis, angle):
        return
    if short_axis <= 0 or long_axis <= 0:
        return
    cv2.ellipse(
        panel,
        (int(round(cx)), int(round(cy))),
        (max(1, int(round(long_axis / 2.0))), max(1, int(round(short_axis / 2.0)))),
        float(angle),
        0,
        360,
        color,
        2,
        cv2.LINE_AA,
    )


def _header(panel: np.ndarray, title: str, lines: list[str]) -> np.ndarray:
    content = cv2.resize(panel, (PANEL_WIDTH, PANEL_HEIGHT), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((PANEL_HEIGHT + HEADER_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    canvas[HEADER_HEIGHT:] = content
    _text(canvas, title, 8, 19, scale=0.52)
    if lines:
        _text(canvas, " | ".join(lines), 8, 42, scale=0.39)
    return canvas


def _eye_panel(eye: str, row: Mapping[str, Any] | None, overlay: np.ndarray | None) -> np.ndarray:
    if overlay is None:
        panel = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), 35, dtype=np.uint8)
    else:
        panel = np.asarray(overlay, dtype=np.uint8).copy()
        if panel.shape != (PANEL_HEIGHT, PANEL_WIDTH, 3):
            panel = cv2.resize(panel, (PANEL_WIDTH, PANEL_HEIGHT), interpolation=cv2.INTER_AREA)

    status = "not_detected" if not row else str(row.get("ritnet_status") or "unknown")
    if row:
        # Only the pupil ellipse is a formal geometric overlay. Iris remains a
        # segmentation class but is not fitted into a diagnostic ellipse/PIR.
        _draw_ellipse(panel, row, "pupil", (0, 0, 255))

    lines = [f"status={status}"]
    if row:
        pupil_d = _finite(row.get("pupil_geom_mean_diameter"))
        maxp = _finite(row.get("ocular_max_probability_mean"))
        entropy = _finite(row.get("ocular_entropy_mean"))
        if pupil_d is not None:
            lines.append(f"pupilD={pupil_d:.2f}px")
        if maxp is not None:
            lines.append(f"maxP={maxp:.3f}")
        if entropy is not None:
            lines.append(f"H={entropy:.3f}")
        failure = str(row.get("ritnet_failure_reason") or "")
        if failure:
            lines.append(failure[:48])
    title = "LEFT EYE" if eye == "frame_left" else "RIGHT EYE"
    return _header(panel, title, lines)


def render_qc_composite(
    *,
    frame_bgr: np.ndarray | None,
    selection: QCSelection,
    coverage_row: Mapping[str, Any],
    eye_metric_rows: Mapping[str, Mapping[str, Any]],
    eye_overlays: Mapping[str, np.ndarray],
    fallback_frame_size: tuple[int, int] = (640, 480),
) -> np.ndarray:
    if frame_bgr is None:
        width, height = map(int, fallback_frame_size)
        original = np.zeros((max(height, 1), max(width, 1), 3), dtype=np.uint8)
        _text(original, "SOURCE FRAME UNAVAILABLE", 20, 40, scale=0.8)
    else:
        original = np.asarray(frame_bgr)
        if original.ndim == 2:
            original = cv2.cvtColor(original.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        elif original.ndim == 3 and original.shape[2] == 3:
            original = original.astype(np.uint8, copy=True)
        else:
            raise ValueError(f"invalid QC source frame shape: {original.shape}")

    original = _draw_source_boxes(original, eye_metric_rows)
    coverage_status = str(coverage_row.get("coverage_status") or "unknown")
    original_panel = _header(
        original,
        "ORIGINAL FRAME",
        [
            f"frame={selection.frame_idx}",
            f"coverage={coverage_status}",
            f"reasons={','.join(selection.reasons)[:90]}",
        ],
    )
    left_panel = _eye_panel(
        "frame_left",
        eye_metric_rows.get("frame_left"),
        eye_overlays.get("frame_left"),
    )
    right_panel = _eye_panel(
        "frame_right",
        eye_metric_rows.get("frame_right"),
        eye_overlays.get("frame_right"),
    )
    return np.ascontiguousarray(np.concatenate([original_panel, left_panel, right_panel], axis=1))
