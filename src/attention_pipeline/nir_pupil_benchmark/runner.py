"""Per-frame runner producing the unified seven-algorithm benchmark rows.

Two lifecycle modes are supported:

- ``independent``: a fresh detector is constructed for every (frame, algorithm)
  call. This is the fair per-frame comparison and removes all cross-frame state
  (PuReST previousPupil, Starburst startPoint, Pupil Labs 2D strong prior).
- ``continuous``: one detector per (subject, eye, algorithm) is kept and frames
  are processed in ``frame_idx`` order. This is the realistic temporal mode used
  for the dedicated temporal window and for jitter/catastrophic-failure metrics.

Main ``runtime_ms`` always measures the plain ``run()``. ``runWithConfidence``
is an optional separate pass with its own ``confidence_runtime_ms``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .adapters import (
    applied_parameter_snapshot,
    apply_scale_params,
    make_detector,
    pupil_diameter_bounds,
    run_detection,
    run_with_confidence,
)
from .core import parse_pupil_result
from .schema import ALGORITHMS, ALGORITHM_SPECS, RESULT_COLUMNS


def scale_params(algorithm: str, width: int, height: int) -> dict[str, Any]:
    """Return scale-rule parameter overrides for an algorithm on a given crop."""
    from .adapters import frozen_pupil_labs_properties, frozen_swirski_params

    if algorithm == "Swirski2D":
        return frozen_swirski_params(width, height)
    if algorithm == "PupilLabs2D":
        return frozen_pupil_labs_properties(width, height)
    return {}


def detect_crop(
    image: np.ndarray,
    algorithm: str,
    *,
    run_confidence: bool = False,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one algorithm on one crop (independent mode) and return fields.

    The returned dict holds the algorithm detection fields only (no frame
    identity); combine with ``assemble_row``.
    """
    if algorithm not in ALGORITHM_SPECS:
        raise KeyError(f"unknown algorithm: {algorithm!r}")
    spec = ALGORITHM_SPECS[algorithm]
    height, width = image.shape[:2]
    overrides = scale_params(algorithm, width, height)
    merged_params = dict(overrides)
    if params:
        merged_params.update(params)

    detector = make_detector(spec, merged_params)

    diameter_min_px = diameter_max_px = None
    if spec.supports_diameter_override:
        radius_min, radius_max = pupil_diameter_bounds(width, height)
        diameter_min_px = float(2 * radius_min)
        diameter_max_px = float(2 * radius_max)

    output = run_detection(
        spec, detector, image,
        diameter_min_px=diameter_min_px, diameter_max_px=diameter_max_px,
    )
    row = _detection_output_to_row(output, spec, width, height)
    actual_params = applied_parameter_snapshot(
        detector,
        spec,
        width=width,
        height=height,
        diameter_min_px=diameter_min_px,
        diameter_max_px=diameter_max_px,
    )

    if run_confidence:
        confidence = run_with_confidence(
            spec, image,
            diameter_min_px=diameter_min_px, diameter_max_px=diameter_max_px,
            params=merged_params,
        )
        row["outline_confidence"] = confidence.outline_confidence
        row["confidence_runtime_ms"] = confidence.confidence_runtime_ms
        if confidence.failure:
            row["failure"] = _join_failures(row.get("failure"), confidence.failure)

    row["params_provenance"] = json.dumps(
        {
            "algorithm": algorithm,
            "scale_rule_overrides": overrides,
            "user_params": params or {},
            "actual_applied": actual_params,
            "scale_rule": "docs/020-nir/030 section 6",
            "mode": "independent",
        },
        ensure_ascii=False,
    )
    return row


def _join_failures(existing: str | None, new: str) -> str:
    return f"{existing}; confidence:{new}" if existing else f"confidence:{new}"


def _detection_output_to_row(
    output: Any, spec: Any, width: int, height: int
) -> dict[str, Any]:
    parsed = parse_pupil_result(output.result, width=float(width), height=float(height))
    outline = parsed["outline_confidence"]
    if outline is not None and not np.isfinite(outline):
        outline = None
    row: dict[str, Any] = {
        "algorithm": spec.name,
        "algorithm_returned": bool(parsed["algorithm_returned"]),
        "official_valid": bool(parsed["official_valid"]),
        "geometry_sane": parsed["geometry_sane"],
        "center_x": parsed["center_x"],
        "center_y": parsed["center_y"],
        "major_axis": parsed["major_axis"],
        "minor_axis": parsed["minor_axis"],
        "angle_deg": parsed["angle_deg"],
        "diameter_geom": parsed["diameter_geom"],
        "area": parsed["area"],
        "runtime_ms": output.runtime_ms,
        "native_confidence": parsed["native_confidence"],
        "outline_confidence": outline,
        "confidence_runtime_ms": None,
        "failure": output.failure,
        "input_width": int(width),
        "input_height": int(height),
    }
    return row


def assemble_row(identity: Mapping[str, Any], detection: Mapping[str, Any]) -> dict[str, Any]:
    """Merge frame identity with detection fields into one unified schema row."""
    row: dict[str, Any] = {}
    for key in RESULT_COLUMNS:
        if key in identity:
            row[key] = identity[key]
        elif key in detection:
            row[key] = detection[key]
        else:
            row[key] = None
    return row


class VideoFrameSource:
    """Decode source-video frames on demand and crop eye regions in memory.

    Avoids materializing PNG crops to disk for full-video runs: each source
    video keeps one open ``cv2.VideoCapture`` and one cached decoded frame.
    ``crop(row)`` uses the row's ``source_video`` / ``frame_idx`` / bbox.
    Must be closed (releases captures) when done.
    """

    def __init__(self) -> None:
        self._caps: dict[str, Any] = {}
        self._last: dict[str, tuple[int, np.ndarray]] = {}

    def gray_frame(self, video: str, frame_idx: int) -> np.ndarray:
        import cv2

        video = str(video)
        frame_idx = int(frame_idx)
        last = self._last.get(video)
        if last is not None and last[0] == frame_idx:
            return last[1]
        cap = self._caps.get(video)
        if cap is None:
            cap = cv2.VideoCapture(video)
            if not cap.isOpened():
                raise RuntimeError(f"cannot open source video: {video}")
            self._caps[video] = cap
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx):
            raise RuntimeError(f"video seek failed: {video} frame {frame_idx}")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"video read failed: {video} frame {frame_idx}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        self._last[video] = (frame_idx, gray)
        return gray

    def crop(self, row: Mapping[str, Any]) -> np.ndarray:
        frame = self.gray_frame(str(row["source_video"]), int(row["frame_idx"]))
        x1, y1, x2, y2 = (
            int(row["bbox_x1"]), int(row["bbox_y1"]), int(row["bbox_x2"]), int(row["bbox_y2"])
        )
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError(f"empty crop: {row.get('input_kind')} frame {row['frame_idx']}")
        return crop

    def close(self) -> None:
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()
        self._last.clear()


def run_crop_list(
    rows: Sequence[Mapping[str, Any]],
    algorithms: Iterable[str],
    *,
    crop_root: str | Path,
    run_confidence: bool = False,
    mode: str = "independent",
    image_source: str = "disk",
) -> pd.DataFrame:
    """Run detection over a manifest of crops and return the unified frame.

    Each input row needs either ``crop_path`` (relative to ``crop_root``) when
    ``image_source="disk"``, or ``source_video`` + ``frame_idx`` + bbox columns
    when ``image_source="video"`` (frames decoded in memory, nothing written).
    In ``continuous`` mode rows are grouped by (subject, eye, sequence_id) and
    processed in frame order.  A continuous sequence must use one fixed source
    coordinate canvas: input dimensions and bbox coordinates cannot change.
    Scale-sensitive properties are still re-applied before every call and the
    actual values are written to provenance.
    """
    if image_source not in ("disk", "video"):
        raise ValueError(f"image_source must be 'disk' or 'video', got {image_source!r}")
    algorithms = list(algorithms)
    for algorithm in algorithms:
        if algorithm not in ALGORITHM_SPECS:
            raise KeyError(f"unknown algorithm: {algorithm!r}")
    crop_root = Path(crop_root)
    frame_source = VideoFrameSource() if image_source == "video" else None
    try:
        return _run_crop_list_impl(
            rows, algorithms, crop_root=crop_root,
            run_confidence=run_confidence, mode=mode, frame_source=frame_source,
        )
    finally:
        if frame_source is not None:
            frame_source.close()


def _run_crop_list_impl(
    rows: Sequence[Mapping[str, Any]],
    algorithms: Iterable[str],
    *,
    crop_root: Path,
    run_confidence: bool,
    mode: str,
    frame_source: VideoFrameSource | None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if mode == "continuous":
        for subject, eye, sequence_id, group in _group_continuous(rows):
            group = sorted(group, key=lambda r: int(r.get("frame_idx", 0)))
            first_image = _load_crop(crop_root, group[0], frame_source=frame_source)
            fw, fh = first_image.shape[1], first_image.shape[0]
            _validate_fixed_continuous_canvas(
                group, crop_root, fw, fh, frame_source=frame_source
            )
            detectors = {
                algorithm: make_detector(
                    ALGORITHM_SPECS[algorithm],
                    scale_params(algorithm, fw, fh),
                )
                for algorithm in algorithms
            }
            for row in group:
                image = _load_crop(crop_root, row, frame_source=frame_source)
                for algorithm in algorithms:
                    spec = ALGORITHM_SPECS[algorithm]
                    applied = apply_scale_params(
                        detectors[algorithm], spec, image.shape[1], image.shape[0]
                    )
                    output = _run_on_shared(detectors[algorithm], spec, image)
                    detection = _detection_output_to_row(
                        output, spec, image.shape[1], image.shape[0]
                    )
                    detection = _maybe_confidence(
                        detection, spec, image, run_confidence, params=applied
                    )
                    radius_min = radius_max = None
                    if spec.supports_diameter_override:
                        radius_min, radius_max = pupil_diameter_bounds(
                            image.shape[1], image.shape[0]
                        )
                    actual = applied_parameter_snapshot(
                        detectors[algorithm],
                        spec,
                        width=image.shape[1],
                        height=image.shape[0],
                        diameter_min_px=float(2 * radius_min) if radius_min else None,
                        diameter_max_px=float(2 * radius_max) if radius_max else None,
                    )
                    detection["params_provenance"] = _provenance(
                        algorithm,
                        image.shape[1],
                        image.shape[0],
                        actual_applied=actual,
                        sequence_id=sequence_id,
                    )
                    records.append(assemble_row(row, detection))
    else:
        for row in rows:
            image = _load_crop(crop_root, row, frame_source=frame_source)
            for algorithm in algorithms:
                detection = detect_crop(image, algorithm, run_confidence=run_confidence)
                records.append(assemble_row(row, detection))

    frame = pd.DataFrame(records, columns=list(RESULT_COLUMNS))
    for column in (
        "center_x", "center_y", "major_axis", "minor_axis", "angle_deg",
        "diameter_geom", "area", "runtime_ms", "native_confidence",
        "outline_confidence", "confidence_runtime_ms",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _group_continuous(rows: Sequence[Mapping[str, Any]]):
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        sequence_id = str(row.get("sequence_id", "")).strip()
        if not sequence_id:
            raise ValueError("continuous mode requires a non-empty sequence_id")
        key = (str(row.get("subject", "")), str(row.get("eye", "")), sequence_id)
        groups.setdefault(key, []).append(dict(row))
    return (
        (subject, eye, sequence_id, items)
        for (subject, eye, sequence_id), items in groups.items()
    )


def _validate_fixed_continuous_canvas(
    rows: Sequence[Mapping[str, Any]], crop_root: Path, width: int, height: int,
    frame_source: VideoFrameSource | None = None,
) -> None:
    bbox_keys = ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
    reference = tuple(rows[0].get(key) for key in bbox_keys)
    for row in rows:
        current = tuple(row.get(key) for key in bbox_keys)
        if current != reference:
            raise ValueError(
                "continuous mode requires a fixed source-coordinate bbox; "
                "moving tight crops would invalidate detector state"
            )
        image = _load_crop(crop_root, row, frame_source=frame_source)
        if image.shape[:2] != (height, width):
            raise ValueError("continuous mode requires constant input dimensions")


def _load_crop(
    crop_root: Path,
    row: Mapping[str, Any],
    frame_source: VideoFrameSource | None = None,
) -> np.ndarray:
    if frame_source is not None:
        return frame_source.crop(row)
    import cv2

    crop_path = row.get("crop_path")
    if not crop_path:
        raise ValueError("each manifest row needs a 'crop_path'")
    path = crop_root / str(crop_path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"could not decode crop image: {path}")
    return image


def _run_on_shared(detector: Any, spec: Any, image: np.ndarray) -> Any:
    if spec.supports_diameter_override:
        radius_min, radius_max = pupil_diameter_bounds(image.shape[1], image.shape[0])
        return run_detection(
            spec, detector, image,
            diameter_min_px=float(2 * radius_min), diameter_max_px=float(2 * radius_max),
        )
    return run_detection(spec, detector, image)


def _maybe_confidence(
    detection: dict[str, Any], spec: Any, image: np.ndarray, run_confidence: bool,
    *, params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not run_confidence:
        return detection
    radius_min = radius_max = None
    if spec.supports_diameter_override:
        radius_min, radius_max = pupil_diameter_bounds(image.shape[1], image.shape[0])
    confidence = run_with_confidence(
        spec, image,
        diameter_min_px=float(2 * radius_min) if radius_min else None,
        diameter_max_px=float(2 * radius_max) if radius_max else None,
        params=params,
    )
    detection["outline_confidence"] = confidence.outline_confidence
    detection["confidence_runtime_ms"] = confidence.confidence_runtime_ms
    if confidence.failure:
        detection["failure"] = _join_failures(detection.get("failure"), confidence.failure)
    return detection


def _provenance(
    algorithm: str,
    width: int,
    height: int,
    *,
    actual_applied: Mapping[str, Any],
    sequence_id: str,
) -> str:
    return json.dumps(
        {
            "algorithm": algorithm,
            "scale_rule_overrides": scale_params(algorithm, width, height),
            "actual_applied": dict(actual_applied),
            "scale_rule": "docs/020-nir/030 section 6",
            "mode": "continuous",
            "sequence_id": sequence_id,
            "coordinate_contract": "fixed_source_pixel_canvas",
        },
        ensure_ascii=False,
    )
