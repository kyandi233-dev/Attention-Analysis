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

    if run_confidence:
        confidence = run_with_confidence(
            spec, image,
            diameter_min_px=diameter_min_px, diameter_max_px=diameter_max_px,
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


def run_crop_list(
    rows: Sequence[Mapping[str, Any]],
    algorithms: Iterable[str],
    *,
    crop_root: str | Path,
    run_confidence: bool = False,
    mode: str = "independent",
) -> pd.DataFrame:
    """Run detection over a manifest of crops and return the unified frame.

    Each input row needs at least ``crop_path`` (relative to ``crop_root``) and
    any identity columns (subject/phase/frame_idx/eye/sample_role/bbox_*). In
    ``continuous`` mode rows are grouped by (subject, eye) and processed in
    ``frame_idx`` order with one detector per algorithm, constructed with the
    scale-rule parameters of the group's first crop.
    """
    algorithms = list(algorithms)
    for algorithm in algorithms:
        if algorithm not in ALGORITHM_SPECS:
            raise KeyError(f"unknown algorithm: {algorithm!r}")
    crop_root = Path(crop_root)

    records: list[dict[str, Any]] = []
    if mode == "continuous":
        for subject, eye, group in _group_by_subject_eye(rows):
            group = sorted(group, key=lambda r: int(r.get("frame_idx", 0)))
            first_image = _load_crop(crop_root, group[0])
            fw, fh = first_image.shape[1], first_image.shape[0]
            detectors = {
                algorithm: make_detector(
                    ALGORITHM_SPECS[algorithm],
                    scale_params(algorithm, fw, fh),
                )
                for algorithm in algorithms
            }
            for row in group:
                image = _load_crop(crop_root, row)
                for algorithm in algorithms:
                    spec = ALGORITHM_SPECS[algorithm]
                    output = _run_on_shared(detectors[algorithm], spec, image)
                    detection = _detection_output_to_row(
                        output, spec, image.shape[1], image.shape[0]
                    )
                    detection = _maybe_confidence(detection, spec, image, run_confidence)
                    detection["params_provenance"] = _provenance(algorithm, image.shape[1], image.shape[0])
                    records.append(assemble_row(row, detection))
    else:
        for row in rows:
            image = _load_crop(crop_root, row)
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


def _group_by_subject_eye(rows: Sequence[Mapping[str, Any]]):
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("subject", "")), str(row.get("eye", "")))
        groups.setdefault(key, []).append(dict(row))
    return ((subject, eye, items) for (subject, eye), items in groups.items())


def _load_crop(crop_root: Path, row: Mapping[str, Any]) -> np.ndarray:
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
    detection: dict[str, Any], spec: Any, image: np.ndarray, run_confidence: bool
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
    )
    detection["outline_confidence"] = confidence.outline_confidence
    detection["confidence_runtime_ms"] = confidence.confidence_runtime_ms
    if confidence.failure:
        detection["failure"] = _join_failures(detection.get("failure"), confidence.failure)
    return detection


def _provenance(algorithm: str, width: int, height: int) -> str:
    return json.dumps(
        {
            "algorithm": algorithm,
            "scale_rule_overrides": scale_params(algorithm, width, height),
            "scale_rule": "docs/020-nir/030 section 6",
            "mode": "continuous",
        },
        ensure_ascii=False,
    )
