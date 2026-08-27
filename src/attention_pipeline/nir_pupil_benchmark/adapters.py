"""Per-algorithm adapters for the seven classical pupil detectors.

All adapters take a grayscale ``uint8`` 2D numpy array (the native-resolution
tight eye crop) and return a raw result object plus wall-clock timing. The
official result-object semantics are normalized separately in ``core``.

Key provenance decisions from the official source audit (docs/020-nir/030):
- Swirski2D ``Radius_Min/Max`` are resolution sensitive and frozen by scale rule;
  ``params.Seed`` must be set >= 0 for RANSAC reproducibility.
- PuRe/PuReST diameter range is overridden via the exposed
  ``minPupilDiameterMM/maxPupilDiameterMM`` fields (the ROI overload is buggy).
- ``runWithConfidence`` is a separate pass with its own timing; it never
  contributes to the main ``runtime_ms``.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .core import _numeric, _object_value
from .schema import ALGORITHM_SPECS, SCALE_RULE, AlgorithmSpec


class UnavailableAlgorithmError(RuntimeError):
    """The backing package for an algorithm is not importable."""


@dataclass
class DetectionOutput:
    algorithm: str
    result: Any  # raw Pupil object or Detector2D dict; None on failure
    runtime_ms: float
    failure: str | None = None
    outline_confidence: float | None = None
    confidence_runtime_ms: float | None = None


def _load_package(package: str):
    import importlib

    try:
        return importlib.import_module(package)
    except ImportError as exc:
        raise UnavailableAlgorithmError(
            f"package '{package}' is not importable (needed by algorithm adapter). "
            f"Run under the pypupilext310 conda env."
        ) from exc


def pupil_diameter_bounds(width: int, height: int) -> tuple[int, int]:
    """Freeze (radius_min, radius_max) in crop pixels from the scale rule.

    Rule provenance: docs/020-nir/030 section 6, based on the sub-031 probe
    (tight crop ~424x187, pupil diameter ~6-16 px). Deliberately conservative
    so the Haar search brackets the true pupil radius.
    """
    base = min(int(width), int(height))
    radius_min = max(2, round(SCALE_RULE["radius_min_fraction"] * base))
    radius_max = max(radius_min + 4, round(SCALE_RULE["radius_max_fraction"] * base))
    return radius_min, radius_max


def _prepare_image(image: np.ndarray) -> np.ndarray:
    """Ensure a 2D uint8 grayscale array for the compiled detectors."""
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.ndim == 3:
        image = image[..., 0] if image.shape[2] == 1 else _rgb_to_gray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim != 2:
        raise ValueError(f"image must be 2D after preparation, got shape {image.shape}")
    return np.ascontiguousarray(image)


def _rgb_to_gray(image: np.ndarray) -> np.ndarray:
    import cv2

    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)


def make_detector(spec: AlgorithmSpec, params: dict[str, Any] | None = None) -> Any:
    """Construct a detector instance with optional parameter overrides."""
    package = _load_package(spec.package)
    cls = getattr(package, spec.class_name)
    merged = dict(spec.default_params)
    if params:
        merged.update(params)

    if spec.name == "PupilLabs2D":
        properties = {
            key: value for key, value in merged.items()
            if key not in ("package", "class_name", "baseSize")
        }
        return cls(properties=properties)

    detector = cls()
    if spec.name == "Swirski2D":
        tracker = detector.params
        for key, value in merged.items():
            if hasattr(tracker, key):
                setattr(tracker, key, value)
        return detector

    # PuRe / PuReST / ElSe / ExCuSe / Starburst: flat fields
    for key, value in merged.items():
        if key == "baseSize":
            detector.baseSize = tuple(value)
        elif hasattr(detector, key):
            setattr(detector, key, value)
    return detector


def _set_pure_diameter_range(
    detector: Any,
    width: int,
    height: int,
    diameter_min_px: float,
    diameter_max_px: float,
) -> None:
    """Override PuRe/PuReST pupil diameter range via the exposed MM fields.

    Official estimate (PuRe.cpp ``estimateParameters``), in working (downscaled)
    pixels:
        maxPupilDiameterPx = diag * (maxPupilDiameterMM / meanCanthiDistanceMM)
        minPupilDiameterPx = (2*diag/3) * (minPupilDiameterMM / meanCanthiDistanceMM)
    Invert for the requested px diameters. The 2/3 factor is the official
    "image contains ~2x canthi distance" assumption and must not be changed.
    """
    scaling_ratio = min(320.0 / float(width), 240.0 / float(height), 1.0)
    work_w = scaling_ratio * float(width)
    work_h = scaling_ratio * float(height)
    diag = math.sqrt(work_w**2 + work_h**2)
    mean_canthi = float(detector.meanCanthiDistanceMM)
    if diag <= 0 or mean_canthi <= 0:
        return
    max_mm = (float(diameter_max_px) * scaling_ratio) * mean_canthi / diag
    min_mm = (float(diameter_min_px) * scaling_ratio) * mean_canthi / (2.0 * diag / 3.0)
    detector.minPupilDiameterMM = float(max(min_mm, 0.3))
    detector.maxPupilDiameterMM = float(max(max_mm, detector.minPupilDiameterMM + 0.1))


def run_detection(
    spec: AlgorithmSpec,
    detector: Any,
    image: np.ndarray,
    *,
    diameter_min_px: float | None = None,
    diameter_max_px: float | None = None,
) -> DetectionOutput:
    """Run the algorithm's plain ``run()`` (never runWithConfidence)."""
    prepared = _prepare_image(image)
    start = time.perf_counter()
    try:
        if spec.supports_diameter_override and diameter_min_px is not None and diameter_max_px is not None:
            _set_pure_diameter_range(
                detector, prepared.shape[1], prepared.shape[0], diameter_min_px, diameter_max_px
            )
        if spec.name == "PupilLabs2D":
            result = detector.detect(prepared)
        else:
            result = detector.run(prepared)
    except Exception as exc:  # noqa: BLE001 - record every failure, never mask it
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return DetectionOutput(
            algorithm=spec.name, result=None, runtime_ms=elapsed_ms,
            failure=f"{type(exc).__name__}: {exc}",
        )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return DetectionOutput(algorithm=spec.name, result=result, runtime_ms=elapsed_ms)


def run_with_confidence(
    spec: AlgorithmSpec,
    image: np.ndarray,
    *,
    diameter_min_px: float | None = None,
    diameter_max_px: float | None = None,
) -> DetectionOutput:
    """Separate outline-confidence pass on a fresh detector.

    Timed independently; the resulting confidence must never be folded into the
    main ``runtime_ms``. Pupil Labs 2D has native confidence and no separate
    outline pass, so this returns a no-op result.
    """
    if spec.name == "PupilLabs2D":
        return DetectionOutput(
            algorithm=spec.name, result=None, runtime_ms=0.0,
            outline_confidence=None, confidence_runtime_ms=0.0,
        )
    prepared = _prepare_image(image)
    detector = make_detector(spec)
    if spec.supports_diameter_override and diameter_min_px is not None and diameter_max_px is not None:
        _set_pure_diameter_range(
            detector, prepared.shape[1], prepared.shape[0], diameter_min_px, diameter_max_px
        )
    start = time.perf_counter()
    try:
        result = detector.runWithConfidence(prepared)
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return DetectionOutput(
            algorithm=spec.name, result=None, runtime_ms=elapsed_ms,
            failure=f"{type(exc).__name__}: {exc}", confidence_runtime_ms=elapsed_ms,
        )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    confidence = _numeric(_object_value(result, ["outline_confidence", "outlineConfidence"]))
    return DetectionOutput(
        algorithm=spec.name, result=result, runtime_ms=elapsed_ms,
        outline_confidence=confidence, confidence_runtime_ms=elapsed_ms,
    )


def frozen_swirski_params(width: int, height: int) -> dict[str, Any]:
    """Return Swirski2D params frozen to the crop scale rule (provenance docs 030)."""
    radius_min, radius_max = pupil_diameter_bounds(width, height)
    return {
        "Radius_Min": radius_min,
        "Radius_Max": radius_max,
        "Seed": SCALE_RULE["seed"],
    }


def frozen_pupil_labs_properties(width: int, height: int) -> dict[str, Any]:
    """Pupil Labs 2D properties frozen to the crop scale rule.

    The default coarse_filter/pupil_size ranges assume a full eye image; on a
    tight crop they must be scaled to the same rule as Swirski2D. The coarse
    filter is expressed in full-resolution diameters.
    """
    radius_min, radius_max = pupil_diameter_bounds(width, height)
    return {
        "coarse_detection": True,
        "coarse_filter_min": 2 * radius_min,
        "coarse_filter_max": 2 * radius_max,
        "pupil_size_min": 2 * radius_min,
        "pupil_size_max": 2 * radius_max,
    }
