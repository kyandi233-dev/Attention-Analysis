"""Lean final hard-label metrics for the production RITnet cohort."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ritnet_native_metrics import _component_metrics, _ellipse_geometry, validate_native_labels


ANALYSIS_DOMAIN_VERSION = "source-backed-output-mask-v2-pupil-geometry-only"
CLASS_NAMES = {
    0: "background",
    1: "sclera",
    2: "iris",
    3: "pupil",
}

PUPIL_GEOMETRY_SUFFIXES = (
    "found",
    "fit_valid",
    "center_x",
    "center_y",
    "short_axis",
    "long_axis",
    "angle_deg",
    "contour_area",
    "ellipse_area",
    "equiv_diameter",
    "geom_mean_diameter",
    "whole_mask_touches_edge",
    "largest_contour_touches_edge",
)


def _validate_analysis_mask(valid_source_mask: np.ndarray | None) -> np.ndarray:
    if valid_source_mask is None:
        return np.ones((400, 640), dtype=bool)
    mask = np.asarray(valid_source_mask)
    if mask.shape != (400, 640):
        raise ValueError(f"valid_source_mask must have shape (400, 640), got {mask.shape}")
    if mask.dtype != np.bool_:
        raise TypeError(f"valid_source_mask must be bool, got {mask.dtype}")
    if not mask.any():
        raise ValueError("valid_source_mask contains no source-backed pixels")
    return np.ascontiguousarray(mask)


def _touches_internal_valid_boundary(structure: np.ndarray, valid: np.ndarray) -> bool:
    """Whether a source-backed structure touches an internal padding boundary."""
    if valid.all() or not structure.any():
        return False
    invalid = (~valid).astype(np.uint8)
    adjacent = cv2.dilate(invalid, np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
    return bool((structure & valid & adjacent).any())


def summarize_final_hard_metrics(
    labels: np.ndarray,
    valid_source_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    labels = validate_native_labels(labels)
    valid = _validate_analysis_mask(valid_source_mask)
    full_source_domain = bool(valid.all())
    valid_count = valid.size if full_source_domain else int(valid.sum())

    observed = labels.reshape(-1) if full_source_domain else labels[valid]
    counts = np.bincount(observed, minlength=4)
    if int(counts[:4].sum()) != valid_count:
        raise AssertionError("valid-domain hard class counts do not sum to valid pixel count")
    result: dict[str, Any] = {}
    for class_id, name in CLASS_NAMES.items():
        count = int(counts[class_id])
        result[f"hard_{name}_pixels"] = count
        result[f"hard_{name}_fraction"] = float(count / valid_count)
    iris_outer_count = int(counts[2] + counts[3])
    ocular_count = int(valid_count - counts[0])
    result["hard_iris_outer_pixels"] = iris_outer_count
    result["hard_iris_outer_fraction"] = float(iris_outer_count / valid_count)
    result["hard_ocular_pixels"] = ocular_count
    result["hard_ocular_fraction"] = float(ocular_count / valid_count)
    pupil = labels == 3
    pupil_analysis = pupil if full_source_domain else (pupil & valid)
    pupil_geom, _ = _ellipse_geometry(pupil_analysis)
    pupil_components, pupil_largest_fraction = _component_metrics(pupil_analysis)
    result["pupil_component_count"] = int(pupil_components)
    result["pupil_largest_component_fraction"] = pupil_largest_fraction
    result["qc_pupil_fragmented"] = bool(pupil_components > 1)
    for suffix in PUPIL_GEOMETRY_SUFFIXES:
        result[f"pupil_{suffix}"] = pupil_geom[suffix]

    if sum(result[f"hard_{name}_pixels"] for name in CLASS_NAMES.values()) != valid_count:
        raise AssertionError("valid-domain hard class counts do not sum to valid pixel count")

    result.update(
        {
            "analysis_domain_version": ANALYSIS_DOMAIN_VERSION,
            "analysis_valid_pixel_count": valid_count,
            "analysis_valid_pixel_fraction": float(valid_count / valid.size),
        }
    )

    invalid = ~valid
    structures = {"pupil": pupil, "iris_outer": labels >= 2, "ocular": labels != 0}
    for name, mask in structures.items():
        result[f"{name}_predicted_in_padding_pixels"] = int((mask & invalid).sum())
        result[f"{name}_touches_valid_domain_edge"] = (
            False if full_source_domain else _touches_internal_valid_boundary(mask, valid)
        )

    return result
