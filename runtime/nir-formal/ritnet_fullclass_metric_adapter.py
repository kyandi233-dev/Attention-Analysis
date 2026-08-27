"""Map audited native-640 hard-label metrics into the fixed final schema.

The underlying geometry implementation remains ``ritnet_native_metrics`` to
avoid duplicating contour/ellipse mathematics. Final scientific metrics are
computed only on pixels backed by real source-video content; replicate padding
is retained for RITnet input context but excluded from analysis denominators and
geometry. Padding overlap is preserved separately as QC evidence.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ritnet_native_metrics import summarize_fullclass_native, validate_native_labels


ANALYSIS_DOMAIN_VERSION = "source-backed-output-mask-v1"
CLASS_NAMES = {
    0: "background",
    1: "sclera",
    2: "iris",
    3: "pupil",
}

DIRECT_MAP = {
    "native_pupil_component_count": "pupil_component_count",
    "native_pupil_largest_component_fraction": "pupil_largest_component_fraction",
    "native_iris_outer_component_count": "iris_outer_component_count",
    "native_iris_outer_largest_component_fraction": "iris_outer_largest_component_fraction",
    "native_ocular_component_count": "ocular_component_count",
    "native_ocular_largest_component_fraction": "ocular_largest_component_fraction",
    "native_pupil_to_iris_diameter_ratio": "pupil_to_iris_diameter_ratio",
    "native_pupil_to_iris_ellipse_area_ratio": "pupil_to_iris_ellipse_area_ratio",
    "native_pupil_to_iris_contour_area_ratio": "pupil_to_iris_contour_area_ratio",
    "native_pupil_center_offset_px": "pupil_center_offset_px",
    "native_pupil_center_offset_norm": "pupil_center_offset_norm",
    "native_pupil_center_in_iris_outer": "pupil_center_in_iris_outer",
    "native_iris_diameter_gt_pupil_diameter": "iris_diameter_gt_pupil_diameter",
    "native_pir_finite": "pir_finite",
    "native_iris_outer_fill_ratio": "iris_outer_fill_ratio",
    "native_ocular_bbox_width": "ocular_bbox_width",
    "native_ocular_bbox_height": "ocular_bbox_height",
    "native_ocular_aperture_height_median": "ocular_aperture_height_median",
    "native_ocular_aperture_height_p90": "ocular_aperture_height_p90",
    "native_ocular_aperture_ratio_median": "ocular_aperture_ratio_median",
    "native_ocular_aperture_ratio_p90": "ocular_aperture_ratio_p90",
    "native_ocular_whole_mask_touches_edge": "ocular_whole_mask_touches_edge",
    "diagnostic_pupil_fragmented": "qc_pupil_fragmented",
    "diagnostic_iris_fragmented": "qc_iris_outer_fragmented",
    "diagnostic_ocular_fragmented": "qc_ocular_fragmented",
}

GEOMETRY_SUFFIXES = (
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
    adjacent_to_invalid = cv2.dilate(invalid, np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
    return bool((structure & valid & adjacent_to_invalid).any())


def summarize_final_hard_metrics(
    labels: np.ndarray,
    valid_source_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    labels = validate_native_labels(labels)
    valid = _validate_analysis_mask(valid_source_mask)
    valid_count = int(valid.sum())
    invalid = ~valid

    # Geometry/components are deliberately fitted only to observed source-backed
    # labels. Invalid/padded positions are set to background for the existing,
    # audited native geometry implementation. Class counts are recalculated
    # below because this temporary background fill must not enter denominators.
    observed_labels = labels.copy()
    observed_labels[invalid] = 0
    native = summarize_fullclass_native(observed_labels)

    result: dict[str, Any] = {target: native[source] for source, target in DIRECT_MAP.items()}
    for final_prefix, native_prefix in (
        ("pupil", "native_pupil"),
        ("iris_outer", "native_iris_outer"),
    ):
        for suffix in GEOMETRY_SUFFIXES:
            result[f"{final_prefix}_{suffix}"] = native[f"{native_prefix}_{suffix}"]

    class_masks = {class_id: (labels == class_id) for class_id in CLASS_NAMES}
    for class_id, name in CLASS_NAMES.items():
        count = int((class_masks[class_id] & valid).sum())
        result[f"hard_{name}_pixels"] = count
        result[f"hard_{name}_fraction"] = float(count / valid_count)

    iris_outer = class_masks[2] | class_masks[3]
    ocular = class_masks[1] | iris_outer
    for name, mask in (("iris_outer", iris_outer), ("ocular", ocular)):
        count = int((mask & valid).sum())
        result[f"hard_{name}_pixels"] = count
        result[f"hard_{name}_fraction"] = float(count / valid_count)

    if sum(result[f"hard_{name}_pixels"] for name in CLASS_NAMES.values()) != valid_count:
        raise AssertionError("valid-domain hard class counts do not sum to valid pixel count")

    result.update(
        {
            "analysis_domain_version": ANALYSIS_DOMAIN_VERSION,
            "analysis_valid_pixel_count": valid_count,
            "analysis_valid_pixel_fraction": float(valid_count / valid.size),
        }
    )

    structures = {
        "pupil": class_masks[3],
        "iris_outer": iris_outer,
        "ocular": ocular,
    }
    for name, mask in structures.items():
        result[f"{name}_predicted_in_padding_pixels"] = int((mask & invalid).sum())
        result[f"{name}_touches_valid_domain_edge"] = _touches_internal_valid_boundary(mask, valid)

    return result
