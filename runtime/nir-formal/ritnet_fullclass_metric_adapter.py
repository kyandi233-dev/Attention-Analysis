"""Map audited native-640 hard-label metrics into the fixed final schema.

The underlying geometry implementation remains ``ritnet_native_metrics`` to
avoid duplicating contour/ellipse mathematics. This adapter explicitly selects
and renames only final-contract fields; historical probability/source fields do
not leak into the new table.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ritnet_native_metrics import summarize_fullclass_native


DIRECT_MAP = {
    "native_background_pixels": "hard_background_pixels",
    "native_background_fraction": "hard_background_fraction",
    "native_sclera_pixels": "hard_sclera_pixels",
    "native_sclera_fraction": "hard_sclera_fraction",
    "native_iris_pixels": "hard_iris_pixels",
    "native_iris_fraction": "hard_iris_fraction",
    "native_pupil_pixels": "hard_pupil_pixels",
    "native_pupil_fraction": "hard_pupil_fraction",
    "native_iris_outer_pixels": "hard_iris_outer_pixels",
    "native_iris_outer_fraction": "hard_iris_outer_fraction",
    "native_ocular_pixels": "hard_ocular_pixels",
    "native_ocular_fraction": "hard_ocular_fraction",
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


def summarize_final_hard_metrics(labels: np.ndarray) -> dict[str, Any]:
    native = summarize_fullclass_native(labels)
    result = {target: native[source] for source, target in DIRECT_MAP.items()}
    for final_prefix, native_prefix in (
        ("pupil", "native_pupil"),
        ("iris_outer", "native_iris_outer"),
    ):
        for suffix in GEOMETRY_SUFFIXES:
            result[f"{final_prefix}_{suffix}"] = native[f"{native_prefix}_{suffix}"]
    return result
