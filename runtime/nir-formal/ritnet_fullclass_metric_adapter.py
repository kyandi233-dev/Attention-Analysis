"""Lean final hard-label metrics for the production RITnet cohort.

Four-class hard segmentation remains intact. Production geometry is deliberately
pupil-only: the project has direct visual evidence that RITnet iris/ocular
segmentation can fragment under glasses/reflections, so iris ellipse/PIR and
ocular-shape geometry are not defensible primary measurements. Artificial
padding remains excluded from scientific denominators and is preserved as QC
facts.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ritnet_native_metrics import _component_metrics, _ellipse_geometry, validate_native_labels


ANALYSIS_DOMAIN_VERSION = "source-backed-output-mask-v3-primary-pupil-topology"
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


def _internal_valid_boundary_adjacency(valid: np.ndarray) -> np.ndarray:
    """Pixels adjacent to artificial padding, computed once per padded eye."""
    invalid = (~valid).astype(np.uint8)
    return cv2.dilate(
        invalid,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ).astype(bool)


def _primary_pupil_component(
    labels: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Select the pupil island belonging to the strongest iris+pupil topology.

    RITnet occasionally predicts multiple disconnected class-3 islands under
    glasses/reflections. Choosing the largest pupil contour alone can therefore
    fit a peripheral false-positive instead of the anatomical pupil. We avoid
    introducing a hand-tuned center/distance threshold: each pupil component is
    associated with the connected component of ``iris_outer = class 2 | class 3``
    that contains it. The pupil embedded in the largest such iris_outer structure
    wins; ties within the same iris_outer structure are resolved by pupil area.

    With zero or one pupil component this is exactly the historical mask, so the
    change only resolves an otherwise ambiguous fragmented-pupil case.
    """
    pupil = np.ascontiguousarray(((labels == 3) & valid).astype(np.uint8))
    pupil_count, pupil_ids, pupil_stats, _ = cv2.connectedComponentsWithStats(
        pupil,
        connectivity=8,
    )
    component_count = max(0, int(pupil_count) - 1)
    if component_count <= 1:
        return pupil.astype(bool)

    iris_outer = np.ascontiguousarray((((labels >= 2) & valid)).astype(np.uint8))
    outer_count, outer_ids, outer_stats, _ = cv2.connectedComponentsWithStats(
        iris_outer,
        connectivity=8,
    )
    if int(outer_count) <= 1:
        # Defensive fallback: every pupil pixel is itself iris_outer, so this
        # should not occur when pupil components exist. Preserve determinism by
        # falling back to the largest pupil component rather than guessing.
        areas = pupil_stats[1:, cv2.CC_STAT_AREA]
        selected_id = int(np.argmax(areas)) + 1
        return pupil_ids == selected_id

    candidates: list[tuple[int, int, int]] = []
    for pupil_id in range(1, int(pupil_count)):
        component = pupil_ids == pupil_id
        containing_outer_ids = outer_ids[component]
        containing_outer_ids = containing_outer_ids[containing_outer_ids > 0]
        if containing_outer_ids.size == 0:
            outer_area = 0
        else:
            counts = np.bincount(containing_outer_ids)
            outer_id = int(np.argmax(counts))
            outer_area = int(outer_stats[outer_id, cv2.CC_STAT_AREA])
        pupil_area = int(pupil_stats[pupil_id, cv2.CC_STAT_AREA])
        # Final -pupil_id term makes the tie-break deterministic without adding
        # a spatial prior that could bias legitimate eccentric gaze positions.
        candidates.append((outer_area, pupil_area, -pupil_id))

    selected_rank = max(candidates)
    selected_id = -int(selected_rank[2])
    return pupil_ids == selected_id


def summarize_final_hard_metrics(
    labels: np.ndarray,
    valid_source_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    labels = validate_native_labels(labels)
    valid = _validate_analysis_mask(valid_source_mask)
    full_source_domain = bool(valid.all())
    valid_count = int(valid.size if full_source_domain else valid.sum())

    # Four-class information is preserved exactly. In the common all-valid case
    # this is one cheap bincount over uint8 labels; padded eyes count only pixels
    # backed by the original source video.
    observed = labels.reshape(-1) if full_source_domain else labels[valid]
    counts_array = np.bincount(observed, minlength=4)
    if int(counts_array[:4].sum()) != valid_count:
        raise AssertionError("valid-domain hard class counts do not sum to valid pixel count")

    result: dict[str, Any] = {}
    for class_id, name in CLASS_NAMES.items():
        count = int(counts_array[class_id])
        result[f"hard_{name}_pixels"] = count
        result[f"hard_{name}_fraction"] = float(count / valid_count)

    iris_outer_count = int(counts_array[2] + counts_array[3])
    ocular_count = int(valid_count - counts_array[0])
    result["hard_iris_outer_pixels"] = iris_outer_count
    result["hard_iris_outer_fraction"] = float(iris_outer_count / valid_count)
    result["hard_ocular_pixels"] = ocular_count
    result["hard_ocular_fraction"] = float(ocular_count / valid_count)

    # Pupil is the only geometric structure retained in the formal NIR cohort.
    # Fragmentation remains a QC fact over *all* source-valid pupil pixels, while
    # geometry is fitted only to the pupil island supported by the strongest
    # iris+pupil connected topology. No iris ellipse or PIR is reintroduced.
    pupil = labels == 3
    pupil_analysis = pupil if full_source_domain else (pupil & valid)
    pupil_components, pupil_largest_fraction = _component_metrics(pupil_analysis)
    primary_pupil = _primary_pupil_component(labels, valid)
    pupil_geom, _ = _ellipse_geometry(primary_pupil)
    result["pupil_component_count"] = int(pupil_components)
    result["pupil_largest_component_fraction"] = pupil_largest_fraction
    result["qc_pupil_fragmented"] = bool(pupil_components > 1)
    for suffix in PUPIL_GEOMETRY_SUFFIXES:
        result[f"pupil_{suffix}"] = pupil_geom[suffix]

    result.update(
        {
            "analysis_domain_version": ANALYSIS_DOMAIN_VERSION,
            "analysis_valid_pixel_count": valid_count,
            "analysis_valid_pixel_fraction": float(valid_count / valid.size),
        }
    )

    if full_source_domain:
        result.update(
            {
                "pupil_predicted_in_padding_pixels": 0,
                "iris_outer_predicted_in_padding_pixels": 0,
                "ocular_predicted_in_padding_pixels": 0,
                "pupil_touches_valid_domain_edge": False,
                "iris_outer_touches_valid_domain_edge": False,
                "ocular_touches_valid_domain_edge": False,
            }
        )
        return result

    invalid = ~valid
    adjacent_to_padding = _internal_valid_boundary_adjacency(valid)
    iris_outer = labels >= 2
    ocular = labels != 0
    structures = {
        "pupil": pupil,
        "iris_outer": iris_outer,
        "ocular": ocular,
    }
    for name, mask in structures.items():
        result[f"{name}_predicted_in_padding_pixels"] = int((mask & invalid).sum())
        result[f"{name}_touches_valid_domain_edge"] = bool(
            (mask & valid & adjacent_to_padding).any()
        )
    return result
