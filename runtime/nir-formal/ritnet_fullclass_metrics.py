"""Pure post-processing helpers for RITnet four-class eye segmentation.

RITnet class mapping is frozen as:
0 background, 1 sclera, 2 iris, 3 pupil.
All geometry returned here is expressed in the configured analysis coordinate
system (320x160 in the current formal NIR pipeline).
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ritnet_fullclass_contract import (
    CLASS_BACKGROUND,
    CLASS_IRIS,
    CLASS_MAPPING,
    CLASS_PUPIL,
    CLASS_SCLERA,
)


def _as_binary(mask: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(mask.astype(np.uint8, copy=False))


def _touches_edge(mask: np.ndarray) -> bool:
    if mask.size == 0 or not mask.any():
        return False
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(
        _as_binary(mask) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _ellipse_geometry(mask: np.ndarray) -> tuple[dict[str, Any], np.ndarray | None]:
    contour = _largest_contour(mask)
    found = contour is not None and cv2.contourArea(contour) >= 5
    result: dict[str, Any] = {
        "found": bool(found),
        "fit_valid": False,
        "center_x": None,
        "center_y": None,
        "axis_a": None,
        "axis_b": None,
        "short_axis": None,
        "long_axis": None,
        "angle_deg": None,
        "contour_area": None,
        "ellipse_area": None,
        "equiv_diameter": None,
        "geom_mean_diameter": None,
        "touches_roi_edge": _touches_edge(mask),
    }
    if not found or contour is None:
        return result, contour

    area = float(cv2.contourArea(contour))
    result["contour_area"] = area
    result["equiv_diameter"] = float(2.0 * np.sqrt(area / np.pi))

    if len(contour) < 5:
        return result, contour

    (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
    axis_a = float(axis_a)
    axis_b = float(axis_b)
    short_axis, long_axis = sorted((axis_a, axis_b))
    ellipse_area = float(np.pi * axis_a * axis_b / 4.0)
    result.update(
        {
            "fit_valid": bool(axis_a > 0 and axis_b > 0),
            "center_x": float(cx),
            "center_y": float(cy),
            "axis_a": axis_a,
            "axis_b": axis_b,
            "short_axis": float(short_axis),
            "long_axis": float(long_axis),
            "angle_deg": float(angle),
            "ellipse_area": ellipse_area,
            "geom_mean_diameter": float(np.sqrt(axis_a * axis_b)),
        }
    )
    return result, contour


def _component_metrics(mask: np.ndarray) -> tuple[int, float | None]:
    binary = _as_binary(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    component_count = max(0, int(count) - 1)
    total = int(binary.sum())
    if component_count == 0 or total <= 0:
        return component_count, None
    areas = stats[1:, cv2.CC_STAT_AREA]
    return component_count, float(np.max(areas) / total)


def _ocular_aperture_metrics(mask: np.ndarray) -> dict[str, Any]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return {
            "bbox_width": None,
            "bbox_height": None,
            "aperture_height_median": None,
            "aperture_height_p90": None,
            "aperture_ratio_median": None,
            "aperture_ratio_p90": None,
            "touches_roi_edge": False,
        }

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    width = x_max - x_min + 1
    height = y_max - y_min + 1

    # Avoid single extreme edge columns. Use the central 80% of the visible
    # ocular horizontal span and summarize per-column vertical apertures.
    left = x_min + int(round(0.10 * max(0, width - 1)))
    right = x_min + int(round(0.90 * max(0, width - 1)))
    heights: list[int] = []
    for x in range(left, right + 1):
        column_y = np.flatnonzero(mask[:, x])
        if column_y.size:
            heights.append(int(column_y[-1] - column_y[0] + 1))

    if heights:
        height_median = float(np.median(heights))
        height_p90 = float(np.percentile(heights, 90))
        ratio_median = float(height_median / width) if width else None
        ratio_p90 = float(height_p90 / width) if width else None
    else:
        height_median = None
        height_p90 = None
        ratio_median = None
        ratio_p90 = None

    return {
        "bbox_width": int(width),
        "bbox_height": int(height),
        "aperture_height_median": height_median,
        "aperture_height_p90": height_p90,
        "aperture_ratio_median": ratio_median,
        "aperture_ratio_p90": ratio_p90,
        "touches_roi_edge": _touches_edge(mask),
    }


def summarize_fullclass(
    labels: np.ndarray,
    pupil_probability: np.ndarray | None,
    analysis_size: tuple[int, int] = (320, 160),
) -> dict[str, Any]:
    """Convert one RITnet label map into full-class ocular metrics.

    ``labels`` is the model-resolution hard argmax label map. It is resized by
    nearest-neighbour interpolation to ``analysis_size`` before all class counts
    and geometry are computed. The current ONNX exposes only the full hard label
    map plus class-3 probability, so iris/sclera/background class probabilities
    are intentionally not invented here.
    """
    analysis_w, analysis_h = map(int, analysis_size)
    resized = cv2.resize(
        np.asarray(labels, dtype=np.uint8),
        (analysis_w, analysis_h),
        interpolation=cv2.INTER_NEAREST,
    )

    masks = {class_id: resized == class_id for class_id in CLASS_MAPPING}
    total_pixels = int(analysis_w * analysis_h)
    counts = {class_id: int(mask.sum()) for class_id, mask in masks.items()}
    ocular = masks[CLASS_SCLERA] | masks[CLASS_IRIS] | masks[CLASS_PUPIL]
    iris_outer = masks[CLASS_IRIS] | masks[CLASS_PUPIL]

    pupil_geom, pupil_contour = _ellipse_geometry(masks[CLASS_PUPIL])
    iris_geom, iris_contour = _ellipse_geometry(iris_outer)
    ocular_components, ocular_largest_fraction = _component_metrics(ocular)
    aperture = _ocular_aperture_metrics(ocular)

    pupil_confidence = None
    if pupil_probability is not None:
        native_pupil = np.asarray(labels) == CLASS_PUPIL
        probs = np.asarray(pupil_probability, dtype=np.float32)
        if native_pupil.any() and probs.shape == native_pupil.shape:
            pupil_confidence = float(probs[native_pupil].mean())
        elif native_pupil.any():
            raise ValueError(
                f"pupil_probability shape {probs.shape} does not match labels {native_pupil.shape}"
            )
        else:
            pupil_confidence = 0.0

    iris_outer_pixels = int(counts[CLASS_IRIS] + counts[CLASS_PUPIL])
    ocular_pixels = int(counts[CLASS_SCLERA] + iris_outer_pixels)

    diameter_ratio = None
    ellipse_area_ratio = None
    contour_area_ratio = None
    center_offset_px = None
    center_offset_norm = None
    pupil_center_in_iris_outer = None

    if pupil_geom["fit_valid"] and iris_geom["fit_valid"]:
        pupil_d = float(pupil_geom["geom_mean_diameter"])
        iris_d = float(iris_geom["geom_mean_diameter"])
        pupil_ellipse_area = float(pupil_geom["ellipse_area"])
        iris_ellipse_area = float(iris_geom["ellipse_area"])
        if iris_d > 0:
            diameter_ratio = float(pupil_d / iris_d)
        if iris_ellipse_area > 0:
            ellipse_area_ratio = float(pupil_ellipse_area / iris_ellipse_area)
        dx = float(pupil_geom["center_x"] - iris_geom["center_x"])
        dy = float(pupil_geom["center_y"] - iris_geom["center_y"])
        center_offset_px = float(np.hypot(dx, dy))
        if iris_d > 0:
            center_offset_norm = float(center_offset_px / iris_d)
        if iris_contour is not None:
            pupil_center_in_iris_outer = bool(
                cv2.pointPolygonTest(
                    iris_contour,
                    (float(pupil_geom["center_x"]), float(pupil_geom["center_y"])),
                    False,
                )
                >= 0
            )

    if (
        pupil_geom["contour_area"] is not None
        and iris_geom["contour_area"] not in (None, 0)
    ):
        contour_area_ratio = float(
            float(pupil_geom["contour_area"]) / float(iris_geom["contour_area"])
        )

    iris_fill_ratio = None
    if iris_geom["ellipse_area"] not in (None, 0):
        iris_fill_ratio = float(iris_outer_pixels / float(iris_geom["ellipse_area"]))

    normalization_valid = bool(
        pupil_geom["fit_valid"]
        and iris_geom["fit_valid"]
        and not pupil_geom["touches_roi_edge"]
        and not iris_geom["touches_roi_edge"]
        and float(iris_geom["geom_mean_diameter"] or 0.0)
        > float(pupil_geom["geom_mean_diameter"] or 0.0)
    )

    result: dict[str, Any] = {
        "analysis_width": analysis_w,
        "analysis_height": analysis_h,
        "background_pixels": counts[CLASS_BACKGROUND],
        "background_fraction": float(counts[CLASS_BACKGROUND] / total_pixels),
        "sclera_pixels": counts[CLASS_SCLERA],
        "sclera_fraction": float(counts[CLASS_SCLERA] / total_pixels),
        "iris_pixels": counts[CLASS_IRIS],
        "iris_fraction": float(counts[CLASS_IRIS] / total_pixels),
        "pupil_pixels": counts[CLASS_PUPIL],
        "pupil_fraction": float(counts[CLASS_PUPIL] / total_pixels),
        "iris_outer_pixels": iris_outer_pixels,
        "iris_outer_fraction": float(iris_outer_pixels / total_pixels),
        "ocular_pixels": ocular_pixels,
        "ocular_fraction": float(ocular_pixels / total_pixels),
        "ocular_component_count": ocular_components,
        "ocular_largest_component_fraction": ocular_largest_fraction,
        "ocular_bbox_width": aperture["bbox_width"],
        "ocular_bbox_height": aperture["bbox_height"],
        "ocular_aperture_height_median": aperture["aperture_height_median"],
        "ocular_aperture_height_p90": aperture["aperture_height_p90"],
        "ocular_aperture_ratio_median": aperture["aperture_ratio_median"],
        "ocular_aperture_ratio_p90": aperture["aperture_ratio_p90"],
        "ocular_touches_roi_edge": aperture["touches_roi_edge"],
        "pupil_confidence": pupil_confidence,
        "pupil_to_iris_diameter_ratio": diameter_ratio,
        "pupil_to_iris_ellipse_area_ratio": ellipse_area_ratio,
        "pupil_to_iris_contour_area_ratio": contour_area_ratio,
        "pupil_center_offset_px": center_offset_px,
        "pupil_center_offset_norm": center_offset_norm,
        "pupil_center_in_iris_outer": pupil_center_in_iris_outer,
        "iris_outer_fill_ratio": iris_fill_ratio,
        "normalization_valid": normalization_valid,
    }

    for prefix, geometry in (("pupil", pupil_geom), ("iris_outer", iris_geom)):
        for key, value in geometry.items():
            result[f"{prefix}_{key}"] = value

    # Make the exact hard-class accounting auditable.
    if (
        result["background_pixels"]
        + result["sclera_pixels"]
        + result["iris_pixels"]
        + result["pupil_pixels"]
        != total_pixels
    ):
        raise AssertionError("RITnet class counts do not sum to the analysis frame size")

    return result
