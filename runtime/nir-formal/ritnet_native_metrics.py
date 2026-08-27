"""Native-resolution post-processing for RITnet four-class eye segmentation.

This module is intentionally separate from the legacy v1.2 320x160 metric path.
"native" means the 640x400 hard-label coordinate system emitted by the frozen
Attention-Analysis RITnet ONNX adapter; it does not mean that these metrics are
upstream RITnet variables.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

NATIVE_LABEL_WIDTH = 640
NATIVE_LABEL_HEIGHT = 400
NATIVE_LABEL_SHAPE = (NATIVE_LABEL_HEIGHT, NATIVE_LABEL_WIDTH)
NATIVE_LABEL_DTYPE = np.dtype(np.uint8)
CLASS_BACKGROUND = 0
CLASS_SCLERA = 1
CLASS_IRIS = 2
CLASS_PUPIL = 3
CLASS_IDS = (CLASS_BACKGROUND, CLASS_SCLERA, CLASS_IRIS, CLASS_PUPIL)


def validate_native_labels(labels: np.ndarray) -> np.ndarray:
    array = np.asarray(labels)
    if array.shape != NATIVE_LABEL_SHAPE:
        raise ValueError(
            f"native RITnet labels must have shape {NATIVE_LABEL_SHAPE}; got {array.shape}"
        )
    if array.dtype != NATIVE_LABEL_DTYPE:
        raise TypeError(
            f"native RITnet labels must be uint8; got {array.dtype}"
        )
    # uint8 guarantees the lower bound. A single max reduction is sufficient to
    # prove the frozen {0,1,2,3} class domain and is cheaper than unique+isin.
    if array.size and int(array.max()) > CLASS_PUPIL:
        raise ValueError("native RITnet labels contain values outside {0,1,2,3}")
    return np.ascontiguousarray(array)


def _binary(mask: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(mask.astype(np.uint8, copy=False))


def _whole_mask_touches_edge(mask: np.ndarray) -> bool:
    if mask.size == 0:
        return False
    return bool(
        mask[0, :].any()
        or mask[-1, :].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
    )


def _contours(mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(
        _binary(mask),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return list(contours)


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours = _contours(mask)
    return max(contours, key=cv2.contourArea) if contours else None


def _largest_contour_touches_edge(contour: np.ndarray | None, shape: tuple[int, int]) -> bool:
    if contour is None or contour.size == 0:
        return False
    h, w = shape
    pts = contour.reshape(-1, 2)
    xs = pts[:, 0]
    ys = pts[:, 1]
    return bool((xs <= 0).any() or (ys <= 0).any() or (xs >= w - 1).any() or (ys >= h - 1).any())


def _component_metrics(mask: np.ndarray) -> tuple[int, float | None]:
    binary = _binary(mask)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    component_count = max(0, int(count) - 1)
    total = int(binary.sum())
    if component_count == 0 or total <= 0:
        return component_count, None
    areas = stats[1:, cv2.CC_STAT_AREA]
    return component_count, float(np.max(areas) / total)


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
        "whole_mask_touches_edge": _whole_mask_touches_edge(mask),
        "largest_contour_touches_edge": _largest_contour_touches_edge(contour, mask.shape),
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
    fit_valid = bool(np.isfinite([cx, cy, axis_a, axis_b, angle]).all() and axis_a > 0 and axis_b > 0)
    result.update(
        {
            "fit_valid": fit_valid,
            "center_x": float(cx) if fit_valid else None,
            "center_y": float(cy) if fit_valid else None,
            "axis_a": axis_a if fit_valid else None,
            "axis_b": axis_b if fit_valid else None,
            "short_axis": float(short_axis) if fit_valid else None,
            "long_axis": float(long_axis) if fit_valid else None,
            "angle_deg": float(angle) if fit_valid else None,
            "ellipse_area": float(np.pi * axis_a * axis_b / 4.0) if fit_valid else None,
            "geom_mean_diameter": float(np.sqrt(axis_a * axis_b)) if fit_valid else None,
        }
    )
    return result, contour


def _ocular_aperture_metrics(mask: np.ndarray) -> dict[str, Any]:
    row_present = np.any(mask, axis=1)
    col_present = np.any(mask, axis=0)
    if not col_present.any():
        return {
            "bbox_width": None,
            "bbox_height": None,
            "aperture_height_median": None,
            "aperture_height_p90": None,
            "aperture_ratio_median": None,
            "aperture_ratio_p90": None,
            "whole_mask_touches_edge": False,
        }

    xs = np.flatnonzero(col_present)
    ys = np.flatnonzero(row_present)
    x_min, x_max = int(xs[0]), int(xs[-1])
    y_min, y_max = int(ys[0]), int(ys[-1])
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    left = x_min + int(round(0.10 * max(0, width - 1)))
    right = x_min + int(round(0.90 * max(0, width - 1)))

    # Compute every central column's top/bottom visible ocular pixel in one
    # vectorized pass instead of up to hundreds of Python flatnonzero calls.
    central = np.asarray(mask[:, left : right + 1], dtype=bool)
    occupied = np.any(central, axis=0)
    if occupied.any():
        top = np.argmax(central, axis=0)
        bottom = central.shape[0] - 1 - np.argmax(central[::-1, :], axis=0)
        heights = (bottom - top + 1)[occupied]
        height_median = float(np.median(heights))
        height_p90 = float(np.percentile(heights, 90))
        ratio_median = float(height_median / width) if width else None
        ratio_p90 = float(height_p90 / width) if width else None
    else:
        height_median = height_p90 = ratio_median = ratio_p90 = None

    return {
        "bbox_width": int(width),
        "bbox_height": int(height),
        "aperture_height_median": height_median,
        "aperture_height_p90": height_p90,
        "aperture_ratio_median": ratio_median,
        "aperture_ratio_p90": ratio_p90,
        "whole_mask_touches_edge": _whole_mask_touches_edge(mask),
    }


def summarize_pupil_probability(
    labels: np.ndarray,
    pupil_probability: np.ndarray | None,
) -> dict[str, Any]:
    labels = validate_native_labels(labels)
    if pupil_probability is None:
        return {
            "native_pupil_probability_available": False,
            "native_pupil_softmax_mean_on_argmax_mask": None,
            "native_pupil_softmax_median_on_argmax_mask": None,
            "native_pupil_softmax_p05_on_argmax_mask": None,
            "native_pupil_softmax_p95_on_argmax_mask": None,
            "native_pupil_softmax_min_on_argmax_mask": None,
            "native_pupil_softmax_max_on_argmax_mask": None,
        }

    probs = np.asarray(pupil_probability)
    if probs.shape != labels.shape:
        raise ValueError(
            f"pupil probability shape {probs.shape} does not match labels {labels.shape}"
        )
    if not np.issubdtype(probs.dtype, np.floating):
        raise TypeError(f"pupil probability must be floating point; got {probs.dtype}")
    if not np.isfinite(probs).all():
        raise ValueError("pupil probability contains non-finite values")
    pupil = labels == CLASS_PUPIL
    if not pupil.any():
        return {
            "native_pupil_probability_available": True,
            "native_pupil_softmax_mean_on_argmax_mask": None,
            "native_pupil_softmax_median_on_argmax_mask": None,
            "native_pupil_softmax_p05_on_argmax_mask": None,
            "native_pupil_softmax_p95_on_argmax_mask": None,
            "native_pupil_softmax_min_on_argmax_mask": None,
            "native_pupil_softmax_max_on_argmax_mask": None,
        }

    values = probs[pupil].astype(np.float64, copy=False)
    return {
        "native_pupil_probability_available": True,
        "native_pupil_softmax_mean_on_argmax_mask": float(np.mean(values)),
        "native_pupil_softmax_median_on_argmax_mask": float(np.median(values)),
        "native_pupil_softmax_p05_on_argmax_mask": float(np.percentile(values, 5)),
        "native_pupil_softmax_p95_on_argmax_mask": float(np.percentile(values, 95)),
        "native_pupil_softmax_min_on_argmax_mask": float(np.min(values)),
        "native_pupil_softmax_max_on_argmax_mask": float(np.max(values)),
    }


def summarize_fullclass_native(
    labels: np.ndarray,
    pupil_probability: np.ndarray | None = None,
    *,
    probability_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive geometry/QC facts directly from one uint8 400x640 hard label map.

    Pupil and iris geometry are always fitted from the same label map. No source
    eyes.csv pupil geometry is accepted by this function.
    """
    labels = validate_native_labels(labels)
    counts_array = np.bincount(labels.reshape(-1), minlength=len(CLASS_IDS))
    counts = {class_id: int(counts_array[class_id]) for class_id in CLASS_IDS}
    total = NATIVE_LABEL_WIDTH * NATIVE_LABEL_HEIGHT
    if int(counts_array.sum()) != total:
        raise AssertionError("RITnet class counts do not sum to 256000 pixels")

    pupil = labels == CLASS_PUPIL
    iris_outer = labels >= CLASS_IRIS
    ocular = labels != CLASS_BACKGROUND
    iris_outer_pixels = int(counts[CLASS_IRIS] + counts[CLASS_PUPIL])
    ocular_pixels = int(total - counts[CLASS_BACKGROUND])

    pupil_geom, _ = _ellipse_geometry(pupil)
    iris_geom, iris_contour = _ellipse_geometry(iris_outer)
    pupil_components, pupil_largest_fraction = _component_metrics(pupil)
    iris_components, iris_largest_fraction = _component_metrics(iris_outer)
    ocular_components, ocular_largest_fraction = _component_metrics(ocular)
    aperture = _ocular_aperture_metrics(ocular)

    diameter_ratio = ellipse_area_ratio = contour_area_ratio = None
    center_offset_px = center_offset_norm = None
    center_in_iris = None
    iris_larger = False
    pir_finite = False

    if pupil_geom["fit_valid"] and iris_geom["fit_valid"]:
        pupil_d = float(pupil_geom["geom_mean_diameter"])
        iris_d = float(iris_geom["geom_mean_diameter"])
        pupil_a = float(pupil_geom["ellipse_area"])
        iris_a = float(iris_geom["ellipse_area"])
        iris_larger = bool(iris_d > pupil_d)
        if iris_d > 0:
            diameter_ratio = float(pupil_d / iris_d)
            center_offset_norm = None
        if iris_a > 0:
            ellipse_area_ratio = float(pupil_a / iris_a)
        dx = float(pupil_geom["center_x"] - iris_geom["center_x"])
        dy = float(pupil_geom["center_y"] - iris_geom["center_y"])
        center_offset_px = float(np.hypot(dx, dy))
        if iris_d > 0:
            center_offset_norm = float(center_offset_px / iris_d)
        if iris_contour is not None:
            center_in_iris = bool(
                cv2.pointPolygonTest(
                    iris_contour,
                    (float(pupil_geom["center_x"]), float(pupil_geom["center_y"])),
                    False,
                ) >= 0
            )
        pir_finite = bool(diameter_ratio is not None and np.isfinite(diameter_ratio))

    if pupil_geom["contour_area"] is not None and iris_geom["contour_area"] not in (None, 0):
        contour_area_ratio = float(
            float(pupil_geom["contour_area"]) / float(iris_geom["contour_area"])
        )

    iris_fill_ratio = None
    if iris_geom["ellipse_area"] not in (None, 0):
        iris_fill_ratio = float(iris_outer_pixels / float(iris_geom["ellipse_area"]))

    result: dict[str, Any] = {
        "native_background_pixels": counts[CLASS_BACKGROUND],
        "native_background_fraction": float(counts[CLASS_BACKGROUND] / total),
        "native_sclera_pixels": counts[CLASS_SCLERA],
        "native_sclera_fraction": float(counts[CLASS_SCLERA] / total),
        "native_iris_pixels": counts[CLASS_IRIS],
        "native_iris_fraction": float(counts[CLASS_IRIS] / total),
        "native_pupil_pixels": counts[CLASS_PUPIL],
        "native_pupil_fraction": float(counts[CLASS_PUPIL] / total),
        "native_iris_outer_pixels": iris_outer_pixels,
        "native_iris_outer_fraction": float(iris_outer_pixels / total),
        "native_ocular_pixels": ocular_pixels,
        "native_ocular_fraction": float(ocular_pixels / total),
        "native_pupil_component_count": pupil_components,
        "native_pupil_largest_component_fraction": pupil_largest_fraction,
        "native_iris_outer_component_count": iris_components,
        "native_iris_outer_largest_component_fraction": iris_largest_fraction,
        "native_ocular_component_count": ocular_components,
        "native_ocular_largest_component_fraction": ocular_largest_fraction,
        "native_ocular_bbox_width": aperture["bbox_width"],
        "native_ocular_bbox_height": aperture["bbox_height"],
        "native_ocular_aperture_height_median": aperture["aperture_height_median"],
        "native_ocular_aperture_height_p90": aperture["aperture_height_p90"],
        "native_ocular_aperture_ratio_median": aperture["aperture_ratio_median"],
        "native_ocular_aperture_ratio_p90": aperture["aperture_ratio_p90"],
        "native_ocular_whole_mask_touches_edge": aperture["whole_mask_touches_edge"],
        "native_pupil_to_iris_diameter_ratio": diameter_ratio,
        "native_pupil_to_iris_ellipse_area_ratio": ellipse_area_ratio,
        "native_pupil_to_iris_contour_area_ratio": contour_area_ratio,
        "native_pupil_center_offset_px": center_offset_px,
        "native_pupil_center_offset_norm": center_offset_norm,
        "native_pupil_center_in_iris_outer": center_in_iris,
        "native_iris_diameter_gt_pupil_diameter": iris_larger,
        "native_pir_finite": pir_finite,
        "native_iris_outer_fill_ratio": iris_fill_ratio,
    }

    for prefix, geometry in (("native_pupil", pupil_geom), ("native_iris_outer", iris_geom)):
        for key, value in geometry.items():
            result[f"{prefix}_{key}"] = value

    if probability_summary is not None and pupil_probability is not None:
        raise ValueError("provide pupil_probability or probability_summary, not both")
    if probability_summary is None:
        probability_summary = summarize_pupil_probability(labels, pupil_probability)
    result.update(probability_summary)

    result.update(
        {
            "gate_pupil_fit_valid": bool(pupil_geom["fit_valid"]),
            "gate_iris_outer_fit_valid": bool(iris_geom["fit_valid"]),
            "gate_pupil_center_in_iris_outer": center_in_iris is True,
            "gate_iris_larger_than_pupil": bool(iris_larger),
            "gate_pir_finite": bool(pir_finite),
            "diagnostic_pupil_whole_mask_edge": bool(pupil_geom["whole_mask_touches_edge"]),
            "diagnostic_pupil_largest_contour_edge": bool(pupil_geom["largest_contour_touches_edge"]),
            "diagnostic_iris_whole_mask_edge": bool(iris_geom["whole_mask_touches_edge"]),
            "diagnostic_iris_largest_contour_edge": bool(iris_geom["largest_contour_touches_edge"]),
            "diagnostic_pupil_fragmented": bool(pupil_components > 1),
            "diagnostic_iris_fragmented": bool(iris_components > 1),
            "diagnostic_ocular_fragmented": bool(ocular_components > 1),
        }
    )

    # Replays the old logical gate on native geometry for sensitivity/reference
    # only. It is NOT bit-identical to v1.2, which used 320x160 geometry.
    result["legacy_v1_strict_valid"] = bool(
        result["gate_pupil_fit_valid"]
        and result["gate_iris_outer_fit_valid"]
        and not result["diagnostic_pupil_whole_mask_edge"]
        and not result["diagnostic_iris_whole_mask_edge"]
        and result["gate_pupil_center_in_iris_outer"]
        and result["gate_iris_larger_than_pupil"]
    )
    return result
