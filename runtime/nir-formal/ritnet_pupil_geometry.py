"""Robust pupil geometry for final RITnet segmentation outputs.

RITnet upstream is a semantic-segmentation model and does not define an ellipse
post-processor. For fragmented pupil masks this module adapts the segmentation-
to-ellipse reference path published by the same eye-segmentation author team in
RSKothari/EllSeg (MIT): semantic valid-boundary extraction followed by ElliFit
and RANSAC outlier removal. Clean single-component pupils intentionally keep the
historical OpenCV fitEllipse geometry so established center/diameter values do
not change unnecessarily.

EllSeg reference: https://github.com/RSKothari/EllSeg
Reference commit inspected: 8f1ea13336fa9c662403bd5823ac13cc6a6dd632
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ritnet_native_metrics import _ellipse_geometry


PUPIL_GEOMETRY_VERSION = "pupil-geometry-v3-single-opencv-fragmented-ellseg-ransac"
ELLSEG_MIN_PUPIL_PIXELS = 50
ELLSEG_RANSAC_MIN_SAMPLE = 15
ELLSEG_RANSAC_ITERATIONS = 40
ELLSEG_RANSAC_THRESHOLD = 5e-3
ELLSEG_RANSAC_MIN_GOOD = 15


@dataclass(frozen=True)
class EllSegFit:
    model: np.ndarray
    error: float
    inlier_count: int
    inlier_fraction: float
    valid_point_count: int


def _empty_geometry(reason: str, *, method: str, valid_point_count: int = 0) -> dict[str, Any]:
    return {
        "found": False,
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
        "whole_mask_touches_edge": False,
        "largest_contour_touches_edge": False,
        "geometry_method": method,
        "geometry_failure_reason": reason,
        "valid_boundary_point_count": int(valid_point_count),
        "ransac_inlier_count": None,
        "ransac_inlier_fraction": None,
        "ellipse_fit_error": None,
        "axis_ratio": None,
        "contour_to_ellipse_area_ratio": None,
    }


def _canonicalize_opencv_geometry(geometry: dict[str, Any]) -> dict[str, Any]:
    result = dict(geometry)
    result.update(
        {
            "geometry_method": "opencv-fitellipse-single-component-v2",
            "geometry_failure_reason": None,
            "valid_boundary_point_count": None,
            "ransac_inlier_count": None,
            "ransac_inlier_fraction": None,
            "ellipse_fit_error": None,
            "axis_ratio": None,
            "contour_to_ellipse_area_ratio": None,
        }
    )
    if not result.get("fit_valid"):
        result["geometry_failure_reason"] = "opencv_fit_invalid"
        return result

    axis_a = float(result["axis_a"])
    axis_b = float(result["axis_b"])
    angle = float(result["angle_deg"])
    if axis_a >= axis_b:
        long_axis, short_axis = axis_a, axis_b
        long_angle = angle
    else:
        long_axis, short_axis = axis_b, axis_a
        long_angle = angle + 90.0
    result["long_axis"] = float(long_axis)
    result["short_axis"] = float(short_axis)
    result["angle_deg"] = float(long_angle % 180.0)
    result["axis_ratio"] = float(short_axis / long_axis) if long_axis > 0 else None
    ellipse_area = float(result["ellipse_area"])
    contour_area = float(result["contour_area"])
    result["contour_to_ellipse_area_ratio"] = (
        float(contour_area / ellipse_area) if ellipse_area > 0 else None
    )
    return result


def _valid_pupil_boundary_points(labels: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of EllSeg getValidPoints for PartSeg pupil edges.

    A valid pupil boundary point must lie on a label edge whose local 3x3
    neighborhood contains neither background (0) nor sclera (1). In the RITnet
    four-class convention this retains the pupil/iris interface and suppresses
    peripheral class-3 fragments attached to non-iris regions.
    """
    label_map = np.asarray(labels, dtype=np.uint8)
    maximum = int(label_map.max()) if label_map.size else 0
    if maximum <= 0:
        return np.empty((0, 2), dtype=np.float64)
    image = np.asarray(np.round(255.0 * label_map.astype(np.float32) / maximum), dtype=np.uint8)
    edges = (cv2.Canny(image, 50, 100) > 0) | (cv2.Canny(255 - image, 50, 100) > 0)
    forbidden = (label_map <= 1).astype(np.uint8)
    near_forbidden = cv2.dilate(forbidden, np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
    accepted = edges & ~near_forbidden
    accepted[[0, -1], :] = False
    accepted[:, [0, -1]] = False
    y, x = np.where(accepted)
    if x.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.ascontiguousarray(np.stack([x, y], axis=1), dtype=np.float64)


def _ellifit(data: np.ndarray) -> tuple[np.ndarray, float]:
    points = np.asarray(data, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] <= 12:
        return -np.ones(5, dtype=np.float64), float("inf")
    xm = float(np.mean(points[:, 0]))
    ym = float(np.mean(points[:, 1]))
    x = points[:, 0] - xm
    y = points[:, 1] - ym
    X = np.stack([x**2, 2*x*y, -2*x, -2*y, -np.ones_like(x)], axis=1)
    Y = -y**2
    try:
        phi = np.linalg.solve(X.T @ X, X.T @ Y)
        denom = float(phi[0] - phi[1] ** 2)
        if abs(denom) < 1e-12:
            raise FloatingPointError("degenerate ElliFit denominator")
        x0 = float((phi[2] - phi[3] * phi[1]) / denom)
        y0 = float((phi[0] * phi[3] - phi[2] * phi[1]) / denom)
        term2 = float(np.sqrt((1 - phi[0]) ** 2 + 4 * phi[1] ** 2))
        term3 = float(phi[4] + y0**2 + x0**2 * phi[0] + 2 * phi[1])
        term1 = float(1 + phi[0])
        den_b = term1 + term2
        den_a = term1 - term2
        if den_a <= 0 or den_b <= 0 or term3 <= 0:
            raise FloatingPointError("non-elliptic ElliFit solution")
        b = float(np.sqrt(2 * term3 / den_b))
        a = float(np.sqrt(2 * term3 / den_a))
        alpha = float(0.5 * np.arctan2(2 * phi[1], 1 - phi[0]))
        model = np.asarray([x0 + xm, y0 + ym, a, b, -alpha], dtype=np.float64)
    except Exception:
        return -np.ones(5, dtype=np.float64), float("inf")
    if not np.isfinite(model).all() or model[2] <= 0 or model[3] <= 0:
        return -np.ones(5, dtype=np.float64), float("inf")
    residual = _ellipse_residual(model, points)
    return model, float(np.mean(residual)) if residual.size else float("inf")


def _ellipse_residual(model: np.ndarray, data: np.ndarray) -> np.ndarray:
    points = np.asarray(data, dtype=np.float64)
    cx, cy, a, b, theta = map(float, model)
    if a <= 0 or b <= 0 or not np.isfinite([cx, cy, a, b, theta]).all():
        return np.full(points.shape[0], np.inf, dtype=np.float64)
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    term1 = dx * np.cos(theta)
    term2 = dy * np.sin(theta)
    term3 = dx * np.sin(theta)
    term4 = dy * np.cos(theta)
    value = (term1 - term2) ** 2 / (a**2) + (term3 + term4) ** 2 / (b**2) - 1.0
    return np.abs(value)


def _deterministic_ellseg_ransac(data: np.ndarray) -> EllSegFit | None:
    points = np.asarray(data, dtype=np.float64)
    count = int(points.shape[0])
    if count < ELLSEG_RANSAC_MIN_SAMPLE:
        return None

    best_model, best_error = _ellifit(points)
    best_residual = _ellipse_residual(best_model, points)
    best_inliers = best_residual < ELLSEG_RANSAC_THRESHOLD

    # EllSeg upstream uses random RANSAC. The formal pipeline freezes the RNG so
    # the same segmentation map always produces the same ellipse.
    rng = np.random.default_rng(0)
    for _ in range(ELLSEG_RANSAC_ITERATIONS + 1):
        sample = rng.choice(count, ELLSEG_RANSAC_MIN_SAMPLE, replace=False)
        candidate, _ = _ellifit(points[sample])
        residual = _ellipse_residual(candidate, points)
        inliers = residual < ELLSEG_RANSAC_THRESHOLD
        inliers[sample] = True
        if int(inliers.sum()) <= ELLSEG_RANSAC_MIN_GOOD:
            continue
        better_model, better_error = _ellifit(points[inliers])
        if better_error < best_error:
            best_model = better_model
            best_error = better_error
            best_residual = _ellipse_residual(best_model, points)
            best_inliers = best_residual < ELLSEG_RANSAC_THRESHOLD

    if not np.isfinite(best_model).all() or np.any(best_model[2:4] <= 0):
        return None
    inlier_count = int(best_inliers.sum())
    if inlier_count < ELLSEG_RANSAC_MIN_GOOD:
        return None
    return EllSegFit(
        model=best_model,
        error=float(best_error),
        inlier_count=inlier_count,
        inlier_fraction=float(inlier_count / count),
        valid_point_count=count,
    )


def _selected_component_from_fit(pupil_mask: np.ndarray, fit: EllSegFit) -> np.ndarray:
    binary = np.ascontiguousarray(pupil_mask.astype(np.uint8))
    count, component_ids, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if int(count) <= 1:
        return pupil_mask.astype(bool)
    cx, cy, a, b, theta = map(float, fit.model)
    yy, xx = np.ogrid[:pupil_mask.shape[0], :pupil_mask.shape[1]]
    dx = xx - cx
    dy = yy - cy
    u = dx * np.cos(theta) - dy * np.sin(theta)
    v = dx * np.sin(theta) + dy * np.cos(theta)
    ellipse_mask = (u * u / (a * a) + v * v / (b * b)) <= 1.0
    ranked: list[tuple[int, int, int]] = []
    for component_id in range(1, int(count)):
        component = component_ids == component_id
        overlap = int(np.count_nonzero(component & ellipse_mask))
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        ranked.append((overlap, area, -component_id))
    chosen = -max(ranked)[2]
    return component_ids == chosen


def _ellseg_fragmented_geometry(labels: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    working = np.asarray(labels, dtype=np.uint8).copy()
    working[~valid] = 0
    pupil_mask = working == 3
    pupil_pixels = int(pupil_mask.sum())
    if pupil_pixels <= ELLSEG_MIN_PUPIL_PIXELS:
        return _empty_geometry(
            "ellseg_insufficient_pupil_pixels",
            method="ellseg-valid-boundary-ellifit-ransac-v1",
        )
    points = _valid_pupil_boundary_points(working)
    if int(points.shape[0]) < ELLSEG_RANSAC_MIN_SAMPLE:
        return _empty_geometry(
            "ellseg_insufficient_valid_boundary_points",
            method="ellseg-valid-boundary-ellifit-ransac-v1",
            valid_point_count=int(points.shape[0]),
        )
    fit = _deterministic_ellseg_ransac(points)
    if fit is None:
        return _empty_geometry(
            "ellseg_ransac_fit_failed",
            method="ellseg-valid-boundary-ellifit-ransac-v1",
            valid_point_count=int(points.shape[0]),
        )

    cx, cy, a, b, theta = map(float, fit.model)
    h, w = working.shape
    if not (0 <= cx < w and 0 <= cy < h and valid[int(round(cy)), int(round(cx))]):
        return _empty_geometry(
            "ellseg_center_outside_valid_source",
            method="ellseg-valid-boundary-ellifit-ransac-v1",
            valid_point_count=fit.valid_point_count,
        )

    selected = _selected_component_from_fit(pupil_mask, fit)
    contours, _ = cv2.findContours(
        np.ascontiguousarray(selected.astype(np.uint8)),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contour = max(contours, key=cv2.contourArea) if contours else None
    contour_area = float(cv2.contourArea(contour)) if contour is not None else None
    axis_a = float(2.0 * a)
    axis_b = float(2.0 * b)
    if axis_a >= axis_b:
        long_axis, short_axis = axis_a, axis_b
        long_angle = np.degrees(theta)
    else:
        long_axis, short_axis = axis_b, axis_a
        long_angle = np.degrees(theta) + 90.0
    ellipse_area = float(np.pi * a * b)
    touches = bool(
        selected[0, :].any()
        or selected[-1, :].any()
        or selected[:, 0].any()
        or selected[:, -1].any()
    )
    return {
        "found": True,
        "fit_valid": True,
        "center_x": cx,
        "center_y": cy,
        "axis_a": axis_a,
        "axis_b": axis_b,
        "short_axis": float(short_axis),
        "long_axis": float(long_axis),
        "angle_deg": float(long_angle % 180.0),
        "contour_area": contour_area,
        "ellipse_area": ellipse_area,
        "equiv_diameter": (
            float(2.0 * np.sqrt(contour_area / np.pi)) if contour_area is not None and contour_area > 0 else None
        ),
        "geom_mean_diameter": float(np.sqrt(axis_a * axis_b)),
        "whole_mask_touches_edge": touches,
        "largest_contour_touches_edge": touches,
        "geometry_method": "ellseg-valid-boundary-ellifit-ransac-v1",
        "geometry_failure_reason": None,
        "valid_boundary_point_count": fit.valid_point_count,
        "ransac_inlier_count": fit.inlier_count,
        "ransac_inlier_fraction": fit.inlier_fraction,
        "ellipse_fit_error": fit.error,
        "axis_ratio": float(short_axis / long_axis) if long_axis > 0 else None,
        "contour_to_ellipse_area_ratio": (
            float(contour_area / ellipse_area) if contour_area is not None and ellipse_area > 0 else None
        ),
    }


def fit_final_pupil_geometry(
    labels: np.ndarray,
    valid_source_mask: np.ndarray,
    *,
    component_count: int,
) -> dict[str, Any]:
    """Fit final pupil geometry without letting fragmented junk win by area."""
    valid = np.asarray(valid_source_mask, dtype=bool)
    pupil = (np.asarray(labels, dtype=np.uint8) == 3) & valid
    if int(component_count) <= 1:
        geometry, _ = _ellipse_geometry(pupil)
        return _canonicalize_opencv_geometry(geometry)
    return _ellseg_fragmented_geometry(labels, valid)
