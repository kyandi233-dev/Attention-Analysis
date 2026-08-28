from __future__ import annotations

import cv2
import numpy as np
import pytest

from benchmark_ritnet_pupil_geometry import (
    METHOD_ELLSEG,
    METHOD_LEGACY,
    METHOD_TOPOLOGY,
    compare_record,
)
from ritnet_pupil_geometry import (
    ELLIFIT_MIN_POINT_COUNT,
    _canonicalize_opencv_geometry,
    _deterministic_ellseg_ransac,
    _ellifit,
    _valid_pupil_boundary_points,
    fit_ellseg_partseg_pupil_geometry,
)


def _ellipse_points(
    *,
    cx: float,
    cy: float,
    a: float,
    b: float,
    theta: float,
    count: int,
) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    x = a * np.cos(t)
    y = b * np.sin(t)
    c = np.cos(theta)
    s = np.sin(theta)
    xr = c * x - s * y + cx
    yr = s * x + c * y + cy
    return np.stack([xr, yr], axis=1)


def _background_fragment_case() -> np.ndarray:
    labels = np.zeros((400, 640), dtype=np.uint8)

    # Main anatomical iris and pupil.
    cv2.ellipse(labels, (230, 195), (105, 66), 8, 0, 360, 2, -1)
    cv2.ellipse(labels, (230, 195), (22, 17), 12, 0, 360, 3, -1)

    # Deliberately larger false pupil island directly exposed to background.
    # Historical largest-contour OpenCV fitting should jump here, while EllSeg
    # PartSeg semantic-boundary filtering should reject its background-facing
    # edge because the local 3x3 neighbourhood contains class 0.
    cv2.ellipse(labels, (535, 305), (36, 28), -5, 0, 360, 3, -1)
    return labels


def test_ellifit_minimum_is_seven_2d_points_not_thirteen_or_fifteen():
    assert ELLIFIT_MIN_POINT_COUNT == 7
    # Use an axis-aligned synthetic ellipse here. EllSeg's published ElliFit
    # parameter/residual code uses an orientation sign convention that is not
    # the object of this minimum-point contract test.
    points = _ellipse_points(cx=201.0, cy=133.0, a=38.0, b=21.0, theta=0.0, count=7)

    model, error = _ellifit(points)

    assert np.isfinite(model).all()
    assert model[0] == pytest.approx(201.0, abs=1e-5)
    assert model[1] == pytest.approx(133.0, abs=1e-5)
    assert error < 1e-8


def test_ellseg_uses_direct_ellifit_when_valid_points_do_not_exceed_15():
    points = _ellipse_points(cx=300.0, cy=180.0, a=42.0, b=19.0, theta=-0.22, count=12)

    fit = _deterministic_ellseg_ransac(points)

    assert fit is not None
    assert fit.used_ransac is False
    assert fit.valid_point_count == 12
    assert fit.inlier_count == 12
    assert fit.inlier_fraction == pytest.approx(1.0)
    assert fit.model[0] == pytest.approx(300.0, abs=1e-5)
    assert fit.model[1] == pytest.approx(180.0, abs=1e-5)


def test_opencv_axis_sorting_rotates_angle_with_the_axis_swap():
    raw = {
        "found": True,
        "fit_valid": True,
        "center_x": 100.0,
        "center_y": 80.0,
        "axis_a": 20.0,
        "axis_b": 50.0,
        "short_axis": 20.0,
        "long_axis": 50.0,
        "angle_deg": 17.0,
        "contour_area": 700.0,
        "ellipse_area": 785.0,
        "equiv_diameter": 29.85,
        "geom_mean_diameter": np.sqrt(1000.0),
        "whole_mask_touches_edge": False,
        "largest_contour_touches_edge": False,
    }

    canonical = _canonicalize_opencv_geometry(raw)

    assert canonical["long_axis"] == pytest.approx(50.0)
    assert canonical["short_axis"] == pytest.approx(20.0)
    assert canonical["angle_deg"] == pytest.approx(107.0)


def test_partseg_boundary_filter_rejects_background_exposed_false_pupil():
    labels = _background_fragment_case()
    points = _valid_pupil_boundary_points(labels)

    assert points.shape[0] >= ELLIFIT_MIN_POINT_COUNT
    # The false island is around x=535 and is directly adjacent to background.
    # No accepted semantic pupil/iris boundary should survive there.
    assert float(points[:, 0].max()) < 400.0

    geometry = fit_ellseg_partseg_pupil_geometry(
        labels,
        np.ones((400, 640), dtype=bool),
    )
    assert geometry["fit_valid"] is True
    assert geometry["center_x"] == pytest.approx(230.0, abs=3.0)
    assert geometry["center_y"] == pytest.approx(195.0, abs=3.0)
    assert geometry["valid_boundary_point_count"] >= ELLIFIT_MIN_POINT_COUNT


def test_three_path_benchmark_exposes_legacy_failure_without_changing_production():
    labels = _background_fragment_case()
    record = {
        "labels": labels,
        "valid": np.ones((400, 640), dtype=bool),
        "source": "synthetic",
        "record_index": 0,
        "subject": "sub-test",
        "phase": "baseline",
        "phase_segment": 1,
        "frame_idx": 123,
        "eye": "frame_left",
        "reasons": "fragmented",
    }

    rows = compare_record(record)
    by_method = {row["method"]: row for row in rows}

    assert set(by_method) == {METHOD_LEGACY, METHOD_TOPOLOGY, METHOD_ELLSEG}
    assert by_method[METHOD_LEGACY]["fit_valid"] is True
    assert float(by_method[METHOD_LEGACY]["center_x"]) > 450.0

    assert by_method[METHOD_TOPOLOGY]["fit_valid"] is True
    assert by_method[METHOD_TOPOLOGY]["center_x"] == pytest.approx(230.0, abs=3.0)

    assert by_method[METHOD_ELLSEG]["fit_valid"] is True
    assert by_method[METHOD_ELLSEG]["center_x"] == pytest.approx(230.0, abs=3.0)


def test_artificial_padding_is_removed_before_ellseg_boundary_fitting():
    labels = _background_fragment_case()
    valid = np.ones((400, 640), dtype=bool)
    valid[:, 450:] = False

    geometry = fit_ellseg_partseg_pupil_geometry(labels, valid)

    assert geometry["fit_valid"] is True
    assert geometry["center_x"] == pytest.approx(230.0, abs=3.0)
    assert geometry["center_y"] == pytest.approx(195.0, abs=3.0)
