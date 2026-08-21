import math

import numpy as np

from attention_pipeline.contracts import EYE_LEFT, EYE_RIGHT
from attention_pipeline.nir.metrics import rolling_perclos
from attention_pipeline.nir.roi import (
    EYE_CORNERS,
    ellipse_from_three_points,
    inverse_affine,
    map_ellipse_to_source,
    normalized_eye_roi,
    roi_border_status,
    transform_points,
)


def test_anatomical_eye_naming_is_explicit():
    assert EYE_CORNERS[EYE_RIGHT] == (33, 133)  # subject right = image left
    assert EYE_CORNERS[EYE_LEFT] == (362, 263)  # subject left = image right


def test_fixed_roi_affine_maps_corners_to_expected_positions():
    image = np.zeros((300, 500, 3), dtype=np.uint8)
    points = np.zeros((478, 2), dtype=np.float32)
    points[33] = (100, 120)
    points[133] = (200, 120)
    roi, affine, distance = normalized_eye_roi(image, points, (33, 133))
    mapped = transform_points(points[[33, 133]], affine)
    assert roi.shape == (160, 320, 3)
    assert math.isclose(distance, 100)
    np.testing.assert_allclose(mapped, [[80, 80], [240, 80]], atol=1e-4)


def test_three_point_ellipse_contract():
    ellipse = ellipse_from_three_points((100, 80), (130, 80), (100, 90))
    assert ellipse["major_diameter"] == 60
    assert ellipse["minor_diameter"] == 20
    assert math.isclose(ellipse["equivalent_diameter"], math.sqrt(1200))


def test_missing_ear_is_not_closed_or_perclos_denominator():
    result = rolling_perclos([0.1, np.nan, 0.4], [0, 100, 200], threshold=0.2, window_sec=1)
    assert math.isclose(result[-1], 0.5)


def test_inverse_affine_round_trips_source_points():
    image = np.zeros((300, 500, 3), dtype=np.uint8)
    points = np.zeros((478, 2), dtype=np.float32)
    points[33], points[133] = (100, 120), (200, 120)
    _, affine, _ = normalized_eye_roi(image, points, (33, 133))
    original = np.array([[100.0, 120.0], [200.0, 120.0], [150.0, 145.0]])
    restored = transform_points(transform_points(original, affine), inverse_affine(affine))
    np.testing.assert_allclose(restored, original, atol=1e-3)


def test_map_ellipse_to_source_preserves_shape_under_similarity():
    scaled = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)
    mapped = map_ellipse_to_source((100, 50), (140, 50), (100, 80), inverse_affine(scaled))
    assert math.isclose(mapped["major_diameter"], 40, abs_tol=1e-3)
    assert math.isclose(mapped["minor_diameter"], 30, abs_tol=1e-3)
    assert math.isclose(mapped["equivalent_diameter"], math.sqrt(1200), abs_tol=1e-3)


def test_roi_border_status_flags_out_of_bounds_corners():
    image = np.zeros((300, 500, 3), dtype=np.uint8)
    points = np.zeros((478, 2), dtype=np.float32)
    points[33], points[133] = (100, 120), (200, 120)
    _, affine, _ = normalized_eye_roi(image, points, (33, 133))
    assert roi_border_status((300, 500), affine) == "ready"
    assert roi_border_status((120, 500), affine) == "border_heavy"

