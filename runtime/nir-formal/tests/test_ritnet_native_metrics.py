from __future__ import annotations

import cv2
import numpy as np
import pytest

from ritnet_native_metrics import summarize_fullclass_native, validate_native_labels


def synthetic_labels() -> np.ndarray:
    labels = np.zeros((400, 640), dtype=np.uint8)
    cv2.ellipse(labels, (320, 200), (240, 120), 0, 0, 360, 1, -1)
    cv2.ellipse(labels, (320, 200), (90, 80), 0, 0, 360, 2, -1)
    cv2.ellipse(labels, (320, 200), (38, 32), 0, 0, 360, 3, -1)
    return labels


def test_native_geometry_and_counts_are_from_same_label():
    labels = synthetic_labels()
    probs = np.full(labels.shape, 0.91, dtype=np.float32)
    result = summarize_fullclass_native(labels, probs)
    assert (
        result["native_background_pixels"]
        + result["native_sclera_pixels"]
        + result["native_iris_pixels"]
        + result["native_pupil_pixels"]
        == 256000
    )
    assert result["native_pupil_fit_valid"] is True
    assert result["native_iris_outer_fit_valid"] is True
    assert 0 < result["native_pupil_to_iris_diameter_ratio"] < 1
    assert result["native_pupil_center_in_iris_outer"] is True
    assert result["native_iris_diameter_gt_pupil_diameter"] is True
    assert result["native_pir_finite"] is True
    assert result["gate_pupil_fit_valid"] is True
    assert result["gate_iris_outer_fit_valid"] is True
    assert "native_primary_valid" not in result
    assert np.isclose(result["native_pupil_softmax_mean_on_argmax_mask"], 0.91, atol=1e-6)


def test_validation_rejects_wrong_shape_dtype_and_domain():
    with pytest.raises(ValueError):
        validate_native_labels(np.zeros((160, 320), dtype=np.uint8))
    with pytest.raises(TypeError):
        validate_native_labels(np.zeros((400, 640), dtype=np.int32))
    invalid = np.zeros((400, 640), dtype=np.uint8)
    invalid[0, 0] = 4
    with pytest.raises(ValueError):
        validate_native_labels(invalid)


def test_whole_mask_edge_is_distinct_from_main_contour_edge():
    labels = synthetic_labels()
    labels[0, 0] = 3
    result = summarize_fullclass_native(labels)
    assert result["native_pupil_component_count"] == 2
    assert result["native_pupil_whole_mask_touches_edge"] is True
    assert result["native_pupil_largest_contour_touches_edge"] is False
    assert result["diagnostic_pupil_fragmented"] is True


def test_probability_available_without_argmax_pupil_does_not_fake_zero_confidence():
    labels = synthetic_labels()
    labels[labels == 3] = 2
    probs = np.full(labels.shape, 0.10, dtype=np.float32)
    result = summarize_fullclass_native(labels, probs)
    assert result["native_pupil_probability_available"] is True
    assert result["native_pupil_softmax_mean_on_argmax_mask"] is None
    assert result["gate_pupil_fit_valid"] is False
    assert result["legacy_v1_strict_valid"] is False


def test_ocular_aperture_is_geometry_only():
    labels = synthetic_labels()
    result = summarize_fullclass_native(labels)
    assert result["native_ocular_bbox_width"] > 0
    assert result["native_ocular_aperture_ratio_median"] > 0
    assert result["native_ocular_aperture_ratio_p90"] >= result["native_ocular_aperture_ratio_median"]
