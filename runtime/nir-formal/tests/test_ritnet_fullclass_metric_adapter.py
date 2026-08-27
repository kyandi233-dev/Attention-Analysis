from __future__ import annotations

import cv2
import numpy as np

from ritnet_fullclass_metric_adapter import summarize_final_hard_metrics


def synthetic_labels():
    labels = np.zeros((400, 640), dtype=np.uint8)
    cv2.ellipse(labels, (320, 200), (220, 90), 0, 0, 360, 1, -1)
    cv2.ellipse(labels, (320, 200), (90, 60), 0, 0, 360, 2, -1)
    cv2.ellipse(labels, (320, 200), (35, 25), 0, 0, 360, 3, -1)
    return labels


def test_adapter_exposes_final_hard_metrics_without_historical_probability_fields():
    result = summarize_final_hard_metrics(synthetic_labels())
    assert result["hard_background_pixels"] > 0
    assert result["hard_sclera_pixels"] > 0
    assert result["hard_iris_pixels"] > 0
    assert result["hard_pupil_pixels"] > 0
    assert result["pupil_fit_valid"] is True
    assert result["iris_outer_fit_valid"] is True
    assert 0 < result["pupil_to_iris_diameter_ratio"] < 1
    assert result["pupil_center_in_iris_outer"] is True
    assert result["analysis_valid_pixel_count"] == 400 * 640
    assert result["analysis_valid_pixel_fraction"] == 1.0
    assert "native_pupil_probability_available" not in result
    assert "source_pupil_confidence" not in result


def test_hard_class_fractions_cover_exactly_one_full_label_plane():
    result = summarize_final_hard_metrics(synthetic_labels())
    hard_sum = sum(
        result[f"hard_{name}_fraction"]
        for name in ("background", "sclera", "iris", "pupil")
    )
    assert abs(hard_sum - 1.0) < 1e-12


def test_padding_predictions_do_not_enter_hard_counts_or_denominator():
    labels = np.zeros((400, 640), dtype=np.uint8)
    labels[:, :80] = 3
    valid = np.ones((400, 640), dtype=bool)
    valid[:, :80] = False

    result = summarize_final_hard_metrics(labels, valid)
    assert result["analysis_valid_pixel_count"] == 400 * 560
    assert result["hard_pupil_pixels"] == 0
    assert result["hard_pupil_fraction"] == 0.0
    assert result["hard_background_pixels"] == 400 * 560
    assert result["hard_background_fraction"] == 1.0
    assert result["pupil_predicted_in_padding_pixels"] == 400 * 80


def test_structure_touching_internal_valid_boundary_is_flagged():
    labels = np.zeros((400, 640), dtype=np.uint8)
    labels[150:250, 80:180] = 1
    labels[175:225, 80:130] = 2
    labels[190:210, 80:100] = 3
    valid = np.ones((400, 640), dtype=bool)
    valid[:, :80] = False

    result = summarize_final_hard_metrics(labels, valid)
    assert result["pupil_touches_valid_domain_edge"] is True
    assert result["iris_outer_touches_valid_domain_edge"] is True
    assert result["ocular_touches_valid_domain_edge"] is True


def test_padding_overlap_is_reported_without_changing_observed_geometry_domain():
    labels = synthetic_labels()
    labels[180:220, :40] = 3
    valid = np.ones((400, 640), dtype=bool)
    valid[:, :50] = False

    full = summarize_final_hard_metrics(synthetic_labels())
    padded = summarize_final_hard_metrics(labels, valid)
    assert padded["pupil_predicted_in_padding_pixels"] > 0
    assert padded["pupil_fit_valid"] is True
    assert padded["pupil_center_x"] == full["pupil_center_x"]
    assert padded["pupil_center_y"] == full["pupil_center_y"]
