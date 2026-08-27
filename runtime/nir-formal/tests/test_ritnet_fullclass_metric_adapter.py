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
    assert "native_pupil_probability_available" not in result
    assert "source_pupil_confidence" not in result


def test_hard_class_fractions_cover_exactly_one_full_label_plane():
    result = summarize_final_hard_metrics(synthetic_labels())
    hard_sum = sum(
        result[f"hard_{name}_fraction"]
        for name in ("background", "sclera", "iris", "pupil")
    )
    assert abs(hard_sum - 1.0) < 1e-12
