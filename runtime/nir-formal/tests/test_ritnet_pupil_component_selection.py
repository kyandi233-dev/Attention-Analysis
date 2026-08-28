from __future__ import annotations

import cv2
import numpy as np
import pytest

from ritnet_fullclass_metric_adapter import (
    ANALYSIS_DOMAIN_VERSION,
    _primary_pupil_component,
    summarize_final_hard_metrics,
)
from ritnet_native_metrics import _ellipse_geometry


def _fragmented_labels() -> np.ndarray:
    labels = np.zeros((400, 640), dtype=np.uint8)
    cv2.ellipse(labels, (260, 190), (105, 62), 0, 0, 360, 2, -1)
    cv2.ellipse(labels, (260, 190), (18, 14), 0, 0, 360, 3, -1)
    cv2.ellipse(labels, (555, 320), (42, 32), 0, 0, 360, 2, -1)
    cv2.ellipse(labels, (555, 320), (27, 21), 0, 0, 360, 3, -1)
    return labels


def test_fragmented_pupil_prefers_component_embedded_in_main_iris_outer_topology():
    labels = _fragmented_labels()
    valid = np.ones((400, 640), dtype=bool)

    historical_geometry, _ = _ellipse_geometry(labels == 3)
    assert historical_geometry["fit_valid"] is True
    assert float(historical_geometry["center_x"]) > 500.0

    selected = _primary_pupil_component(labels, valid)
    selected_geometry, _ = _ellipse_geometry(selected)
    assert selected_geometry["fit_valid"] is True
    assert float(selected_geometry["center_x"]) == pytest.approx(260.0, abs=1.0)
    assert float(selected_geometry["center_y"]) == pytest.approx(190.0, abs=1.0)

    metrics = summarize_final_hard_metrics(labels, valid_source_mask=valid)
    assert metrics["pupil_component_count"] == 2
    assert metrics["qc_pupil_fragmented"] is True
    assert metrics["pupil_center_x"] == pytest.approx(260.0, abs=1.0)
    assert metrics["pupil_center_y"] == pytest.approx(190.0, abs=1.0)
    assert metrics["analysis_domain_version"] == ANALYSIS_DOMAIN_VERSION
    assert ANALYSIS_DOMAIN_VERSION == "source-backed-output-mask-v3-primary-pupil-topology"


def test_single_pupil_component_keeps_historical_geometry_exactly():
    labels = np.zeros((400, 640), dtype=np.uint8)
    cv2.ellipse(labels, (310, 205), (110, 65), 13, 0, 360, 2, -1)
    cv2.ellipse(labels, (310, 205), (31, 22), 17, 0, 360, 3, -1)
    valid = np.ones((400, 640), dtype=bool)

    historical, _ = _ellipse_geometry(labels == 3)
    metrics = summarize_final_hard_metrics(labels, valid_source_mask=valid)

    assert metrics["pupil_component_count"] == 1
    assert metrics["qc_pupil_fragmented"] is False
    for field in (
        "center_x",
        "center_y",
        "short_axis",
        "long_axis",
        "angle_deg",
        "contour_area",
        "ellipse_area",
        "equiv_diameter",
        "geom_mean_diameter",
    ):
        assert metrics[f"pupil_{field}"] == pytest.approx(historical[field], rel=0, abs=1e-12)


def test_padding_cannot_make_a_pupil_component_win_primary_selection():
    labels = _fragmented_labels()
    valid = np.ones((400, 640), dtype=bool)
    valid[:, 500:] = False

    metrics = summarize_final_hard_metrics(labels, valid_source_mask=valid)
    assert metrics["pupil_component_count"] == 1
    assert metrics["qc_pupil_fragmented"] is False
    assert metrics["pupil_center_x"] == pytest.approx(260.0, abs=1.0)
    assert metrics["pupil_predicted_in_padding_pixels"] > 0
