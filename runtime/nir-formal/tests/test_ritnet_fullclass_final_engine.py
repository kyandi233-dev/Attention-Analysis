from __future__ import annotations

import cv2
import numpy as np
import pytest

from ritnet_fullclass_final_engine import (
    _complete_batch,
    _final_roi_config,
    _group_remaining_rows,
    _source_base_row,
)


def source(frame=100, eye="frame_left"):
    return {
        "phase": "block1",
        "phase_segment": 1,
        "frame_idx": frame,
        "video_time_ms": 1000,
        "unix_ms": 2000,
        "phase_time_ms": 500,
        "eye": eye,
        "source": "yolo",
        "redetect_reason": "tracker_disabled",
        "frame_status": "two_eyes",
        "status": "observed",
        "anchor_yolo_confidence": 0.9,
        "bbox_x1": 10,
        "bbox_y1": 20,
        "bbox_x2": 50,
        "bbox_y2": 60,
        "yolo_batch_size": 8,
        "pupil_confidence": 0.999,
    }


def synthetic_labels():
    labels = np.zeros((400, 640), dtype=np.uint8)
    cv2.ellipse(labels, (320, 200), (220, 90), 0, 0, 360, 1, -1)
    cv2.ellipse(labels, (320, 200), (90, 60), 0, 0, 360, 2, -1)
    cv2.ellipse(labels, (320, 200), (35, 25), 0, 0, 360, 3, -1)
    return labels


def class_probability(count):
    probability = np.zeros((count, 4, 400, 640), dtype=np.float32)
    probability[:, 0] = 0.50
    probability[:, 1] = 0.30
    probability[:, 2] = 0.15
    probability[:, 3] = 0.05
    return probability


class FakeRuntime:
    def __init__(self, labels):
        self.labels = np.asarray(labels, dtype=np.uint8)

    def infer_batch(self, rois):
        count = len(rois)
        return {
            "labels": np.stack([self.labels] * count),
            "class_probability": class_probability(count),
            "max_probability": np.full((count, 400, 640), 0.50, dtype=np.float32),
            "top1_top2_margin": np.full((count, 400, 640), 0.20, dtype=np.float32),
            "entropy": np.full((count, 400, 640), 1.142120, dtype=np.float32),
        }, {"valid_batch_size": count}


def final_roi_config(*, historical_horizontal=9.0, historical_vertical=9.0):
    return {
        "roi": {
            "width": 320,
            "height": 160,
            "expand_horizontal_each_side": historical_horizontal,
            "expand_vertical_each_side": historical_vertical,
        },
        "fullclass": {
            "roi": {
                "target_width": 640,
                "target_height": 400,
                "aspect_ratio": 1.6,
                "expand_horizontal_each_side": 0.30,
                "expand_vertical_each_side": 0.45,
                "padding_mode": "replicate",
            }
        },
    }


def test_source_base_row_keeps_only_final_source_provenance():
    row = _source_base_row("sub-031", source())
    assert row["subject"] == "sub-031"
    assert row["yolo_confidence"] == 0.9
    assert "pupil_confidence" not in row


def test_final_roi_contract_is_independent_of_historical_roi_block():
    config_a = final_roi_config(historical_horizontal=0.01, historical_vertical=0.02)
    config_b = final_roi_config(historical_horizontal=8.0, historical_vertical=9.0)
    assert _final_roi_config(config_a) == _final_roi_config(config_b)
    assert _final_roi_config(config_a)["target_width"] == 640
    assert _final_roi_config(config_a)["target_height"] == 400
    assert _final_roi_config(config_a)["expand_horizontal_each_side"] == 0.30
    assert _final_roi_config(config_a)["expand_vertical_each_side"] == 0.45


def test_final_roi_contract_fails_closed_when_missing_or_wrong():
    with pytest.raises(ValueError, match="fullclass.roi"):
        _final_roi_config({"roi": {"width": 320, "height": 160}, "fullclass": {}})

    wrong = final_roi_config()
    wrong["fullclass"]["roi"]["target_height"] = 401
    with pytest.raises(ValueError):
        _final_roi_config(wrong)

    wrong = final_roi_config()
    wrong["fullclass"]["roi"]["padding_mode"] = "reflect101"
    with pytest.raises(ValueError):
        _final_roi_config(wrong)


def test_group_remaining_rows_preserves_source_ordinal_groups():
    rows = (
        source(10, "frame_left"),
        source(10, "frame_right"),
        source(11, "frame_left"),
        source(12, "frame_left"),
    )
    groups = list(_group_remaining_rows(rows, 1))
    assert groups[0][0] == 10
    assert [ordinal for ordinal, _ in groups[0][1]] == [1]
    assert groups[1][0] == 11
    assert groups[2][0] == 12


def test_complete_batch_keeps_four_classes_and_pupil_only_geometry():
    labels = synthetic_labels()
    valid = np.ones((400, 640), dtype=bool)
    base = _source_base_row("sub-031", source(10, "frame_left"))
    items = [{
        "ordinal": 0,
        "base": base,
        "roi": np.zeros((100, 160), dtype=np.uint8),
        "valid_source_mask": valid,
    }]

    completed = _complete_batch(
        items=items,
        runtime=FakeRuntime(labels),
        boundary_band_px=5,
        low_max_probability_threshold=None,
    )
    row = completed[0][1]
    assert row["ritnet_status"] == "success"
    assert row["hard_background_pixels"] > 0
    assert row["hard_sclera_pixels"] > 0
    assert row["hard_iris_pixels"] > 0
    assert row["hard_pupil_pixels"] > 0
    assert row["pupil_fit_valid"] is True
    assert row["pupil_geom_mean_diameter"] > 0
    assert "iris_outer_fit_valid" not in row
    assert "pupil_to_iris_diameter_ratio" not in row
    assert "ocular_aperture_ratio_median" not in row
    assert row["soft_background_fraction"] == pytest.approx(0.50)
    assert row["soft_sclera_fraction"] == pytest.approx(0.30)
    assert row["soft_iris_fraction"] == pytest.approx(0.15)
    assert row["soft_pupil_fraction"] == pytest.approx(0.05)


def test_complete_batch_excludes_padding_from_scientific_hard_counts():
    labels = np.zeros((400, 640), dtype=np.uint8)
    labels[:, :80] = 3
    valid = np.ones((400, 640), dtype=bool)
    valid[:, :80] = False
    base = _source_base_row("sub-031", source(10, "frame_left"))
    items = [{
        "ordinal": 0,
        "base": base,
        "roi": np.zeros((100, 160), dtype=np.uint8),
        "valid_source_mask": valid,
    }]

    completed = _complete_batch(
        items=items,
        runtime=FakeRuntime(labels),
        boundary_band_px=5,
        low_max_probability_threshold=None,
    )
    row = completed[0][1]
    assert row["analysis_valid_pixel_count"] == 400 * 560
    assert row["hard_pupil_pixels"] == 0
    assert row["pupil_predicted_in_padding_pixels"] == 400 * 80
