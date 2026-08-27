from __future__ import annotations

import cv2
import numpy as np

from ritnet_fullclass_final_engine import _complete_batch, _group_remaining_rows, _source_base_row


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


def test_source_base_row_selects_only_final_source_provenance():
    row = _source_base_row("sub-031", source())
    assert row["subject"] == "sub-031"
    assert row["yolo_confidence"] == 0.9
    assert "pupil_confidence" not in row


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


def test_complete_batch_maps_success_outputs_and_keeps_failed_row_in_order():
    labels = synthetic_labels()

    class FakeRuntime:
        def infer_batch(self, rois):
            count = len(rois)
            return {
                "labels": np.stack([labels] * count),
                "soft_class_fraction": np.tile(
                    np.asarray([[0.5, 0.3, 0.15, 0.05]], dtype=np.float32), (count, 1)
                ),
                "max_probability": np.full((count, 400, 640), 0.9, dtype=np.float32),
                "top1_top2_margin": np.full((count, 400, 640), 0.7, dtype=np.float32),
                "entropy": np.full((count, 400, 640), 0.3, dtype=np.float32),
            }, {"valid_batch_size": count}

    good_base = _source_base_row("sub-031", source(10, "frame_left"))
    failed_base = _source_base_row("sub-031", source(10, "frame_right"))
    failed_base["ritnet_status"] = "failed"
    failed_base["ritnet_failure_reason"] = "roi_invalid:test"
    good_base_2 = _source_base_row("sub-031", source(11, "frame_left"))

    items = [
        {"ordinal": 0, "base": good_base, "roi": np.zeros((100, 160), dtype=np.uint8)},
        {"ordinal": 1, "base": failed_base, "roi": None},
        {"ordinal": 2, "base": good_base_2, "roi": np.zeros((100, 160), dtype=np.uint8)},
    ]
    completed = _complete_batch(
        items=items,
        runtime=FakeRuntime(),
        boundary_band_px=5,
        low_max_probability_threshold=None,
    )
    assert [ordinal for ordinal, _ in completed] == [0, 1, 2]
    assert completed[0][1]["ritnet_status"] == "success"
    assert completed[1][1]["ritnet_status"] == "failed"
    assert completed[2][1]["ritnet_status"] == "success"
    assert completed[0][1]["hard_pupil_pixels"] > 0
    assert completed[0][1]["ocular_max_probability_mean"] == 0.9
    assert completed[0][1]["soft_pupil_fraction"] == np.float32(0.05)
