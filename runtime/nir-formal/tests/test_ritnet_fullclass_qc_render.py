from __future__ import annotations

import numpy as np

from ritnet_fullclass_qc import QCSelection
from ritnet_fullclass_qc_render import (
    HEADER_HEIGHT,
    PANEL_HEIGHT,
    PANEL_WIDTH,
    QC_COMPOSITE_VERSION,
    render_qc_composite,
)


def metric(eye, *, success=True):
    return {
        "eye": eye,
        "ritnet_status": "success" if success else "failed",
        "ritnet_failure_reason": None if success else "roi_invalid:test",
        "yolo_bbox_x1": 20,
        "yolo_bbox_y1": 20,
        "yolo_bbox_x2": 80,
        "yolo_bbox_y2": 60,
        "roi_source_x1": 10,
        "roi_source_y1": 10,
        "roi_source_x2": 100,
        "roi_source_y2": 80,
        "pupil_fit_valid": True,
        "pupil_center_x": 320,
        "pupil_center_y": 200,
        "pupil_short_axis": 40,
        "pupil_long_axis": 60,
        "pupil_angle_deg": 15,
        "iris_outer_fit_valid": True,
        "iris_outer_center_x": 320,
        "iris_outer_center_y": 200,
        "iris_outer_short_axis": 120,
        "iris_outer_long_axis": 180,
        "iris_outer_angle_deg": 10,
        "pupil_to_iris_diameter_ratio": 0.33,
        "ocular_aperture_ratio_median": 0.44,
        "ocular_max_probability_mean": 0.90,
        "ocular_entropy_mean": 0.20,
    }


def selection():
    return QCSelection(
        phase="block1",
        phase_segment=1,
        frame_idx=123,
        reasons=("fixed_anchor", "temporal_jump"),
        eyes=("frame_left",),
    )


def test_composite_contains_original_and_two_eye_panels():
    frame = np.full((240, 320, 3), 80, dtype=np.uint8)
    overlay = np.full((400, 640, 3), 120, dtype=np.uint8)
    output = render_qc_composite(
        frame_bgr=frame,
        selection=selection(),
        coverage_row={"coverage_status": "both_eyes_success"},
        eye_metric_rows={
            "frame_left": metric("frame_left"),
            "frame_right": metric("frame_right"),
        },
        eye_overlays={"frame_left": overlay, "frame_right": overlay},
    )
    assert output.shape == (PANEL_HEIGHT + HEADER_HEIGHT, PANEL_WIDTH * 3, 3)
    assert output.dtype == np.uint8
    assert output.any()
    assert QC_COMPOSITE_VERSION.endswith("v1")


def test_composite_supports_yolo_miss_with_no_eye_rows():
    frame = np.full((240, 320, 3), 80, dtype=np.uint8)
    miss = QCSelection(
        phase="block1",
        phase_segment=1,
        frame_idx=50,
        reasons=("fixed_anchor", "yolo_no_eye"),
        eyes=(),
    )
    output = render_qc_composite(
        frame_bgr=frame,
        selection=miss,
        coverage_row={"coverage_status": "yolo_no_eye"},
        eye_metric_rows={},
        eye_overlays={},
    )
    assert output.shape == (PANEL_HEIGHT + HEADER_HEIGHT, PANEL_WIDTH * 3, 3)
    assert output[:, PANEL_WIDTH:, :].mean() > 0


def test_composite_marks_unavailable_source_frame_without_crashing():
    unavailable = QCSelection(
        phase="block1",
        phase_segment=1,
        frame_idx=99,
        reasons=("final_video_decode_failed",),
        eyes=("frame_left", "frame_right"),
    )
    output = render_qc_composite(
        frame_bgr=None,
        selection=unavailable,
        coverage_row={"coverage_status": "final_video_decode_failed"},
        eye_metric_rows={
            "frame_left": metric("frame_left", success=False),
            "frame_right": metric("frame_right", success=False),
        },
        eye_overlays={},
        fallback_frame_size=(320, 240),
    )
    assert output.shape == (PANEL_HEIGHT + HEADER_HEIGHT, PANEL_WIDTH * 3, 3)
    assert output.any()
