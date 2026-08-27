from __future__ import annotations

import numpy as np
import pytest

from ritnet_fullclass_qc import (
    QC_SELECTION_VERSION,
    build_qc_selections,
    render_qc_images,
)


def coverage(frame, *, status="both_eyes_success", fixed=False, phase="block1", segment=1):
    return {
        "phase": phase,
        "phase_segment": segment,
        "frame_idx": frame,
        "coverage_status": status,
        "fixed_qc_anchor": fixed,
    }


def eye(frame, eye_name="frame_left", *, phase="block1", segment=1, **updates):
    row = {
        "phase": phase,
        "phase_segment": segment,
        "frame_idx": frame,
        "eye": eye_name,
        "ritnet_status": "success",
        "qc_pupil_fragmented": False,
        "qc_iris_outer_fragmented": False,
        "qc_ocular_fragmented": False,
        "pupil_touches_valid_domain_edge": False,
        "iris_outer_touches_valid_domain_edge": False,
        "ocular_touches_valid_domain_edge": False,
        "temporal_anomaly": False,
        "pupil_predicted_in_padding_pixels": 0,
        "iris_outer_predicted_in_padding_pixels": 0,
        "ocular_predicted_in_padding_pixels": 0,
        "low_max_probability_threshold": None,
        "ocular_low_max_probability_fraction": None,
    }
    row.update(updates)
    return row


def test_fixed_anchor_is_selected_even_with_zero_eye_rows():
    selections = build_qc_selections(
        frame_coverage_rows=[coverage(10, status="yolo_no_eye", fixed=True)],
        eye_metric_rows=[],
        anomaly_limit_per_reason_per_phase=2,
        max_image_count=10,
    )
    assert len(selections) == 1
    assert selections[0].key == ("block1", 1, 10)
    assert selections[0].reasons == ("fixed_anchor", "yolo_no_eye")
    assert selections[0].eyes == ()


def test_same_frame_merges_both_eye_anomalies_into_one_qc_image():
    selections = build_qc_selections(
        frame_coverage_rows=[coverage(20)],
        eye_metric_rows=[
            eye(20, "frame_left", qc_pupil_fragmented=True),
            eye(20, "frame_right", temporal_anomaly=True),
        ],
        anomaly_limit_per_reason_per_phase=5,
        max_image_count=10,
    )
    assert len(selections) == 1
    assert set(selections[0].reasons) == {"pupil_fragmented", "temporal_jump"}
    assert selections[0].eyes == ("frame_left", "frame_right")


def test_failure_and_padding_reasons_use_current_final_fields():
    selections = build_qc_selections(
        frame_coverage_rows=[
            coverage(30, status="final_video_decode_failed"),
            coverage(31, status="roi_invalid"),
            coverage(32),
        ],
        eye_metric_rows=[
            eye(
                32,
                ocular_predicted_in_padding_pixels=12,
                low_max_probability_threshold=0.60,
                ocular_low_max_probability_fraction=0.25,
            )
        ],
        anomaly_limit_per_reason_per_phase=5,
        max_image_count=10,
    )
    by_frame = {item.frame_idx: set(item.reasons) for item in selections}
    assert "final_video_decode_failed" in by_frame[30]
    assert "roi_invalid" in by_frame[31]
    assert by_frame[32] == {"prediction_in_artificial_padding", "low_model_confidence"}


def test_anomaly_examples_are_spread_across_phase_not_only_first_rows():
    frames = [coverage(index) for index in range(10)]
    eyes = [eye(index, temporal_anomaly=True) for index in range(10)]
    selections = build_qc_selections(
        frame_coverage_rows=frames,
        eye_metric_rows=eyes,
        anomaly_limit_per_reason_per_phase=3,
        max_image_count=20,
    )
    assert [item.frame_idx for item in selections] == [0, 4, 9]
    assert all(item.reasons == ("temporal_jump",) for item in selections)


def test_global_image_limit_merges_reasons_and_caps_anomalies():
    frames = [coverage(index, fixed=index in {0, 9}) for index in range(10)]
    eyes = [
        eye(index, temporal_anomaly=True, qc_ocular_fragmented=True)
        for index in range(10)
    ]
    selections = build_qc_selections(
        frame_coverage_rows=frames,
        eye_metric_rows=eyes,
        anomaly_limit_per_reason_per_phase=10,
        max_image_count=4,
    )
    assert len(selections) == 4
    assert {0, 9}.issubset({item.frame_idx for item in selections})


def test_fixed_anchor_overflow_fails_closed_instead_of_silently_dropping_qc():
    with pytest.raises(RuntimeError, match="fixed QC anchors alone exceed"):
        build_qc_selections(
            frame_coverage_rows=[coverage(i, fixed=True) for i in range(3)],
            eye_metric_rows=[],
            anomaly_limit_per_reason_per_phase=1,
            max_image_count=2,
        )


def test_eye_metric_must_reference_full_frame_timeline():
    with pytest.raises(ValueError, match="absent from coverage"):
        build_qc_selections(
            frame_coverage_rows=[coverage(1)],
            eye_metric_rows=[eye(2, temporal_anomaly=True)],
            anomaly_limit_per_reason_per_phase=2,
            max_image_count=5,
        )


def test_render_qc_images_keeps_four_class_overlay_contract():
    labels = np.zeros((400, 640), dtype=np.uint8)
    labels[100:300, 100:500] = 1
    labels[150:250, 220:420] = 2
    labels[180:220, 290:350] = 3
    roi = np.full((200, 320), 128, dtype=np.uint8)
    color, overlay = render_qc_images(roi, labels)
    assert color.shape == (400, 640, 3)
    assert overlay.shape == (400, 640, 3)
    assert color.dtype == np.uint8
    assert overlay.dtype == np.uint8
    assert QC_SELECTION_VERSION.startswith("full-frame-timeline")
