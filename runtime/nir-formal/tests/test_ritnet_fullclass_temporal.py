from __future__ import annotations

import pytest

from ritnet_fullclass_temporal import add_temporal_facts


def row(frame, eye="frame_left", phase="block1", segment=1, status="success", value=0.2):
    return {
        "frame_idx": frame,
        "eye": eye,
        "phase": phase,
        "phase_segment": segment,
        "unix_ms": 1000 + frame * 33,
        "video_time_ms": frame * 33,
        "ritnet_status": status,
        "hard_pupil_fraction": value,
        "hard_iris_outer_fraction": 0.4 + value,
        "hard_ocular_fraction": 0.7,
        "pupil_to_iris_diameter_ratio": 0.3 + value,
        "ocular_aperture_ratio_median": 0.2,
        "pupil_center_x": 100 + frame,
        "pupil_center_y": 50,
        "ocular_max_probability_mean": 0.9,
        "ocular_top1_top2_margin_mean": 0.7,
        "ocular_entropy_mean": 0.2,
    }


def test_consecutive_same_eye_gets_delta():
    output = add_temporal_facts([row(10, value=0.20), row(11, value=0.25)])
    assert output[0]["temporal_reset_reason"] == "first_observation"
    assert output[1]["temporal_reset_reason"] is None
    assert output[1]["temporal_prev_frame_idx"] == 10
    assert output[1]["temporal_frame_gap"] == 1
    assert output[1]["delta_hard_pupil_fraction"] == pytest.approx(0.05)
    assert output[1]["delta_pupil_center_distance_px"] == pytest.approx(1.0)
    assert output[1]["temporal_anomaly"] is None


def test_missing_eye_frame_resets_instead_of_fake_one_frame_jump():
    output = add_temporal_facts([row(10), row(12, value=0.8)])
    assert output[1]["temporal_frame_gap"] == 2
    assert output[1]["temporal_reset_reason"] == "nonconsecutive_frame_gap"
    assert output[1]["delta_hard_pupil_fraction"] is None


def test_phase_boundary_resets_even_when_frame_is_consecutive():
    output = add_temporal_facts([row(10, phase="practice"), row(11, phase="block1")])
    assert output[1]["temporal_reset_reason"] == "phase_or_segment_boundary"
    assert output[1]["delta_hard_pupil_fraction"] is None


def test_left_and_right_histories_are_independent():
    output = add_temporal_facts(
        [
            row(10, eye="frame_left", value=0.1),
            row(10, eye="frame_right", value=0.4),
            row(11, eye="frame_left", value=0.2),
            row(11, eye="frame_right", value=0.5),
        ]
    )
    assert output[2]["delta_hard_pupil_fraction"] == pytest.approx(0.1)
    assert output[3]["delta_hard_pupil_fraction"] == pytest.approx(0.1)


def test_failed_ritnet_row_breaks_temporal_chain():
    output = add_temporal_facts(
        [row(10), row(11, status="failed"), row(12, value=0.4)]
    )
    assert output[1]["temporal_reset_reason"] == "ritnet_not_success"
    assert output[2]["temporal_reset_reason"] == "ritnet_not_success"
    assert output[2]["delta_hard_pupil_fraction"] is None


def test_reordered_same_eye_is_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        add_temporal_facts([row(11), row(10)])
