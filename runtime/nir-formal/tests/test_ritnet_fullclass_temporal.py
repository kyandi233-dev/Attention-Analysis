from __future__ import annotations

import pytest

from ritnet_fullclass_temporal import (
    TEMPORAL_ANOMALY_THRESHOLD,
    TEMPORAL_JUMP_SCORE_CAP,
    TEMPORAL_QC_VERSION,
    add_temporal_facts,
)


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


def stable_rows(start=10, stop=20):
    return [row(frame, value=0.20 + 0.01 * (frame - start)) for frame in range(start, stop)]


def test_consecutive_same_eye_gets_delta():
    output = add_temporal_facts([row(10, value=0.20), row(11, value=0.25)])
    assert output[0]["temporal_reset_reason"] == "first_observation"
    assert output[1]["temporal_reset_reason"] is None
    assert output[1]["temporal_prev_frame_idx"] == 10
    assert output[1]["temporal_frame_gap"] == 1
    assert output[1]["delta_hard_pupil_fraction"] == pytest.approx(0.05)
    assert output[1]["delta_pupil_center_distance_px"] == pytest.approx(1.0)
    assert output[1]["temporal_jump_score"] is None
    assert output[1]["temporal_anomaly"] is None
    assert output[1]["temporal_qc_version"] == TEMPORAL_QC_VERSION


def test_missing_eye_frame_resets_instead_of_fake_one_frame_jump():
    output = add_temporal_facts([row(10), row(12, value=0.8)])
    assert output[1]["temporal_frame_gap"] == 2
    assert output[1]["temporal_reset_reason"] == "nonconsecutive_frame_gap"
    assert output[1]["delta_hard_pupil_fraction"] is None
    assert output[1]["temporal_jump_score"] is None
    assert output[1]["temporal_anomaly"] is None


def test_phase_boundary_resets_even_when_frame_is_consecutive():
    output = add_temporal_facts([row(10, phase="practice"), row(11, phase="block1")])
    assert output[1]["temporal_reset_reason"] == "phase_or_segment_boundary"
    assert output[1]["delta_hard_pupil_fraction"] is None
    assert output[1]["temporal_jump_score"] is None


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
    assert output[2]["temporal_jump_score"] is None


def test_rolling_mad_marks_stable_delta_as_non_anomalous_after_baseline_warmup():
    output = add_temporal_facts(stable_rows())
    last = output[-1]
    assert last["temporal_reset_reason"] is None
    assert last["temporal_jump_score"] == pytest.approx(0.0)
    assert last["temporal_anomaly"] is False


def test_rolling_mad_marks_large_jump_without_deleting_row():
    rows = stable_rows()
    rows.append(row(20, value=0.80))
    output = add_temporal_facts(rows)
    jump = output[-1]
    assert jump["frame_idx"] == 20
    assert jump["ritnet_status"] == "success"
    assert jump["temporal_reset_reason"] is None
    assert jump["temporal_jump_score"] == pytest.approx(TEMPORAL_JUMP_SCORE_CAP)
    assert jump["temporal_jump_score"] >= TEMPORAL_ANOMALY_THRESHOLD
    assert jump["temporal_anomaly"] is True


def test_gap_clears_robust_history_before_large_value():
    rows = stable_rows()
    rows.append(row(21, value=0.90))
    output = add_temporal_facts(rows)
    after_gap = output[-1]
    assert after_gap["temporal_reset_reason"] == "nonconsecutive_frame_gap"
    assert after_gap["temporal_jump_score"] is None
    assert after_gap["temporal_anomaly"] is None


def test_reordered_same_eye_is_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        add_temporal_facts([row(11), row(10)])
