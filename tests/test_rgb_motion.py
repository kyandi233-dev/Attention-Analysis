from __future__ import annotations

import csv

import numpy as np

from attention_pipeline.rgb.behavior import BehaviorIndex
from attention_pipeline.rgb.motion import measure_motion_pair


def test_motion_pair_preserves_raw_difference_and_rate():
    previous = np.zeros((2, 2), dtype=np.uint8)
    current = np.full((2, 2), 10, dtype=np.uint8)
    result = measure_motion_pair(
        current,
        previous,
        dt_ms=32,
        median_interval_ms=32.0,
        previous_capture_idx=10,
        current_capture_idx=11,
        previous_gray_mean=0.0,
        pixel_diff_threshold=5,
    )
    assert result["motion_valid"] is True
    assert result["gap_before"] is False
    assert result["irregular_dt"] is False
    assert result["mean_abs_difference"] == 10.0
    assert result["sum_abs_difference"] == 40
    assert result["max_abs_difference"] == 10
    assert result["changed_pixel_ratio"] == 1.0
    assert np.isclose(result["global_motion_energy"], 10.0 / 255.0)
    assert np.isclose(result["global_motion_energy_per_sec"], (10.0 / 255.0) / 0.032)


def test_motion_pair_marks_timestamp_or_capture_gap_missing():
    previous = np.zeros((2, 2), dtype=np.uint8)
    current = np.full((2, 2), 20, dtype=np.uint8)

    time_gap = measure_motion_pair(
        current,
        previous,
        dt_ms=200,
        median_interval_ms=32.0,
        previous_capture_idx=10,
        current_capture_idx=11,
        previous_gray_mean=0.0,
        gap_reset_ms=100,
    )
    assert time_gap["gap_before"] is True
    assert time_gap["gap_reason"] == "timestamp_gap"
    assert time_gap["motion_valid"] is False
    assert time_gap["mean_abs_difference"] is None
    assert time_gap["gray_mean"] == 20.0

    capture_gap = measure_motion_pair(
        current,
        previous,
        dt_ms=32,
        median_interval_ms=32.0,
        previous_capture_idx=10,
        current_capture_idx=12,
        previous_gray_mean=0.0,
        gap_reset_ms=100,
    )
    assert capture_gap["gap_before"] is True
    assert capture_gap["gap_reason"] == "capture_index_gap"
    assert capture_gap["capture_missing_frame_indices_before"] == 1
    assert capture_gap["motion_valid"] is False


def test_behavior_index_distinguishes_trial_probe_and_recovery(tmp_path):
    path = tmp_path / "sub-001_Block1_B_beh.csv"
    fields = [
        "block_num",
        "condition",
        "trial_num",
        "cycle_num",
        "position_in_cycle",
        "stimulus_name",
        "is_no_go",
        "correct",
        "is_probe",
        "probe_response",
        "probe_vigilance",
        "probe_onset_time",
        "probe_response_time",
        "absolute_onset_time",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "block_num": 1,
                "condition": "B",
                "trial_num": 1,
                "cycle_num": 1,
                "position_in_cycle": 1,
                "stimulus_name": "01_mango",
                "is_no_go": "False",
                "correct": "True",
                "is_probe": "True",
                "probe_response": 2,
                "probe_vigilance": 3,
                "probe_onset_time": 2200,
                "probe_response_time": 2500,
                "absolute_onset_time": 1000,
            }
        )
        writer.writerow(
            {
                "block_num": 1,
                "condition": "B",
                "trial_num": 2,
                "cycle_num": 1,
                "position_in_cycle": 2,
                "stimulus_name": "02_grape",
                "is_no_go": "False",
                "correct": "True",
                "is_probe": "False",
                "absolute_onset_time": 3000,
            }
        )

    index = BehaviorIndex.from_csv(path)
    assert index is not None

    trial = index.context_at(1100, trial_duration_ms=1150)
    assert trial["trial_num"] == 1
    assert trial["correct"] == 1
    assert trial["trial_active"] is True
    assert trial["behavior_state"] == "trial"

    probe = index.context_at(2300, trial_duration_ms=1150)
    assert probe["trial_num"] == 1
    assert probe["probe_active"] is True
    assert probe["behavior_state"] == "probe"

    recovery = index.context_at(2700, trial_duration_ms=1150)
    assert recovery["trial_num"] == 1
    assert recovery["trial_active"] is False
    assert recovery["probe_active"] is False
    assert recovery["behavior_state"] == "post_probe_recovery"
