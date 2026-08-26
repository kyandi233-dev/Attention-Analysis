from __future__ import annotations

import numpy as np
import pandas as pd

from attention_pipeline.rgb.motion_qc import summarize_motion_table


def test_motion_qc_summarizes_dt_motion_and_gaps():
    table = pd.DataFrame(
        {
            "video_frame_position": [0, 1, 2, 3, 4],
            "capture_frame_idx": [0, 1, 2, 3, 4],
            "unix_ms": [1000, 1032, 1064, 1128, 1328],
            "dt_ms": [np.nan, 32.0, 32.0, 64.0, 200.0],
            "phase": ["baseline", "baseline", "block1", "block1", "block1"],
            "block": [np.nan, np.nan, 1.0, 1.0, 1.0],
            "trial_num": [np.nan, np.nan, 1.0, 1.0, 1.0],
            "behavior_state": [None, None, "trial", "trial", "trial"],
            "motion_valid": [False, True, True, True, False],
            "global_motion_energy": [np.nan, 0.01, 0.02, 0.05, np.nan],
            "global_motion_energy_per_sec": [np.nan, 0.3125, 0.625, 0.78125, np.nan],
            "changed_pixel_ratio": [np.nan, 0.02, 0.03, 0.08, np.nan],
            "gray_mean_delta": [np.nan, 0.5, -1.0, 3.0, np.nan],
            "gray_mean": [100.0, 100.5, 99.5, 102.5, 103.0],
            "irregular_dt": [False, False, False, True, True],
            "gap_before": [False, False, False, False, True],
            "gap_reason": ["analysis_start", "", "", "", "timestamp_gap"],
        }
    )

    summary = summarize_motion_table(table, subject="sub-999")

    assert summary["subject"] == "sub-999"
    assert summary["rows"] == 5
    assert summary["motion_valid_rows"] == 3
    assert summary["gap_rows"] == 1
    assert summary["irregular_dt_rows"] == 2
    assert summary["dt_bands"] == {
        "le_40_ms": 2,
        "41_to_48_ms": 0,
        "49_to_66_ms": 1,
        "67_to_100_ms": 0,
        "gt_100_ms": 1,
    }
    assert summary["global_motion_energy"]["max"] == 0.05
    assert summary["samples"]["largest_timestamp_gaps"][0]["gap_reason"] == "timestamp_gap"
    assert summary["samples"]["highest_motion_energy"][0]["video_frame_position"] == 3
