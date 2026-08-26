from __future__ import annotations

import pandas as pd

from attention_pipeline.rgb.motion_review import select_motion_review_rows


def test_motion_review_selects_extreme_and_typical_rows():
    table = pd.DataFrame(
        {
            "video_frame_position": list(range(10, 20)),
            "capture_frame_idx": list(range(10, 20)),
            "unix_ms": [1000 + i * 33 for i in range(10)],
            "dt_ms": [33] * 10,
            "phase": ["block1"] * 10,
            "block": [1] * 10,
            "trial_num": list(range(1, 11)),
            "behavior_state": ["trial"] * 10,
            "motion_valid": [True] * 10,
            "global_motion_energy": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.020],
            "global_motion_energy_per_sec": [0.03] * 10,
            "changed_pixel_ratio": [0.001] * 10,
            "gray_mean_delta": [0.0, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -3.0, 0.2],
            "gray_mean": [150.0] * 10,
            "irregular_dt": [False] * 10,
        }
    )

    rows = select_motion_review_rows(table)
    categories = {row["review_category"] for row in rows}
    positions = [int(row["video_frame_position"]) for row in rows]

    assert "highest_motion" in categories
    assert "largest_brightness_change" in categories
    assert {"motion_p50", "motion_p90", "motion_p99"} & categories
    assert 19 in positions  # highest motion
    assert 18 in positions  # largest absolute brightness change
    assert len(positions) == len(set(positions))
