import numpy as np
import pandas as pd
import pytest

from attention_pipeline.nir_pipeline_validation.pupil_validation import attach_visual_with_temporal_gate


def test_pre_stimulus_window_cannot_use_current_or_future_brightness():
    trials = pd.DataFrame([
        {"session_id": "s1", "analysis_group_token": "g1", "block_num": 1, "trial_num": 1,
         "global_trial_index": 1, "absolute_onset_time": 1000, "stimulus_name": "a", "stimulus_size": 1},
        {"session_id": "s1", "analysis_group_token": "g1", "block_num": 1, "trial_num": 2,
         "global_trial_index": 2, "absolute_onset_time": 2000, "stimulus_name": "b", "stimulus_size": 1},
    ])
    windows = pd.DataFrame([
        {"session_id": "s1", "analysis_group_token": "g1", "block_num": 1, "trial_num": 2,
         "global_trial_index": 2, "window_name": "pre_200ms", "window_start_offset_ms": -200,
         "window_end_offset_ms": 0},
        {"session_id": "s1", "analysis_group_token": "g1", "block_num": 1, "trial_num": 2,
         "global_trial_index": 2, "window_name": "post_200_1150ms", "window_start_offset_ms": 200,
         "window_end_offset_ms": 1150},
    ])
    visual = pd.DataFrame([
        {"stimulus_name": "a", "stimulus_size": 1, "relative_luminance": 0.1},
        {"stimulus_name": "b", "stimulus_size": 1, "relative_luminance": 0.9},
    ])
    linked, audit = attach_visual_with_temporal_gate(windows, trials, visual)
    current = "current_visual__relative_luminance__mean"
    previous = "previous_visual__relative_luminance__mean"
    pre = linked[linked.window_name.eq("pre_200ms")].iloc[0]
    post = linked[linked.window_name.eq("post_200_1150ms")].iloc[0]

    assert np.isnan(pre[current])
    assert pre[previous] == pytest.approx(0.1)
    assert pre["current_visual_allowed"] == False
    assert pre["visual_temporal_tolerance_ms"] == 0.0
    assert post[current] == pytest.approx(0.9)
    assert post["current_visual_allowed"] == True
    assert set(audit["future_or_current_brightness_in_pre_window"]) == {False}
