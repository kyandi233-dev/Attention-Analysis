from __future__ import annotations

import numpy as np
import pandas as pd

from attention_pipeline.nir_pipeline_validation.extended import (
    advanced_behavior_summary,
    attach_visual_covariates,
    nogo_precursor_trajectory,
    track_robustness,
    trial_dynamic_feature_long,
    window_duration_sec,
)
from attention_pipeline.nir_pipeline_validation.extended_plots import (
    plot_dynamic_feature_matrix,
)


def _trial_level() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": ["sub-031"] * 6,
            "block_num": [1] * 6,
            "trial_num": [1, 2, 3, 4, 5, 6],
            "global_trial_index": [1, 2, 3, 4, 5, 6],
            "is_no_go": [0, 0, 1, 0, 0, 1],
            "correct": [1, 1, 1, 1, 1, 0],
            "commission": [0, 0, 0, 0, 0, 1],
            "omission": [0, 0, 0, 0, 0, 0],
            "rt": [400.0, 380.0, np.nan, 360.0, 340.0, 250.0],
            "time_in_block_sec": [1, 2, 3, 4, 5, 6],
            "stimulus_name": ["a.png", "b.png", "nogo.png", "a.png", "b.png", "nogo.png"],
            "stimulus_size": [100, 100, 100, 120, 120, 120],
            "prestimulus_press_flag": [False] * 6,
            "carryover_candidate_flag": [False] * 6,
            "ambiguous_omission_flag": [False] * 6,
            "anticipatory_candidate_flag": [False, False, False, True, False, False],
            "multiple_keypress_flag": [False] * 6,
        }
    )


def _trial_windows() -> pd.DataFrame:
    rows = []
    for track, offset in (("binocular_primary", 0.0), ("left_primary", 0.01)):
        for trial in range(1, 7):
            for window, scale in (("pre_1s", 1.0), ("pre_5s", 5.0)):
                rows.append(
                    {
                        "subject": "sub-031",
                        "block_num": 1,
                        "trial_num": trial,
                        "global_trial_index": trial,
                        "track": track,
                        "window_name": window,
                        "pupil_median": 0.01 * trial + offset,
                        "pupil_mean": 0.01 * trial + offset,
                        "pupil_mad": 0.001 * scale * trial,
                        "pupil_iqr": 0.002 * scale * trial,
                        "pupil_sd": 0.003 * scale * trial,
                        "pupil_p10": 0.01 * trial - 0.005,
                        "pupil_p90": 0.01 * trial + 0.005,
                        "pupil_slope_per_sec": 0.0001 * trial,
                        "pupil_diff_mad": 0.0002 * trial,
                        "pupil_diff_rate_mad_per_sec": 0.0003 * trial,
                    }
                )
    return pd.DataFrame(rows)


def test_window_duration_parses_state_windows_only():
    assert window_duration_sec("pre_1s") == 1.0
    assert window_duration_sec("pre_60s") == 60.0
    assert window_duration_sec("pre_200ms") is None


def test_advanced_behavior_summary_contains_high_order_metrics():
    result = advanced_behavior_summary(_trial_level()).iloc[0]
    assert result["n_go"] == 4
    assert result["n_nogo"] == 2
    assert np.isfinite(result["dprime"])
    assert "rt_cv" in result.index
    assert "exg_tau" in result.index


def test_trial_dynamic_feature_long_preserves_feature_families():
    result = trial_dynamic_feature_long(
        _trial_level(),
        _trial_windows(),
        tracks=["binocular_primary"],
        window_names=["pre_1s", "pre_5s"],
    )
    assert {"pupil_median", "pupil_mad", "pupil_slope_per_sec", "pupil_diff_rate_mad_per_sec"}.issubset(
        set(result["feature"])
    )
    assert {"pre_1s", "pre_5s"} == set(result["window_name"])


def test_nogo_precursor_trajectory_has_negative_lags_and_event_zero():
    result = nogo_precursor_trajectory(
        _trial_level(),
        _trial_windows(),
        track="binocular_primary",
        window_name="pre_5s",
        n_preceding_go=2,
    )
    assert {-2, -1, 0}.issubset(set(result["lag"]))
    assert {"correct_inhibition", "commission"} == set(result["event_outcome"])


def test_track_robustness_compares_primary_and_eye_track():
    correlations, agreement = track_robustness(
        _trial_windows(),
        window_name="pre_5s",
        main_track="binocular_primary",
        tracks=["binocular_primary", "left_primary"],
    )
    assert not correlations.empty
    assert set(agreement["comparison_track"]) == {"binocular_primary", "left_primary"}


def test_visual_covariates_attach_current_and_previous_stimulus():
    visual = pd.DataFrame(
        {
            "stimulus_name": ["a.png", "a.png", "b.png", "b.png", "nogo.png", "nogo.png"],
            "stimulus_size_pct": [100, 120, 100, 120, 100, 120],
            "central_rel_lum_mean": [0.2, 0.21, 0.3, 0.31, 0.4, 0.41],
            "central_rms_contrast": [0.1] * 6,
        }
    )
    result = attach_visual_covariates(
        _trial_level(),
        _trial_windows(),
        visual,
        track="binocular_primary",
        window_name="pre_5s",
    )
    assert "current_central_rel_lum_mean" in result.columns
    assert "previous_central_rel_lum_mean" in result.columns
    assert result["current_central_rel_lum_mean"].notna().all()


def test_dynamic_feature_plot_is_code_generated(tmp_path):
    dynamic = trial_dynamic_feature_long(
        _trial_level(),
        _trial_windows(),
        tracks=["binocular_primary"],
        window_names=["pre_1s", "pre_5s"],
    )
    outputs = plot_dynamic_feature_matrix(
        dynamic,
        track="binocular_primary",
        base=tmp_path / "dynamic",
        formats=["png"],
        dpi=80,
    )
    assert len(outputs) == 1
    assert (tmp_path / "dynamic.png").is_file()
