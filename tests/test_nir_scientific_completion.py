import numpy as np
import pandas as pd

from attention_pipeline.nir_pipeline_validation.scientific_completion import (
    aggregate_probe_visual_exposure,
    decompose_pupil_within_between,
    dynamic_feature_admission_registry,
    fit_visual_adjustment_models,
)


def test_within_between_decomposition_separates_participant_mean_and_deviation():
    frame = pd.DataFrame(
        {
            "analysis_group_token": ["p1", "p1", "p2", "p2"],
            "session_id": ["s1", "s1", "s2", "s2"],
            "pupil_median": [1.0, 3.0, 10.0, 14.0],
        }
    )
    out = decompose_pupil_within_between(frame)
    assert out.loc[0, "pupil_median__participant_mean"] == 2.0
    assert out.loc[1, "pupil_median__within_participant"] == 1.0
    assert out.loc[2, "pupil_median__participant_mean"] == 12.0
    assert np.allclose(
        out.groupby("analysis_group_token")["pupil_median__within_participant"].mean(),
        0.0,
    )


def test_probe_visual_exposure_excludes_anchor_and_future_trials():
    probe = pd.DataFrame(
        {
            "session_id": ["s1"],
            "analysis_group_token": ["p1"],
            "block_num": [1],
            "probe_index_global": [1],
            "probe_index_in_block": [1],
            "probe_trial_num": [4],
            "probe_onset_ms": [4000.0],
            "window_name": ["pre_3s"],
            "window_start_ms": [1000.0],
            "window_end_ms": [4000.0],
        }
    )
    trials = pd.DataFrame(
        {
            "session_id": ["s1"] * 5,
            "analysis_group_token": ["p1"] * 5,
            "block_num": [1] * 5,
            "trial_num": [1, 2, 3, 4, 5],
            "absolute_onset_time": [0.0, 1000.0, 2000.0, 3000.0, 4000.0],
            "stimulus_name": ["a", "b", "c", "anchor", "future"],
            "stimulus_size": [1] * 5,
        }
    )
    visual = pd.DataFrame(
        {
            "stimulus_name": ["a", "b", "c", "anchor", "future"],
            "stimulus_size": [1] * 5,
            "brightness": [10.0, 20.0, 30.0, 999.0, 9999.0],
            "contrast": [1.0, 2.0, 3.0, 99.0, 999.0],
        }
    )
    out = aggregate_probe_visual_exposure(probe, trials, visual)
    row = out.iloc[0]
    # Only trials 2 and 3 are inside [1000, 4000) and before anchor trial 4.
    assert row["visual_exposure_trial_n"] == 2
    assert row["probe_exposure__brightness__mean"] == 25.0
    assert bool(row["strict_pre_probe_verified"])
    assert bool(row["anchoring_probe_trial_excluded"])


def test_visual_adjustment_models_are_participant_clustered_and_keep_failures_separate():
    rng = np.random.default_rng(7)
    windows = []
    trials = []
    for participant in range(8):
        for trial in range(10):
            visual = (trial - 5) / 2.0
            windows.append(
                {
                    "session_id": f"s{participant}",
                    "analysis_group_token": f"p{participant}",
                    "block_num": 1,
                    "trial_num": trial + 1,
                    "global_trial_index": trial + 1,
                    "track": "binocular_primary",
                    "window_name": "pre_5s",
                    "pupil_median": 0.5 * visual + rng.normal(0, 0.2),
                    "previous_visual__brightness__mean": visual,
                    "pupil_valid_fraction": 0.9,
                    "internal_coverage_fraction": 0.95,
                }
            )
            trials.append(
                {
                    "session_id": f"s{participant}",
                    "analysis_group_token": f"p{participant}",
                    "block_num": 1,
                    "trial_num": trial + 1,
                    "global_trial_index": trial + 1,
                    "is_no_go": int(trial % 5 == 0),
                }
            )
    result, failures = fit_visual_adjustment_models(
        pd.DataFrame(windows), pd.DataFrame(trials)
    )
    assert failures.empty
    assert set(result["model_stage"]) == {"unadjusted", "visual_time_quality_adjusted"}
    assert set(result["participant_group_n"]) == {8}
    assert not result["current_trial_visual_used"].any()


def test_dynamic_registry_fails_closed_for_recovery_and_frequency():
    registry = dynamic_feature_admission_registry().set_index("feature")
    assert registry.loc["recovery_magnitude_or_time", "status"] == "not_admitted"
    assert registry.loc["frequency_domain_pupil", "status"] == "not_admitted"
