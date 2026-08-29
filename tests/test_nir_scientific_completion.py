import numpy as np
import pandas as pd

from attention_pipeline.nir_pipeline_validation.scientific_completion import (
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


def test_visual_adjustment_models_are_participant_clustered_and_use_only_previous_visual():
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
                    "previous_visual__central_rel_lum_mean__mean": visual,
                    "current_visual__central_rel_lum_mean__mean": 9999.0,
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
    assert not result["controls"].str.contains("current_visual", regex=False).any()


def test_dynamic_registry_fails_closed_for_recovery_and_frequency():
    registry = dynamic_feature_admission_registry().set_index("feature")
    assert registry.loc["recovery_magnitude_or_time", "status"] == "not_admitted"
    assert registry.loc["frequency_domain_pupil", "status"] == "not_admitted"
