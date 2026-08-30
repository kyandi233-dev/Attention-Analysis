import numpy as np
import pandas as pd

from attention_pipeline.behavior_formal.science_v3_extensions import (
    build_error_trajectories,
    cluster_bootstrap_b1_b2,
    fit_block_cycle_gee,
    repeat_stability_boundary,
)


def test_b1_b2_bootstrap_clusters_repeated_sessions_by_participant():
    pairs = pd.DataFrame([
        {"repeat_participant_id": "p1", "session_id": "s1", "metric": "omission_rate", "b2_minus_b1": 0.1},
        {"repeat_participant_id": "p1", "session_id": "s2", "metric": "omission_rate", "b2_minus_b1": 0.3},
        {"repeat_participant_id": "p2", "session_id": "s3", "metric": "omission_rate", "b2_minus_b1": -0.2},
    ])
    result, failures = cluster_bootstrap_b1_b2(pairs, iterations=200, seed=7)
    assert failures.empty
    row = result.iloc[0]
    assert row["participant_group_n"] == 2
    assert row["session_pair_n"] == 3
    # Participant p1 contributes its mean (0.2), not two independent people.
    assert row["estimate_b2_minus_b1"] == np.mean([0.2, -0.2])
    assert row["cluster_unit"] == "repeat_participant_id"


def test_error_trajectory_centers_rt_within_participant_and_separates_error_types():
    rows = []
    for trial in range(1, 7):
        nogo = int(trial in {3, 6})
        rows.append({
            "repeat_participant_id": "p1",
            "session_id": "s1",
            "block_num": 1,
            "trial_num": trial,
            "is_no_go": nogo,
            "omission": int(trial == 4),
            "commission": int(trial == 3),
            "correct": int(trial not in {3, 4}),
            "rt": np.nan if nogo or trial == 4 else 300 + trial * 10,
        })
    events = build_error_trajectories(pd.DataFrame(rows), relative_trials=range(-1, 2))
    assert set(events["error_type"]) == {"go_omission", "nogo_commission"}
    assert events["error_event_id"].nunique() == 2
    valid = events["correct_go_rt_ms"].notna()
    assert np.allclose(
        events.loc[valid, "correct_go_rt_centered_ms"],
        events.loc[valid, "correct_go_rt_ms"] - events.loc[valid, "participant_correct_go_rt_median_ms"],
    )


def test_cycle_gee_failure_is_explicit_for_insufficient_data():
    cycle = pd.DataFrame([
        {"repeat_participant_id": "p1", "session_id": "s1", "block_id": "B1", "cycle_bin": 1, "omission_rate": 0.0},
        {"repeat_participant_id": "p1", "session_id": "s1", "block_id": "B1", "cycle_bin": 2, "omission_rate": 0.1},
    ])
    results, failures = fit_block_cycle_gee(cycle, metrics=["omission_rate"])
    assert results.empty
    assert not failures.empty
    assert set(failures["status"]) == {"not_estimable"}


def test_unbalanced_repeat_visits_are_descriptive_not_a_reliability_gate():
    session = pd.DataFrame([
        {"repeat_participant_id": "p1", "session_id": "s1"},
        {"repeat_participant_id": "p1", "session_id": "s2"},
        {"repeat_participant_id": "p1", "session_id": "s3"},
        {"repeat_participant_id": "p2", "session_id": "s4"},
    ])
    boundary = repeat_stability_boundary(session)
    assert boundary.iloc[0]["repeated_participant_group_n"] == 1
    assert boundary.iloc[0]["max_sessions_per_participant"] == 3
    assert boundary.iloc[0]["status"] == "requires_metric_specific_reliability_design"
