import numpy as np
import pandas as pd
import pytest

from attention_pipeline.behavior_formal.science_v3 import (
    BehaviorContractError,
    BehaviorScienceConfig,
    aggregate_behavior_metrics,
    build_multiscale_tables,
    build_probe_windows,
    fit_q1_nominal,
    fit_q2_ordinal,
    participant_disjoint_folds,
    validate_topology,
)


def _trials():
    rows = []
    for session, participant in [("sub-001", "p1"), ("sub-002", "p2")]:
        for block in [1, 2]:
            for trial in range(1, 13):
                nogo = int(trial % 5 == 0)
                probe = int(trial in {7, 12})
                rows.append({
                    "subject": session,
                    "repeat_participant_id": participant,
                    "block_num": block,
                    "trial_num": trial,
                    "absolute_onset_time": block * 100000 + trial * 1000,
                    "probe_onset_time": block * 100000 + trial * 1000 + 800 if probe else np.nan,
                    "is_no_go": nogo,
                    "correct": 0 if (nogo and trial == 10) else 1,
                    "omission": 1 if (not nogo and trial == 11) else 0,
                    "commission": 1 if (nogo and trial == 10) else 0,
                    "response": 0 if nogo else 1,
                    "rt": np.nan if nogo else 300 + trial * 5,
                    "time_in_block_sec": trial,
                    "cycle_bin": 1 if trial <= 6 else 2,
                    "is_probe": probe,
                    "probe_response": (trial % 4) + 1 if probe else np.nan,
                    "probe_vigilance": (trial % 4) + 1 if probe else np.nan,
                })
    return pd.DataFrame(rows)


def test_metrics_keep_go_and_nogo_errors_separate_and_complete():
    d = _trials()
    m = aggregate_behavior_metrics(d)
    assert m["omission_denominator"] == int((d.is_no_go == 0).sum())
    assert m["commission_denominator"] == int((d.is_no_go == 1).sum())
    expected = {
        "go_correct_rt_mean_ms", "go_correct_rt_median_ms", "go_correct_rt_sd_ms",
        "go_correct_rt_mad_ms", "go_correct_rt_iqr_ms", "go_correct_rt_cv",
        "go_correct_rt_theilsen_slope_ms_per_s", "dprime_loglinear", "criterion_c", "beta",
    }
    assert expected.issubset(m)


def test_multiscale_tables_preserve_participant_session_block_cycle_hierarchy():
    tables = build_multiscale_tables(_trials())
    assert set(tables) == {"session", "block", "cycle"}
    assert tables["session"]["session_id"].nunique() == 2
    assert tables["session"]["repeat_participant_id"].nunique() == 2
    assert set(tables["block"]["observation_unit"]) == {"block"}


def test_probe_windows_exclude_anchor_and_do_not_inflate_primary_n():
    primary, sensitivity = build_probe_windows(_trials())
    assert primary["probe_event_id"].is_unique
    assert set(sensitivity["window_seconds_nominal"]) == {10, 20, 30}
    assert primary["anchor_trial_excluded"].all()
    first = primary.sort_values(["session_id", "block_id", "probe_order_in_block"]).iloc[0]
    assert first["trial_opportunities"] <= 6
    assert primary["formal_independent_sample"].all()
    assert not sensitivity["formal_independent_sample"].any()


def test_participant_disjoint_folds_never_split_one_group():
    primary, _ = build_probe_windows(_trials())
    folded = participant_disjoint_folds(primary, n_splits=2)
    assert folded.groupby("repeat_participant_id")["fold_id"].nunique().eq(1).all()


def test_topology_allows_unbalanced_one_to_many_visits_without_a_group_size_gate():
    session = pd.DataFrame({
        "repeat_participant_id": ["p1", "p1", "p1", "p2", "p2", "p3"],
        "session_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
    })
    observed = validate_topology(session, {"sessions": 6, "analysis_groups": 3})
    assert observed["repeated_participant_groups"] == 2
    assert observed["max_sessions_per_participant"] == 3
    assert observed["group_size_distribution"] == {"1": 1, "2": 1, "3": 1}
    with pytest.raises(BehaviorContractError):
        validate_topology(session, {"sessions": 6, "analysis_groups": 4})


def test_q1_q2_failure_rows_are_not_estimable_not_empty_success():
    primary, _ = build_probe_windows(_trials(), BehaviorScienceConfig(min_model_rows=999))
    q1, f1 = fit_q1_nominal(primary, BehaviorScienceConfig(min_model_rows=999))
    q2, f2 = fit_q2_ordinal(primary, BehaviorScienceConfig(min_model_rows=999))
    assert q1.empty and q2.empty
    assert not f1.empty and not f2.empty
    assert set(f1["status"]) == {"not_estimable"}
    assert set(f2["status"]) == {"not_estimable"}
