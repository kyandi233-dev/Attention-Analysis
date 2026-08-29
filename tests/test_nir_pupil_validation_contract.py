import json

import numpy as np
import pandas as pd
import pytest

from attention_pipeline.nir_pipeline_validation.pupil_validation import (
    ValidationContractError,
    _admission_windows,
    attach_visual_with_temporal_gate,
    load_analysis_table_cohort,
    participant_exclusive_prediction,
)


def _trial_rows():
    return pd.DataFrame([
        {
            "session_id": "sub-001",
            "analysis_group_token": "g1",
            "block_num": 1,
            "trial_num": 1,
            "global_trial_index": 1,
            "absolute_onset_time": 1000,
            "stimulus_name": "a.png",
            "stimulus_size": 1,
            "is_no_go": 0,
            "omission": 0,
            "commission": 0,
        },
        {
            "session_id": "sub-001",
            "analysis_group_token": "g1",
            "block_num": 1,
            "trial_num": 2,
            "global_trial_index": 2,
            "absolute_onset_time": 2000,
            "stimulus_name": "b.png",
            "stimulus_size": 1,
            "is_no_go": 1,
            "omission": 0,
            "commission": 1,
        },
    ])


def _window_rows():
    base = {
        "session_id": "sub-001",
        "analysis_group_token": "g1",
        "block_num": 1,
        "trial_num": 2,
        "global_trial_index": 2,
        "track": "left_primary",
        "pupil_median": 12.0,
        "pupil_mad": 1.0,
        "pupil_iqr": 2.0,
        "pupil_slope_per_sec": 0.1,
    }
    return pd.DataFrame([
        {**base, "window_name": "pre_200ms", "window_start_offset_ms": -200, "window_end_offset_ms": 0},
        {**base, "window_name": "post_200_1150ms", "window_start_offset_ms": 200, "window_end_offset_ms": 1150},
    ])


def test_validation_refuses_session_without_stage_manifest(tmp_path):
    session = tmp_path / "sessions" / "sub-001"
    session.mkdir(parents=True)
    (session / "sub-001_analysis_tables_completion.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    with pytest.raises(ValidationContractError, match="manifest/completion"):
        load_analysis_table_cohort(tmp_path)


def test_validation_refuses_non_pupil_signal_semantics(tmp_path):
    session = tmp_path / "sessions" / "sub-001"
    session.mkdir(parents=True)
    (session / "sub-001_analysis_tables_manifest.json").write_text(
        json.dumps({"signal_semantics": "pir"}), encoding="utf-8"
    )
    (session / "sub-001_analysis_tables_completion.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    with pytest.raises(ValidationContractError, match="non-pupil-only"):
        load_analysis_table_cohort(tmp_path)


def test_current_visual_is_rejected_from_pre_stimulus_window():
    trials = _trial_rows()
    windows = _window_rows()
    visual = pd.DataFrame([
        {"stimulus_name": "a.png", "stimulus_size": 1, "brightness": 10.0},
        {"stimulus_name": "b.png", "stimulus_size": 1, "brightness": 20.0},
    ])
    linked, audit = attach_visual_with_temporal_gate(windows, trials, visual)
    current = "current_visual__brightness__mean"
    previous = "previous_visual__brightness__mean"
    pre = linked[linked["window_name"].eq("pre_200ms")].iloc[0]
    post = linked[linked["window_name"].eq("post_200_1150ms")].iloc[0]
    assert np.isnan(pre[current])
    assert pre[previous] == pytest.approx(10.0)
    assert post[current] == pytest.approx(20.0)
    assert pre["visual_temporal_gate_status"] == "current_rejected_pre_stimulus"
    assert not audit["future_or_current_brightness_in_pre_window"].any()


def test_admission_windows_carry_separate_sart_targets():
    windows = _window_rows()
    out = _admission_windows(_trial_rows(), windows)
    assert {"go_omission_target", "nogo_commission_target", "track", "analysis_group_token"}.issubset(out.columns)
    nogo = out[out["trial_num"].eq(2)].iloc[0]
    assert np.isnan(nogo["go_omission_target"])
    assert nogo["nogo_commission_target"] == 1


def test_prediction_failure_is_explicit_not_estimable():
    trials = _trial_rows().iloc[[0]].copy()
    windows = pd.DataFrame([
        {
            "session_id": "sub-001",
            "analysis_group_token": "g1",
            "block_num": 1,
            "trial_num": 1,
            "global_trial_index": 1,
            "track": "binocular_primary",
            "window_name": "pre_200ms",
            "pupil_median": 12.0,
            "pupil_mad": 1.0,
            "pupil_iqr": 2.0,
            "pupil_slope_per_sec": 0.1,
        }
    ])
    metrics, failures = participant_exclusive_prediction(trials, windows)
    assert metrics.empty
    assert not failures.empty
    assert set(failures["status"]) == {"not_estimable"}
