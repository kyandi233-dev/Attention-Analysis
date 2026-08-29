from __future__ import annotations

import numpy as np

from attention_pipeline.nir_behavior.features import summarize_signal


def test_dynamic_window_features_have_defined_direction_and_units() -> None:
    times_ms = np.array([0.0, 1000.0, 2000.0, 3000.0])
    values = np.array([10.0, 12.0, 11.0, 14.0])
    out = summarize_signal(times_ms, values, "pupil")

    assert out["pupil_peak_to_trough"] == 4.0
    assert out["pupil_dynamic_velocity_status"] == "estimable"
    assert out["pupil_dynamic_velocity_pair_n"] == 3
    assert out["pupil_dilation_step_n"] == 2
    assert out["pupil_constriction_step_n"] == 1
    assert np.isclose(out["pupil_dilation_velocity_median_per_sec"], 2.5)
    assert np.isclose(out["pupil_constriction_velocity_median_per_sec"], 1.0)


def test_dynamic_velocity_fails_closed_with_too_few_pairs() -> None:
    out = summarize_signal(np.array([0.0, 1000.0]), np.array([10.0, 11.0]), "pupil")
    assert out["pupil_dynamic_velocity_status"] == "not_estimable_low_valid_pairs"
    assert out["pupil_dynamic_velocity_pair_n"] == 1
    assert out["pupil_dilation_velocity_median_per_sec"] is None
    assert out["pupil_constriction_velocity_median_per_sec"] is None


def test_dynamic_velocity_ignores_nonpositive_time_deltas() -> None:
    times_ms = np.array([0.0, 0.0, 1000.0, 2000.0])
    values = np.array([10.0, 20.0, 21.0, 20.0])
    out = summarize_signal(times_ms, values, "pupil")
    assert out["pupil_dynamic_velocity_pair_n"] == 2
    assert out["pupil_dynamic_velocity_status"] == "estimable"
