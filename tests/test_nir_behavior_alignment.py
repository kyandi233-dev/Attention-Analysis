from __future__ import annotations

import numpy as np
import pandas as pd

from attention_pipeline.nir_behavior.alignment import (
    build_nir_indices,
    build_probe_windows,
    build_trial_windows,
)
from attention_pipeline.nir_behavior.alignment_v12 import augment_window_metadata
from attention_pipeline.nir_behavior.behavior_qc import add_behavior_qc
from attention_pipeline.nir_behavior.contract import WindowSpec
from attention_pipeline.nir_behavior.coverage import build_window_coverage_report
from attention_pipeline.nir_behavior.diagnostics import _break_large_gaps


def _behavior() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": ["sub-031"] * 4,
            "block_num": [1, 1, 1, 1],
            "trial_num": [1, 2, 3, 4],
            "global_trial_index": [1, 2, 3, 4],
            "block_onset_time": [900.0, 900.0, 900.0, 900.0],
            "absolute_onset_time": [1000.0, 2150.0, 3300.0, 4450.0],
            "stimulus_name": ["go1", "go2", "nogo", "go3"],
            "stimulus_size": [100, 100, 100, 100],
            "is_no_go": [0, 0, 1, 0],
            "response": [1, 0, 1, 1],
            "rt": [300.0, np.nan, 250.0, 350.0],
            "response_time": [1300.0, np.nan, 3550.0, 4800.0],
            "correct": [1, 0, 0, 1],
            "commission": [0, 0, 1, 0],
            "omission": [0, 1, 0, 0],
            "raw_keypresses": ["1300", "", "3550;4400", "4800"],
            "prestimulus_press_ms": [np.nan, 2145.0, np.nan, np.nan],
            "is_probe": [0, 1, 0, 1],
            "probe_response": [np.nan, 2, np.nan, 1],
            "probe_rt": [np.nan, 700, np.nan, 600],
            "probe_vigilance": [np.nan, 3, np.nan, 4],
            "probe_vigilance_rt": [np.nan, 500, np.nan, 550],
            "probe_onset_time": [np.nan, 3000.0, np.nan, 5600.0],
        }
    )


def _nir() -> pd.DataFrame:
    times = np.arange(500.0, 6000.0, 100.0)
    rows = []
    for eye, offset in (("left", 0.0), ("right", 0.01)):
        for idx, time in enumerate(times):
            rows.append(
                {
                    "subject": "sub-031",
                    "block_num": 1,
                    "eye": eye,
                    "unix_ms": time,
                    "frame_idx": idx,
                    "fullclass_pupil_to_iris_diameter_ratio": 0.4
                    + offset
                    + idx * 0.0001,
                    "fullclass_normalization_valid": True,
                    "fullclass_ocular_aperture_ratio_median": 0.30 + offset,
                }
            )
    return pd.DataFrame(rows)


def test_behavior_qc_preserves_scoring_and_flags_prestimulus() -> None:
    frame = add_behavior_qc(_behavior())
    assert frame.loc[1, "omission"] == 1
    assert bool(frame.loc[1, "prestimulus_press_flag"])
    assert bool(frame.loc[1, "ambiguous_omission_flag"])
    assert frame.loc[1, "prestimulus_delta_to_onset_ms"] == -5.0
    assert frame.loc[2, "n_raw_keypresses"] == 2


def test_trial_window_keeps_eyes_separate() -> None:
    trials = add_behavior_qc(_behavior())
    indices = build_nir_indices(_nir(), "sub-031")
    specs = [WindowSpec("pre_1s", "state", -1000, 0)]
    result = build_trial_windows(trials, indices, specs)
    assert len(result) == len(trials) * 2
    assert set(result["eye"]) == {"left", "right"}
    assert result["pir_median"].notna().any()


def test_probe_window_marks_previous_probe_crossing() -> None:
    trials = add_behavior_qc(_behavior())
    indices = build_nir_indices(_nir(), "sub-031")
    specs = [WindowSpec("pre_3s", "probe_state", -3000, 0)]
    result = build_probe_windows(trials, indices, specs)
    second_probe = result[result["probe_index_global"] == 2]
    assert len(second_probe) == 2
    assert second_probe["window_crosses_previous_probe"].all()


def test_schema_v2_renames_oar_availability_and_marks_boundary_truncation() -> None:
    trials = add_behavior_qc(_behavior())
    indices = build_nir_indices(_nir(), "sub-031")
    specs = [WindowSpec("pre_1s", "state", -1000, 0)]
    windows = build_trial_windows(trials, indices, specs)
    augmented = augment_window_metadata(windows, trials, indices)

    assert "oar_available_fraction" in augmented.columns
    assert "oar_valid_fraction" not in augmented.columns
    first = augmented[
        (augmented["trial_num"] == 1) & (augmented["eye"] == "left")
    ].iloc[0]
    assert bool(first["window_truncated_by_block_start"])
    assert np.isclose(first["requested_duration_sec"], 1.0)
    assert np.isclose(first["available_duration_sec"], 0.1)
    assert np.isclose(first["available_duration_fraction"], 0.1)


def test_schema_v2_internal_coverage_uses_available_duration() -> None:
    trials = add_behavior_qc(_behavior())
    indices = build_nir_indices(_nir(), "sub-031")
    specs = [WindowSpec("pre_1s", "state", -1000, 0)]
    windows = build_trial_windows(trials, indices, specs)
    augmented = augment_window_metadata(windows, trials, indices)
    first = augmented[
        (augmented["trial_num"] == 1) & (augmented["eye"] == "left")
    ].iloc[0]
    assert np.isclose(first["sampling_rate_hz_estimate"], 10.0)
    assert np.isclose(first["expected_nir_rows_available"], 1.0)
    assert first["n_nir_rows_available"] == 1
    assert np.isclose(first["internal_coverage_fraction"], 1.0)


def test_schema_v2_detects_internal_gap_without_boundary_truncation() -> None:
    trials = add_behavior_qc(_behavior())
    nir = _nir()
    gap = (
        nir["unix_ms"].ge(1500.0)
        & nir["unix_ms"].lt(1900.0)
        & nir["eye"].eq("left")
    )
    indices = build_nir_indices(nir.loc[~gap].copy(), "sub-031")
    specs = [WindowSpec("pre_1s", "state", -1000, 0)]
    windows = augment_window_metadata(
        build_trial_windows(trials, indices, specs), trials, indices
    )
    second = windows[
        (windows["trial_num"] == 2) & (windows["eye"] == "left")
    ].iloc[0]
    assert not bool(second["window_truncated_by_block_start"])
    assert not bool(second["window_truncated_by_block_end"])
    assert second["internal_coverage_fraction"] < 1.0
    assert second["max_temporal_gap_sec"] >= 0.5


def test_coverage_report_keeps_block_eye_window_groups() -> None:
    trials = add_behavior_qc(_behavior())
    indices = build_nir_indices(_nir(), "sub-031")
    specs = [WindowSpec("pre_1s", "state", -1000, 0)]
    windows = augment_window_metadata(
        build_trial_windows(trials, indices, specs), trials, indices
    )
    report = build_window_coverage_report(windows, level="trial")
    assert len(report) == 2
    assert set(report["eye"]) == {"left", "right"}
    assert report["n_windows"].eq(4).all()
    assert report["pir_valid_fraction_median"].eq(1.0).all()
    assert report["oar_available_fraction_median"].eq(1.0).all()
    assert report["nir_rows_per_available_sec_median"].gt(0).all()
    assert report["boundary_truncated_window_fraction"].gt(0).all()


def test_diagnostic_line_breaks_long_missing_intervals() -> None:
    x = np.array([0.0, 1.0, 5.0, 6.0])
    y = np.array([0.3, 0.31, 0.29, 0.30])
    out_x, out_y = _break_large_gaps(x, y, max_gap_sec=2.5)
    assert len(out_x) == 5
    assert np.isnan(out_x[2])
    assert np.isnan(out_y[2])
