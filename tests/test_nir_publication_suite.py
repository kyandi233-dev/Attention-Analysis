from __future__ import annotations

import numpy as np
import pandas as pd

from attention_pipeline.nir_pipeline_validation.landscape import (
    build_event_catalogs,
    continuous_event_trajectory,
    feature_redundancy,
    global_pir_trajectory,
    window_effect_stability,
)
from attention_pipeline.nir_pipeline_validation.publication_figures import (
    figure01_global_landscape,
)


def _time_on_task() -> pd.DataFrame:
    rows = []
    for subject, offset in (("sub-031", 0.0), ("sub-032", 0.01)):
        for block in (1, 2):
            for sec in range(0, 120):
                rows.append(
                    {
                        "subject": subject,
                        "block_num": block,
                        "track": "binocular_primary",
                        "time_in_block_mid_sec": sec + 0.5,
                        "pir_median": offset + 0.0001 * sec + 0.005 * (block - 1),
                        "pir_valid_fraction": 0.95,
                    }
                )
    return pd.DataFrame(rows)


def _trial_level() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": ["sub-031"] * 6,
            "block_num": [1] * 6,
            "trial_num": [1, 2, 3, 4, 5, 6],
            "global_trial_index": [1, 2, 3, 4, 5, 6],
            "is_no_go": [0, 1, 0, 1, 0, 0],
            "correct": [1, 1, 1, 0, 1, 0],
            "commission": [0, 0, 0, 1, 0, 0],
            "omission": [0, 0, 0, 0, 0, 1],
            "rt": [400.0, np.nan, 380.0, 300.0, 360.0, np.nan],
            "absolute_onset_time": [10000, 12000, 14000, 16000, 18000, 20000],
            "prestimulus_press_flag": [False] * 6,
            "carryover_candidate_flag": [False] * 6,
            "ambiguous_omission_flag": [False] * 6,
        }
    )


def _trial_windows() -> pd.DataFrame:
    rows = []
    outcomes = {1: 0.01, 2: 0.02, 3: 0.03, 4: 0.05, 5: 0.02, 6: 0.06}
    for trial, value in outcomes.items():
        for window in ("pre_1s", "pre_5s", "pre_10s"):
            rows.append(
                {
                    "subject": "sub-031",
                    "block_num": 1,
                    "trial_num": trial,
                    "global_trial_index": trial,
                    "track": "binocular_primary",
                    "window_name": window,
                    "pir_median": value,
                    "pir_mean": value,
                    "pir_mad": value / 10,
                    "pir_iqr": value / 8,
                    "pir_sd": value / 7,
                    "pir_slope_per_sec": value / 100,
                    "pir_diff_mad": value / 50,
                    "pir_diff_rate_mad_per_sec": value / 40,
                }
            )
    return pd.DataFrame(rows)


def _probe_windows() -> pd.DataFrame:
    rows = []
    for idx, onset in enumerate((13000, 19000), start=1):
        for window in ("pre_10s", "pre_20s"):
            rows.append(
                {
                    "subject": "sub-031",
                    "block_num": 1,
                    "track": "binocular_primary",
                    "window_name": window,
                    "probe_index_global": idx,
                    "probe_index_in_block": idx,
                    "probe_onset_ms": onset,
                    "probe_response": idx,
                    "probe_vigilance": 3 + idx,
                    "probe_rt": 500 + 10 * idx,
                    "probe_vigilance_rt": 600 + 10 * idx,
                    "pir_median": 0.01 * idx,
                }
            )
    return pd.DataFrame(rows)


def test_global_trajectory_offsets_block2_after_block1():
    detail, summary = global_pir_trajectory(
        _time_on_task(),
        track="binocular_primary",
        display_gap_sec=60,
        summary_bin_sec=10,
    )
    b1_max = detail.loc[detail["block_num"].eq(1), "global_time_sec"].max()
    b2_min = detail.loc[detail["block_num"].eq(2), "global_time_sec"].min()
    assert b2_min > b1_max
    assert b2_min - b1_max >= 59
    assert {1, 2} == set(summary["block_num"].astype(int))


def test_continuous_event_trajectory_uses_real_time_bins():
    continuous = pd.DataFrame(
        {
            "subject": ["sub-031"] * 81,
            "block_num": [1] * 81,
            "unix_ms": np.arange(0, 8100, 100),
            "pir": np.linspace(-0.1, 0.1, 81),
        }
    )
    events = pd.DataFrame(
        {
            "subject": ["sub-031"],
            "block_num": [1],
            "event_id": ["e1"],
            "event_onset_ms": [5000.0],
            "event_condition": ["commission"],
        }
    )
    result = continuous_event_trajectory(
        continuous,
        events,
        start_sec=-3,
        end_sec=1,
        bin_sec=1,
    )
    assert set(result["time_bin_mid_sec"]) == {-2.5, -1.5, -0.5, 0.5}
    assert result["n_rows"].sum() == 40
    assert result["pir_median"].notna().all()


def test_event_catalogs_preserve_nogo_omission_and_probe_conditions():
    catalogs = build_event_catalogs(
        _trial_level(),
        _probe_windows(),
        track="binocular_primary",
        max_go_reference_per_subject_block=2,
    )
    assert {"correct_inhibition", "commission"} == set(catalogs["nogo"]["event_condition"])
    assert "clean_omission" in set(catalogs["omission"]["event_condition"])
    assert any(str(value).startswith("response_") for value in catalogs["probe"]["event_condition"])


def test_feature_redundancy_is_within_person_centered():
    within, between = feature_redundancy(
        _trial_windows(),
        track="binocular_primary",
        window_name="pre_5s",
    )
    assert not within.empty
    assert not between.empty
    assert {"pir_median", "pir_mad"}.issubset(set(within["feature_a"]))


def test_window_stability_keeps_prespecified_windows():
    effects, summary = window_effect_stability(
        _trial_level(),
        _trial_windows(),
        _probe_windows(),
        track="binocular_primary",
    )
    assert {1.0, 5.0, 10.0}.issubset(set(effects["window_sec"]))
    assert set(summary["window_sec"]).issubset({1.0, 5.0, 10.0, 20.0})


def test_publication_figure_exports_fixed_canvas_vector_and_raster(tmp_path):
    detail, summary = global_pir_trajectory(
        _time_on_task(), track="binocular_primary", display_gap_sec=60, summary_bin_sec=10
    )
    distribution = (
        _time_on_task()
        .groupby(["subject", "block_num"], as_index=False)["pir_median"]
        .median()
    )
    transition = detail.copy()
    transition["transition_time_sec"] = np.where(
        transition["block_num"].eq(1),
        transition["time_in_block_sec"] - transition.loc[transition["block_num"].eq(1), "time_in_block_sec"].max(),
        transition["time_in_block_sec"],
    )
    outputs = figure01_global_landscape(
        detail,
        summary,
        distribution,
        transition,
        base=tmp_path / "Figure01",
        formats=["pdf", "png"],
        raster_dpi=120,
    )
    assert len(outputs) == 2
    assert (tmp_path / "Figure01.pdf").is_file()
    assert (tmp_path / "Figure01.png").is_file()
