from __future__ import annotations

import numpy as np
import pandas as pd

from attention_pipeline.nir_pipeline_validation.probe_analysis import (
    canonical_probe_response,
    probe_event_table,
    probe_response_subject_summary,
    probe_response_vigilance_table,
    probe_response_window_table,
)


def test_probe_response_codes_preserve_raw_categories_without_semantic_labels():
    series = pd.Series([1, 2.0, 3.5, "custom", np.nan])
    result = canonical_probe_response(series)
    assert result.iloc[0] == "1"
    assert result.iloc[1] == "2"
    assert result.iloc[2] == "3.5"
    assert result.iloc[3] == "custom"
    assert pd.isna(result.iloc[4])


def _probe_windows() -> pd.DataFrame:
    rows = []
    for response, vigilance, probe_index in [(1, 2, 1), (2, 4, 2)]:
        for window, pir, rt in [("pre_10s", 0.1 * response, 420 + 10 * response), ("pre_20s", 0.2 * response, 430 + 10 * response)]:
            rows.append(
                {
                    "subject": "sub-031",
                    "block_num": 1,
                    "probe_index_global": probe_index,
                    "probe_index_in_block": probe_index,
                    "probe_onset_ms": 1000 * probe_index,
                    "probe_response": response,
                    "probe_rt": 500 + response,
                    "probe_vigilance": vigilance,
                    "probe_vigilance_rt": 600 + response,
                    "track": "binocular_primary",
                    "window_name": window,
                    "pir_median": pir,
                    "pir_valid_fraction": 0.9,
                    "internal_coverage_fraction": 1.0,
                    "n_trials": 10,
                    "n_go": 8,
                    "n_nogo": 2,
                    "n_commission": 1 if response == 2 else 0,
                    "n_omission": response - 1,
                    "n_ambiguous_omission": 0,
                    "n_anticipatory_candidate": response - 1,
                    "go_rt_median_ms": rt,
                    "go_rt_mad_ms": 25.0,
                }
            )
    return pd.DataFrame(rows)


def test_probe_event_table_deduplicates_windows_and_keeps_both_probe_dimensions():
    events = probe_event_table(_probe_windows(), track="binocular_primary")
    assert len(events) == 2
    assert events["probe_response_code"].tolist() == ["1", "2"]
    assert events["probe_vigilance"].tolist() == [2, 4]


def test_probe_response_summaries_cover_distribution_pir_behavior_and_joint_structure():
    windows = _probe_windows()
    events = probe_event_table(windows, track="binocular_primary")
    response = probe_response_subject_summary(events)
    assert response["n_response"].sum() == 2
    assert np.isclose(response["response_fraction"].sum(), 1.0)

    window_summary = probe_response_window_table(windows, track="binocular_primary")
    assert set(window_summary["window_name"]) == {"pre_10s", "pre_20s"}
    row = window_summary[
        window_summary["probe_response_code"].astype(str).eq("2")
        & window_summary["window_name"].eq("pre_20s")
    ].iloc[0]
    assert np.isclose(row["pir_median"], 0.4)
    assert np.isclose(row["commission_rate_window"], 0.5)
    assert np.isclose(row["omission_rate_window"], 0.125)

    joint = probe_response_vigilance_table(events)
    assert joint["n_probes"].sum() == 2
    assert set(joint["probe_vigilance"]) == {2.0, 4.0}
