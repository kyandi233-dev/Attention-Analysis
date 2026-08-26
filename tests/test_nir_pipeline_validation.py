from __future__ import annotations

import numpy as np
import pandas as pd

from attention_pipeline.nir_pipeline_validation.analysis import (
    add_within_between,
    behavior_subject_summary,
    trial_outcome_label,
)
from attention_pipeline.nir_pipeline_validation.plots import plot_pipeline_schematic


def test_trial_outcome_label_covers_sart_cases():
    frame = pd.DataFrame(
        {
            "is_no_go": [0, 0, 1, 1],
            "commission": [0, 0, 0, 1],
            "omission": [0, 1, 0, 0],
        }
    )
    assert trial_outcome_label(frame).tolist() == [
        "go_correct",
        "go_omission",
        "nogo_correct",
        "nogo_commission",
    ]


def test_within_between_split_centers_each_subject():
    frame = pd.DataFrame(
        {
            "subject": ["sub-031", "sub-031", "sub-032", "sub-032"],
            "pir_median": [1.0, 3.0, 10.0, 14.0],
        }
    )
    result = add_within_between(frame)
    assert np.allclose(result["pir_median_between"], [2.0, 2.0, 12.0, 12.0])
    assert np.allclose(result["pir_median_within"], [-1.0, 1.0, -2.0, 2.0])
    centered = result.groupby("subject")["pir_median_within"].mean()
    assert np.allclose(centered.to_numpy(), 0.0)


def test_behavior_summary_separates_go_and_nogo_rates():
    trials = pd.DataFrame(
        {
            "subject": ["sub-031"] * 4,
            "block_num": [1] * 4,
            "is_no_go": [0, 0, 1, 1],
            "correct": [1, 0, 1, 0],
            "commission": [0, 0, 0, 1],
            "omission": [0, 1, 0, 0],
            "rt": [400.0, np.nan, np.nan, 300.0],
            "is_probe": [0, 0, 1, 0],
        }
    )
    summary = behavior_subject_summary(trials).iloc[0]
    assert summary["n_trials"] == 4
    assert np.isclose(summary["commission_rate"], 0.5)
    assert np.isclose(summary["omission_rate"], 0.5)
    assert summary["go_rt_median_ms"] == 400.0
    assert summary["n_probes"] == 1


def test_pipeline_schematic_is_code_generated(tmp_path):
    outputs = plot_pipeline_schematic(
        base=tmp_path / "schematic",
        formats=["png"],
        dpi=80,
    )
    assert len(outputs) == 1
    assert (tmp_path / "schematic.png").is_file()
