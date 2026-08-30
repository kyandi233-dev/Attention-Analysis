from __future__ import annotations

import numpy as np
import pandas as pd

from attention_pipeline.nir_pipeline_validation.analysis import (
    add_within_between,
    behavior_subject_summary,
    omission_qc_type,
    trial_outcome_label,
)
from attention_pipeline.nir_pipeline_validation.plots import (
    plot_omission_subtypes,
    plot_pipeline_schematic,
)


def test_trial_outcome_label_covers_sart_program_scoring_cases():
    frame = pd.DataFrame(
        {
            "is_no_go": [0, 0, 1, 1],
            "commission": [0, 0, 0, 1],
            "omission": [0, 1, 0, 0],
        }
    )
    assert trial_outcome_label(frame).tolist() == [
        "go_correct",
        "go_omission_program",
        "nogo_correct",
        "nogo_commission",
    ]


def test_omission_qc_type_preserves_clean_and_ambiguous_motor_timing_patterns():
    frame = pd.DataFrame(
        {
            "is_no_go": [0, 0, 0, 0, 0, 1],
            "omission": [1, 1, 1, 1, 0, 0],
            "prestimulus_press_flag": [False, True, False, True, False, False],
            "carryover_candidate_flag": [False, False, True, True, False, False],
            "ambiguous_omission_flag": [False, True, True, True, False, False],
        }
    )
    assert omission_qc_type(frame).tolist() == [
        "clean_omission",
        "prestimulus_associated_omission",
        "carryover_associated_omission",
        "prestimulus_and_carryover_associated_omission",
        "not_go_omission",
        "not_go_omission",
    ]


def test_omission_qc_type_does_not_silently_call_missing_qc_clean():
    frame = pd.DataFrame(
        {
            "is_no_go": [0, 0],
            "omission": [1, 0],
        }
    )
    assert omission_qc_type(frame).tolist() == [
        "go_omission_unclassified_qc_missing",
        "not_go_omission",
    ]


def test_within_between_split_centers_each_subject():
    frame = pd.DataFrame(
        {
            "subject": ["sub-031", "sub-031", "sub-032", "sub-032"],
            "pupil_median": [1.0, 3.0, 10.0, 14.0],
        }
    )
    result = add_within_between(frame)
    assert np.allclose(result["pupil_median_between"], [2.0, 2.0, 12.0, 12.0])
    assert np.allclose(result["pupil_median_within"], [-1.0, 1.0, -2.0, 2.0])
    centered = result.groupby("subject")["pupil_median_within"].mean()
    assert np.allclose(centered.to_numpy(), 0.0)


def test_behavior_summary_separates_program_and_qc_aware_omissions():
    trials = pd.DataFrame(
        {
            "subject": ["sub-031"] * 6,
            "block_num": [1] * 6,
            "is_no_go": [0, 0, 0, 0, 1, 1],
            "correct": [1, 0, 0, 1, 1, 0],
            "commission": [0, 0, 0, 0, 0, 1],
            "omission": [0, 1, 1, 0, 0, 0],
            "rt": [400.0, np.nan, np.nan, 420.0, np.nan, 300.0],
            "is_probe": [0, 0, 0, 0, 1, 0],
            "prestimulus_press_flag": [False, False, True, False, False, False],
            "carryover_candidate_flag": [False, False, False, False, False, False],
            "ambiguous_omission_flag": [False, False, True, False, False, False],
            "anticipatory_candidate_flag": [False, False, True, False, False, False],
            "multiple_keypress_flag": [False] * 6,
        }
    )
    summary = behavior_subject_summary(trials).iloc[0]
    assert summary["n_trials"] == 6
    assert np.isclose(summary["commission_rate"], 0.5)
    assert summary["omission_program_n"] == 2
    assert summary["clean_omission_n"] == 1
    assert summary["ambiguous_omission_n"] == 1
    assert summary["prestimulus_associated_omission_n"] == 1
    assert summary["go_rt_median_ms"] == 410.0
    assert summary["n_probes"] == 1


def test_pipeline_schematic_is_code_generated(tmp_path):
    outputs = plot_pipeline_schematic(
        base=tmp_path / "schematic",
        formats=["png"],
        dpi=80,
    )
    assert len(outputs) == 1
    assert (tmp_path / "schematic.png").is_file()


def test_omission_subtype_figure_is_code_generated(tmp_path):
    trial = pd.DataFrame(
        {
            "subject": ["sub-031", "sub-032", "sub-033"],
            "omission_qc_type": [
                "clean_omission",
                "prestimulus_associated_omission",
                "carryover_associated_omission",
            ],
            "pupil_median": [0.1, 0.2, 0.3],
        }
    )
    outputs = plot_omission_subtypes(
        trial,
        base=tmp_path / "omission",
        formats=["png"],
        dpi=80,
    )
    assert len(outputs) == 1
    assert (tmp_path / "omission.png").is_file()
