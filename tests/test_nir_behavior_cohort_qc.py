from __future__ import annotations

import numpy as np
import pandas as pd

from attention_pipeline.nir_behavior.behavior_qc import add_behavior_qc
from attention_pipeline.nir_behavior.contract import OAR_COLUMN, PIR_COLUMN, PIR_VALID_COLUMN
from attention_pipeline.nir_behavior_cohort.qc import (
    add_review_scores,
    summarize_behavior_block,
    summarize_eye_block,
)


def test_eye_block_qc_uses_only_valid_finite_pir():
    frame = pd.DataFrame(
        {
            "unix_ms": [1000, 1100, 1200, 1300, 1400],
            PIR_COLUMN: [0.30, 0.31, 9.99, np.nan, 0.32],
            PIR_VALID_COLUMN: [True, True, False, True, True],
            OAR_COLUMN: [0.40, 0.41, 0.42, 0.43, 0.44],
            "roi_clipped": [False, False, True, False, False],
            "ritnet_found": [True, True, True, True, True],
            "fullclass_ocular_component_count": [1, 2, 2, 1, 1],
            "fullclass_ocular_largest_component_fraction": [1.0, 0.95, 0.80, 1.0, 1.0],
        }
    )
    row = summarize_eye_block(frame, subject="sub-031", block_num=1, eye="left")
    assert row["n_nir_rows"] == 5
    assert row["pir_n"] == 3
    assert np.isclose(row["pir_usable_fraction"], 3 / 5)
    assert np.isclose(row["pir_median"], 0.31)
    assert np.isclose(row["roi_clipped_fraction"], 1 / 5)
    assert np.isclose(row["ocular_fragmented_candidate_fraction"], 1 / 5)
    assert np.isclose(row["sampling_rate_hz_estimate"], 10.0)


def test_behavior_qc_keeps_raw_omission_and_builds_candidate_subtypes():
    trials = pd.DataFrame(
        {
            "subject": ["sub-031"] * 4,
            "block_num": [1] * 4,
            "trial_num": [1, 2, 3, 4],
            "is_no_go": [0, 0, 0, 1],
            "response": [0, 0, 1, 1],
            "rt": [np.nan, np.nan, 120.0, 250.0],
            "omission": [1, 1, 0, 0],
            "commission": [0, 0, 0, 1],
            "correct": [0, 0, 1, 0],
            "is_probe": [0, 0, 0, 0],
            "raw_keypresses": [np.nan, np.nan, "2120", "3250"],
            "prestimulus_press_ms": [990.0, np.nan, np.nan, np.nan],
            "absolute_onset_time": [1000.0, 2000.0, 2000.0, 3000.0],
        }
    )
    enriched = add_behavior_qc(trials, carryover_ms=200)
    summary = summarize_behavior_block(enriched)
    assert summary["raw_omission_count"] == 2
    assert summary["prestimulus_only_omission_candidate_count"] == 1
    assert summary["clean_omission_candidate_count"] == 1
    assert summary["commission_count"] == 1
    assert summary["rt_candidate_lt_150_flag_count"] == 1


def test_review_scores_are_continuous_not_exclusion_flags():
    qc = pd.DataFrame(
        {
            "pir_usable_fraction": [0.2, 0.5, 0.8],
            "max_temporal_gap_sec": [0.1, 0.2, 2.0],
        }
    )
    out = add_review_scores(qc)
    assert "pir_usable_fraction_robust_z" in out.columns
    assert "max_temporal_gap_sec_robust_z" in out.columns
    assert not any("exclude" in column for column in out.columns)
