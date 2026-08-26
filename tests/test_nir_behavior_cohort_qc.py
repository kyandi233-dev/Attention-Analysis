from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from attention_pipeline.nir_behavior.behavior_qc import add_behavior_qc
from attention_pipeline.nir_behavior.contract import (
    OAR_COLUMN,
    OAR_P90_COLUMN,
    PIR_COLUMN,
    PIR_VALID_COLUMN,
)
from attention_pipeline.nir_behavior_cohort.qc import (
    add_review_scores,
    summarize_behavior_block,
    summarize_eye_block,
)
from attention_pipeline.nir_behavior_cohort.runner import (
    _write_phase1_review_bundle,
)


def test_eye_block_qc_uses_only_valid_finite_pir_and_keeps_fullclass_auxiliary_signals():
    frame = pd.DataFrame(
        {
            "unix_ms": [1000, 1100, 1200, 1300, 1400],
            PIR_COLUMN: [0.30, 0.31, 9.99, np.nan, 0.32],
            PIR_VALID_COLUMN: [True, True, False, True, True],
            OAR_COLUMN: [0.40, 0.41, 0.42, 0.43, 0.44],
            OAR_P90_COLUMN: [0.50, 0.51, 0.52, 0.53, 0.54],
            "roi_clipped": [False, False, True, False, False],
            "ritnet_found": [True, True, True, True, True],
            "fullclass_ocular_component_count": [1, 2, 2, 1, 1],
            "fullclass_ocular_largest_component_fraction": [
                1.0,
                0.95,
                0.80,
                1.0,
                1.0,
            ],
            "fullclass_ocular_fraction": [0.30, 0.31, 0.32, 0.33, 0.34],
            "fullclass_iris_outer_fraction": [0.10, 0.11, 0.12, 0.13, 0.14],
            "fullclass_pupil_fraction": [0.03, 0.031, 0.032, 0.033, 0.034],
        }
    )
    row = summarize_eye_block(
        frame,
        subject="sub-031",
        block_num=1,
        eye="left",
    )
    assert row["n_nir_rows"] == 5
    assert row["unique_unix_ms_count"] == 5
    assert row["pir_n"] == 3
    assert np.isclose(row["pir_usable_fraction"], 3 / 5)
    assert np.isclose(row["pir_median"], 0.31)
    assert np.isclose(row["roi_clipped_fraction"], 1 / 5)
    assert np.isclose(
        row["ocular_fragmented_candidate_fraction"],
        1 / 5,
    )
    assert np.isclose(row["sampling_rate_hz_estimate"], 10.0)
    assert np.isclose(row["rows_per_sec_observed"], 10.0)
    assert np.isclose(
        row["internal_coverage_fraction_estimate"],
        1.0,
    )
    assert np.isclose(row["oar_p90_median"], 0.52)
    assert np.isclose(
        row["oar_p90_minus_median_median"],
        0.10,
    )
    assert np.isclose(
        row["ocular_fraction_available_fraction"],
        1.0,
    )
    assert np.isclose(row["ocular_fraction_median"], 0.32)
    assert np.isclose(row["iris_outer_fraction_median"], 0.12)


def test_behavior_qc_keeps_raw_scoring_and_builds_candidate_subtypes_and_rates():
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
            "is_probe": [0, 1, 0, 0],
            "probe_response": [np.nan, 2, np.nan, np.nan],
            "probe_vigilance": [np.nan, 3, np.nan, np.nan],
            "raw_keypresses": [np.nan, np.nan, "2120", "3250"],
            "prestimulus_press_ms": [990.0, np.nan, np.nan, np.nan],
            "absolute_onset_time": [1000.0, 2000.0, 2000.0, 3000.0],
        }
    )
    enriched = add_behavior_qc(trials, carryover_ms=200)
    summary = summarize_behavior_block(enriched)
    assert summary["raw_omission_count"] == 2
    assert np.isclose(summary["raw_omission_rate_go"], 2 / 3)
    assert summary["prestimulus_only_omission_candidate_count"] == 1
    assert summary["clean_omission_candidate_count"] == 1
    assert np.isclose(
        summary["clean_omission_candidate_rate_go"],
        1 / 3,
    )
    assert summary["commission_count"] == 1
    assert np.isclose(summary["commission_rate_nogo"], 1.0)
    assert summary["rt_candidate_lt_150_flag_count"] == 1
    assert summary["go_response_rt_ms_n"] == 1
    assert np.isclose(summary["go_response_rt_ms_median"], 120.0)
    assert summary["nogo_commission_rt_ms_n"] == 1
    assert np.isclose(
        summary["nogo_commission_rt_ms_median"],
        250.0,
    )
    assert summary["probe_response_2_count"] == 1
    assert summary["probe_vigilance_3_count"] == 1


def test_review_scores_are_continuous_not_exclusion_flags():
    qc = pd.DataFrame(
        {
            "pir_usable_fraction": [0.2, 0.5, 0.8],
            "internal_coverage_fraction_estimate": [0.8, 0.9, 1.0],
            "max_temporal_gap_sec": [0.1, 0.2, 2.0],
            "oar_p90_median": [0.4, 0.5, 0.6],
        }
    )
    out = add_review_scores(qc)
    assert "pir_usable_fraction_robust_z" in out.columns
    assert "internal_coverage_fraction_estimate_robust_z" in out.columns
    assert "max_temporal_gap_sec_robust_z" in out.columns
    assert "oar_p90_median_robust_z" in out.columns
    assert not any("exclude" in column for column in out.columns)


def test_phase1_review_bundle_contains_only_small_review_artifacts(tmp_path: Path):
    output_root = tmp_path / "cohort"
    inventory = output_root / "00_inventory"
    qc = output_root / "01_qc"
    provenance = output_root / "provenance"
    inventory.mkdir(parents=True)
    qc.mkdir(parents=True)
    provenance.mkdir(parents=True)

    (inventory / "cohort_preflight_summary.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (inventory / "cohort_discovery.csv").write_text(
        "subject\nsub-031\n",
        encoding="utf-8",
    )
    for name in (
        "subject_eye_block_qc.csv",
        "subject_qc.csv",
        "behavior_cohort_qc.csv",
        "cohort_anomaly_flags.csv",
    ):
        (qc / name).write_text("subject\nsub-031\n", encoding="utf-8")
    (provenance / "cohort_manifest.json").write_text(
        "{}\n",
        encoding="utf-8",
    )

    review = _write_phase1_review_bundle(
        output_root=output_root,
        inventory_dir=inventory,
        qc_dir=qc,
        provenance_dir=provenance,
    )

    expected = {
        "cohort_preflight_summary.json",
        "cohort_discovery.csv",
        "subject_eye_block_qc.csv",
        "subject_qc.csv",
        "behavior_cohort_qc.csv",
        "cohort_anomaly_flags.csv",
        "cohort_manifest.json",
        "README.md",
    }
    assert {path.name for path in review.iterdir()} == expected
    assert not any("fullclass" in path.name.lower() for path in review.iterdir())
