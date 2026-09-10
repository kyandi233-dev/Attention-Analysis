from __future__ import annotations

import numpy as np
import pandas as pd

from attention_pipeline.behavior_formal.behavior_error_taxonomy import (
    CURRENT_PRIMARY_OMISSION_ENDPOINT_METRICS,
    OMISSION_PARTITION_RATE_METRICS,
    OMISSION_QC_RATE_METRICS,
    TAXONOMY_RATE_METRICS,
)
from attention_pipeline.behavior_formal.omission_candidate_validation import validate_omission_candidates


def _frame() -> pd.DataFrame:
    rows = []
    for participant, base in (("P1", 0.02), ("P2", 0.08), ("P3", 0.14)):
        for visit in (1, 2):
            row = {
                "participant_group_id": participant,
                "repeat_participant_id": participant,
                "session_id": f"{participant}-s{visit}",
                "block_id": "B1",
            }
            clean = base + 0.01 * visit
            ambiguous = base / 2 + 0.005 * visit
            row["clean_go_omission_rate"] = clean
            row["timing_ambiguous_go_omission_rate"] = ambiguous
            row["raw_go_omission_rate"] = clean + ambiguous
            for i, metric in enumerate(OMISSION_QC_RATE_METRICS):
                row[metric] = base / 3 + 0.003 * visit + 0.002 * i
            rows.append(row)
    return pd.DataFrame(rows)


def test_omission_audit_preserves_current_roles_but_has_no_full_cohort_selection_authority() -> None:
    frame = _frame()
    validation, redundancy = validate_omission_candidates(
        {"session": frame, "block": frame.copy(), "cycle": frame.copy()},
        frame.copy(),
    )
    session = validation[validation["scale"].eq("session")]
    assert set(session["metric"]) == set(TAXONOMY_RATE_METRICS)
    assert session["between_participant_variance"].notna().all()
    assert session["within_participant_variance"].notna().all()
    assert session["selection_authority"].eq("descriptive_only").all()
    assert session["automatic_drop_allowed"].eq(False).all()

    primary = session[session["metric"].isin(CURRENT_PRIMARY_OMISSION_ENDPOINT_METRICS)]
    assert primary["endpoint_role"].eq("current_primary_omission_endpoint").all()
    assert primary["endpoint_status"].eq("prespecified_not_pvalue_selected").all()

    partition_qc = session[session["metric"].isin(OMISSION_PARTITION_RATE_METRICS[1:])]
    assert partition_qc["endpoint_role"].eq("descriptive_qc_sensitivity_partition").all()
    assert partition_qc["endpoint_status"].eq("not_a_primary_endpoint").all()

    qc = session[session["metric"].isin(OMISSION_QC_RATE_METRICS)]
    assert qc["endpoint_role"].eq("qc_or_timing_diagnostic").all()
    assert qc["endpoint_status"].eq("not_a_primary_endpoint").all()
    assert session["selection_contract"].str.contains("descriptive only", regex=False).all()
    assert not redundancy.empty
    assert redundancy["automatic_drop_allowed"].eq(False).all()
    assert redundancy["selection_authority"].eq("descriptive_only").all()


def test_omission_partition_redundancy_is_labeled_structural_not_drop_rule() -> None:
    frame = _frame()
    _, redundancy = validate_omission_candidates({"session": frame}, frame.iloc[0:0].copy())
    partition_pairs = redundancy[
        redundancy["metric_a"].isin(OMISSION_PARTITION_RATE_METRICS)
        & redundancy["metric_b"].isin(OMISSION_PARTITION_RATE_METRICS)
    ]
    assert not partition_pairs.empty
    assert partition_pairs["structural_same_denominator_pair"].eq(True).all()
    assert partition_pairs["automatic_drop_allowed"].eq(False).all()
    assert partition_pairs["redundancy_interpretation"].str.contains("structurally", regex=False).all()


def test_floor_effect_and_low_coverage_are_review_flags_not_admission_gates() -> None:
    frame = _frame()
    frame["clean_go_omission_rate"] = 0.0
    frame.loc[0:1, "clean_go_omission_rate"] = np.nan
    validation, _ = validate_omission_candidates({"session": frame}, frame.iloc[0:0].copy())
    row = validation[
        (validation["scale"].eq("session"))
        & (validation["metric"].eq("clean_go_omission_rate"))
    ].iloc[0]
    assert row["coverage"] < 0.80
    assert bool(row["below_historical_80pct_coverage_reference"])
    assert "strong_floor_effect" in row["candidate_reasons"]
    assert row["candidate_status"] == "descriptive_audit_only"
    assert row["selection_authority"] == "descriptive_only"
    assert not bool(row["automatic_drop_allowed"])
