from __future__ import annotations

import numpy as np
import pandas as pd

from attention_pipeline.behavior_formal.behavior_error_taxonomy import (
    FORMAL_OMISSION_ENDPOINT_METRICS,
    OMISSION_QC_RATE_METRICS,
    TAXONOMY_RATE_METRICS,
)
from attention_pipeline.behavior_formal.omission_candidate_validation import validate_omission_candidates


def _frame() -> pd.DataFrame:
    rows = []
    for participant, base in (("P1", 0.02), ("P2", 0.08), ("P3", 0.14)):
        for visit in (1, 2):
            row = {
                "repeat_participant_id": participant,
                "session_id": f"{participant}-s{visit}",
                "block_id": "B1",
            }
            # Make raw = clean + timing_ambiguous while still giving the other
            # QC candidates nonconstant values.
            clean = base + 0.01 * visit
            ambiguous = base / 2 + 0.005 * visit
            row["clean_go_omission_rate"] = clean
            row["timing_ambiguous_go_omission_rate"] = ambiguous
            row["raw_go_omission_rate"] = clean + ambiguous
            for i, metric in enumerate(OMISSION_QC_RATE_METRICS):
                row[metric] = base / 3 + 0.003 * visit + 0.002 * i
            rows.append(row)
    return pd.DataFrame(rows)


def test_omission_endpoints_get_within_between_and_role_is_prespecified() -> None:
    frame = _frame()
    validation, redundancy = validate_omission_candidates(
        {"session": frame, "block": frame.copy(), "cycle": frame.copy()},
        frame.copy(),
    )
    session = validation[validation["scale"].eq("session")]
    assert set(session["metric"]) == set(TAXONOMY_RATE_METRICS)
    assert session["between_participant_variance"].notna().all()
    assert session["within_participant_variance"].notna().all()

    formal = session[session["metric"].isin(FORMAL_OMISSION_ENDPOINT_METRICS)]
    assert formal["endpoint_role"].eq("prespecified_formal_endpoint").all()
    assert formal["endpoint_status"].eq("prespecified_not_pvalue_selected").all()

    qc = session[session["metric"].isin(OMISSION_QC_RATE_METRICS)]
    assert qc["endpoint_role"].eq("qc_or_timing_diagnostic").all()
    assert qc["endpoint_status"].eq("not_a_primary_endpoint").all()
    assert session["selection_contract"].str.contains("never outcome p-value", regex=False).all()
    assert not redundancy.empty
    assert redundancy["automatic_drop_allowed"].eq(False).all()


def test_formal_omission_redundancy_is_labeled_structural_not_drop_rule() -> None:
    frame = _frame()
    _, redundancy = validate_omission_candidates({"session": frame}, frame.iloc[0:0].copy())
    formal_pairs = redundancy[
        redundancy["metric_a"].isin(FORMAL_OMISSION_ENDPOINT_METRICS)
        & redundancy["metric_b"].isin(FORMAL_OMISSION_ENDPOINT_METRICS)
    ]
    assert not formal_pairs.empty
    assert formal_pairs["structural_same_denominator_pair"].eq(True).all()
    assert formal_pairs["automatic_drop_allowed"].eq(False).all()
    assert formal_pairs["redundancy_interpretation"].str.contains("structurally", regex=False).all()


def test_floor_effect_is_review_flag_not_automatic_exclusion_for_formal_endpoint() -> None:
    frame = _frame()
    frame["clean_go_omission_rate"] = 0.0
    frame["raw_go_omission_rate"] = frame["timing_ambiguous_go_omission_rate"]
    validation, _ = validate_omission_candidates({"session": frame}, frame.iloc[0:0].copy())
    row = validation[
        (validation["scale"].eq("session"))
        & (validation["metric"].eq("clean_go_omission_rate"))
    ].iloc[0]
    assert np.isclose(row["floor_fraction"], 1.0)
    assert "strong_floor_effect" in row["candidate_reasons"]
    assert row["endpoint_role"] == "prespecified_formal_endpoint"
    assert row["endpoint_status"] == "prespecified_not_pvalue_selected"
    assert row["candidate_status"] == "formal_endpoint_needs_real_data_stability_review"
