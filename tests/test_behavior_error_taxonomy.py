from __future__ import annotations

import numpy as np
import pandas as pd

from attention_pipeline.behavior_formal.behavior_error_taxonomy import (
    FORMAL_OMISSION_ENDPOINT_METRICS,
    add_omission_taxonomy,
    build_taxonomy_validation,
    enrich_multiscale_taxonomy,
    summarize_error_taxonomy,
)


def _trials() -> pd.DataFrame:
    return pd.DataFrame([
        {"repeat_participant_id": "P1", "session_id": "sub-031", "block_num": 1, "cycle_bin": 1, "trial_num": 1, "is_no_go": 0, "omission": 1, "response": 0, "rt": np.nan, "prestimulus_press_flag": True,  "carryover_candidate_flag": False, "anticipatory_candidate_flag": True},
        {"repeat_participant_id": "P1", "session_id": "sub-031", "block_num": 1, "cycle_bin": 1, "trial_num": 2, "is_no_go": 0, "omission": 1, "response": 0, "rt": np.nan, "prestimulus_press_flag": False, "carryover_candidate_flag": True,  "anticipatory_candidate_flag": True},
        {"repeat_participant_id": "P1", "session_id": "sub-031", "block_num": 1, "cycle_bin": 1, "trial_num": 3, "is_no_go": 0, "omission": 1, "response": 0, "rt": np.nan, "prestimulus_press_flag": True,  "carryover_candidate_flag": True,  "anticipatory_candidate_flag": True},
        {"repeat_participant_id": "P1", "session_id": "sub-031", "block_num": 1, "cycle_bin": 1, "trial_num": 4, "is_no_go": 0, "omission": 1, "response": 0, "rt": np.nan, "prestimulus_press_flag": False, "carryover_candidate_flag": False, "anticipatory_candidate_flag": False},
        {"repeat_participant_id": "P1", "session_id": "sub-031", "block_num": 1, "cycle_bin": 1, "trial_num": 5, "is_no_go": 0, "omission": 0, "response": 1, "rt": 1200.0, "prestimulus_press_flag": False, "carryover_candidate_flag": False, "anticipatory_candidate_flag": False},
        {"repeat_participant_id": "P1", "session_id": "sub-031", "block_num": 1, "cycle_bin": 1, "trial_num": 6, "is_no_go": 1, "omission": 0, "response": 1, "rt": 400.0, "prestimulus_press_flag": False, "carryover_candidate_flag": False, "anticipatory_candidate_flag": False},
    ])


def test_raw_omission_is_preserved_and_clean_plus_ambiguous_partition_it() -> None:
    raw = _trials()
    out = add_omission_taxonomy(raw)
    assert out["omission"].equals(raw["omission"])
    omission = out[out["raw_go_omission_flag"]]
    assert len(omission) == 4
    assert int(omission["clean_go_omission_flag"].sum()) == 1
    assert int(omission["timing_ambiguous_go_omission_flag"].sum()) == 3
    assert (
        omission["clean_go_omission_flag"].astype(int)
        + omission["timing_ambiguous_go_omission_flag"].astype(int)
    ).eq(1).all()

    # Finer timing subtypes remain mutually exclusive inside ambiguous omissions.
    ambiguous = omission[omission["timing_ambiguous_go_omission_flag"]]
    subtype_flags = [
        "omission_prestimulus_only_ambiguity_flag",
        "omission_carryover_only_ambiguity_flag",
        "omission_prestimulus_and_carryover_ambiguity_flag",
    ]
    assert (ambiguous[subtype_flags].sum(axis=1) == 1).all()


def test_formal_omission_rates_share_go_denominator_and_sum_to_raw() -> None:
    out = add_omission_taxonomy(_trials())
    summary = summarize_error_taxonomy(out)
    assert summary["omission_taxonomy_denominator"] == 5
    assert summary["raw_go_omission_n"] == 4
    assert summary["clean_go_omission_n"] == 1
    assert summary["timing_ambiguous_go_omission_n"] == 3
    assert np.isclose(summary["raw_go_omission_rate"], 4 / 5)
    assert np.isclose(summary["clean_go_omission_rate"], 1 / 5)
    assert np.isclose(summary["timing_ambiguous_go_omission_rate"], 3 / 5)
    assert np.isclose(
        summary["clean_go_omission_rate"] + summary["timing_ambiguous_go_omission_rate"],
        summary["raw_go_omission_rate"],
    )
    assert bool(summary["omission_primary_partition_check"])
    assert bool(summary["omission_subtype_partition_check"])
    # Compatibility aliases must remain exact aliases, not additional outcomes.
    assert summary["omission_motor_timing_ambiguous_n"] == summary["timing_ambiguous_go_omission_n"]
    assert summary["omission_no_detected_motor_timing_ambiguity_n"] == summary["clean_go_omission_n"]


def test_multiscale_taxonomy_derives_block_id_from_raw_block_num() -> None:
    out = add_omission_taxonomy(_trials())
    tables = {
        "session": pd.DataFrame([{"repeat_participant_id": "P1", "session_id": "sub-031"}]),
        "block": pd.DataFrame([{"repeat_participant_id": "P1", "session_id": "sub-031", "block_id": "B1"}]),
        "cycle": pd.DataFrame([{"repeat_participant_id": "P1", "session_id": "sub-031", "block_id": "B1", "cycle_bin": 1}]),
    }
    enriched = enrich_multiscale_taxonomy(out, tables)
    assert enriched["block"]["timing_ambiguous_go_omission_n"].iloc[0] == 3
    assert enriched["cycle"]["clean_go_omission_n"].iloc[0] == 1
    assert enriched["session"]["raw_go_omission_n"].iloc[0] == 4


def test_taxonomy_validation_marks_only_three_omission_rates_as_formal_endpoints() -> None:
    out = add_omission_taxonomy(_trials())
    tables = {
        "session": pd.DataFrame([{"repeat_participant_id": "P1", "session_id": "sub-031"}]),
        "block": pd.DataFrame([{"repeat_participant_id": "P1", "session_id": "sub-031", "block_id": "B1"}]),
        "cycle": pd.DataFrame([{"repeat_participant_id": "P1", "session_id": "sub-031", "block_id": "B1", "cycle_bin": 1}]),
    }
    enriched = enrich_multiscale_taxonomy(out, tables)
    validation = build_taxonomy_validation(enriched, pd.DataFrame())
    formal = validation[validation["endpoint_role"].eq("prespecified_formal_endpoint")]
    assert set(formal["metric"].unique()) == set(FORMAL_OMISSION_ENDPOINT_METRICS)
    qc = validation[validation["endpoint_role"].eq("qc_or_timing_diagnostic")]
    assert "late_go_response_candidate_rate" in set(qc["metric"])
