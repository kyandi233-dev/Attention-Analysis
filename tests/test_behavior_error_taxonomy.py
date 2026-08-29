from __future__ import annotations

import numpy as np
import pandas as pd

from attention_pipeline.behavior_formal.behavior_error_taxonomy import (
    add_omission_taxonomy,
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


def test_raw_omission_is_preserved_and_mutually_exclusive_subtypes_partition_it() -> None:
    raw = _trials()
    out = add_omission_taxonomy(raw)
    assert out["omission"].equals(raw["omission"])
    omission = out[out["raw_go_omission_flag"]]
    assert len(omission) == 4
    assert omission["omission_subtype"].nunique() == 4
    subtype_flags = [
        "omission_no_detected_motor_timing_ambiguity_flag",
        "omission_prestimulus_only_ambiguity_flag",
        "omission_carryover_only_ambiguity_flag",
        "omission_prestimulus_and_carryover_ambiguity_flag",
    ]
    assert (omission[subtype_flags].sum(axis=1) == 1).all()


def test_omission_taxonomy_rates_use_go_opportunities_not_nogo_opportunities() -> None:
    out = add_omission_taxonomy(_trials())
    summary = summarize_error_taxonomy(out)
    assert summary["omission_taxonomy_denominator"] == 5
    assert summary["omission_motor_timing_ambiguous_n"] == 3
    assert np.isclose(summary["omission_motor_timing_ambiguous_rate"], 3 / 5)
    assert bool(summary["omission_subtype_partition_check"])
    assert summary["late_go_response_candidate_n"] == 1


def test_multiscale_taxonomy_derives_block_id_from_raw_block_num() -> None:
    out = add_omission_taxonomy(_trials())
    tables = {
        "session": pd.DataFrame([{"repeat_participant_id": "P1", "session_id": "sub-031"}]),
        "block": pd.DataFrame([{"repeat_participant_id": "P1", "session_id": "sub-031", "block_id": "B1"}]),
        "cycle": pd.DataFrame([{"repeat_participant_id": "P1", "session_id": "sub-031", "block_id": "B1", "cycle_bin": 1}]),
    }
    enriched = enrich_multiscale_taxonomy(out, tables)
    assert enriched["block"]["omission_motor_timing_ambiguous_n"].iloc[0] == 3
    assert enriched["cycle"]["omission_no_detected_motor_timing_ambiguity_n"].iloc[0] == 1
