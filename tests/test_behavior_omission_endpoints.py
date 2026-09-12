from __future__ import annotations

import numpy as np
import pandas as pd

from attention_pipeline.behavior_formal.behavior_error_taxonomy import OMISSION_PARTITION_RATE_METRICS
from attention_pipeline.behavior_formal.omission_endpoints import (
    build_omission_b1_b2_pairs,
    formal_omission_partition_audit,
)


def _block() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "repeat_participant_id": "P1", "session_id": "sub-031", "block_id": "B1",
            "raw_go_omission_rate": 0.10, "clean_go_omission_rate": 0.06,
            "timing_ambiguous_go_omission_rate": 0.04,
        },
        {
            "repeat_participant_id": "P1", "session_id": "sub-031", "block_id": "B2",
            "raw_go_omission_rate": 0.14, "clean_go_omission_rate": 0.08,
            "timing_ambiguous_go_omission_rate": 0.06,
        },
    ])


def test_omission_b1_b2_pairs_retain_full_partition_with_current_roles() -> None:
    pairs, failures = build_omission_b1_b2_pairs(_block())
    assert failures.empty
    assert set(pairs["metric"]) == set(OMISSION_PARTITION_RATE_METRICS)
    raw = pairs[pairs["metric"].eq("raw_go_omission_rate")].iloc[0]
    assert np.isclose(raw["b2_minus_b1"], 0.04)
    assert raw["endpoint_role"] == "current_primary_omission_endpoint"
    qc = pairs[pairs["metric"].isin(["clean_go_omission_rate", "timing_ambiguous_go_omission_rate"])]
    assert qc["endpoint_role"].eq("descriptive_qc_sensitivity_partition").all()


def test_partition_audit_accepts_exact_raw_clean_ambiguous_identity() -> None:
    audit = formal_omission_partition_audit(_block())
    assert audit["status"].iloc[0] == "complete"
    assert bool(audit["partition_exact_within_1e_12"].iloc[0])
    assert np.isclose(audit["max_absolute_partition_error"].iloc[0], 0.0)


def test_partition_audit_detects_broken_identity() -> None:
    frame = _block()
    frame.loc[0, "clean_go_omission_rate"] = 0.05
    audit = formal_omission_partition_audit(frame)
    assert not bool(audit["partition_exact_within_1e_12"].iloc[0])
    assert audit["max_absolute_partition_error"].iloc[0] > 0
