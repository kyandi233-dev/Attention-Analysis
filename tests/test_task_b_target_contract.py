import numpy as np
import pandas as pd

from attention_pipeline.multimodal_formal.analysis_sets import build_analysis_sets
from attention_pipeline.multimodal_formal.quality_admission import audit_quality


def _behavior_with_targets():
    return pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s2", "s2"],
            "participant_group_id": ["p1", "p1", "p2", "p2"],
            "block_id": ["b1", "b1", "b1", "b1"],
            "probe_index_in_block": [1, 2, 1, 2],
            "probe_event_id": [
                "s1|b1|probe|1",
                "s1|b1|probe|2",
                "s2|b1|probe|1",
                "s2|b1|probe|2",
            ],
            "probe_order_in_block": [1, 2, 1, 2],
            "q1_nominal_4class": [1, np.nan, 2, 5],
            "q2_ordinal_4level": [1, 2, 3, 4],
            "window_name": ["pre_30s"] * 4,
            "b": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_formal_identity_retains_authoritative_probe_metadata_and_targets():
    behavior = _behavior_with_targets()
    result = audit_quality({"behavior": behavior}, {"behavior": ["b"]})
    identity = result["formal_probe_identity"]
    for column in (
        "probe_event_id",
        "probe_order_in_block",
        "q1_nominal_4class",
        "q2_ordinal_4level",
        "window_name",
    ):
        assert column in identity.columns


def test_required_q1_target_missing_or_invalid_never_enters_analysis_set():
    behavior = _behavior_with_targets()
    result = audit_quality({"behavior": behavior}, {"behavior": ["b"]})
    sets, summary = build_analysis_sets(
        result["formal_probe_identity"],
        result["probe_feature_status"],
        {
            "q1_behavior": {
                "models": ["M0"],
                "required_features": {"behavior": ["b"]},
                "required_outcomes": ["q1_nominal_4class"],
            }
        },
    )

    by_probe = sets.set_index(["session_id", "probe_index_in_block"])
    assert bool(by_probe.loc[("s1", 1), "included_complete"])
    assert not bool(by_probe.loc[("s1", 2), "included_complete"])
    assert bool(by_probe.loc[("s2", 1), "included_complete"])
    assert not bool(by_probe.loc[("s2", 2), "included_complete"])
    assert not bool(by_probe.loc[("s1", 2), "included_missing_aware"])
    assert not bool(by_probe.loc[("s2", 2), "included_missing_aware"])
    assert "outcome:q1_nominal_4class:missing_or_invalid" in by_probe.loc[
        ("s1", 2), "complete_exclusion_reason"
    ]
    assert "outcome:q1_nominal_4class:missing_or_invalid" in by_probe.loc[
        ("s2", 2), "missing_aware_exclusion_reason"
    ]
    complete = summary.query(
        "analysis_set_id == 'q1_behavior' and membership == 'included_complete'"
    ).iloc[0]
    assert complete.probe_n == 2
    assert complete.participant_group_n == 2
