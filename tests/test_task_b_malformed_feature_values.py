import numpy as np
import pandas as pd

from attention_pipeline.multimodal_formal.analysis_sets import build_analysis_sets
from attention_pipeline.multimodal_formal.quality_admission import audit_quality


def _behavior_with_value_states() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s2", "s2"],
            "participant_group_id": ["p1", "p1", "p2", "p2"],
            "block_id": ["b1", "b1", "b1", "b1"],
            "probe_index_in_block": [1, 2, 1, 2],
            "window_name": ["pre_30s"] * 4,
            "b": [1.0, "corrupt", np.nan, np.inf],
        }
    )


def test_quality_audit_separates_parse_invalid_missing_and_nonfinite_values() -> None:
    behavior = _behavior_with_value_states()
    audit = audit_quality({"behavior": behavior}, {"behavior": ["b"]})
    status = audit["probe_feature_status"].sort_values(
        ["session_id", "probe_index_in_block"]
    )

    by_probe = {
        (row.session_id, int(row.probe_index_in_block)): row
        for row in status.itertuples(index=False)
    }

    available = by_probe[("s1", 1)]
    malformed = by_probe[("s1", 2)]
    missing = by_probe[("s2", 1)]
    nonfinite = by_probe[("s2", 2)]

    assert available.missing_kind == "available"
    assert bool(available.feature_computable)
    assert bool(available.eligible_for_missing_strategy)

    assert malformed.missing_kind == "feature_parse_invalid"
    assert not bool(malformed.feature_computable)
    assert not bool(malformed.eligible_for_missing_strategy)

    assert missing.missing_kind == "single_feature_missing"
    assert not bool(missing.feature_computable)
    assert bool(missing.eligible_for_missing_strategy)

    assert nonfinite.missing_kind == "feature_nonfinite_invalid"
    assert not bool(nonfinite.feature_computable)
    assert not bool(nonfinite.eligible_for_missing_strategy)


def test_malformed_and_nonfinite_values_never_enter_missing_aware_analysis_set() -> None:
    behavior = _behavior_with_value_states()
    audit = audit_quality({"behavior": behavior}, {"behavior": ["b"]})
    analysis_sets, _ = build_analysis_sets(
        audit["formal_probe_identity"],
        audit["probe_feature_status"],
        {
            "behavior_only": {
                "models": ["B"],
                "required_features": {"behavior": ["b"]},
            }
        },
    )

    membership = {
        (row.session_id, int(row.probe_index_in_block)): bool(row.included_missing_aware)
        for row in analysis_sets.itertuples(index=False)
    }
    assert membership[("s1", 1)]
    assert not membership[("s1", 2)]
    assert membership[("s2", 1)]
    assert not membership[("s2", 2)]
