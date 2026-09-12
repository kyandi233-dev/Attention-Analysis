import numpy as np
import pandas as pd

from attention_pipeline.multimodal_formal.analysis_sets import build_analysis_sets
from attention_pipeline.multimodal_formal.quality_admission import audit_quality


def _identity() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": ["s1", "s1"],
            "participant_group_id": ["p1", "p1"],
            "block_id": ["b1", "b1"],
            "probe_index_in_block": [1, 2],
        }
    )


def test_analysis_sets_do_not_treat_serialized_false_as_true() -> None:
    identity = _identity()
    status = identity.copy()
    status["modality"] = "behavior"
    status["feature"] = "b"
    status["feature_computable"] = ["False", "True"]
    status["eligible_for_missing_strategy"] = ["False", "True"]
    status["missing_kind"] = ["single_feature_missing", "available"]

    sets, _ = build_analysis_sets(
        identity,
        status,
        {
            "behavior_only": {
                "models": ["behavior_reference"],
                "required_features": {"behavior": ["b"]},
            }
        },
    )
    first = sets.sort_values("probe_index_in_block").iloc[0]
    second = sets.sort_values("probe_index_in_block").iloc[1]

    assert not bool(first["included_complete"])
    assert not bool(first["included_missing_aware"])
    assert bool(second["included_complete"])
    assert bool(second["included_missing_aware"])


def test_only_genuine_residual_na_is_missing_strategy_eligible() -> None:
    behavior = pd.DataFrame(
        {
            "session_id": ["s1"] * 4,
            "participant_group_id": ["p1"] * 4,
            "block_id": ["b1"] * 4,
            "probe_index_in_block": [1, 2, 3, 4],
            "window_name": ["pre_30s"] * 4,
            "candidate": ["not-a-number", np.inf, np.nan, 1.5],
        }
    )

    result = audit_quality({"behavior": behavior}, {"behavior": ["candidate"]})
    status = result["probe_feature_status"].sort_values("probe_index_in_block")

    assert status["missing_kind"].tolist() == [
        "feature_value_invalid_non_numeric",
        "feature_value_invalid_nonfinite",
        "single_feature_missing",
        "available",
    ]
    assert status["eligible_for_missing_strategy"].tolist() == [False, False, True, True]
    assert status["feature_computable"].tolist() == [False, False, False, True]
