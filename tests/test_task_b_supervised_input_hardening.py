from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attention_pipeline.multimodal_formal.analysis_sets import build_analysis_sets
from attention_pipeline.multimodal_formal.quality_admission import audit_quality
from attention_pipeline.multimodal_formal.supervised_input import (
    SupervisedInputMaterializationError,
    materialize_supervised_input,
)


def _tables() -> dict[str, pd.DataFrame]:
    behavior = pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s2", "s2"],
            "participant_group_id": ["p1", "p1", "p2", "p2"],
            "block_id": ["b1", "b1", "b1", "b1"],
            "probe_index_in_block": [1, 2, 1, 2],
            "probe_event_id": ["s1|b1|p1", "s1|b1|p2", "s2|b1|p1", "s2|b1|p2"],
            "probe_order_in_block": [1, 2, 1, 2],
            "probe_time_ms": [1000, 2000, 1000, 2000],
            "q1_nominal_4class": [1, 2, 1, 3],
            "q2_ordinal_4level": [4, 3, 4, 2],
            "window_name": ["pre_30s"] * 4,
            "b": [1.0, 2.0, 3.0, 4.0],
        }
    )
    nir = behavior[
        ["session_id", "participant_group_id", "block_id", "probe_index_in_block"]
    ].copy()
    nir["source_present"] = True
    nir["source_readable"] = True
    nir["window_name"] = "pre_30s"
    nir["n"] = [10.0, np.nan, 12.0, 13.0]
    return {"behavior": behavior, "nir": nir}


def _outputs():
    tables = _tables()
    audit = audit_quality(tables, {"behavior": ["b"], "nir": ["n"]})
    sets, _ = build_analysis_sets(
        audit["formal_probe_identity"],
        audit["probe_feature_status"],
        {
            "behavior_plus_nir": {
                "models": ["M0", "M1"],
                "required_features": {"behavior": ["b"], "nir": ["n"]},
                "required_outcomes": ["q1_nominal_4class"],
            }
        },
    )
    return tables, audit, sets


def test_csv_style_false_membership_is_not_cast_to_true() -> None:
    tables, audit, sets = _outputs()
    reloaded = sets.copy()
    for column in ("included_complete", "included_missing_aware"):
        reloaded[column] = reloaded[column].map({True: "True", False: "False"})
    status = audit["probe_feature_status"].copy()
    status["eligible_for_missing_strategy"] = status["eligible_for_missing_strategy"].map(
        {True: "True", False: "False"}
    )

    complete = materialize_supervised_input(
        reloaded,
        status,
        analysis_set_id="behavior_plus_nir",
        membership_type="included_complete",
        probe_metadata=tables["behavior"],
    )
    missing_aware = materialize_supervised_input(
        reloaded,
        status,
        analysis_set_id="behavior_plus_nir",
        membership_type="included_missing_aware",
        probe_metadata=tables["behavior"],
    )
    assert len(complete) == 3
    assert len(missing_aware) == 4


def test_behavior_authority_metadata_disagreement_fails_closed() -> None:
    tables, audit, sets = _outputs()
    authority = tables["behavior"].copy()
    authority.loc[0, "q1_nominal_4class"] = 4

    with pytest.raises(SupervisedInputMaterializationError, match="q1_nominal_4class disagrees"):
        materialize_supervised_input(
            sets,
            audit["probe_feature_status"],
            analysis_set_id="behavior_plus_nir",
            membership_type="included_complete",
            probe_metadata=authority,
        )


def test_materialized_output_requires_probe_event_and_q1_for_task_a() -> None:
    tables, audit, sets = _outputs()
    without_event = sets.drop(columns=["probe_event_id"])
    without_event_metadata = tables["behavior"].drop(columns=["probe_event_id"])

    with pytest.raises(SupervisedInputMaterializationError, match="missing required metadata"):
        materialize_supervised_input(
            without_event,
            audit["probe_feature_status"],
            analysis_set_id="behavior_plus_nir",
            membership_type="included_complete",
            probe_metadata=without_event_metadata,
        )


def test_malformed_comparison_models_fails_before_task_a() -> None:
    tables, audit, sets = _outputs()
    corrupted = sets.copy()
    corrupted["comparison_models"] = "not-json"

    with pytest.raises(SupervisedInputMaterializationError, match="comparison_models.*not valid JSON"):
        materialize_supervised_input(
            corrupted,
            audit["probe_feature_status"],
            analysis_set_id="behavior_plus_nir",
            membership_type="included_complete",
            probe_metadata=tables["behavior"],
        )
