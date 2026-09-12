import json

import numpy as np
import pandas as pd
import pytest

from attention_pipeline.multimodal_formal.analysis_sets import build_analysis_sets
from attention_pipeline.multimodal_formal.prediction_archive import (
    normalize_task_a_predictions,
    validate_prediction_archive,
)
from attention_pipeline.multimodal_formal.quality_admission import audit_quality
from attention_pipeline.multimodal_formal.supervised_input import materialize_supervised_input
from attention_pipeline.supervised_learning.entrypoint import _validate_analysis_set_feature_scope
from attention_pipeline.supervised_learning.feature_schemes import FeatureScheme
from attention_pipeline.supervised_learning.runner import run_nested_loso
from attention_pipeline.supervised_learning.task import SupervisedLearningContractError


ANALYSIS_SET_ID = "behavior_vs_sensor_x"
COMPARISON_SPEC = {
    ANALYSIS_SET_ID: {
        "models": ["B", "B+x"],
        "required_features": {"behavior": ["b"], "sensor": ["x"]},
        "required_outcomes": ["q1_nominal_4class"],
    }
}


def _vertical_tables() -> dict[str, pd.DataFrame]:
    behavior_rows = []
    sensor_rows = []
    for participant_index in range(6):
        participant = f"p{participant_index + 1}"
        session = f"s{participant_index + 1}"
        for probe_index, q1 in enumerate((1, 2, 1, 2), start=1):
            key = {
                "session_id": session,
                "participant_group_id": participant,
                "block_id": "b1",
                "probe_index_in_block": probe_index,
            }
            behavior_rows.append(
                {
                    **key,
                    "probe_event_id": f"{session}|B1|probe|{probe_index}",
                    "probe_order_in_block": probe_index,
                    "q1_nominal_4class": q1,
                    "q2_ordinal_4level": 2,
                    "window_name": "pre_30s",
                    "b": float(participant_index) + 0.1 * probe_index,
                }
            )
            x = float(q1 == 1) + 0.05 * participant_index
            if participant_index == 0 and probe_index == 4:
                x = np.nan
            sensor_rows.append(
                {
                    **key,
                    "source_present": True,
                    "source_readable": True,
                    "window_name": "pre_30s",
                    "x": x,
                }
            )
    return {
        "behavior": pd.DataFrame(behavior_rows),
        "sensor": pd.DataFrame(sensor_rows),
    }


def _build_vertical_contract():
    audited = audit_quality(
        _vertical_tables(),
        {"behavior": ["b"], "sensor": ["x"]},
    )
    analysis_sets, _ = build_analysis_sets(
        audited["formal_probe_identity"],
        audited["probe_feature_status"],
        COMPARISON_SPEC,
    )
    schemes = {
        "B": [FeatureScheme("behavior-reference-v1", ("b",), modality_blocks=("behavior",))],
        "B+x": [FeatureScheme("behavior-plus-x-v1", ("b", "x"), modality_blocks=("behavior", "sensor"))],
    }
    return audited, analysis_sets, schemes


def _materialize(membership_type: str):
    audited, analysis_sets, schemes = _build_vertical_contract()
    frame = materialize_supervised_input(
        analysis_sets,
        audited["probe_feature_status"],
        analysis_set_id=ANALYSIS_SET_ID,
        membership_type=membership_type,
        probe_metadata=audited["formal_probe_identity"],
    )
    return audited, analysis_sets, schemes, frame


def _run_and_validate(membership_type: str):
    _, analysis_sets, schemes, frame = _materialize(membership_type)

    # This is the actual Task-A boundary contract that had previously only been
    # tested against A-side fixtures. The required_features mapping here is
    # produced by Task B, persisted by materialization, and checked against the
    # exact predictor union of the models that will run.
    required_columns, predictor_union = _validate_analysis_set_feature_scope(
        frame,
        schemes,
        analysis_set_id=ANALYSIS_SET_ID,
    )
    assert required_columns == ("b", "x")
    assert predictor_union == ("b", "x")

    result = run_nested_loso(
        frame,
        model_feature_schemes=schemes,
        c_candidates=(0.1,),
        inner_splits=5,
        max_iter=500,
        seed=20260912,
        run_id=f"vertical-{membership_type}",
    )
    normalized = normalize_task_a_predictions(result.predictions)
    audit = validate_prediction_archive(
        normalized,
        analysis_sets,
        membership_column=membership_type,
        requested_analysis_set_ids=[ANALYSIS_SET_ID],
    )
    return frame, result, audit


def test_task_b_materialized_complete_membership_runs_through_task_a_and_archive():
    frame, result, audit = _run_and_validate("included_complete")
    assert len(frame) == 23
    assert not frame[["b", "x"]].isna().any().any()
    assert set(result.predictions["model_id"]) == {"B", "B+x"}
    assert not result.predictions["model_failed"].any()
    assert audit["status"] == "PASS_PREDICTION_ARCHIVE"


def test_task_b_materialized_missing_aware_membership_runs_through_train_fold_imputation():
    frame, result, audit = _run_and_validate("included_missing_aware")
    assert len(frame) == 24
    assert frame["x"].isna().sum() == 1
    assert not result.predictions["model_failed"].any()
    assert audit["status"] == "PASS_PREDICTION_ARCHIVE"


def test_task_a_rejects_task_b_required_features_that_exceed_actual_model_union():
    _, _, schemes, frame = _materialize("included_complete")
    bad = frame.copy()
    bad["required_features"] = json.dumps(
        {"behavior": ["b"], "sensor": ["x", "unused"]},
        sort_keys=True,
    )
    with pytest.raises(SupervisedLearningContractError, match="required_features does not match"):
        _validate_analysis_set_feature_scope(
            bad,
            schemes,
            analysis_set_id=ANALYSIS_SET_ID,
        )


def test_task_b_serialized_false_states_fail_closed_before_materialization():
    audited, _, _ = _build_vertical_contract()
    status = audited["probe_feature_status"].copy()
    target = (
        status["session_id"].eq("s1")
        & status["block_id"].eq("b1")
        & status["probe_index_in_block"].eq(1)
        & status["modality"].eq("sensor")
        & status["feature"].eq("x")
    )
    assert target.sum() == 1
    status.loc[target, "feature_computable"] = "False"
    status.loc[target, "eligible_for_missing_strategy"] = "0"
    status.loc[target, "missing_kind"] = "feature_value_invalid_non_numeric"

    sets, _ = build_analysis_sets(
        audited["formal_probe_identity"],
        status,
        COMPARISON_SPEC,
    )
    row = sets[
        sets["session_id"].eq("s1")
        & sets["block_id"].eq("b1")
        & sets["probe_index_in_block"].eq(1)
    ].iloc[0]
    assert not bool(row["included_complete"])
    assert not bool(row["included_missing_aware"])
