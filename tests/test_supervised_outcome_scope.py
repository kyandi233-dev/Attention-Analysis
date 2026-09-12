import pandas as pd
import pytest

from attention_pipeline.supervised_learning.outcome_scope import (
    validate_task_a_required_outcomes,
)
from attention_pipeline.supervised_learning.task import SupervisedLearningContractError


def _frame(required_outcomes: str = '["q1_nominal_4class"]') -> pd.DataFrame:
    return pd.DataFrame(
        {
            "analysis_set_id": ["behavior_vs_x", "behavior_vs_x"],
            "required_outcomes": [required_outcomes, required_outcomes],
            "q1_nominal_4class": [1, 2],
        }
    )


def test_current_task_a_accepts_q1_authoritative_source_only():
    outcomes = validate_task_a_required_outcomes(_frame())
    assert outcomes == ("q1_nominal_4class",)


def test_current_task_a_rejects_q2_as_extra_sample_defining_outcome():
    frame = _frame('["q1_nominal_4class", "q2_ordinal_4level"]')
    with pytest.raises(
        SupervisedLearningContractError,
        match="does not match the frozen Task-A target source",
    ):
        validate_task_a_required_outcomes(frame)


def test_current_task_a_rejects_inconsistent_required_outcomes_within_analysis_set():
    frame = _frame()
    frame.loc[1, "required_outcomes"] = '["q1_nominal_4class", "q2_ordinal_4level"]'
    with pytest.raises(
        SupervisedLearningContractError,
        match="one analysis_set_id must declare one required_outcomes list",
    ):
        validate_task_a_required_outcomes(frame)


def test_current_task_a_requires_required_outcomes_provenance():
    frame = _frame().drop(columns=["required_outcomes"])
    with pytest.raises(
        SupervisedLearningContractError,
        match="must retain required_outcomes",
    ):
        validate_task_a_required_outcomes(frame)
