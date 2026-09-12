"""Formal outcome-scope contract for the current Task-A Q1 supervised task."""
from __future__ import annotations

import json
from collections.abc import Sequence

import pandas as pd

from .task import Q1_BINARY_SPEC, SupervisedLearningContractError

FORMAL_REQUIRED_OUTCOMES_COLUMN = "required_outcomes"


def _parse_required_outcomes(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise SupervisedLearningContractError("required_outcomes contains a blank value")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SupervisedLearningContractError(
                f"required_outcomes must be a JSON list: {exc}"
            ) from exc
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        parsed = list(value)
    else:
        raise SupervisedLearningContractError("required_outcomes must be a JSON/list sequence")
    if not isinstance(parsed, list) or not parsed:
        raise SupervisedLearningContractError("required_outcomes must contain at least one outcome")
    outcomes = tuple(str(item).strip() for item in parsed)
    if any(not item for item in outcomes):
        raise SupervisedLearningContractError("required_outcomes contains blank outcome names")
    if len(set(outcomes)) != len(outcomes):
        raise SupervisedLearningContractError("required_outcomes contains duplicate outcome names")
    return outcomes


def validate_task_a_required_outcomes(frame: pd.DataFrame) -> tuple[str, ...]:
    """Fail closed if Task-B sample eligibility used outcomes outside Task A's target source.

    Current Task A predicts binary Q1 but derives that label from the authoritative
    four-class source column ``q1_nominal_4class``. Future Q1 four-class prediction
    can therefore reuse the same source-level analysis-set contract. Q2 must not
    filter the current Q1 training population.
    """
    if FORMAL_REQUIRED_OUTCOMES_COLUMN not in frame.columns:
        raise SupervisedLearningContractError(
            "formal Task-A input must retain required_outcomes from the Task-B analysis-set contract"
        )
    raw = frame[FORMAL_REQUIRED_OUTCOMES_COLUMN]
    if raw.isna().any():
        raise SupervisedLearningContractError("required_outcomes contains missing values")
    parsed_rows = [_parse_required_outcomes(value) for value in raw.tolist()]
    unique = set(parsed_rows)
    if len(unique) != 1:
        raise SupervisedLearningContractError(
            f"one analysis_set_id must declare one required_outcomes list; got {sorted(unique)}"
        )
    outcomes = parsed_rows[0]
    expected = (Q1_BINARY_SPEC.source_column,)
    if outcomes != expected:
        raise SupervisedLearningContractError(
            "Task-B required_outcomes does not match the frozen Task-A target source; "
            f"expected={list(expected)}, observed={list(outcomes)}"
        )
    return outcomes
