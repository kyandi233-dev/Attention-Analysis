"""Behavior feature-role contract for the first supervised-learning round.

This module does not redefine the historical behavior inference outputs. It only
states which already-produced behavior fields are scientifically eligible for the
current first-round Q1 supervised-learning interface.
"""
from __future__ import annotations

from collections.abc import Iterable

from .behavior_error_taxonomy import OMISSION_QC_RATE_METRICS


FIRST_ROUND_SUPERVISED_OMISSION_PREDICTORS = (
    "raw_go_omission_rate",
)

FIRST_ROUND_OMISSION_DESCRIPTIVE_QC_ONLY = tuple(dict.fromkeys((
    "clean_go_omission_rate",
    "timing_ambiguous_go_omission_rate",
    *OMISSION_QC_RATE_METRICS,
)))

OMISSION_COMPATIBILITY_ALIASES = {
    "omission_rate": "raw_go_omission_rate",
    "omission_no_detected_motor_timing_ambiguity_rate": "clean_go_omission_rate",
    "omission_motor_timing_ambiguous_rate": "timing_ambiguous_go_omission_rate",
}


class BehaviorSupervisedContractError(ValueError):
    """Raised when a proposed supervised behavior scheme violates frozen roles."""


def omission_supervised_role(metric: str) -> str:
    name = str(metric)
    if name in FIRST_ROUND_SUPERVISED_OMISSION_PREDICTORS:
        return "first_round_supervised_predictor"
    if name in FIRST_ROUND_OMISSION_DESCRIPTIVE_QC_ONLY:
        return "descriptive_qc_sensitivity_only"
    if name in OMISSION_COMPATIBILITY_ALIASES:
        return "compatibility_alias_not_independent_predictor"
    return "not_an_omission_contract_field"


def validate_first_round_omission_predictors(columns: Iterable[str]) -> tuple[str, ...]:
    """Validate omission-related columns for the current supervised mainline.

    Raw Go omission is the only eligible omission predictor. Clean/timing-
    ambiguous components and finer motor-timing QC rates remain available
    upstream for description/QC, and compatibility aliases must never enter as
    duplicate predictors.
    """
    normalized = tuple(str(column) for column in columns)
    forbidden = [
        column
        for column in normalized
        if column in FIRST_ROUND_OMISSION_DESCRIPTIVE_QC_ONLY
        or column in OMISSION_COMPATIBILITY_ALIASES
    ]
    if forbidden:
        raise BehaviorSupervisedContractError(
            "first-round supervised omission predictors must use only "
            f"raw_go_omission_rate; forbidden fields: {sorted(set(forbidden))}"
        )
    return normalized
