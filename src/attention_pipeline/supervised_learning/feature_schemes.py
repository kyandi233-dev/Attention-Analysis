"""Predeclared feature-scheme contracts for the current Q1 mainline.

Feature schemes define scientifically allowed candidate representations before a
model sees outer-test data.  They do not perform empirical filtering themselves;
training-split-only column handling belongs to ``preprocessing.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from .task import Q1_BINARY_SPEC, SupervisedLearningContractError


# Frozen first-round exclusions from the current 1.15.5/1.15.6 method state.
_MAINLINE_FORBIDDEN_COLUMNS: dict[str, str] = {
    Q1_BINARY_SPEC.source_column: "outcome label cannot be used as a predictor",
    "q2_ordinal_4level": "Q2 is interpretation/construct validation, not a first-round Q1 predictor",
    "omission_rate": "historical alias duplicates raw_go_omission_rate",
    "clean_go_omission_rate": "clean omission is descriptive/QC only in the first-round mainline",
    "timing_ambiguous_go_omission_rate": "timing-ambiguous omission is descriptive/QC only in the first-round mainline",
    "hard_pupil_fraction": "historical validated field is outside the first-round NIR mainline",
    "participant_group_id": "participant identity is a grouping key, not a predictor",
    "repeat_participant_id": "participant identity alias is not a predictor",
    "session_id": "session locator is not a predictor",
    "block_id": "block locator is not a predictor",
    "probe_event_id": "probe locator is not a predictor",
    "probe_id": "probe locator is not a predictor",
    "window_name": "window locator is not a predictor",
}
_FORBIDDEN_SUFFIXES = ("_within", "_between")


@dataclass(frozen=True)
class FeatureScheme:
    """One scientifically predeclared candidate predictor representation."""

    feature_set_id: str
    columns: tuple[str, ...]
    description: str = ""
    modality_blocks: tuple[str, ...] = ()

    def audit_dict(self) -> dict[str, object]:
        return {
            "feature_set_id": self.feature_set_id,
            "columns": list(self.columns),
            "description": self.description,
            "modality_blocks": list(self.modality_blocks),
        }


def validate_mainline_feature_scheme(scheme: FeatureScheme) -> None:
    """Fail closed when a candidate violates the current first-round contract."""
    if not scheme.feature_set_id.strip():
        raise SupervisedLearningContractError("feature_set_id must be non-empty")
    if not scheme.columns:
        raise SupervisedLearningContractError(
            f"feature scheme {scheme.feature_set_id} must declare at least one predictor"
        )
    if len(set(scheme.columns)) != len(scheme.columns):
        raise SupervisedLearningContractError(
            f"feature scheme {scheme.feature_set_id} contains duplicate columns"
        )

    for col in scheme.columns:
        if col in _MAINLINE_FORBIDDEN_COLUMNS:
            raise SupervisedLearningContractError(
                f"feature {col} is forbidden in first-round mainline: {_MAINLINE_FORBIDDEN_COLUMNS[col]}"
            )
        if col.endswith(_FORBIDDEN_SUFFIXES):
            raise SupervisedLearningContractError(
                f"feature {col} is forbidden in zero-calibration mainline: participant-specific within/between features are disabled"
            )


def feature_scheme_from_mapping(raw: Mapping[str, Any]) -> FeatureScheme:
    """Parse one configuration mapping without silently dropping fields/columns."""
    if "feature_set_id" not in raw:
        raise SupervisedLearningContractError("feature scheme requires feature_set_id")
    columns_raw = raw.get("columns")
    if not isinstance(columns_raw, Sequence) or isinstance(columns_raw, (str, bytes)):
        raise SupervisedLearningContractError("feature scheme columns must be a sequence")
    blocks_raw = raw.get("modality_blocks", ())
    if not isinstance(blocks_raw, Sequence) or isinstance(blocks_raw, (str, bytes)):
        raise SupervisedLearningContractError("modality_blocks must be a sequence")

    scheme = FeatureScheme(
        feature_set_id=str(raw["feature_set_id"]),
        columns=tuple(str(c) for c in columns_raw),
        description=str(raw.get("description", "")),
        modality_blocks=tuple(str(v) for v in blocks_raw),
    )
    validate_mainline_feature_scheme(scheme)
    return scheme


def load_feature_schemes(section: Mapping[str, Any]) -> list[FeatureScheme]:
    """Load declared candidates; an intentionally pending empty interface returns []."""
    candidates = section.get("candidates", [])
    if candidates is None:
        candidates = []
    if not isinstance(candidates, list):
        raise SupervisedLearningContractError("feature_schemes.candidates must be a list")
    schemes = [feature_scheme_from_mapping(item) for item in candidates]
    ids = [scheme.feature_set_id for scheme in schemes]
    if len(set(ids)) != len(ids):
        raise SupervisedLearningContractError("feature_set_id values must be unique")
    return schemes


def require_scheme_columns(frame: pd.DataFrame, scheme: FeatureScheme) -> None:
    """Require upstream producers/B-layer joins to supply every declared column."""
    missing = sorted(set(scheme.columns) - set(frame.columns))
    if missing:
        raise SupervisedLearningContractError(
            f"feature scheme {scheme.feature_set_id} missing upstream columns: {missing}"
        )
