"""Structural feature-scheme contracts for the current Q1 supervised core.

Scientific eligibility is frozen upstream in the feature registry. This module
only guards against structural leakage/misuse (outcomes, generated predictions,
identity/audit keys, and participant-specific zero-calibration features) and
checks that declared predictor columns are present. Scientific inclusion choices
belong to the frozen registry rather than a second Task A blacklist.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from .task import Q1_BINARY_SPEC, SupervisedLearningContractError


_STRUCTURAL_FORBIDDEN_COLUMNS: dict[str, str] = {
    Q1_BINARY_SPEC.source_column: "outcome label cannot be used as a predictor",
    "q1_binary": "derived outcome label cannot be used as a predictor",
    Q1_BINARY_SPEC.positive_probability_name: "supervised output probability cannot be recycled as a predictor",
    "predicted_q1_binary": "supervised output label cannot be recycled as a predictor",
    "participant_group_id": "participant identity is a grouping key, not a predictor",
    "repeat_participant_id": "participant identity alias is not a predictor",
    "session_id": "session locator is not a predictor",
    "block_id": "block locator is not a predictor",
    "probe_event_id": "probe locator is not a predictor",
    "probe_id": "probe locator is not a predictor",
    "probe_order_in_block": "probe locator/order is audit metadata, not a predictor",
    "probe_index_in_block": "probe locator/order is audit metadata, not a predictor",
    "probe_index_global": "probe locator/order is audit metadata, not a predictor",
    "probe_time_ms": "raw probe timestamp is audit metadata, not a predictor",
    "probe_onset_unix_ms": "raw probe timestamp is audit metadata, not a predictor",
    "window_name": "window locator is not a predictor",
    "window_start_unix_ms": "window boundary is audit metadata, not a predictor",
    "window_effective_start_unix_ms": "window boundary is audit metadata, not a predictor",
    "window_end_unix_ms": "window boundary is audit metadata, not a predictor",
    "analysis_set_id": "analysis-set identity is audit metadata, not a predictor",
    "comparison_models": "comparison-plan metadata is not a predictor",
    "run_id": "run identity is audit metadata, not a predictor",
    "model_id": "model identity is audit metadata, not a predictor",
    "outer_fold_group": "validation-fold identity is audit metadata, not a predictor",
}
_FORBIDDEN_SUFFIXES = ("_within", "_between")


@dataclass(frozen=True)
class FeatureScheme:
    """One predeclared candidate predictor representation."""

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
    """Fail closed on structural leakage or malformed candidate definitions."""
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
    normalized_blocks = tuple(str(block).strip() for block in scheme.modality_blocks)
    if any(not block for block in normalized_blocks):
        raise SupervisedLearningContractError(
            f"feature scheme {scheme.feature_set_id} contains blank modality_blocks"
        )
    if len(set(normalized_blocks)) != len(normalized_blocks):
        raise SupervisedLearningContractError(
            f"feature scheme {scheme.feature_set_id} contains duplicate modality_blocks"
        )

    for col in scheme.columns:
        if col in _STRUCTURAL_FORBIDDEN_COLUMNS:
            raise SupervisedLearningContractError(
                f"feature {col} is structurally forbidden as a predictor: {_STRUCTURAL_FORBIDDEN_COLUMNS[col]}"
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
