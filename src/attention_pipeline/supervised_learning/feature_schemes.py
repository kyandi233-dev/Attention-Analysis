"""Structural feature-scheme contracts for the current Q1 supervised core.

Scientific eligibility is frozen upstream in the feature registry. This module
only guards against structural leakage/misuse and keeps scientific modality
separate from hardware provenance. ``modality_blocks`` is retained only as an
explicit deprecated compatibility field for historical hand-written schemes;
it may contain scientific modalities, never device names.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from .task import Q1_BINARY_SPEC, SupervisedLearningContractError


ALLOWED_SCIENTIFIC_MODALITIES = frozenset(
    {"behavior", "ocular", "movement", "cardiopulmonary"}
)
ALLOWED_SENSOR_DEVICES = frozenset({"nir", "rgb", "mmwave"})

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
    "required_features": "comparison-sample metadata is not a predictor",
    "required_feature_records": "comparison-sample metadata is not a predictor",
    "run_id": "run identity is audit metadata, not a predictor",
    "model_id": "model identity is audit metadata, not a predictor",
    "outer_fold_group": "validation-fold identity is audit metadata, not a predictor",
}
_FORBIDDEN_SUFFIXES = ("_within", "_between")


def _normalize_unique(values: Sequence[object], *, field: str, scheme_id: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise SupervisedLearningContractError(f"{field} must be a sequence")
    normalized = tuple(str(value).strip() for value in values)
    if any(not value for value in normalized):
        raise SupervisedLearningContractError(
            f"feature scheme {scheme_id} contains blank {field}"
        )
    if len(set(normalized)) != len(normalized):
        raise SupervisedLearningContractError(
            f"feature scheme {scheme_id} contains duplicate {field}"
        )
    return normalized


@dataclass(frozen=True)
class FeatureScheme:
    """One predeclared candidate predictor representation.

    ``modalities`` describes scientific information content. ``required_devices``
    describes hardware needed to produce the predictors. ``modality_blocks`` is a
    deprecated compatibility input and is never populated by new registry-backed
    plans.
    """

    feature_set_id: str
    columns: tuple[str, ...]
    description: str = ""
    modalities: tuple[str, ...] = ()
    required_devices: tuple[str, ...] = ()
    modality_blocks: tuple[str, ...] = ()

    @property
    def effective_modalities(self) -> tuple[str, ...]:
        return self.modalities if self.modalities else self.modality_blocks

    @property
    def uses_deprecated_modality_blocks(self) -> bool:
        return bool(self.modality_blocks)

    def audit_dict(self) -> dict[str, object]:
        return {
            "feature_set_id": self.feature_set_id,
            "columns": list(self.columns),
            "description": self.description,
            "modalities": list(self.effective_modalities),
            "required_devices": list(self.required_devices),
            "deprecated_modality_blocks": list(self.modality_blocks),
            "uses_deprecated_modality_blocks": self.uses_deprecated_modality_blocks,
        }


def validate_mainline_feature_scheme(scheme: FeatureScheme) -> None:
    """Fail closed on structural leakage or malformed candidate definitions."""
    scheme_id = scheme.feature_set_id.strip()
    if not scheme_id:
        raise SupervisedLearningContractError("feature_set_id must be non-empty")
    if not scheme.columns:
        raise SupervisedLearningContractError(
            f"feature scheme {scheme.feature_set_id} must declare at least one predictor"
        )
    columns = _normalize_unique(scheme.columns, field="columns", scheme_id=scheme_id)

    current_modalities = _normalize_unique(
        scheme.modalities, field="modalities", scheme_id=scheme_id
    ) if scheme.modalities else ()
    legacy_modalities = _normalize_unique(
        scheme.modality_blocks, field="modality_blocks", scheme_id=scheme_id
    ) if scheme.modality_blocks else ()
    if current_modalities and legacy_modalities:
        raise SupervisedLearningContractError(
            f"feature scheme {scheme_id} cannot declare both modalities and deprecated modality_blocks"
        )
    modalities = current_modalities or legacy_modalities
    unknown_modalities = sorted(set(modalities) - ALLOWED_SCIENTIFIC_MODALITIES)
    if unknown_modalities:
        raise SupervisedLearningContractError(
            f"feature scheme {scheme_id} has non-scientific modality values {unknown_modalities}; "
            "device names must be declared in required_devices"
        )

    devices = _normalize_unique(
        scheme.required_devices, field="required_devices", scheme_id=scheme_id
    ) if scheme.required_devices else ()
    unknown_devices = sorted(set(devices) - ALLOWED_SENSOR_DEVICES)
    if unknown_devices:
        raise SupervisedLearningContractError(
            f"feature scheme {scheme_id} has unknown required_devices {unknown_devices}"
        )

    for col in columns:
        if col in _STRUCTURAL_FORBIDDEN_COLUMNS:
            raise SupervisedLearningContractError(
                f"feature {col} is structurally forbidden as a predictor: {_STRUCTURAL_FORBIDDEN_COLUMNS[col]}"
            )
        if col.endswith(_FORBIDDEN_SUFFIXES):
            raise SupervisedLearningContractError(
                f"feature {col} is forbidden in zero-calibration mainline: participant-specific within/between features are disabled"
            )


def feature_scheme_from_mapping(raw: Mapping[str, Any]) -> FeatureScheme:
    """Parse one configuration mapping without silently conflating devices/modalities."""
    if "feature_set_id" not in raw:
        raise SupervisedLearningContractError("feature scheme requires feature_set_id")
    columns_raw = raw.get("columns")
    if not isinstance(columns_raw, Sequence) or isinstance(columns_raw, (str, bytes)):
        raise SupervisedLearningContractError("feature scheme columns must be a sequence")

    modalities_raw = raw.get("modalities", ())
    legacy_blocks_raw = raw.get("modality_blocks", ())
    devices_raw = raw.get("required_devices", ())
    for field, value in (
        ("modalities", modalities_raw),
        ("modality_blocks", legacy_blocks_raw),
        ("required_devices", devices_raw),
    ):
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise SupervisedLearningContractError(f"{field} must be a sequence")

    scheme = FeatureScheme(
        feature_set_id=str(raw["feature_set_id"]),
        columns=tuple(str(c) for c in columns_raw),
        description=str(raw.get("description", "")),
        modalities=tuple(str(v) for v in modalities_raw),
        required_devices=tuple(str(v) for v in devices_raw),
        modality_blocks=tuple(str(v) for v in legacy_blocks_raw),
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
