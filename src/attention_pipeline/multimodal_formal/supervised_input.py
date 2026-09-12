"""Materialize one comparison-specific Task-A input table from Task-B audit outputs.

This layer performs no imputation, scaling, feature selection, or model fitting. It
only selects an already-defined analysis-set membership and pivots the exact audited
feature values into a one-row-per-probe wide table for the supervised-learning core.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .alignment import KEY_COLUMNS


KEYS = list(KEY_COLUMNS)
GROUP = "participant_group_id"
ALLOWED_MEMBERSHIP_TYPES = frozenset({"included_complete", "included_missing_aware"})
_OPTIONAL_PROBE_METADATA = (
    "probe_event_id",
    "probe_order_in_block",
    "probe_index_global",
    "probe_time_ms",
    "probe_onset_unix_ms",
    "q1_nominal_4class",
    "q2_ordinal_4level",
    "window_name",
)
_TASK_A_REQUIRED_METADATA = ("probe_event_id", "q1_nominal_4class")
_NUMERIC_METADATA = {
    "probe_order_in_block",
    "probe_index_global",
    "probe_time_ms",
    "probe_onset_unix_ms",
    "q1_nominal_4class",
    "q2_ordinal_4level",
}


class SupervisedInputMaterializationError(ValueError):
    """Raised when Task-B outputs cannot be materialized without ambiguity."""


def _single_nonblank_value(frame: pd.DataFrame, column: str, *, context: str) -> str:
    if column not in frame.columns:
        raise SupervisedInputMaterializationError(f"{context} missing required column: {column}")
    raw = frame[column]
    if raw.isna().any():
        raise SupervisedInputMaterializationError(f"{context} {column} contains missing values")
    values = raw.astype(str).str.strip()
    if values.eq("").any():
        raise SupervisedInputMaterializationError(f"{context} {column} contains blank values")
    unique = values.drop_duplicates().tolist()
    if len(unique) != 1:
        raise SupervisedInputMaterializationError(
            f"{context} requires one {column}; got {unique}"
        )
    return str(unique[0])


def _strict_bool(values: pd.Series, *, context: str) -> pd.Series:
    if values.isna().any():
        raise SupervisedInputMaterializationError(f"{context} contains missing boolean values")
    normalized = values.astype("string").str.strip().str.lower()
    true_values = {"true", "1", "1.0", "yes"}
    false_values = {"false", "0", "0.0", "no"}
    valid = normalized.isin(true_values | false_values)
    if not valid.all():
        bad = sorted(normalized.loc[~valid].dropna().unique().tolist())
        raise SupervisedInputMaterializationError(
            f"{context} contains invalid boolean values: {bad}"
        )
    return normalized.isin(true_values)


def _parse_string_list(value: object, *, context: str) -> list[str]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise SupervisedInputMaterializationError(f"{context} is not valid JSON") from exc
    if not isinstance(parsed, list) or not parsed:
        raise SupervisedInputMaterializationError(f"{context} must be a non-empty JSON list")
    cleaned = [str(item).strip() for item in parsed]
    if any(not item for item in cleaned) or len(cleaned) != len(set(cleaned)):
        raise SupervisedInputMaterializationError(
            f"{context} contains blank or duplicate values"
        )
    return cleaned


def _parse_required_features(value: object) -> dict[str, list[str]]:
    if pd.isna(value):
        raise SupervisedInputMaterializationError("required_features is missing")
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise SupervisedInputMaterializationError("required_features is not valid JSON") from exc
    if not isinstance(parsed, Mapping) or not parsed:
        raise SupervisedInputMaterializationError("required_features must be a non-empty mapping")

    normalized: dict[str, list[str]] = {}
    seen_feature_names: set[str] = set()
    for raw_modality, raw_features in parsed.items():
        modality = str(raw_modality).strip()
        if not modality or not isinstance(raw_features, list) or not raw_features:
            raise SupervisedInputMaterializationError(
                "required_features must map nonblank modality names to non-empty lists"
            )
        features = [str(feature).strip() for feature in raw_features]
        if any(not feature for feature in features) or len(features) != len(set(features)):
            raise SupervisedInputMaterializationError(
                f"required_features[{modality}] contains blank or duplicate feature names"
            )
        overlap = seen_feature_names & set(features)
        if overlap:
            raise SupervisedInputMaterializationError(
                "Task-A wide feature columns must be unique across modalities; "
                f"duplicate names: {sorted(overlap)}"
            )
        seen_feature_names.update(features)
        normalized[modality] = features
    return normalized


def _validate_probe_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    required = KEYS + [GROUP]
    missing = sorted(set(required) - set(metadata.columns))
    if missing:
        raise SupervisedInputMaterializationError(
            f"probe_metadata missing canonical keys: {missing}"
        )
    if metadata[required].isna().any().any():
        raise SupervisedInputMaterializationError("probe_metadata contains null canonical keys")
    if metadata.duplicated(required).any():
        raise SupervisedInputMaterializationError("probe_metadata contains duplicate canonical probe keys")
    keep = required + [column for column in _OPTIONAL_PROBE_METADATA if column in metadata.columns]
    return metadata[keep].copy()


def _merge_probe_metadata(base: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    key = KEYS + [GROUP]
    overlapping = [
        column for column in _OPTIONAL_PROBE_METADATA if column in base.columns and column in metadata.columns
    ]
    metadata_columns = [column for column in _OPTIONAL_PROBE_METADATA if column in metadata.columns]
    authority = metadata[key + metadata_columns].copy()
    renamed = {column: f"__authority_{column}" for column in overlapping}
    authority = authority.rename(columns=renamed)
    merged = base.merge(authority, on=key, how="left", validate="one_to_one")

    for column in overlapping:
        authority_column = f"__authority_{column}"
        left = merged[column]
        right = merged[authority_column]
        comparable = left.notna() & right.notna()
        if column in _NUMERIC_METADATA:
            left_value = pd.to_numeric(left, errors="coerce")
            right_value = pd.to_numeric(right, errors="coerce")
            mismatch = comparable & (
                left_value.isna()
                | right_value.isna()
                | ~np.isclose(left_value, right_value, rtol=0.0, atol=1e-6, equal_nan=False)
            )
        else:
            left_value = left.astype("string").str.strip()
            right_value = right.astype("string").str.strip()
            mismatch = comparable & left_value.ne(right_value)
        if mismatch.any():
            raise SupervisedInputMaterializationError(
                f"probe_metadata {column} disagrees with analysis-set authority"
            )
        merged = merged.drop(columns=[authority_column])

    return merged


def materialize_supervised_input(
    analysis_sets: pd.DataFrame,
    probe_feature_status: pd.DataFrame,
    *,
    analysis_set_id: str,
    membership_type: str,
    probe_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the exact wide probe table consumed by one Task-A run.

    ``included_complete`` requires every audited feature value to be finite.
    ``included_missing_aware`` may retain only residual single-feature missing values
    that Task B has already marked eligible for a training-fold missing-data strategy.
    The Task-B ``required_outcomes`` declaration is preserved unchanged so Task A
    can verify that sample eligibility was defined by its own authoritative target.
    """
    set_id = str(analysis_set_id).strip()
    membership = str(membership_type).strip()
    if not set_id:
        raise SupervisedInputMaterializationError("analysis_set_id must be nonblank")
    if membership not in ALLOWED_MEMBERSHIP_TYPES:
        raise SupervisedInputMaterializationError(
            f"unsupported membership_type={membership!r}; expected one of {sorted(ALLOWED_MEMBERSHIP_TYPES)}"
        )
    if analysis_sets.empty:
        raise SupervisedInputMaterializationError("analysis_sets is empty")
    if probe_feature_status.empty:
        raise SupervisedInputMaterializationError("probe_feature_status is empty")

    set_required = set(
        KEYS
        + [
            GROUP,
            "analysis_set_id",
            membership,
            "comparison_models",
            "required_features",
            "required_outcomes",
        ]
    )
    missing_set = sorted(set_required - set(analysis_sets.columns))
    if missing_set:
        raise SupervisedInputMaterializationError(
            f"analysis_sets missing required columns: {missing_set}"
        )
    status_required = set(
        KEYS
        + [
            GROUP,
            "modality",
            "feature",
            "value",
            "feature_computable",
            "eligible_for_missing_strategy",
            "missing_kind",
        ]
    )
    missing_status = sorted(status_required - set(probe_feature_status.columns))
    if missing_status:
        raise SupervisedInputMaterializationError(
            f"probe_feature_status missing required columns: {missing_status}"
        )

    scoped = analysis_sets[analysis_sets["analysis_set_id"].astype(str).str.strip().eq(set_id)].copy()
    if scoped.empty:
        raise SupervisedInputMaterializationError(
            f"analysis_set_id={set_id!r} is not present in analysis_sets"
        )
    if scoped.duplicated(KEYS + [GROUP]).any():
        raise SupervisedInputMaterializationError(
            f"analysis_set_id={set_id!r} contains duplicate probe membership rows"
        )

    comparison_models = _single_nonblank_value(
        scoped, "comparison_models", context=f"analysis_set_id={set_id}"
    )
    _parse_string_list(comparison_models, context=f"comparison_models for {set_id}")
    required_features_raw = _single_nonblank_value(
        scoped, "required_features", context=f"analysis_set_id={set_id}"
    )
    required_features = _parse_required_features(required_features_raw)
    required_outcomes_raw = _single_nonblank_value(
        scoped, "required_outcomes", context=f"analysis_set_id={set_id}"
    )
    _parse_string_list(required_outcomes_raw, context=f"required_outcomes for {set_id}")

    membership_mask = _strict_bool(
        scoped[membership], context=f"analysis_set_id={set_id} {membership}"
    )
    members = scoped.loc[membership_mask].copy()
    if members.empty:
        raise SupervisedInputMaterializationError(
            f"analysis_set_id={set_id!r} has no probes in membership_type={membership}"
        )

    base_columns = KEYS + [GROUP, "analysis_set_id"]
    base_columns += [column for column in _OPTIONAL_PROBE_METADATA if column in members.columns]
    base = members[base_columns].copy()
    base["membership_type"] = membership
    base["comparison_models"] = comparison_models
    base["required_features"] = required_features_raw
    base["required_outcomes"] = required_outcomes_raw

    if probe_metadata is not None:
        metadata = _validate_probe_metadata(probe_metadata)
        base = _merge_probe_metadata(base, metadata)

    missing_task_a = sorted(set(_TASK_A_REQUIRED_METADATA) - set(base.columns))
    if missing_task_a:
        raise SupervisedInputMaterializationError(
            f"materialized Task-A input missing required metadata: {missing_task_a}"
        )
    if base[list(_TASK_A_REQUIRED_METADATA)].isna().any().any():
        raise SupervisedInputMaterializationError(
            "materialized Task-A input contains missing probe_event_id or q1_nominal_4class"
        )
    if base["probe_event_id"].astype("string").str.strip().eq("").any():
        raise SupervisedInputMaterializationError(
            "materialized Task-A input contains blank probe_event_id"
        )
    q1 = pd.to_numeric(base["q1_nominal_4class"], errors="coerce")
    if q1.isna().any() or not q1.isin([1, 2, 3, 4]).all():
        raise SupervisedInputMaterializationError(
            "materialized Task-A input requires q1_nominal_4class in {1,2,3,4}"
        )

    status_key = KEYS + [GROUP, "modality", "feature"]
    if probe_feature_status.duplicated(status_key).any():
        raise SupervisedInputMaterializationError(
            "probe_feature_status contains duplicate probe/modality/feature rows"
        )

    for modality, features in required_features.items():
        for feature in features:
            status = probe_feature_status[
                probe_feature_status["modality"].astype(str).str.strip().eq(modality)
                & probe_feature_status["feature"].astype(str).str.strip().eq(feature)
            ][
                KEYS
                + [
                    GROUP,
                    "value",
                    "feature_computable",
                    "eligible_for_missing_strategy",
                    "missing_kind",
                ]
            ].copy()
            if status.empty:
                raise SupervisedInputMaterializationError(
                    f"required feature has no Task-B status rows: {modality}:{feature}"
                )
            status = status.rename(
                columns={
                    "value": feature,
                    "feature_computable": f"__{feature}_computable",
                    "eligible_for_missing_strategy": f"__{feature}_missing_eligible",
                    "missing_kind": f"__{feature}_missing_kind",
                }
            )
            base = base.merge(status, on=KEYS + [GROUP], how="left", validate="one_to_one")

            helper_columns = [
                f"__{feature}_computable",
                f"__{feature}_missing_eligible",
                f"__{feature}_missing_kind",
            ]
            if base[helper_columns].isna().any().any():
                raise SupervisedInputMaterializationError(
                    f"selected membership lacks Task-B status rows for required feature {modality}:{feature}"
                )

            numeric = pd.to_numeric(base[feature], errors="coerce")
            finite = pd.Series(np.isfinite(numeric), index=base.index)
            if membership == "included_complete":
                if not finite.all():
                    raise SupervisedInputMaterializationError(
                        f"included_complete contains non-finite required feature {modality}:{feature}"
                    )
            else:
                missing_value = ~finite
                if missing_value.any():
                    eligible = _strict_bool(
                        base[f"__{feature}_missing_eligible"],
                        context=f"{modality}:{feature} eligible_for_missing_strategy",
                    )
                    kind = base[f"__{feature}_missing_kind"].astype(str).str.strip()
                    illegal = missing_value & (~eligible | kind.ne("single_feature_missing"))
                    if illegal.any():
                        raise SupervisedInputMaterializationError(
                            "included_missing_aware contains a missing value outside residual "
                            f"single-feature missingness for {modality}:{feature}"
                        )
            base[feature] = numeric
            base = base.drop(columns=helper_columns)

    sort_columns = [GROUP, "session_id", "block_id", "probe_index_in_block"]
    return base.sort_values(sort_columns, kind="stable").reset_index(drop=True)
