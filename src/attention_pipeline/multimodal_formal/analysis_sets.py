"""Comparison-specific analysis-set construction for Task B.

``required_features`` is the scientific-modality -> predictor-column sample
contract consumed by Task A. Task B source lookup is separate: new comparison
specs may provide explicit per-predictor ``required_feature_records`` containing
feature identity, scientific modality, source namespace, and predictor column.
This prevents RGB/NIR/mmWave producer namespaces from being mistaken for
scientific modalities.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .alignment import KEY_COLUMNS

KEYS = list(KEY_COLUMNS)
GROUP = "participant_group_id"


def _strict_bool_scalar(value: Any, *, field: str) -> bool:
    """Parse serialized Task-B booleans without Python truthiness shortcuts."""
    if pd.isna(value):
        raise ValueError(f"{field} contains missing boolean state")
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value) in (0.0, 1.0):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0", "yes"}:
        return True
    if text in {"false", "0", "0.0", "no"}:
        return False
    raise ValueError(f"{field} contains invalid boolean state: {value!r}")


def _normalize_required_features(raw: Any) -> dict[str, list[str]]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError("comparison spec requires non-empty required_features mapping")
    normalized: dict[str, list[str]] = {}
    seen_columns: set[str] = set()
    for modality, features in raw.items():
        scientific_modality = str(modality).strip()
        if not scientific_modality or not isinstance(features, list) or not features:
            raise ValueError(
                f"{scientific_modality or modality}: required_features must be a non-empty list"
            )
        columns = [str(value).strip() for value in features]
        if any(not value for value in columns) or len(columns) != len(set(columns)):
            raise ValueError(
                f"{scientific_modality}: required_features contains blank or duplicate predictor columns"
            )
        overlap = seen_columns & set(columns)
        if overlap:
            raise ValueError(
                f"required_features repeats predictor columns across scientific modalities: {sorted(overlap)}"
            )
        seen_columns.update(columns)
        normalized[scientific_modality] = columns
    return normalized


def _legacy_feature_records(required: dict[str, list[str]]) -> list[dict[str, str]]:
    """Explicitly label the historical source-key contract; never reinterpret devices."""
    return [
        {
            "feature_id": f"legacy::{modality}::{column}",
            "scientific_modality": modality,
            "source_namespace": modality,
            "predictor_column": column,
            "identity_mode": "legacy_source_feature",
        }
        for modality, columns in required.items()
        for column in columns
    ]


def _normalize_feature_records(
    raw: Any,
    required: dict[str, list[str]],
) -> tuple[list[dict[str, str]], str]:
    if raw is None:
        return _legacy_feature_records(required), "legacy_source_feature"
    if not isinstance(raw, list) or not raw:
        raise ValueError("required_feature_records must be a non-empty list when declared")

    records: list[dict[str, str]] = []
    seen_keys: set[tuple[str, str]] = set()
    seen_columns: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("required_feature_records entries must be mappings")
        record = {
            "feature_id": str(item.get("feature_id", "")).strip(),
            "scientific_modality": str(item.get("scientific_modality", "")).strip(),
            "source_namespace": str(item.get("source_namespace", "")).strip(),
            "predictor_column": str(item.get("predictor_column", "")).strip(),
            "identity_mode": "explicit_per_feature",
        }
        missing = [
            key
            for key in ("feature_id", "scientific_modality", "source_namespace", "predictor_column")
            if not record[key]
        ]
        if missing:
            raise ValueError(f"required_feature_records entry has blank fields: {missing}")
        key = (record["feature_id"], record["predictor_column"])
        if key in seen_keys:
            raise ValueError(f"duplicate feature identity record: {key}")
        seen_keys.add(key)
        if record["predictor_column"] in seen_columns:
            raise ValueError(
                f"predictor column {record['predictor_column']} is assigned to multiple feature identity records"
            )
        seen_columns.add(record["predictor_column"])
        expected_columns = required.get(record["scientific_modality"])
        if expected_columns is None or record["predictor_column"] not in expected_columns:
            raise ValueError(
                "required_feature_records scientific_modality/predictor_column does not match required_features: "
                f"{record['scientific_modality']}:{record['predictor_column']}"
            )
        records.append(record)

    required_columns = {
        column for columns in required.values() for column in columns
    }
    if seen_columns != required_columns:
        raise ValueError(
            "required_feature_records must cover exactly the required_features predictor union; "
            f"missing={sorted(required_columns - seen_columns)}, extra={sorted(seen_columns - required_columns)}"
        )
    return records, "explicit_per_feature"


def _normalize_spec(
    spec: dict[str, Any],
) -> tuple[dict[str, list[str]], list[dict[str, str]], str, list[str], list[str]]:
    required = _normalize_required_features(spec.get("required_features"))
    feature_records, identity_mode = _normalize_feature_records(
        spec.get("required_feature_records"), required
    )
    models = [str(x) for x in spec.get("models", [])]
    outcomes = [str(x) for x in spec.get("required_outcomes", [])]
    if len(models) != len(set(models)):
        raise ValueError("comparison models contain duplicates")
    if len(outcomes) != len(set(outcomes)):
        raise ValueError("required_outcomes contain duplicates")
    return required, feature_records, identity_mode, models, outcomes


def _aggregate_reasons(rows: pd.DataFrame, mask_column: str) -> str:
    failed = rows[~rows[mask_column].astype(bool)]
    if failed.empty:
        return ""
    parts = []
    for _, row in failed.iterrows():
        parts.append(
            f"{row['scientific_modality']}:{row['feature_id']}:{row['predictor_column']}:{row['missing_kind']}"
        )
    return "|".join(sorted(set(parts)))


def _outcome_valid(value: Any, column: str) -> bool:
    """Validate only frozen target coding; targets are never imputed by Task B."""
    if pd.isna(value):
        return False
    if column in {"q1_nominal_4class", "q2_ordinal_4level"}:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return bool(np.isfinite(numeric) and numeric in {1, 2, 3, 4})
    return True


def build_analysis_sets(
    formal_identity: pd.DataFrame,
    probe_feature_status: pd.DataFrame,
    comparison_specs: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build strict-complete and missing-aware sets per requested model comparison.

    The status table's historical ``modality`` column is treated only as a producer
    source namespace. New formal specs must use ``required_feature_records`` to map
    each scientific predictor to that source explicitly. Legacy specs remain
    executable but are labelled ``legacy_source_feature`` rather than silently
    reclassified as scientific modalities.
    """
    required_identity = KEYS + [GROUP]
    if not set(required_identity) <= set(formal_identity.columns):
        raise ValueError("formal_identity missing required keys")
    if formal_identity[required_identity].isna().any().any():
        raise ValueError("formal_identity contains null keys")
    if formal_identity.duplicated(KEYS).any():
        raise ValueError("formal_identity contains duplicate probe keys")

    status_required = set(
        required_identity
        + [
            "modality",
            "feature",
            "feature_computable",
            "eligible_for_missing_strategy",
            "missing_kind",
        ]
    )
    if not status_required <= set(probe_feature_status.columns):
        missing = sorted(status_required - set(probe_feature_status.columns))
        raise ValueError(f"probe_feature_status missing columns: {missing}")
    status_key = required_identity + ["modality", "feature"]
    if probe_feature_status.duplicated(status_key).any():
        raise ValueError("probe_feature_status contains duplicate probe/source-namespace/predictor rows")

    status_index = probe_feature_status.set_index(status_key, verify_integrity=True)
    set_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for analysis_set_id, raw_spec in comparison_specs.items():
        required, feature_records, identity_mode, models, outcomes = _normalize_spec(raw_spec)
        missing_outcome_columns = sorted(set(outcomes) - set(formal_identity.columns))
        if missing_outcome_columns:
            raise ValueError(
                f"analysis_set_id={analysis_set_id} missing required outcome columns: "
                f"{missing_outcome_columns}"
            )
        required_modalities = sorted(required)
        required_features_json = json.dumps(required, ensure_ascii=False, sort_keys=True)
        required_feature_records_json = json.dumps(
            feature_records, ensure_ascii=False, sort_keys=True
        )
        current_rows: list[dict[str, Any]] = []

        for _, identity_row in formal_identity.iterrows():
            identity_values = identity_row.to_dict()
            identity_key = tuple(identity_row[column] for column in required_identity)
            chosen: list[dict[str, Any]] = []
            for feature_record in feature_records:
                source_namespace = feature_record["source_namespace"]
                predictor_column = feature_record["predictor_column"]
                lookup = identity_key + (source_namespace, predictor_column)
                if lookup in status_index.index:
                    row = status_index.loc[lookup]
                    chosen.append(
                        {
                            **feature_record,
                            "feature_computable": _strict_bool_scalar(
                                row["feature_computable"], field="feature_computable"
                            ),
                            "eligible_for_missing_strategy": _strict_bool_scalar(
                                row["eligible_for_missing_strategy"],
                                field="eligible_for_missing_strategy",
                            ),
                            "missing_kind": str(row["missing_kind"]),
                        }
                    )
                else:
                    chosen.append(
                        {
                            **feature_record,
                            "feature_computable": False,
                            "eligible_for_missing_strategy": False,
                            "missing_kind": "status_missing",
                        }
                    )

            selected = pd.DataFrame(chosen)
            feature_complete = bool(selected["feature_computable"].all())
            feature_missing_aware = bool(selected["eligible_for_missing_strategy"].all())
            outcome_failures = [
                outcome
                for outcome in outcomes
                if not _outcome_valid(identity_row[outcome], outcome)
            ]
            outcome_valid = not outcome_failures
            complete = feature_complete and outcome_valid
            missing_aware = feature_missing_aware and outcome_valid

            complete_reason = _aggregate_reasons(selected, "feature_computable")
            missing_aware_reason = _aggregate_reasons(
                selected, "eligible_for_missing_strategy"
            )
            if outcome_failures:
                target_reason = "|".join(
                    f"outcome:{name}:missing_or_invalid" for name in outcome_failures
                )
                complete_reason = "|".join(
                    part for part in (complete_reason, target_reason) if part
                )
                missing_aware_reason = "|".join(
                    part for part in (missing_aware_reason, target_reason) if part
                )

            record = dict(identity_values)
            record.update(
                {
                    "analysis_set_id": str(analysis_set_id),
                    "comparison_models": json.dumps(models, ensure_ascii=False),
                    "required_features": required_features_json,
                    "required_feature_records": required_feature_records_json,
                    "feature_identity_mode": identity_mode,
                    "required_modalities": json.dumps(required_modalities, ensure_ascii=False),
                    "required_outcomes": json.dumps(outcomes, ensure_ascii=False),
                    "required_feature_n": len(feature_records),
                    "required_outcome_n": len(outcomes),
                    "outcome_valid": outcome_valid,
                    "invalid_or_missing_outcome_n": len(outcome_failures),
                    "included_complete": complete,
                    "included_missing_aware": missing_aware,
                    "missing_feature_n": int((~selected["feature_computable"]).sum()),
                    "complete_exclusion_reason": complete_reason,
                    "missing_aware_exclusion_reason": missing_aware_reason,
                    "complete_rule": "all_required_features_computable_and_required_outcomes_valid",
                    "missing_aware_rule": (
                        "all_required_features_have_valid_measurement_opportunity_qc_"
                        "and_support_plus_required_outcomes_valid"
                    ),
                }
            )
            current_rows.append(record)

        current = pd.DataFrame(current_rows)
        set_rows.extend(current_rows)
        for membership in ("included_complete", "included_missing_aware"):
            chosen_rows = current[current[membership]]
            summary_rows.append(
                {
                    "analysis_set_id": str(analysis_set_id),
                    "membership": membership,
                    "probe_n": int(len(chosen_rows)),
                    "session_n": int(chosen_rows["session_id"].nunique()),
                    "participant_group_n": int(chosen_rows[GROUP].nunique()),
                    "required_features": required_features_json,
                    "required_feature_records": required_feature_records_json,
                    "feature_identity_mode": identity_mode,
                    "required_outcomes": json.dumps(outcomes, ensure_ascii=False),
                }
            )

    return pd.DataFrame(set_rows), pd.DataFrame(summary_rows)


def write_analysis_sets(
    output_dir: str | Path,
    analysis_sets: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "analysis_sets": str(output / "analysis_sets.csv"),
        "analysis_set_summary": str(output / "analysis_set_summary.csv"),
    }
    analysis_sets.to_csv(paths["analysis_sets"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["analysis_set_summary"], index=False, encoding="utf-8-sig")
    return paths
