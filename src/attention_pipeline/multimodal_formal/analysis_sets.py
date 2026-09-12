"""Comparison-specific analysis-set construction for Task B."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .alignment import KEY_COLUMNS

KEYS = list(KEY_COLUMNS)
GROUP = "participant_group_id"


def _normalize_spec(
    spec: dict[str, Any],
) -> tuple[dict[str, list[str]], list[str], list[str]]:
    required = spec.get("required_features")
    if not isinstance(required, dict) or not required:
        raise ValueError("comparison spec requires non-empty required_features mapping")
    normalized: dict[str, list[str]] = {}
    for modality, features in required.items():
        if not isinstance(features, list) or not features:
            raise ValueError(f"{modality}: required_features must be a non-empty list")
        normalized[str(modality)] = [str(x) for x in features]
    models = [str(x) for x in spec.get("models", [])]
    outcomes = [str(x) for x in spec.get("required_outcomes", [])]
    if len(models) != len(set(models)):
        raise ValueError("comparison models contain duplicates")
    if len(outcomes) != len(set(outcomes)):
        raise ValueError("required_outcomes contain duplicates")
    return normalized, models, outcomes


def _strict_bool_scalar(value: Any, *, context: str) -> bool:
    """Parse persisted booleans without Python's truthy-string semantics."""
    if pd.isna(value):
        raise ValueError(f"{context} contains a missing boolean value")
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value) in {0.0, 1.0}:
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0", "yes"}:
        return True
    if text in {"false", "0", "0.0", "no"}:
        return False
    raise ValueError(f"{context} contains invalid boolean value: {value!r}")


def _aggregate_reasons(rows: pd.DataFrame, mask_column: str) -> str:
    failed = rows[~rows[mask_column].astype(bool)]
    if failed.empty:
        return ""
    parts = []
    for _, row in failed.iterrows():
        parts.append(f"{row['modality']}:{row['feature']}:{row['missing_kind']}")
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

    The missing-aware set still requires real/readable/aligned source, explicit native
    QC and producer-level mathematical support. It only tolerates residual single-feature
    missingness that remains eligible for a later training-fold missing-data strategy.
    Structural modality absence and missing/invalid target labels are never admitted.
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
        raise ValueError("probe_feature_status contains duplicate probe/feature rows")

    status_index = probe_feature_status.set_index(status_key, verify_integrity=True)
    set_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for analysis_set_id, raw_spec in comparison_specs.items():
        required, models, outcomes = _normalize_spec(raw_spec)
        missing_outcome_columns = sorted(set(outcomes) - set(formal_identity.columns))
        if missing_outcome_columns:
            raise ValueError(
                f"analysis_set_id={analysis_set_id} missing required outcome columns: "
                f"{missing_outcome_columns}"
            )
        required_pairs = [(m, f) for m, features in required.items() for f in features]
        required_modalities = sorted(required)
        current_rows: list[dict[str, Any]] = []

        for _, identity_row in formal_identity.iterrows():
            identity_values = identity_row.to_dict()
            identity_key = tuple(identity_row[column] for column in required_identity)
            chosen: list[dict[str, Any]] = []
            for modality, feature in required_pairs:
                lookup = identity_key + (modality, feature)
                if lookup in status_index.index:
                    row = status_index.loc[lookup]
                    chosen.append(
                        {
                            "modality": modality,
                            "feature": feature,
                            "feature_computable": _strict_bool_scalar(
                                row["feature_computable"],
                                context=f"{analysis_set_id}:{modality}:{feature}:feature_computable",
                            ),
                            "eligible_for_missing_strategy": _strict_bool_scalar(
                                row["eligible_for_missing_strategy"],
                                context=(
                                    f"{analysis_set_id}:{modality}:{feature}:"
                                    "eligible_for_missing_strategy"
                                ),
                            ),
                            "missing_kind": str(row["missing_kind"]),
                        }
                    )
                else:
                    chosen.append(
                        {
                            "modality": modality,
                            "feature": feature,
                            "feature_computable": False,
                            "eligible_for_missing_strategy": False,
                            "missing_kind": "status_missing",
                        }
                    )

            selected = pd.DataFrame(chosen)
            feature_complete = bool(selected["feature_computable"].all())
            feature_missing_aware = bool(
                selected["eligible_for_missing_strategy"].all()
            )
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
                    "required_modalities": json.dumps(
                        required_modalities, ensure_ascii=False
                    ),
                    "required_outcomes": json.dumps(outcomes, ensure_ascii=False),
                    "required_feature_n": len(required_pairs),
                    "required_outcome_n": len(outcomes),
                    "outcome_valid": outcome_valid,
                    "invalid_or_missing_outcome_n": len(outcome_failures),
                    "included_complete": complete,
                    "included_missing_aware": missing_aware,
                    "missing_feature_n": int(
                        (~selected["feature_computable"]).sum()
                    ),
                    "complete_exclusion_reason": complete_reason,
                    "missing_aware_exclusion_reason": missing_aware_reason,
                    "complete_rule": (
                        "all_required_features_computable_and_required_outcomes_valid"
                    ),
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
