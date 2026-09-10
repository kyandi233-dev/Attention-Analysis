"""Validation and export contract for one-row-per-probe supervised predictions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .alignment import KEY_COLUMNS

KEYS = list(KEY_COLUMNS)
GROUP = "participant_group_id"
REQUIRED_PREDICTION_COLUMNS = KEYS + [
    GROUP,
    "analysis_set_id",
    "outer_fold_group",
    "outcome",
    "model_id",
    "y_true",
    "y_pred",
    "probability_positive",
]
PREDICTION_KEY = KEYS + [GROUP, "analysis_set_id", "outcome", "model_id"]


def _declared_models(analysis_sets: pd.DataFrame) -> dict[str, list[str]]:
    """Read per-set comparison models when analysis-set metadata declares them."""
    if "comparison_models" not in analysis_sets.columns:
        return {}
    declared: dict[str, list[str]] = {}
    for set_id, rows in analysis_sets.groupby("analysis_set_id", sort=False):
        values = rows["comparison_models"].dropna().astype(str).unique().tolist()
        if not values:
            continue
        if len(values) != 1:
            raise ValueError(f"inconsistent comparison_models within analysis_set_id={set_id}")
        raw = values[0]
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid comparison_models JSON for {set_id}") from exc
        if not isinstance(parsed, list) or any(not isinstance(x, str) or not x for x in parsed):
            raise ValueError(f"comparison_models must be a JSON string list for {set_id}")
        if len(parsed) != len(set(parsed)):
            raise ValueError(f"duplicate model ids in comparison_models for {set_id}")
        declared[str(set_id)] = parsed
    return declared


def validate_prediction_archive(
    predictions: pd.DataFrame,
    analysis_sets: pd.DataFrame,
    *,
    membership_column: str = "included_complete",
    require_complete: bool = True,
) -> dict[str, Any]:
    """Validate identity, LOSO ownership, probability semantics and set coverage."""
    missing = sorted(set(REQUIRED_PREDICTION_COLUMNS) - set(predictions.columns))
    if missing:
        raise ValueError(f"predictions missing columns: {missing}")
    if membership_column not in analysis_sets.columns:
        raise ValueError(f"analysis_sets missing membership column: {membership_column}")
    set_required = set(KEYS + [GROUP, "analysis_set_id", membership_column])
    if not set_required <= set(analysis_sets.columns):
        raise ValueError("analysis_sets missing identity/set columns")
    if predictions[REQUIRED_PREDICTION_COLUMNS].isna().any().any():
        raise ValueError("predictions contain null required values")
    if predictions.duplicated(PREDICTION_KEY).any():
        raise ValueError("duplicate per-probe prediction key")
    if analysis_sets.duplicated(KEYS + [GROUP, "analysis_set_id"]).any():
        raise ValueError("duplicate analysis-set membership rows")

    if not predictions["outer_fold_group"].astype(str).eq(predictions[GROUP].astype(str)).all():
        raise ValueError("LOSO outer_fold_group must equal the held-out participant_group_id")

    y_true = pd.to_numeric(predictions["y_true"], errors="coerce")
    y_pred = pd.to_numeric(predictions["y_pred"], errors="coerce")
    if not y_true.isin([0, 1]).all() or not y_pred.isin([0, 1]).all():
        raise ValueError("binary prediction archive requires y_true/y_pred in {0,1}")
    proba = pd.to_numeric(predictions["probability_positive"], errors="coerce")
    if not np.isfinite(proba).all() or not proba.between(0, 1).all():
        raise ValueError("probability_positive must be finite and in [0,1]")

    declared = _declared_models(analysis_sets)
    for set_id, models in declared.items():
        if not models:
            continue
        observed_models = set(
            predictions.loc[predictions["analysis_set_id"].astype(str).eq(set_id), "model_id"]
            .astype(str)
            .unique()
        )
        undeclared = observed_models - set(models)
        if undeclared:
            raise ValueError(
                f"prediction model_id not declared for {set_id}: {sorted(undeclared)}"
            )

    membership = analysis_sets[KEYS + [GROUP, "analysis_set_id", membership_column]].copy()
    merged = predictions.merge(
        membership,
        on=KEYS + [GROUP, "analysis_set_id"],
        how="left",
        validate="many_to_one",
    )
    if merged[membership_column].isna().any():
        raise ValueError("prediction references unknown analysis_set/probe membership")
    if not merged[membership_column].astype(bool).all():
        raise ValueError("prediction emitted for probe outside requested analysis-set membership")

    expected = membership[membership[membership_column].astype(bool)][
        KEYS + [GROUP, "analysis_set_id"]
    ]
    coverage_rows: list[dict[str, Any]] = []

    # Outcomes are observed archive dimensions; within each outcome, every model declared
    # by the comparison contract must cover the same requested analysis set. This catches
    # an entire missing model, not only missing individual probe rows.
    for set_id, set_predictions in predictions.groupby("analysis_set_id", sort=False):
        expected_set = expected[expected["analysis_set_id"].eq(set_id)]
        outcomes = set_predictions["outcome"].astype(str).unique().tolist()
        expected_models = declared.get(str(set_id)) or sorted(
            set_predictions["model_id"].astype(str).unique().tolist()
        )
        for outcome in outcomes:
            outcome_rows = set_predictions[set_predictions["outcome"].astype(str).eq(outcome)]
            for model_id in expected_models:
                rows = outcome_rows[outcome_rows["model_id"].astype(str).eq(model_id)]
                actual = rows[KEYS + [GROUP, "analysis_set_id"]]
                joined = expected_set.merge(
                    actual,
                    on=KEYS + [GROUP, "analysis_set_id"],
                    how="outer",
                    indicator=True,
                )
                missing_n = int(joined["_merge"].eq("left_only").sum())
                extra_n = int(joined["_merge"].eq("right_only").sum())
                if require_complete and (missing_n or extra_n):
                    raise ValueError(
                        f"incomplete prediction coverage for {set_id}/{outcome}/{model_id}: "
                        f"missing={missing_n}, extra={extra_n}"
                    )
                coverage_rows.append(
                    {
                        "analysis_set_id": set_id,
                        "outcome": outcome,
                        "model_id": model_id,
                        "expected_probe_n": int(len(expected_set)),
                        "predicted_probe_n": int(len(actual)),
                        "missing_probe_n": missing_n,
                        "extra_probe_n": extra_n,
                    }
                )

    return {
        "status": "PASS_PREDICTION_ARCHIVE",
        "membership_column": membership_column,
        "prediction_n": int(len(predictions)),
        "analysis_set_n": int(predictions["analysis_set_id"].nunique()),
        "model_n": int(predictions["model_id"].nunique()),
        "outcome_n": int(predictions["outcome"].nunique()),
        "coverage": coverage_rows,
    }


def write_prediction_archive(
    output_dir: str | Path,
    predictions: pd.DataFrame,
    analysis_sets: pd.DataFrame,
    *,
    membership_column: str = "included_complete",
    require_complete: bool = True,
) -> dict[str, str]:
    audit = validate_prediction_archive(
        predictions,
        analysis_sets,
        membership_column=membership_column,
        require_complete=require_complete,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "probe_predictions.csv"
    audit_path = output / "prediction_archive_audit.json"
    predictions.sort_values(PREDICTION_KEY).to_csv(
        archive_path, index=False, encoding="utf-8-sig"
    )
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "probe_predictions": str(archive_path),
        "prediction_archive_audit": str(audit_path),
    }
