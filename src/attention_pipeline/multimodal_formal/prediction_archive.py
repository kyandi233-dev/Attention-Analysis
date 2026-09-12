"""Validation and export contract for one-row-per-probe supervised predictions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .alignment import KEY_COLUMNS

KEYS = list(KEY_COLUMNS)
GROUP = "participant_group_id"
REQUIRED_IDENTITY_COLUMNS = KEYS + [
    GROUP,
    "analysis_set_id",
    "outer_fold_group",
    "outcome",
    "model_id",
    "y_true",
]
PREDICTION_KEY = KEYS + [GROUP, "analysis_set_id", "outcome", "model_id"]
TASK_A_Q1_OUTCOME = "q1_equals_1_vs_2_3_4"
TASK_A_Q1_PROBABILITY_COLUMN = "p_q1_equals_1"
Q1_AUTHORITY_COLUMN = "q1_nominal_4class"
MEMBERSHIP_TYPE_COLUMN = "membership_type"
REQUIRED_TASK_A_AUDIT_COLUMNS = (
    "run_id",
    "feature_set_id",
    MEMBERSHIP_TYPE_COLUMN,
    "model_failed",
    "failure_reason",
)
_ALLOWED_MEMBERSHIP_COLUMNS = frozenset({"included_complete", "included_missing_aware"})


def _parse_json_string_list(value: object, *, context: str) -> list[str]:
    if pd.isna(value):
        return []
    raw = str(value).strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON list for {context}") from exc
    if not isinstance(parsed, list) or any(not isinstance(x, str) or not x.strip() for x in parsed):
        raise ValueError(f"{context} must be a JSON string list")
    cleaned = [x.strip() for x in parsed]
    if len(cleaned) != len(set(cleaned)):
        raise ValueError(f"duplicate values in {context}")
    return cleaned


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
        declared[str(set_id)] = _parse_json_string_list(
            values[0], context=f"comparison_models for {set_id}"
        )
    return declared


def _declared_outcomes(analysis_sets: pd.DataFrame) -> dict[str, list[str]]:
    """Map Task-B required outcomes onto archive outcome identifiers.

    The first-round supervised archive currently has one formal target: binary Q1.
    Task B records that target upstream as ``q1_nominal_4class``; the archive records
    the derived task name ``q1_equals_1_vs_2_3_4``.
    """
    if "required_outcomes" not in analysis_sets.columns:
        return {}
    declared: dict[str, list[str]] = {}
    for set_id, rows in analysis_sets.groupby("analysis_set_id", sort=False):
        values = rows["required_outcomes"].dropna().astype(str).unique().tolist()
        if not values:
            continue
        if len(values) != 1:
            raise ValueError(f"inconsistent required_outcomes within analysis_set_id={set_id}")
        raw = _parse_json_string_list(values[0], context=f"required_outcomes for {set_id}")
        mapped: list[str] = []
        for outcome in raw:
            if outcome == Q1_AUTHORITY_COLUMN:
                mapped.append(TASK_A_Q1_OUTCOME)
            else:
                mapped.append(outcome)
        declared[str(set_id)] = mapped
    return declared


def _probe_index_from_task_a(frame: pd.DataFrame) -> pd.Series:
    if "probe_index_in_block" in frame.columns:
        return pd.to_numeric(frame["probe_index_in_block"], errors="coerce").astype("Int64")
    if "probe_order_in_block" in frame.columns:
        return pd.to_numeric(frame["probe_order_in_block"], errors="coerce").astype("Int64")
    if "probe_event_id" in frame.columns:
        extracted = frame["probe_event_id"].astype("string").str.extract(r"\|probe\|(\d+)$", expand=False)
        return pd.to_numeric(extracted, errors="coerce").astype("Int64")
    return pd.Series(pd.NA, index=frame.index, dtype="Int64")


def _require_nonblank(frame: pd.DataFrame, column: str, *, mask: pd.Series | None = None) -> None:
    values = frame[column] if mask is None else frame.loc[mask, column]
    if values.isna().any() or values.astype("string").str.strip().eq("").any():
        raise ValueError(f"prediction archive requires nonblank {column}")


def normalize_task_a_predictions(
    predictions: pd.DataFrame,
    *,
    outcome: str = TASK_A_Q1_OUTCOME,
    probability_column: str = TASK_A_Q1_PROBABILITY_COLUMN,
) -> pd.DataFrame:
    """Normalize current Task-A runner output into the Task-B archive schema.

    This is a schema adapter only. It does not recompute labels/probabilities, alter folds,
    discard failed model rows, or perform any training/model selection. Formal Task-A
    audit fields are required rather than silently synthesized by the adapter.
    """
    required = {
        "session_id",
        "block_id",
        GROUP,
        "analysis_set_id",
        "outer_fold_group",
        "model_id",
        "q1_binary",
        "predicted_q1_binary",
        probability_column,
        *REQUIRED_TASK_A_AUDIT_COLUMNS,
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Task-A predictions missing columns for archive adapter: {missing}")

    out = predictions.copy()
    out["probe_index_in_block"] = _probe_index_from_task_a(out)
    if out["probe_index_in_block"].isna().any():
        raise ValueError(
            "Task-A predictions require probe_order_in_block, probe_index_in_block, "
            "or parseable probe_event_id"
        )
    out["outcome"] = str(outcome)
    out["y_true"] = pd.to_numeric(out["q1_binary"], errors="coerce").astype("Int64")
    out["y_pred"] = pd.to_numeric(out["predicted_q1_binary"], errors="coerce").astype("Int64")
    out["probability_positive"] = pd.to_numeric(out[probability_column], errors="coerce")
    return out


def _resolve_requested_sets(
    analysis_sets: pd.DataFrame,
    requested_analysis_set_ids: Sequence[str] | None,
) -> list[str]:
    available = analysis_sets["analysis_set_id"].dropna().astype(str).drop_duplicates().tolist()
    if requested_analysis_set_ids is None:
        requested = available
    else:
        requested = [str(x).strip() for x in requested_analysis_set_ids]
        if not requested or any(not x for x in requested):
            raise ValueError("requested_analysis_set_ids must contain nonblank set ids")
        if len(requested) != len(set(requested)):
            raise ValueError("requested_analysis_set_ids contains duplicates")
        unknown = sorted(set(requested) - set(available))
        if unknown:
            raise ValueError(f"requested analysis_set_id not present in analysis_sets: {unknown}")
    if not requested:
        raise ValueError("analysis_sets contains no requested analysis_set_id")
    return requested


def _validate_q1_authority_consistency(analysis_sets: pd.DataFrame) -> None:
    """Ensure copied Behavior-authoritative Q1 cannot disagree across analysis sets."""
    if Q1_AUTHORITY_COLUMN not in analysis_sets.columns:
        return
    authority = analysis_sets[KEYS + [GROUP, Q1_AUTHORITY_COLUMN]].copy()
    authority = authority.dropna(subset=[Q1_AUTHORITY_COLUMN])
    if authority.empty:
        return
    numeric = pd.to_numeric(authority[Q1_AUTHORITY_COLUMN], errors="coerce")
    if numeric.isna().any() or not numeric.isin([1, 2, 3, 4]).all():
        raise ValueError("analysis_sets contains invalid authoritative Q1 values")
    authority[Q1_AUTHORITY_COLUMN] = numeric.astype(int)
    counts = authority.groupby(KEYS + [GROUP], dropna=False)[Q1_AUTHORITY_COLUMN].nunique()
    if (counts > 1).any():
        raise ValueError("authoritative Q1 disagrees across analysis_set_id copies of the same probe")


def validate_prediction_archive(
    predictions: pd.DataFrame,
    analysis_sets: pd.DataFrame,
    *,
    membership_column: str = "included_complete",
    require_complete: bool = True,
    requested_analysis_set_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate identity, authority, LOSO ownership and expected archive coverage.

    Expected coverage is generated from the requested Task-B analysis sets, their
    declared models and their required outcomes. It is never inferred only from rows
    that happen to be present in ``predictions``.
    """
    if predictions.empty:
        raise ValueError("predictions archive is empty")
    if analysis_sets.empty:
        raise ValueError("analysis_sets is empty")
    if membership_column not in _ALLOWED_MEMBERSHIP_COLUMNS:
        raise ValueError(
            f"unsupported membership_column={membership_column!r}; expected one of {sorted(_ALLOWED_MEMBERSHIP_COLUMNS)}"
        )

    required_prediction_columns = set(
        REQUIRED_IDENTITY_COLUMNS
        + ["y_pred", "probability_positive", *REQUIRED_TASK_A_AUDIT_COLUMNS]
    )
    missing = sorted(required_prediction_columns - set(predictions.columns))
    if missing:
        raise ValueError(f"predictions missing columns: {missing}")
    if membership_column not in analysis_sets.columns:
        raise ValueError(f"analysis_sets missing membership column: {membership_column}")
    set_required = set(KEYS + [GROUP, "analysis_set_id", membership_column])
    if not set_required <= set(analysis_sets.columns):
        raise ValueError("analysis_sets missing identity/set columns")
    if predictions[REQUIRED_IDENTITY_COLUMNS].isna().any().any():
        raise ValueError("predictions contain null required identity/label values")
    if predictions.duplicated(PREDICTION_KEY).any():
        raise ValueError("duplicate per-probe prediction key")
    if analysis_sets.duplicated(KEYS + [GROUP, "analysis_set_id"]).any():
        raise ValueError("duplicate analysis-set membership rows")

    _require_nonblank(predictions, "run_id")
    _require_nonblank(predictions, MEMBERSHIP_TYPE_COLUMN)
    if not predictions["model_failed"].isin([True, False, 0, 1]).all():
        raise ValueError("prediction model_failed must be explicit boolean values")
    failed = predictions["model_failed"].astype(bool)
    successful = ~failed
    _require_nonblank(predictions, "feature_set_id", mask=successful)

    requested_sets = _resolve_requested_sets(analysis_sets, requested_analysis_set_ids)
    scoped_sets = analysis_sets[analysis_sets["analysis_set_id"].astype(str).isin(requested_sets)].copy()
    observed_sets = set(predictions["analysis_set_id"].astype(str).unique().tolist())
    unexpected_sets = sorted(observed_sets - set(requested_sets))
    if unexpected_sets:
        raise ValueError(f"predictions contain analysis_set_id outside requested scope: {unexpected_sets}")

    membership_values = predictions[MEMBERSHIP_TYPE_COLUMN].astype(str).str.strip().unique().tolist()
    if len(membership_values) != 1 or membership_values[0] != membership_column:
        raise ValueError(
            f"prediction membership_type must equal requested membership_column={membership_column}; "
            f"got {membership_values}"
        )

    if not predictions["outer_fold_group"].astype(str).eq(predictions[GROUP].astype(str)).all():
        raise ValueError("LOSO outer_fold_group must equal the held-out participant_group_id")

    y_true = pd.to_numeric(predictions["y_true"], errors="coerce")
    if not y_true.isin([0, 1]).all():
        raise ValueError("binary prediction archive requires y_true in {0,1}")

    y_pred = pd.to_numeric(predictions.loc[successful, "y_pred"], errors="coerce")
    if not y_pred.isin([0, 1]).all():
        raise ValueError("successful binary predictions require y_pred in {0,1}")
    proba = pd.to_numeric(predictions.loc[successful, "probability_positive"], errors="coerce")
    if not np.isfinite(proba).all() or not proba.between(0, 1).all():
        raise ValueError("successful probability_positive must be finite and in [0,1]")
    success_reason = predictions.loc[successful, "failure_reason"].astype("string").fillna("").str.strip()
    if success_reason.ne("").any():
        raise ValueError("successful model rows must not contain failure_reason")

    if failed.any():
        failed_y = pd.to_numeric(predictions.loc[failed, "y_pred"], errors="coerce")
        failed_p = pd.to_numeric(predictions.loc[failed, "probability_positive"], errors="coerce")
        if failed_y.notna().any() or failed_p.notna().any():
            raise ValueError("failed model rows must not contain fabricated predictions/probabilities")
        reason = predictions.loc[failed, "failure_reason"].astype("string").fillna("").str.strip()
        if reason.eq("").any():
            raise ValueError("failed model rows require failure_reason")

    declared_models = _declared_models(scoped_sets)
    declared_outcomes = _declared_outcomes(scoped_sets)
    for set_id, set_predictions in predictions.groupby("analysis_set_id", sort=False):
        models = declared_models.get(str(set_id), [])
        if models:
            observed_models = set(set_predictions["model_id"].astype(str).unique().tolist())
            undeclared = observed_models - set(models)
            if undeclared:
                raise ValueError(f"prediction model_id not declared for {set_id}: {sorted(undeclared)}")
        expected_outcomes = declared_outcomes.get(str(set_id), [])
        if expected_outcomes:
            observed_outcomes = set(set_predictions["outcome"].astype(str).unique().tolist())
            undeclared_outcomes = observed_outcomes - set(expected_outcomes)
            if undeclared_outcomes:
                raise ValueError(
                    f"prediction outcome not declared for {set_id}: {sorted(undeclared_outcomes)}"
                )

    membership_columns = KEYS + [GROUP, "analysis_set_id", membership_column]
    if Q1_AUTHORITY_COLUMN in scoped_sets.columns:
        membership_columns.append(Q1_AUTHORITY_COLUMN)
    membership = scoped_sets[membership_columns].copy()
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

    _validate_q1_authority_consistency(scoped_sets)
    if Q1_AUTHORITY_COLUMN in merged.columns:
        q1_rows = merged["outcome"].astype(str).isin({TASK_A_Q1_OUTCOME, "q1_binary"})
        if q1_rows.any():
            q1 = pd.to_numeric(merged.loc[q1_rows, Q1_AUTHORITY_COLUMN], errors="coerce")
            if q1.isna().any() or not q1.isin([1, 2, 3, 4]).all():
                raise ValueError("Q1 prediction rows lack a valid Behavior-authoritative Q1 label")
            expected_y = q1.eq(1).astype(int).reset_index(drop=True)
            observed_y = pd.to_numeric(merged.loc[q1_rows, "y_true"], errors="coerce").astype(int).reset_index(drop=True)
            if not observed_y.equals(expected_y):
                raise ValueError("prediction y_true disagrees with Behavior-authoritative Q1 label")

    expected_membership = membership[membership[membership_column].astype(bool)][
        KEYS + [GROUP, "analysis_set_id"]
    ]
    coverage_rows: list[dict[str, Any]] = []

    for set_id in requested_sets:
        expected_set = expected_membership[expected_membership["analysis_set_id"].astype(str).eq(set_id)]
        set_predictions = predictions[predictions["analysis_set_id"].astype(str).eq(set_id)]
        expected_models = declared_models.get(str(set_id), [])
        if not expected_models:
            expected_models = sorted(set_predictions["model_id"].astype(str).unique().tolist())
        expected_outcomes = declared_outcomes.get(str(set_id), [])
        if not expected_outcomes:
            expected_outcomes = sorted(set_predictions["outcome"].astype(str).unique().tolist())
        if not expected_models:
            raise ValueError(f"no expected models declared or observed for analysis_set_id={set_id}")
        if not expected_outcomes:
            raise ValueError(f"no expected outcomes declared or observed for analysis_set_id={set_id}")

        for outcome in expected_outcomes:
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
                        "failed_probe_n": int(rows["model_failed"].astype(bool).sum()),
                        "missing_probe_n": missing_n,
                        "extra_probe_n": extra_n,
                    }
                )

    return {
        "status": "PASS_PREDICTION_ARCHIVE",
        "membership_column": membership_column,
        "requested_analysis_set_ids": requested_sets,
        "run_ids": sorted(predictions["run_id"].astype(str).str.strip().unique().tolist()),
        "prediction_n": int(len(predictions)),
        "successful_prediction_n": int(successful.sum()),
        "failed_prediction_n": int(failed.sum()),
        "analysis_set_n": int(len(requested_sets)),
        "model_n": int(predictions["model_id"].nunique()),
        "outcome_n": int(predictions["outcome"].nunique()),
        "authoritative_q1_checked": bool(Q1_AUTHORITY_COLUMN in merged.columns),
        "coverage": coverage_rows,
    }


def write_prediction_archive(
    output_dir: str | Path,
    predictions: pd.DataFrame,
    analysis_sets: pd.DataFrame,
    *,
    membership_column: str = "included_complete",
    require_complete: bool = True,
    requested_analysis_set_ids: Sequence[str] | None = None,
) -> dict[str, str]:
    audit = validate_prediction_archive(
        predictions,
        analysis_sets,
        membership_column=membership_column,
        require_complete=require_complete,
        requested_analysis_set_ids=requested_analysis_set_ids,
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
