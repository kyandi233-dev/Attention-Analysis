"""Audit-safe output writing for Task A supervised-learning runs."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .evaluation import (
    MEMBERSHIP_COLUMN,
    EvaluationContractError,
    evaluate_prediction_archive,
    paired_log_loss_increment,
)
from .runner import SupervisedRunResult
from .task import SupervisedLearningContractError
from .trajectory import (
    build_probe_trajectory,
    session_discrimination_audit,
    summarize_session_discrimination,
)


PREDICTIONS_FILENAME = "probe_predictions.csv"
FOLD_AUDITS_FILENAME = "fold_audits.json"
FAILURES_FILENAME = "failures.csv"
PARTICIPANT_SCORES_FILENAME = "participant_log_loss.csv"
MODEL_SCORES_FILENAME = "model_evaluation.csv"
BOOTSTRAP_FILENAME = "participant_bootstrap.json"
PAIRED_PARTICIPANT_INCREMENTS_FILENAME = "paired_participant_increments.csv"
PAIRED_MODEL_INCREMENTS_FILENAME = "paired_model_increments.csv"
PAIRED_BOOTSTRAP_FILENAME = "paired_increment_bootstrap.json"
PROBE_TRAJECTORY_FILENAME = "probe_trajectory.csv"
SESSION_DISCRIMINATION_FILENAME = "session_discrimination.csv"
MANIFEST_FILENAME = "run_manifest.json"

_FAILURE_COLUMNS = (
    "run_id",
    "analysis_set_id",
    MEMBERSHIP_COLUMN,
    "model_id",
    "outer_fold_group",
    "n_outer_train_rows",
    "n_outer_test_rows",
    "reason",
)
_FOLD_KEY = ("model_id", "outer_fold_group")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NA:
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _key_set(frame: pd.DataFrame, columns: tuple[str, str]) -> set[tuple[str, str]]:
    if frame.empty:
        return set()
    return {
        (str(model_id), str(group_id))
        for model_id, group_id in frame.loc[:, list(columns)].itertuples(index=False, name=None)
    }


def _validate_prediction_contract(result: SupervisedRunResult) -> None:
    predictions = result.predictions
    required = {
        "run_id",
        "analysis_set_id",
        MEMBERSHIP_COLUMN,
        "participant_group_id",
        "model_id",
        "outer_fold_group",
        "session_id",
        "block_id",
        "probe_event_id",
        "q1_nominal_4class",
        "q1_binary",
        "feature_set_id",
        "selected_c",
        "p_q1_equals_1",
        "predicted_q1_binary",
        "model_failed",
        "failure_reason",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise SupervisedLearningContractError(
            f"prediction output missing required columns: {missing}"
        )

    n_input = result.metadata.get("n_input_rows")
    n_models = result.metadata.get("n_models")
    n_groups = result.metadata.get("n_participant_groups")
    if n_input is None or n_models is None or n_groups is None:
        raise SupervisedLearningContractError(
            "run metadata must contain n_input_rows, n_models and n_participant_groups"
        )
    expected_predictions = int(n_input) * int(n_models)
    if len(predictions) != expected_predictions:
        raise SupervisedLearningContractError(
            f"prediction row count mismatch: got {len(predictions)}, expected {expected_predictions}"
        )

    duplicate_key = ["model_id", "session_id", "block_id", "probe_event_id"]
    if predictions.duplicated(duplicate_key).any():
        raise SupervisedLearningContractError(
            f"duplicate prediction rows for key {duplicate_key}"
        )

    expected_audits = int(n_groups) * int(n_models)
    if len(result.fold_audits) != expected_audits:
        raise SupervisedLearningContractError(
            f"fold audit count mismatch: got {len(result.fold_audits)}, expected {expected_audits}"
        )
    audit_frame = pd.DataFrame(result.fold_audits)
    audit_required = {
        "run_id",
        "analysis_set_id",
        MEMBERSHIP_COLUMN,
        "model_id",
        "outer_fold_group",
        "failed",
        "reason",
    }
    missing_audit = sorted(audit_required - set(audit_frame.columns))
    if missing_audit:
        raise SupervisedLearningContractError(
            f"fold audit output missing required fields: {missing_audit}"
        )
    if audit_frame.duplicated(list(_FOLD_KEY)).any():
        raise SupervisedLearningContractError(
            f"duplicate fold audit rows for key {list(_FOLD_KEY)}"
        )

    failures = result.failures
    if failures.empty and len(failures.columns) == 0:
        failure_keys: set[tuple[str, str]] = set()
    else:
        missing_failure = sorted(set(_FAILURE_COLUMNS) - set(failures.columns))
        if missing_failure:
            raise SupervisedLearningContractError(
                f"failure table missing required columns: {missing_failure}"
            )
        if failures.duplicated(list(_FOLD_KEY)).any():
            raise SupervisedLearningContractError(
                f"duplicate failure rows for key {list(_FOLD_KEY)}"
            )
        failure_keys = _key_set(failures, _FOLD_KEY)

    failed_audit_keys = _key_set(audit_frame.loc[audit_frame["failed"].astype(bool)], _FOLD_KEY)
    failed_prediction_keys = _key_set(predictions.loc[predictions["model_failed"].astype(bool)], _FOLD_KEY)
    if failure_keys != failed_audit_keys or failure_keys != failed_prediction_keys:
        raise SupervisedLearningContractError(
            "failed-fold keys disagree across predictions, fold audits and failure table"
        )

    successful = predictions.loc[~predictions["model_failed"].astype(bool)]
    if not successful.empty:
        p = pd.to_numeric(successful["p_q1_equals_1"], errors="coerce")
        if p.isna().any() or ((p < 0.0) | (p > 1.0)).any():
            raise SupervisedLearningContractError(
                "successful prediction rows must contain finite p_q1_equals_1 within [0, 1]"
            )
        predicted = pd.to_numeric(successful["predicted_q1_binary"], errors="coerce")
        if predicted.isna().any() or not predicted.isin([0, 1]).all():
            raise SupervisedLearningContractError(
                "successful prediction rows must contain binary predicted_q1_binary"
            )
        selected_c = pd.to_numeric(successful["selected_c"], errors="coerce")
        if selected_c.isna().any() or (selected_c <= 0).any():
            raise SupervisedLearningContractError(
                "successful prediction rows must contain a finite positive selected_c"
            )
        if successful["feature_set_id"].isna().any() or successful["feature_set_id"].astype(str).str.strip().eq("").any():
            raise SupervisedLearningContractError(
                "successful prediction rows must contain a non-empty feature_set_id"
            )


def _paired_comparison_outputs(
    predictions: pd.DataFrame,
    paired_specs: object,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    """Evaluate declared comparison pairs without silently dropping invalid pairs."""
    participant_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    bootstrap_records: list[dict[str, object]] = []
    if paired_specs in (None, []):
        return (
            pd.DataFrame(
                columns=[
                    "comparison_type",
                    "feature_id",
                    "baseline_model_id",
                    "added_model_id",
                    "analysis_set_id",
                    MEMBERSHIP_COLUMN,
                    "participant_group_id",
                    "n_probes",
                    "mean_log_loss_increment",
                ]
            ),
            pd.DataFrame(
                columns=[
                    "comparison_type",
                    "feature_id",
                    "baseline_model_id",
                    "added_model_id",
                    "analysis_set_id",
                    MEMBERSHIP_COLUMN,
                    "overall_log_loss_increment",
                    "status",
                    "reason",
                ]
            ),
            [],
        )
    if not isinstance(paired_specs, list):
        raise SupervisedLearningContractError("paired_comparisons metadata must be a list")

    known_models = set(predictions["model_id"].astype(str).unique().tolist())
    for raw in paired_specs:
        if not isinstance(raw, Mapping):
            raise SupervisedLearningContractError("each paired comparison must be a mapping")
        comparison_type = str(raw.get("comparison_type", "")).strip()
        feature_id = str(raw.get("feature_id", "")).strip()
        baseline_model_id = str(raw.get("baseline_model_id", "")).strip()
        added_model_id = str(raw.get("added_model_id", "")).strip()
        if not comparison_type or not baseline_model_id or not added_model_id:
            raise SupervisedLearningContractError(
                "paired comparison requires comparison_type, baseline_model_id and added_model_id"
            )
        missing_models = sorted({baseline_model_id, added_model_id} - known_models)
        if missing_models:
            raise SupervisedLearningContractError(
                f"paired comparison references model IDs absent from OOF archive: {missing_models}"
            )
        baseline = predictions.loc[predictions["model_id"].astype(str).eq(baseline_model_id)].copy()
        added = predictions.loc[predictions["model_id"].astype(str).eq(added_model_id)].copy()
        try:
            paired = paired_log_loss_increment(baseline, added)
            participant = paired.participant_increments.copy()
            participant.insert(0, "feature_id", feature_id)
            participant.insert(0, "comparison_type", comparison_type)
            participant_frames.append(participant)
            summary_rows.append(
                {
                    "comparison_type": comparison_type,
                    "feature_id": feature_id,
                    "baseline_model_id": paired.baseline_model_id,
                    "added_model_id": paired.added_model_id,
                    "analysis_set_id": paired.analysis_set_id,
                    MEMBERSHIP_COLUMN: paired.membership_type,
                    "overall_log_loss_increment": paired.overall_increment,
                    "status": "estimable",
                    "reason": "",
                }
            )
            bootstrap_records.append(
                {
                    "comparison_type": comparison_type,
                    "feature_id": feature_id,
                    "baseline_model_id": paired.baseline_model_id,
                    "added_model_id": paired.added_model_id,
                    "analysis_set_id": paired.analysis_set_id,
                    MEMBERSHIP_COLUMN: paired.membership_type,
                    **paired.bootstrap,
                }
            )
        except EvaluationContractError as exc:
            analysis_values = predictions.loc[
                predictions["model_id"].astype(str).isin([baseline_model_id, added_model_id]),
                "analysis_set_id",
            ].dropna().astype(str).str.strip().drop_duplicates().tolist()
            membership_values = predictions.loc[
                predictions["model_id"].astype(str).isin([baseline_model_id, added_model_id]),
                MEMBERSHIP_COLUMN,
            ].dropna().astype(str).str.strip().drop_duplicates().tolist()
            summary_rows.append(
                {
                    "comparison_type": comparison_type,
                    "feature_id": feature_id,
                    "baseline_model_id": baseline_model_id,
                    "added_model_id": added_model_id,
                    "analysis_set_id": analysis_values[0] if len(analysis_values) == 1 else None,
                    MEMBERSHIP_COLUMN: membership_values[0] if len(membership_values) == 1 else None,
                    "overall_log_loss_increment": np.nan,
                    "status": "not_estimable",
                    "reason": str(exc),
                }
            )

    participant_output = (
        pd.concat(participant_frames, ignore_index=True)
        if participant_frames
        else pd.DataFrame(
            columns=[
                "comparison_type",
                "feature_id",
                "baseline_model_id",
                "added_model_id",
                "analysis_set_id",
                MEMBERSHIP_COLUMN,
                "participant_group_id",
                "n_probes",
                "mean_log_loss_increment",
            ]
        )
    )
    return participant_output, pd.DataFrame(summary_rows), bootstrap_records


def write_supervised_run(
    result: SupervisedRunResult,
    *,
    output_root: str | Path,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Write one immutable run directory and return its manifest."""
    _validate_prediction_contract(result)
    run_id = str(result.metadata.get("run_id", "")).strip()
    if not run_id:
        raise SupervisedLearningContractError(
            "result metadata must contain a non-empty run_id"
        )

    evaluation = evaluate_prediction_archive(result.predictions)
    paired_participant, paired_summary, paired_bootstrap = _paired_comparison_outputs(
        result.predictions,
        result.metadata.get("paired_comparisons"),
    )
    trajectory = build_probe_trajectory(result.predictions)
    session_discrimination = summarize_session_discrimination(result.predictions)
    session_audit = session_discrimination_audit(session_discrimination)

    run_root = Path(output_root) / run_id
    if run_root.exists():
        raise FileExistsError(
            f"run_id already exists; refusing to overwrite: {run_root}"
        )
    run_root.mkdir(parents=True, exist_ok=False)

    predictions_path = run_root / PREDICTIONS_FILENAME
    audits_path = run_root / FOLD_AUDITS_FILENAME
    failures_path = run_root / FAILURES_FILENAME
    participant_scores_path = run_root / PARTICIPANT_SCORES_FILENAME
    model_scores_path = run_root / MODEL_SCORES_FILENAME
    bootstrap_path = run_root / BOOTSTRAP_FILENAME
    paired_participant_path = run_root / PAIRED_PARTICIPANT_INCREMENTS_FILENAME
    paired_summary_path = run_root / PAIRED_MODEL_INCREMENTS_FILENAME
    paired_bootstrap_path = run_root / PAIRED_BOOTSTRAP_FILENAME
    trajectory_path = run_root / PROBE_TRAJECTORY_FILENAME
    session_discrimination_path = run_root / SESSION_DISCRIMINATION_FILENAME
    manifest_path = run_root / MANIFEST_FILENAME

    result.predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    trajectory.to_csv(trajectory_path, index=False, encoding="utf-8-sig")
    session_discrimination.to_csv(session_discrimination_path, index=False, encoding="utf-8-sig")

    failures = result.failures.copy()
    if failures.empty and len(failures.columns) == 0:
        failures = pd.DataFrame(columns=list(_FAILURE_COLUMNS))
    failures.to_csv(failures_path, index=False, encoding="utf-8-sig")

    evaluation.participant_scores.to_csv(participant_scores_path, index=False, encoding="utf-8-sig")
    evaluation.model_scores.to_csv(model_scores_path, index=False, encoding="utf-8-sig")
    bootstrap_path.write_text(
        json.dumps(evaluation.bootstrap_records, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    paired_participant.to_csv(paired_participant_path, index=False, encoding="utf-8-sig")
    paired_summary.to_csv(paired_summary_path, index=False, encoding="utf-8-sig")
    paired_bootstrap_path.write_text(
        json.dumps(paired_bootstrap, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    audits_path.write_text(
        json.dumps(result.fold_audits, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    n_failed_rows = int(result.predictions["model_failed"].astype(bool).sum())
    n_failed_folds = int(len(failures))
    n_estimable_models = int(evaluation.model_scores["status"].eq("estimable").sum())
    n_declared_comparisons = int(len(paired_summary))
    n_estimable_comparisons = int(paired_summary["status"].eq("estimable").sum()) if not paired_summary.empty else 0
    manifest: dict[str, object] = {
        **result.metadata,
        "status": "complete" if n_failed_folds == 0 else "partial_with_failures",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_prediction_rows": int(len(result.predictions)),
        "n_failed_prediction_rows": n_failed_rows,
        "n_failed_folds": n_failed_folds,
        "n_estimable_models": n_estimable_models,
        "n_declared_paired_comparisons": n_declared_comparisons,
        "n_estimable_paired_comparisons": n_estimable_comparisons,
        "outer_evaluation": {
            "primary_probability_metric": "log_loss",
            "aggregation": "participant_equal_within_participant_probe_equal",
            "pooled_probe_metric_role": "descriptive_only",
            "bootstrap_method": "fixed_oof_participant_cluster_percentile",
            "bootstrap_replicates": 1000,
            "bootstrap_seed": 20260830,
            "bootstrap_confidence_level": 0.95,
            "bootstrap_retrain_models": False,
            "paired_increment_definition": "baseline_log_loss_minus_added_log_loss",
            "positive_increment_interpretation": "added_model_has_lower_loss",
        },
        "trajectory_reporting": {
            "role": "exploratory_probe_sampled_reporting",
            "continuous_real_time_tracking_claim": False,
            "observed_q1_four_class_sequence_available": True,
            "predicted_q1_four_class_probabilities_available": False,
            "q2_context_available": "q2_ordinal_4level" in result.predictions.columns,
            "questionnaire_join_performed": False,
            **session_audit,
        },
        "outputs": {
            "probe_predictions": PREDICTIONS_FILENAME,
            "fold_audits": FOLD_AUDITS_FILENAME,
            "failures": FAILURES_FILENAME,
            "participant_log_loss": PARTICIPANT_SCORES_FILENAME,
            "model_evaluation": MODEL_SCORES_FILENAME,
            "participant_bootstrap": BOOTSTRAP_FILENAME,
            "paired_participant_increments": PAIRED_PARTICIPANT_INCREMENTS_FILENAME,
            "paired_model_increments": PAIRED_MODEL_INCREMENTS_FILENAME,
            "paired_increment_bootstrap": PAIRED_BOOTSTRAP_FILENAME,
            "probe_trajectory": PROBE_TRAJECTORY_FILENAME,
            "session_discrimination": SESSION_DISCRIMINATION_FILENAME,
            "manifest": MANIFEST_FILENAME,
        },
    }
    if provenance:
        manifest["provenance"] = dict(provenance)

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return manifest
