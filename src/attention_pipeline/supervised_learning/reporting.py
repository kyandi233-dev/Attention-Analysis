"""Audit-safe output writing for Task A supervised-learning runs."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .runner import SupervisedRunResult
from .task import SupervisedLearningContractError


PREDICTIONS_FILENAME = "probe_predictions.csv"
FOLD_AUDITS_FILENAME = "fold_audits.json"
FAILURES_FILENAME = "failures.csv"
MANIFEST_FILENAME = "run_manifest.json"

_FAILURE_COLUMNS = (
    "run_id",
    "analysis_set_id",
    "model_id",
    "outer_fold_group",
    "n_outer_train_rows",
    "n_outer_test_rows",
    "reason",
)


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


def _validate_prediction_contract(result: SupervisedRunResult) -> None:
    predictions = result.predictions
    required = {
        "run_id",
        "model_id",
        "outer_fold_group",
        "session_id",
        "block_id",
        "probe_event_id",
        "q1_nominal_4class",
        "q1_binary",
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
    if n_input is None or n_models is None:
        raise SupervisedLearningContractError(
            "run metadata must contain n_input_rows and n_models"
        )
    expected = int(n_input) * int(n_models)
    if len(predictions) != expected:
        raise SupervisedLearningContractError(
            f"prediction row count mismatch: got {len(predictions)}, expected {expected}"
        )

    duplicate_key = ["model_id", "session_id", "block_id", "probe_event_id"]
    if predictions.duplicated(duplicate_key).any():
        raise SupervisedLearningContractError(
            f"duplicate prediction rows for key {duplicate_key}"
        )


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

    run_root = Path(output_root) / run_id
    if run_root.exists():
        raise FileExistsError(
            f"run_id already exists; refusing to overwrite: {run_root}"
        )
    run_root.mkdir(parents=True, exist_ok=False)

    predictions_path = run_root / PREDICTIONS_FILENAME
    audits_path = run_root / FOLD_AUDITS_FILENAME
    failures_path = run_root / FAILURES_FILENAME
    manifest_path = run_root / MANIFEST_FILENAME

    result.predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    failures = result.failures.copy()
    if failures.empty and len(failures.columns) == 0:
        failures = pd.DataFrame(columns=list(_FAILURE_COLUMNS))
    failures.to_csv(failures_path, index=False, encoding="utf-8-sig")

    audits_path.write_text(
        json.dumps(
            result.fold_audits,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )

    n_failed_rows = int(result.predictions["model_failed"].astype(bool).sum())
    n_failed_folds = int(len(failures))
    manifest: dict[str, object] = {
        **result.metadata,
        "status": "complete" if n_failed_folds == 0 else "partial_with_failures",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_prediction_rows": int(len(result.predictions)),
        "n_failed_prediction_rows": n_failed_rows,
        "n_failed_folds": n_failed_folds,
        "outputs": {
            "probe_predictions": PREDICTIONS_FILENAME,
            "fold_audits": FOLD_AUDITS_FILENAME,
            "failures": FAILURES_FILENAME,
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
