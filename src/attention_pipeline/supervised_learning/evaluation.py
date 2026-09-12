"""Archive-driven outer evaluation for the FocusWave Q1 supervised task.

Evaluation is deliberately separated from model fitting. It consumes legal OOF
probe predictions, computes probe-level probability loss, aggregates within each
participant, then gives participants equal weight. Participant-cluster
bootstrap resamples the already-frozen participant summaries; models are not
retrained inside bootstrap replicates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .task import Q1_BINARY_SPEC, SupervisedLearningContractError


DEFAULT_BOOTSTRAP_REPLICATES = 1000
DEFAULT_BOOTSTRAP_SEED = 20260830
DEFAULT_CONFIDENCE_LEVEL = 0.95
MEMBERSHIP_COLUMN = "membership_type"
ALLOWED_MEMBERSHIP_TYPES = frozenset({"included_complete", "included_missing_aware"})
PAIR_KEY_COLUMNS = (
    "participant_group_id",
    "session_id",
    "block_id",
    "probe_event_id",
)


class EvaluationContractError(SupervisedLearningContractError):
    """Raised when an OOF archive is not legal for the requested evaluation."""


@dataclass
class ArchiveEvaluationResult:
    """Participant-level and overall evaluation reconstructed from OOF rows."""

    participant_scores: pd.DataFrame
    model_scores: pd.DataFrame
    bootstrap_records: list[dict[str, object]] = field(default_factory=list)


@dataclass
class PairedIncrementResult:
    """Matched participant-level loss improvement for two OOF model archives."""

    baseline_model_id: str
    added_model_id: str
    analysis_set_id: str
    membership_type: str
    participant_increments: pd.DataFrame
    overall_increment: float
    bootstrap: dict[str, object]


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, context: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise EvaluationContractError(f"{context} missing required columns: {missing}")


def _single_nonblank_value(frame: pd.DataFrame, column: str, *, context: str) -> str:
    if column not in frame.columns:
        raise EvaluationContractError(f"{context} missing required column: {column}")
    raw = frame[column]
    if raw.isna().any():
        raise EvaluationContractError(f"{context} {column} contains missing values")
    values = raw.astype(str).str.strip()
    if values.eq("").any():
        raise EvaluationContractError(f"{context} {column} contains blank values")
    unique = values.drop_duplicates().tolist()
    if len(unique) != 1:
        raise EvaluationContractError(f"{context} requires one {column}; got {unique}")
    return str(unique[0])


def _strict_bool_series(series: pd.Series, *, context: str, column: str) -> pd.Series:
    """Accept real booleans or unambiguous serialized boolean values only."""
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapped = normalized.map({"true": True, "false": False, "1": True, "0": False})
    if mapped.isna().any():
        bad = sorted(normalized[mapped.isna()].drop_duplicates().tolist())
        raise EvaluationContractError(f"{context} {column} contains invalid boolean values: {bad}")
    return mapped.astype(bool)


def _validate_complete_single_model(
    frame: pd.DataFrame, *, context: str
) -> tuple[pd.DataFrame, str, str, str]:
    required = {
        *PAIR_KEY_COLUMNS,
        "analysis_set_id",
        MEMBERSHIP_COLUMN,
        "model_id",
        "outer_fold_group",
        "q1_binary",
        Q1_BINARY_SPEC.positive_probability_name,
        "model_failed",
    }
    _require_columns(frame, required, context=context)
    if frame.empty:
        raise EvaluationContractError(f"{context} is empty")
    model_id = _single_nonblank_value(frame, "model_id", context=context)
    analysis_set_id = _single_nonblank_value(frame, "analysis_set_id", context=context)
    membership_type = _single_nonblank_value(frame, MEMBERSHIP_COLUMN, context=context)
    if membership_type not in ALLOWED_MEMBERSHIP_TYPES:
        raise EvaluationContractError(
            f"{context} has unsupported {MEMBERSHIP_COLUMN}={membership_type!r}; "
            f"expected one of {sorted(ALLOWED_MEMBERSHIP_TYPES)}"
        )
    if frame.duplicated(list(PAIR_KEY_COLUMNS)).any():
        raise EvaluationContractError(f"{context} contains duplicate probe keys: {list(PAIR_KEY_COLUMNS)}")
    if frame["participant_group_id"].isna().any():
        raise EvaluationContractError(f"{context} participant_group_id contains missing values")
    participants = frame["participant_group_id"].astype(str)
    outer_groups = frame["outer_fold_group"].astype(str)
    if not participants.equals(outer_groups):
        raise EvaluationContractError(
            f"{context} outer_fold_group must equal held-out participant_group_id on every OOF row"
        )
    failed = _strict_bool_series(frame["model_failed"], context=context, column="model_failed")
    if failed.any():
        failed_groups = sorted(frame.loc[failed, "participant_group_id"].astype(str).unique().tolist())
        raise EvaluationContractError(
            f"{context} has failed OOF rows; participant-equal evaluation cannot silently drop them: {failed_groups}"
        )
    y = pd.to_numeric(frame["q1_binary"], errors="coerce")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise EvaluationContractError(f"{context} q1_binary must contain only 0/1")
    p = pd.to_numeric(frame[Q1_BINARY_SPEC.positive_probability_name], errors="coerce")
    if p.isna().any() or ((p < 0.0) | (p > 1.0)).any():
        raise EvaluationContractError(
            f"{context} {Q1_BINARY_SPEC.positive_probability_name} must contain finite probabilities in [0, 1]"
        )
    out = frame.copy()
    out["q1_binary"] = y.astype(int)
    out[Q1_BINARY_SPEC.positive_probability_name] = p.astype(float)
    out["model_failed"] = failed
    return out, model_id, analysis_set_id, membership_type


def binary_probe_log_loss(
    y_true: Sequence[int] | np.ndarray, p_positive: Sequence[float] | np.ndarray
) -> np.ndarray:
    """Return one binary negative-log-likelihood value per probe."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_positive, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise EvaluationContractError("binary probe loss inputs must be aligned 1D arrays")
    if not np.isin(y, [0, 1]).all():
        raise EvaluationContractError("binary probe loss labels must be 0/1")
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise EvaluationContractError("binary probe loss probabilities must be finite in [0, 1]")
    eps = np.finfo(float).eps
    clipped = np.clip(p, eps, 1.0 - eps)
    return -(y * np.log(clipped) + (1 - y) * np.log(1.0 - clipped))


def participant_log_loss(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compute probe-equal within-participant and participant-equal overall log loss."""
    data, model_id, analysis_set_id, membership_type = _validate_complete_single_model(
        frame, context="OOF model archive"
    )
    data = data.copy()
    data["probe_log_loss"] = binary_probe_log_loss(
        data["q1_binary"].to_numpy(dtype=int),
        data[Q1_BINARY_SPEC.positive_probability_name].to_numpy(dtype=float),
    )
    participant = (
        data.groupby("participant_group_id", sort=True, as_index=False)
        .agg(
            n_probes=("probe_log_loss", "size"),
            mean_log_loss=("probe_log_loss", "mean"),
        )
    )
    participant.insert(0, MEMBERSHIP_COLUMN, membership_type)
    participant.insert(0, "analysis_set_id", analysis_set_id)
    participant.insert(0, "model_id", model_id)
    overall = float(participant["mean_log_loss"].mean())
    pooled_probe = float(data["probe_log_loss"].mean())
    return participant, {
        "model_id": model_id,
        "analysis_set_id": analysis_set_id,
        MEMBERSHIP_COLUMN: membership_type,
        "metric": "log_loss",
        "aggregation": "participant_equal_within_participant_probe_equal",
        "n_participants": int(len(participant)),
        "n_probes": int(len(data)),
        "participant_equal_log_loss": overall,
        "pooled_probe_log_loss_descriptive": pooled_probe,
    }


def fixed_oof_participant_bootstrap(
    participant_values: Sequence[float] | np.ndarray,
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> dict[str, object]:
    """Percentile CI from participant-level values without model retraining."""
    values = np.asarray(participant_values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise EvaluationContractError("participant bootstrap requires a non-empty 1D participant-value vector")
    if not np.isfinite(values).all():
        raise EvaluationContractError("participant bootstrap values must be finite")
    if int(replicates) < 1:
        raise EvaluationContractError("participant bootstrap replicates must be positive")
    if not 0.0 < float(confidence_level) < 1.0:
        raise EvaluationContractError("confidence_level must be between 0 and 1")

    rng = np.random.default_rng(int(seed))
    n = len(values)
    estimates = np.empty(int(replicates), dtype=float)
    for index in range(int(replicates)):
        draw = rng.integers(0, n, size=n)
        estimates[index] = float(np.mean(values[draw]))

    alpha = (1.0 - float(confidence_level)) / 2.0
    lower, upper = np.quantile(estimates, [alpha, 1.0 - alpha])
    return {
        "method": "fixed_oof_participant_cluster_percentile",
        "retrain_within_bootstrap": False,
        "resampling_unit": "participant",
        "replicates": int(replicates),
        "seed": int(seed),
        "confidence_level": float(confidence_level),
        "point_estimate": float(np.mean(values)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n_participants": int(n),
        "n_valid_replicates": int(replicates),
    }


def evaluate_prediction_archive(
    predictions: pd.DataFrame,
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> ArchiveEvaluationResult:
    """Evaluate every model in a prediction archive, recording failures explicitly."""
    _require_columns(predictions, ["model_id"], context="prediction archive")
    if predictions.empty:
        raise EvaluationContractError("prediction archive is empty")

    participant_frames: list[pd.DataFrame] = []
    score_rows: list[dict[str, object]] = []
    bootstrap_records: list[dict[str, object]] = []
    for model_id, model_frame in predictions.groupby("model_id", sort=True, dropna=False):
        model_name = str(model_id)
        try:
            participant, overall = participant_log_loss(model_frame)
            bootstrap = fixed_oof_participant_bootstrap(
                participant["mean_log_loss"].to_numpy(dtype=float),
                replicates=replicates,
                seed=seed,
                confidence_level=confidence_level,
            )
            participant_frames.append(participant)
            score_rows.append({**overall, "status": "estimable", "reason": ""})
            bootstrap_records.append(
                {
                    "model_id": model_name,
                    "analysis_set_id": overall["analysis_set_id"],
                    MEMBERSHIP_COLUMN: overall[MEMBERSHIP_COLUMN],
                    "metric": "participant_equal_log_loss",
                    **bootstrap,
                }
            )
        except EvaluationContractError as exc:
            analysis_values = (
                model_frame["analysis_set_id"].dropna().astype(str).str.strip().drop_duplicates().tolist()
                if "analysis_set_id" in model_frame.columns
                else []
            )
            membership_values = (
                model_frame[MEMBERSHIP_COLUMN].dropna().astype(str).str.strip().drop_duplicates().tolist()
                if MEMBERSHIP_COLUMN in model_frame.columns
                else []
            )
            score_rows.append(
                {
                    "model_id": model_name,
                    "analysis_set_id": analysis_values[0] if len(analysis_values) == 1 else None,
                    MEMBERSHIP_COLUMN: membership_values[0] if len(membership_values) == 1 else None,
                    "metric": "log_loss",
                    "aggregation": "participant_equal_within_participant_probe_equal",
                    "n_participants": int(model_frame["participant_group_id"].nunique())
                    if "participant_group_id" in model_frame.columns
                    else 0,
                    "n_probes": int(len(model_frame)),
                    "participant_equal_log_loss": np.nan,
                    "pooled_probe_log_loss_descriptive": np.nan,
                    "status": "not_estimable",
                    "reason": str(exc),
                }
            )

    participant_scores = (
        pd.concat(participant_frames, ignore_index=True)
        if participant_frames
        else pd.DataFrame(
            columns=[
                "model_id",
                "analysis_set_id",
                MEMBERSHIP_COLUMN,
                "participant_group_id",
                "n_probes",
                "mean_log_loss",
            ]
        )
    )
    return ArchiveEvaluationResult(
        participant_scores=participant_scores,
        model_scores=pd.DataFrame(score_rows),
        bootstrap_records=bootstrap_records,
    )


def paired_log_loss_increment(
    baseline: pd.DataFrame,
    added: pd.DataFrame,
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> PairedIncrementResult:
    """Compare matched OOF archives as baseline loss minus added-model loss.

    Positive increment means the added model has lower loss. The two inputs must
    contain exactly the same analysis-set membership, participant/probe members,
    labels and outer-fold assignment; mismatches are a hard not-comparable error.
    """
    base, baseline_model_id, base_set, base_membership = _validate_complete_single_model(
        baseline, context="baseline OOF archive"
    )
    aug, added_model_id, added_set, added_membership = _validate_complete_single_model(
        added, context="added-model OOF archive"
    )
    if base_set != added_set:
        raise EvaluationContractError(
            f"paired comparison requires the same analysis_set_id; got {base_set!r} vs {added_set!r}"
        )
    if base_membership != added_membership:
        raise EvaluationContractError(
            f"paired comparison requires the same {MEMBERSHIP_COLUMN}; "
            f"got {base_membership!r} vs {added_membership!r}"
        )

    compare_columns = [*PAIR_KEY_COLUMNS, "outer_fold_group", "q1_binary"]
    base_sorted = base.sort_values(list(PAIR_KEY_COLUMNS)).reset_index(drop=True)
    aug_sorted = aug.sort_values(list(PAIR_KEY_COLUMNS)).reset_index(drop=True)
    if len(base_sorted) != len(aug_sorted):
        raise EvaluationContractError(
            f"paired comparison probe count mismatch: baseline={len(base_sorted)}, added={len(aug_sorted)}"
        )
    for column in compare_columns:
        left = base_sorted[column].astype(str).reset_index(drop=True)
        right = aug_sorted[column].astype(str).reset_index(drop=True)
        if not left.equals(right):
            raise EvaluationContractError(
                f"paired comparison mismatch in {column}; direct increment is not comparable"
            )

    y = base_sorted["q1_binary"].to_numpy(dtype=int)
    baseline_loss = binary_probe_log_loss(
        y,
        base_sorted[Q1_BINARY_SPEC.positive_probability_name].to_numpy(dtype=float),
    )
    added_loss = binary_probe_log_loss(
        y,
        aug_sorted[Q1_BINARY_SPEC.positive_probability_name].to_numpy(dtype=float),
    )
    paired = pd.DataFrame(
        {
            "participant_group_id": base_sorted["participant_group_id"].astype(str),
            "probe_increment": baseline_loss - added_loss,
        }
    )
    participant = (
        paired.groupby("participant_group_id", sort=True, as_index=False)
        .agg(n_probes=("probe_increment", "size"), mean_log_loss_increment=("probe_increment", "mean"))
    )
    participant.insert(0, MEMBERSHIP_COLUMN, base_membership)
    participant.insert(0, "analysis_set_id", base_set)
    participant.insert(0, "added_model_id", added_model_id)
    participant.insert(0, "baseline_model_id", baseline_model_id)
    values = participant["mean_log_loss_increment"].to_numpy(dtype=float)
    bootstrap = fixed_oof_participant_bootstrap(
        values,
        replicates=replicates,
        seed=seed,
        confidence_level=confidence_level,
    )
    bootstrap["paired_model_resampling"] = True
    bootstrap["increment_definition"] = "baseline_log_loss_minus_added_log_loss"
    return PairedIncrementResult(
        baseline_model_id=baseline_model_id,
        added_model_id=added_model_id,
        analysis_set_id=base_set,
        membership_type=base_membership,
        participant_increments=participant,
        overall_increment=float(np.mean(values)),
        bootstrap=bootstrap,
    )
