"""Nested participant-equal model selection and complete outer-training refit.

The selection API receives only outer-training rows. Every inner split fits its
own participant-equal preprocessing state on inner-training participants, fits
logistic regression with equal total training mass per participant, and scores
candidates by participant-macro validation log loss. Outer-test rows are accepted
only by the final refit/predict function and never by model selection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

from .feature_schemes import FeatureScheme, require_scheme_columns
from .preprocessing import (
    PreprocessingContractError,
    apply_preprocessing,
    fit_preprocessing,
    participant_equal_row_weights,
    participant_weight_audit,
)
from .task import Q1_BINARY_SPEC, SupervisedLearningContractError, positive_class_probability


DEFAULT_C_CANDIDATES = (0.01, 0.1, 1.0, 10.0)
DEFAULT_INNER_SPLITS = 5
DEFAULT_MAX_ITER = 2000
SELECTION_METRIC = "participant_macro_log_loss"
_TEST_OUTCOME_COLUMNS = frozenset({Q1_BINARY_SPEC.source_column, "q1_binary"})


class ModelSelectionError(RuntimeError):
    """Raised when nested selection cannot produce a valid candidate."""


_EXPECTED_MODEL_FAILURES = (
    ModelSelectionError,
    SupervisedLearningContractError,
    ValueError,
    FloatingPointError,
    np.linalg.LinAlgError,
)


@dataclass
class ModelSelectionResult:
    """Winner plus complete inner-development audit."""

    feature_scheme: FeatureScheme
    selected_c: float
    candidate_participant_macro_log_loss: dict[str, float]
    inner_fold_audits: list[dict[str, object]] = field(default_factory=list)
    failed_candidates: dict[str, list[str]] = field(default_factory=dict)

    @property
    def candidate_mean_log_loss(self) -> dict[str, float]:
        """Deprecated compatibility alias for earlier Task A callers."""
        return self.candidate_participant_macro_log_loss

    def audit_dict(self) -> dict[str, object]:
        return {
            "feature_scheme": self.feature_scheme.audit_dict(),
            "selected_c": float(self.selected_c),
            "candidate_participant_macro_log_loss": dict(self.candidate_participant_macro_log_loss),
            "candidate_mean_log_loss": dict(self.candidate_participant_macro_log_loss),
            "inner_fold_audits": list(self.inner_fold_audits),
            "failed_candidates": dict(self.failed_candidates),
            "selection_metric": SELECTION_METRIC,
        }


def _binary_labels(y: Sequence[object] | np.ndarray, expected_n: int) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1 or len(arr) != expected_n:
        raise SupervisedLearningContractError("binary labels must be a 1D array aligned with frame rows")
    numeric = pd.to_numeric(pd.Series(arr), errors="coerce")
    if numeric.isna().any():
        raise SupervisedLearningContractError("binary labels contain missing/non-numeric values")
    labels = numeric.astype(int).to_numpy()
    unexpected = sorted(set(labels.tolist()) - set(Q1_BINARY_SPEC.class_labels))
    if unexpected:
        raise SupervisedLearningContractError(f"unexpected binary labels: {unexpected}")
    return labels


def _fit_logistic(
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    c: float,
    max_iter: int,
    seed: int,
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> LogisticRegression:
    if len(np.unique(y)) < 2:
        raise ModelSelectionError("training split contains only one binary class")
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if weights is not None:
        if weights.ndim != 1 or len(weights) != len(y):
            raise ModelSelectionError("sample_weight must be 1D and aligned with training labels")
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ModelSelectionError("sample_weight must contain finite positive values")
    model = LogisticRegression(
        C=float(c),
        solver="lbfgs",
        max_iter=int(max_iter),
        random_state=int(seed),
    )
    model.fit(x, y, sample_weight=weights)
    return model


def _participant_validation_log_losses(
    y_valid: np.ndarray,
    p_positive: np.ndarray,
    valid_groups: Sequence[object] | np.ndarray,
) -> dict[str, float]:
    """Return one probe-equal mean probability loss per validation participant."""
    groups = np.asarray(valid_groups).astype(str)
    probabilities = np.asarray(p_positive, dtype=float)
    if groups.ndim != 1 or probabilities.ndim != 1:
        raise ModelSelectionError("validation groups and probabilities must be 1D")
    if len(groups) != len(y_valid) or len(probabilities) != len(y_valid):
        raise ModelSelectionError("validation groups/probabilities must align with labels")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ModelSelectionError("validation probabilities must be finite values in [0, 1]")

    participant_losses: dict[str, float] = {}
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        aligned = np.column_stack([1.0 - probabilities[mask], probabilities[mask]])
        loss = float(log_loss(y_valid[mask], aligned, labels=list(Q1_BINARY_SPEC.class_labels)))
        if not np.isfinite(loss):
            raise ModelSelectionError(f"validation log loss is non-finite for participant {group}")
        participant_losses[str(group)] = loss
    return participant_losses


def select_logistic_model(
    outer_train: pd.DataFrame,
    y_outer_train: Sequence[object] | np.ndarray,
    *,
    feature_schemes: Sequence[FeatureScheme],
    group_col: str = "participant_group_id",
    c_candidates: Sequence[float] = DEFAULT_C_CANDIDATES,
    n_splits: int = DEFAULT_INNER_SPLITS,
    max_iter: int = DEFAULT_MAX_ITER,
    seed: int = 20260910,
) -> ModelSelectionResult:
    """Select feature scheme and C by nested participant-grouped CV."""
    if group_col not in outer_train.columns:
        raise SupervisedLearningContractError(f"missing grouping column: {group_col}")
    if outer_train[group_col].isna().any():
        raise SupervisedLearningContractError(f"{group_col} contains missing participant IDs")
    if not feature_schemes:
        raise ModelSelectionError("no feature schemes were supplied for nested selection")
    candidates = tuple(float(c) for c in c_candidates)
    if not candidates or any((not np.isfinite(c) or c <= 0) for c in candidates):
        raise ModelSelectionError("Logistic C candidates must be finite positive values")
    if int(n_splits) < 2:
        raise ModelSelectionError("inner grouped CV requires at least 2 splits")

    labels = _binary_labels(y_outer_train, len(outer_train))
    groups = outer_train[group_col].astype(str).to_numpy()
    unique_groups = sorted(set(groups.tolist()))
    if len(unique_groups) < int(n_splits):
        raise ModelSelectionError(
            f"inner grouped CV needs {n_splits} participant groups; got {len(unique_groups)}"
        )

    splitter = GroupKFold(n_splits=int(n_splits))
    splits = list(splitter.split(outer_train, labels, groups))
    participant_losses: dict[tuple[str, float], dict[str, float]] = {
        (scheme.feature_set_id, c): {} for scheme in feature_schemes for c in candidates
    }
    failures: dict[tuple[str, float], list[str]] = {
        (scheme.feature_set_id, c): [] for scheme in feature_schemes for c in candidates
    }
    fold_audits: list[dict[str, object]] = []

    for scheme_index, scheme in enumerate(feature_schemes):
        require_scheme_columns(outer_train, scheme)
        for fold_index, (train_idx, valid_idx) in enumerate(splits):
            inner_train = outer_train.iloc[train_idx].copy()
            inner_valid = outer_train.iloc[valid_idx].copy()
            y_train = labels[train_idx]
            y_valid = labels[valid_idx]
            train_groups = sorted(inner_train[group_col].astype(str).unique().tolist())
            valid_groups = sorted(inner_valid[group_col].astype(str).unique().tolist())
            if set(train_groups) & set(valid_groups):
                raise ModelSelectionError("inner training/validation participant overlap detected")

            train_weights = participant_equal_row_weights(
                inner_train,
                group_col=group_col,
                normalize_mean_one=True,
            )
            audit: dict[str, object] = {
                "scheme_index": int(scheme_index),
                "feature_set_id": scheme.feature_set_id,
                "inner_fold": int(fold_index),
                "train_group_ids": train_groups,
                "validation_group_ids": valid_groups,
                "training_weights": participant_weight_audit(inner_train, train_weights, group_col=group_col),
                "preprocessing": None,
                "loss_by_c": {},
                "participant_loss_by_c": {},
                "failure_by_c": {},
            }
            try:
                state = fit_preprocessing(inner_train, columns=scheme.columns, group_col=group_col)
                x_train = apply_preprocessing(inner_train, state, group_col=group_col)
                x_valid = apply_preprocessing(inner_valid, state, group_col=group_col)
                audit["preprocessing"] = state.audit_dict()
            except (PreprocessingContractError, SupervisedLearningContractError) as exc:
                reason = f"{type(exc).__name__}: {exc}"
                for c in candidates:
                    failures[(scheme.feature_set_id, c)].append(f"inner_fold={fold_index}: {reason}")
                    audit["failure_by_c"][str(c)] = reason
                fold_audits.append(audit)
                continue

            validation_group_rows = inner_valid[group_col].astype(str).to_numpy()
            for c_index, c in enumerate(candidates):
                key = (scheme.feature_set_id, c)
                try:
                    model = _fit_logistic(
                        x_train,
                        y_train,
                        c=c,
                        max_iter=max_iter,
                        seed=seed + scheme_index * 1000 + fold_index * 100 + c_index,
                        sample_weight=train_weights,
                    )
                    p_positive = positive_class_probability(model.predict_proba(x_valid), model.classes_)
                    per_participant = _participant_validation_log_losses(
                        y_valid,
                        p_positive,
                        validation_group_rows,
                    )
                    overlap = set(participant_losses[key]) & set(per_participant)
                    if overlap:
                        raise ModelSelectionError(
                            f"validation participants appeared in more than one inner fold: {sorted(overlap)}"
                        )
                    participant_losses[key].update(per_participant)
                    fold_macro = float(np.mean(list(per_participant.values())))
                    audit["loss_by_c"][str(c)] = fold_macro
                    audit["participant_loss_by_c"][str(c)] = dict(per_participant)
                except _EXPECTED_MODEL_FAILURES as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    failures[key].append(f"inner_fold={fold_index}: {reason}")
                    audit["failure_by_c"][str(c)] = reason
            fold_audits.append(audit)

    scores: dict[str, float] = {}
    eligible: list[tuple[float, int, int, FeatureScheme, float]] = []
    required_validation_groups = set(unique_groups)
    for scheme_index, scheme in enumerate(feature_schemes):
        for c_index, c in enumerate(candidates):
            key = (scheme.feature_set_id, c)
            score_key = f"{scheme.feature_set_id}|C={c:g}"
            observed_groups = set(participant_losses[key])
            if failures[key] or observed_groups != required_validation_groups:
                if observed_groups != required_validation_groups and not failures[key]:
                    missing_groups = sorted(required_validation_groups - observed_groups)
                    extra_groups = sorted(observed_groups - required_validation_groups)
                    failures[key].append(
                        f"incomplete_validation_participants: missing={missing_groups}, extra={extra_groups}"
                    )
                continue
            macro_loss = float(np.mean([participant_losses[key][group] for group in unique_groups]))
            if not np.isfinite(macro_loss):
                failures[key].append("participant_macro_log_loss_nonfinite")
                continue
            scores[score_key] = macro_loss
            eligible.append((macro_loss, scheme_index, c_index, scheme, c))

    failed_candidates = {
        f"{scheme_id}|C={c:g}": reasons
        for (scheme_id, c), reasons in failures.items()
        if reasons
    }
    if not eligible:
        raise ModelSelectionError(
            "all feature-scheme/C candidates failed inner grouped CV; "
            f"failures={failed_candidates}"
        )

    _, _, _, winner_scheme, winner_c = min(eligible, key=lambda item: (item[0], item[1], item[2]))
    return ModelSelectionResult(
        feature_scheme=winner_scheme,
        selected_c=float(winner_c),
        candidate_participant_macro_log_loss=scores,
        inner_fold_audits=fold_audits,
        failed_candidates=failed_candidates,
    )


def refit_logistic_and_predict(
    outer_train: pd.DataFrame,
    y_outer_train: Sequence[object] | np.ndarray,
    outer_test_features: pd.DataFrame,
    *,
    feature_scheme: FeatureScheme,
    selected_c: float,
    group_col: str = "participant_group_id",
    max_iter: int = DEFAULT_MAX_ITER,
    seed: int = 20260910,
) -> dict[str, object]:
    """Refit winner with participant-equal weights and predict label-free test rows."""
    leaked = sorted(_TEST_OUTCOME_COLUMNS & set(outer_test_features.columns))
    if leaked:
        raise SupervisedLearningContractError(
            f"outer_test_features must be outcome-free; leaked columns: {leaked}"
        )
    require_scheme_columns(outer_train, feature_scheme)
    require_scheme_columns(outer_test_features, feature_scheme)
    labels = _binary_labels(y_outer_train, len(outer_train))
    state = fit_preprocessing(outer_train, columns=feature_scheme.columns, group_col=group_col)
    x_train = apply_preprocessing(outer_train, state, group_col=group_col)
    x_test = apply_preprocessing(outer_test_features, state, group_col=group_col)
    train_weights = participant_equal_row_weights(
        outer_train,
        group_col=group_col,
        normalize_mean_one=True,
    )
    model = _fit_logistic(
        x_train,
        labels,
        c=float(selected_c),
        max_iter=max_iter,
        seed=seed,
        sample_weight=train_weights,
    )
    raw_proba = model.predict_proba(x_test)
    p_positive = positive_class_probability(raw_proba, model.classes_)
    predicted = model.predict(x_test).astype(int)

    if model.coef_.shape != (1, x_train.shape[1]) or model.intercept_.shape != (1,):
        raise ModelSelectionError(
            f"unexpected binary logistic coefficient shape coef={model.coef_.shape}, intercept={model.intercept_.shape}"
        )
    standardized_coefficients = {
        str(column): float(value)
        for column, value in zip(x_train.columns.tolist(), model.coef_[0].tolist(), strict=True)
    }

    return {
        "feature_set_id": feature_scheme.feature_set_id,
        "selected_c": float(selected_c),
        "n_train_rows": int(len(outer_train)),
        "n_test_rows": int(len(outer_test_features)),
        "train_group_ids": list(state.fit_group_ids),
        "test_group_ids": sorted(outer_test_features[group_col].astype(str).unique().tolist()),
        "training_weights": participant_weight_audit(outer_train, train_weights, group_col=group_col),
        "preprocessing": state.audit_dict(),
        "model_classes": [int(v) for v in model.classes_.tolist()],
        "coefficient_scale": "post_imputation_participant_equal_standardized_predictors",
        "standardized_coefficients": standardized_coefficients,
        "intercept": float(model.intercept_[0]),
        "p_positive": p_positive,
        "predicted_label": predicted,
    }
