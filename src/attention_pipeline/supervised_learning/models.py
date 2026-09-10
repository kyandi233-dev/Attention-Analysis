"""Nested model selection and complete outer-training refit.

The selection API receives only outer-training rows.  Every inner split fits its
own preprocessing state on inner-training participants, applies that frozen state
to inner validation, and evaluates predeclared feature schemes plus Logistic C.
Outer-test rows are accepted only by the final refit/predict function and never by
model selection.
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
from .preprocessing import PreprocessingContractError, apply_preprocessing, fit_preprocessing
from .task import Q1_BINARY_SPEC, SupervisedLearningContractError, positive_class_probability


DEFAULT_C_CANDIDATES = (0.01, 0.1, 1.0, 10.0)
DEFAULT_INNER_SPLITS = 5
DEFAULT_MAX_ITER = 2000


class ModelSelectionError(RuntimeError):
    """Raised when nested selection cannot produce a valid candidate."""


@dataclass
class ModelSelectionResult:
    """Winner plus complete inner-development audit."""

    feature_scheme: FeatureScheme
    selected_c: float
    candidate_mean_log_loss: dict[str, float]
    inner_fold_audits: list[dict[str, object]] = field(default_factory=list)
    failed_candidates: dict[str, list[str]] = field(default_factory=dict)

    def audit_dict(self) -> dict[str, object]:
        return {
            "feature_scheme": self.feature_scheme.audit_dict(),
            "selected_c": float(self.selected_c),
            "candidate_mean_log_loss": dict(self.candidate_mean_log_loss),
            "inner_fold_audits": list(self.inner_fold_audits),
            "failed_candidates": dict(self.failed_candidates),
            "selection_metric": "mean_inner_log_loss",
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


def _fit_logistic(x: pd.DataFrame, y: np.ndarray, *, c: float, max_iter: int, seed: int) -> LogisticRegression:
    if len(np.unique(y)) < 2:
        raise ModelSelectionError("training split contains only one binary class")
    model = LogisticRegression(
        C=float(c),
        penalty="l2",
        solver="lbfgs",
        max_iter=int(max_iter),
        random_state=int(seed),
    )
    model.fit(x, y)
    return model


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
    """Select feature scheme and C using genuinely nested grouped CV.

    Preprocessing is fitted once per (scheme, inner split), never on all outer
    training rows before inner validation.  A candidate is eligible only when it
    succeeds on every declared inner split; failures are preserved in the audit.
    """
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
    losses: dict[tuple[str, float], list[float]] = {
        (scheme.feature_set_id, c): [] for scheme in feature_schemes for c in candidates
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

            audit: dict[str, object] = {
                "scheme_index": int(scheme_index),
                "feature_set_id": scheme.feature_set_id,
                "inner_fold": int(fold_index),
                "train_group_ids": train_groups,
                "validation_group_ids": valid_groups,
                "preprocessing": None,
                "loss_by_c": {},
                "failure_by_c": {},
            }
            try:
                state = fit_preprocessing(
                    inner_train,
                    columns=scheme.columns,
                    group_col=group_col,
                )
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

            for c_index, c in enumerate(candidates):
                try:
                    model = _fit_logistic(
                        x_train,
                        y_train,
                        c=c,
                        max_iter=max_iter,
                        seed=seed + scheme_index * 1000 + fold_index * 100 + c_index,
                    )
                    p_positive = positive_class_probability(model.predict_proba(x_valid), model.classes_)
                    aligned = np.column_stack([1.0 - p_positive, p_positive])
                    loss = float(log_loss(y_valid, aligned, labels=list(Q1_BINARY_SPEC.class_labels)))
                    if not np.isfinite(loss):
                        raise ModelSelectionError("inner validation log loss is non-finite")
                    losses[(scheme.feature_set_id, c)].append(loss)
                    audit["loss_by_c"][str(c)] = loss
                except Exception as exc:  # preserve candidate-level failure without contaminating peers
                    reason = f"{type(exc).__name__}: {exc}"
                    failures[(scheme.feature_set_id, c)].append(f"inner_fold={fold_index}: {reason}")
                    audit["failure_by_c"][str(c)] = reason
            fold_audits.append(audit)

    mean_scores: dict[str, float] = {}
    eligible: list[tuple[float, int, int, FeatureScheme, float]] = []
    for scheme_index, scheme in enumerate(feature_schemes):
        for c_index, c in enumerate(candidates):
            key = (scheme.feature_set_id, c)
            score_key = f"{scheme.feature_set_id}|C={c:g}"
            if failures[key] or len(losses[key]) != len(splits):
                continue
            mean_loss = float(np.mean(losses[key]))
            mean_scores[score_key] = mean_loss
            eligible.append((mean_loss, scheme_index, c_index, scheme, c))

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

    # Stable tie-break: mean loss, then declared scheme order, then declared C order.
    _, _, _, winner_scheme, winner_c = min(eligible, key=lambda item: (item[0], item[1], item[2]))
    return ModelSelectionResult(
        feature_scheme=winner_scheme,
        selected_c=float(winner_c),
        candidate_mean_log_loss=mean_scores,
        inner_fold_audits=fold_audits,
        failed_candidates=failed_candidates,
    )


def refit_logistic_and_predict(
    outer_train: pd.DataFrame,
    y_outer_train: Sequence[object] | np.ndarray,
    outer_test: pd.DataFrame,
    *,
    feature_scheme: FeatureScheme,
    selected_c: float,
    group_col: str = "participant_group_id",
    max_iter: int = DEFAULT_MAX_ITER,
    seed: int = 20260910,
) -> dict[str, object]:
    """Refit winner on complete outer training data and predict untouched test rows.

    The function intentionally accepts no outer-test labels, so test outcomes
    cannot influence preprocessing, fitting, thresholding or hyperparameters.
    """
    require_scheme_columns(outer_train, feature_scheme)
    require_scheme_columns(outer_test, feature_scheme)
    labels = _binary_labels(y_outer_train, len(outer_train))
    state = fit_preprocessing(
        outer_train,
        columns=feature_scheme.columns,
        group_col=group_col,
    )
    x_train = apply_preprocessing(outer_train, state, group_col=group_col)
    x_test = apply_preprocessing(outer_test, state, group_col=group_col)
    model = _fit_logistic(
        x_train,
        labels,
        c=float(selected_c),
        max_iter=max_iter,
        seed=seed,
    )
    raw_proba = model.predict_proba(x_test)
    p_positive = positive_class_probability(raw_proba, model.classes_)
    predicted = model.predict(x_test).astype(int)
    return {
        "feature_set_id": feature_scheme.feature_set_id,
        "selected_c": float(selected_c),
        "n_train_rows": int(len(outer_train)),
        "n_test_rows": int(len(outer_test)),
        "train_group_ids": list(state.fit_group_ids),
        "test_group_ids": sorted(outer_test[group_col].astype(str).unique().tolist()),
        "preprocessing": state.audit_dict(),
        "model_classes": [int(v) for v in model.classes_.tolist()],
        "p_positive": p_positive,
        "predicted_label": predicted,
    }
