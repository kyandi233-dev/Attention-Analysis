"""Training-split-only preprocessing for FocusWave supervised learning.

The zero-calibration mainline uses participant identity only to define the
training distribution: participants contribute equal total mass to imputation,
centering and scaling.  Validation/test values are transformed only with the
state frozen on the corresponding training split; there is no participant-
specific centering, calibration or test-side adaptation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


class PreprocessingContractError(ValueError):
    """Raised when numeric feature preprocessing cannot be applied safely."""


@dataclass
class PreprocessingState:
    """Serializable state fitted on one training split only."""

    input_columns: tuple[str, ...] = ()
    output_columns: tuple[str, ...] = ()
    dropped_columns: dict[str, str] = field(default_factory=dict)
    imputation_medians: dict[str, float] = field(default_factory=dict)
    imputation_observed_participant_count: dict[str, int] = field(default_factory=dict)
    imputation_valid_observation_count: dict[str, int] = field(default_factory=dict)
    standardization_mean: dict[str, float] = field(default_factory=dict)
    standardization_std: dict[str, float] = field(default_factory=dict)
    fit_group_ids: tuple[str, ...] = ()
    n_fit_rows: int = 0

    def audit_dict(self) -> dict[str, object]:
        return {
            "input_columns": list(self.input_columns),
            "output_columns": list(self.output_columns),
            "dropped_columns": dict(self.dropped_columns),
            "imputation_medians": dict(self.imputation_medians),
            "imputation_observed_participant_count": dict(self.imputation_observed_participant_count),
            "imputation_valid_observation_count": dict(self.imputation_valid_observation_count),
            "standardization_mean": dict(self.standardization_mean),
            "standardization_std": dict(self.standardization_std),
            "fit_group_ids": list(self.fit_group_ids),
            "n_fit_rows": int(self.n_fit_rows),
            "participant_equal_preprocessing": True,
            "participant_specific_transform": False,
        }


def _validate_group_column(frame: pd.DataFrame, group_col: str) -> pd.Series:
    if group_col not in frame.columns:
        raise PreprocessingContractError(f"missing grouping column: {group_col}")
    if frame[group_col].isna().any():
        raise PreprocessingContractError(f"{group_col} contains missing participant IDs")
    groups = frame[group_col].astype(str)
    if groups.str.strip().eq("").any():
        raise PreprocessingContractError(f"{group_col} contains blank participant IDs")
    return groups


def participant_equal_row_weights(
    frame: pd.DataFrame,
    *,
    group_col: str = "participant_group_id",
    normalize_mean_one: bool = True,
) -> np.ndarray:
    """Return row weights giving every participant equal total training mass.

    A participant with ``n_i`` rows receives raw row weight ``1 / n_i``.  The
    optional common rescaling to mean weight one preserves relative weights while
    keeping the numerical scale of regularized model fitting stable.
    """
    if frame.empty:
        raise PreprocessingContractError("participant-equal weights require at least one training row")
    groups = _validate_group_column(frame, group_col)
    counts = groups.value_counts(dropna=False)
    weights = groups.map(lambda value: 1.0 / float(counts.loc[value])).to_numpy(dtype=float)
    if normalize_mean_one:
        mean_weight = float(np.mean(weights))
        if not np.isfinite(mean_weight) or mean_weight <= 0:
            raise PreprocessingContractError("participant-equal row weights have invalid mean")
        weights = weights / mean_weight
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise PreprocessingContractError("participant-equal row weights must be finite and positive")
    return weights


def participant_weight_audit(
    frame: pd.DataFrame,
    weights: Sequence[float] | np.ndarray,
    *,
    group_col: str = "participant_group_id",
) -> dict[str, object]:
    """Summarize participant-equal weights for fold-level audit."""
    groups = _validate_group_column(frame, group_col).reset_index(drop=True)
    arr = np.asarray(weights, dtype=float)
    if arr.ndim != 1 or len(arr) != len(frame):
        raise PreprocessingContractError("sample weights must be a 1D array aligned with frame rows")
    if not np.isfinite(arr).all() or np.any(arr <= 0):
        raise PreprocessingContractError("sample weights must be finite and positive")
    audit_frame = pd.DataFrame({"group": groups, "weight": arr})
    totals = audit_frame.groupby("group", sort=True)["weight"].sum()
    return {
        "n_rows": int(len(frame)),
        "n_participants": int(totals.size),
        "mean_row_weight": float(np.mean(arr)),
        "participant_total_weight": {str(group): float(value) for group, value in totals.items()},
        "participant_total_weight_min": float(totals.min()),
        "participant_total_weight_max": float(totals.max()),
    }


def _validate_columns(frame: pd.DataFrame, columns: Sequence[str], group_col: str) -> tuple[str, ...]:
    names = tuple(str(c) for c in columns)
    if not names:
        raise PreprocessingContractError("at least one feature column is required")
    if len(set(names)) != len(names):
        raise PreprocessingContractError("feature columns contain duplicates")
    required = {group_col, *names}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PreprocessingContractError(f"missing required preprocessing columns: {missing}")
    _validate_group_column(frame, group_col)
    return names


def _numeric_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    numeric = pd.DataFrame(index=frame.index)
    for col in columns:
        converted = pd.to_numeric(frame[col], errors="coerce")
        invalid = frame[col].notna() & converted.isna()
        if invalid.any():
            bad = frame.loc[invalid, col].astype(str).drop_duplicates().tolist()
            raise PreprocessingContractError(f"non-numeric values in feature {col}: {bad}")
        numeric[col] = converted.astype(float)
    return numeric


def _participant_equal_weighted_median(values: pd.Series, groups: pd.Series) -> float:
    """Feature-wise median where each participant with an observation has equal mass."""
    observed_mask = values.notna()
    observed_values = values.loc[observed_mask].astype(float).reset_index(drop=True)
    observed_groups = groups.loc[observed_mask].astype(str).reset_index(drop=True)
    if observed_values.empty:
        return float("nan")

    counts = observed_groups.value_counts(dropna=False)
    raw_weights = observed_groups.map(lambda value: 1.0 / float(counts.loc[value])).to_numpy(dtype=float)
    weights = raw_weights / float(np.sum(raw_weights))
    order = np.argsort(observed_values.to_numpy(dtype=float), kind="mergesort")
    sorted_values = observed_values.to_numpy(dtype=float)[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    index = int(np.searchsorted(cumulative, 0.5, side="left"))
    if index >= len(sorted_values):
        index = len(sorted_values) - 1

    value = float(sorted_values[index])
    if (
        np.isclose(cumulative[index], 0.5, rtol=0.0, atol=1e-12)
        and index + 1 < len(sorted_values)
        and not np.isclose(sorted_values[index], sorted_values[index + 1], rtol=0.0, atol=0.0)
    ):
        value = float((sorted_values[index] + sorted_values[index + 1]) / 2.0)
    return value


def fit_preprocessing(
    train: pd.DataFrame,
    *,
    columns: Sequence[str],
    group_col: str = "participant_group_id",
    drop_all_missing: bool = True,
    drop_zero_variance: bool = True,
) -> PreprocessingState:
    """Fit participant-equal imputation/scaling on training rows only.

    No global coverage percentage is used. A low-coverage feature can remain if
    it is scientifically admitted upstream and has enough observed variation in
    the current training split to estimate the declared transformations.
    Structural whole-modality missingness must be handled upstream and must not
    be converted into apparent measurements by this residual-feature imputer.
    """
    names = _validate_columns(train, columns, group_col)
    numeric = _numeric_frame(train, names)
    groups = train[group_col].astype(str)
    state = PreprocessingState(
        input_columns=names,
        fit_group_ids=tuple(sorted(groups.unique().tolist())),
        n_fit_rows=int(len(train)),
    )

    kept: list[str] = []
    for col in names:
        observed_mask = numeric[col].notna()
        observed = numeric.loc[observed_mask, col]
        state.imputation_valid_observation_count[col] = int(observed_mask.sum())
        state.imputation_observed_participant_count[col] = int(groups.loc[observed_mask].nunique())
        if drop_all_missing and observed.empty:
            state.dropped_columns[col] = "all_missing_in_training"
            continue
        if drop_zero_variance and observed.nunique(dropna=True) < 2:
            state.dropped_columns[col] = "zero_variance_in_training"
            continue
        median = _participant_equal_weighted_median(numeric[col], groups)
        if not np.isfinite(median):
            state.dropped_columns[col] = "nonfinite_training_participant_equal_median"
            continue
        state.imputation_medians[col] = median
        kept.append(col)

    imputed = numeric[kept].copy()
    for col in kept:
        imputed[col] = imputed[col].fillna(state.imputation_medians[col])

    row_weights = participant_equal_row_weights(train, group_col=group_col, normalize_mean_one=True)
    weight_sum = float(np.sum(row_weights))
    final: list[str] = []
    for col in kept:
        values = imputed[col].to_numpy(dtype=float)
        mean = float(np.sum(row_weights * values) / weight_sum)
        variance = float(np.sum(row_weights * np.square(values - mean)) / weight_sum)
        std = float(np.sqrt(max(variance, 0.0)))
        if not np.isfinite(mean) or not np.isfinite(std):
            state.dropped_columns[col] = "nonfinite_participant_equal_standardization_state"
            continue
        if std <= 1e-12:
            state.dropped_columns[col] = "zero_variance_after_imputation"
            continue
        state.standardization_mean[col] = mean
        state.standardization_std[col] = std
        final.append(col)

    if not final:
        raise PreprocessingContractError("no usable features remain in this training split")
    state.output_columns = tuple(final)
    return state


def apply_preprocessing(
    frame: pd.DataFrame,
    state: PreprocessingState,
    *,
    group_col: str = "participant_group_id",
) -> pd.DataFrame:
    """Apply an already-fitted training state without learning from these rows."""
    _validate_columns(frame, state.input_columns, group_col)
    numeric = _numeric_frame(frame, state.input_columns)
    out = pd.DataFrame(index=frame.index)
    for col in state.output_columns:
        values = numeric[col].fillna(state.imputation_medians[col])
        out[col] = (values - state.standardization_mean[col]) / state.standardization_std[col]
    if not np.isfinite(out.to_numpy(dtype=float)).all():
        raise PreprocessingContractError("preprocessed output contains non-finite values")
    return out.loc[:, list(state.output_columns)].astype(float)
