"""Training-split-only preprocessing for FocusWave supervised learning.

This module intentionally has no participant-specific within/between transform.
The zero-calibration mainline may inspect participant IDs only to audit which
participants were used to fit preprocessing state; participant identity never
changes feature values.
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
            "standardization_mean": dict(self.standardization_mean),
            "standardization_std": dict(self.standardization_std),
            "fit_group_ids": list(self.fit_group_ids),
            "n_fit_rows": int(self.n_fit_rows),
            "participant_specific_transform": False,
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
    if frame[group_col].isna().any():
        raise PreprocessingContractError(f"{group_col} contains missing participant IDs")
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


def fit_preprocessing(
    train: pd.DataFrame,
    *,
    columns: Sequence[str],
    group_col: str = "participant_group_id",
    drop_all_missing: bool = True,
    drop_zero_variance: bool = True,
) -> PreprocessingState:
    """Fit imputation, scaling and data-dependent column state on training rows.

    No global coverage percentage is used.  A low-coverage feature can remain if
    it is scientifically admitted upstream and has enough observed variation in
    the current training split to estimate the declared transformations.
    """
    names = _validate_columns(train, columns, group_col)
    numeric = _numeric_frame(train, names)
    state = PreprocessingState(
        input_columns=names,
        fit_group_ids=tuple(sorted(train[group_col].astype(str).unique().tolist())),
        n_fit_rows=int(len(train)),
    )

    kept: list[str] = []
    for col in names:
        observed = numeric[col].dropna()
        if drop_all_missing and observed.empty:
            state.dropped_columns[col] = "all_missing_in_training"
            continue
        if drop_zero_variance and observed.nunique(dropna=True) < 2:
            state.dropped_columns[col] = "zero_variance_in_training"
            continue
        median = float(observed.median()) if len(observed) else np.nan
        if not np.isfinite(median):
            state.dropped_columns[col] = "nonfinite_training_median"
            continue
        state.imputation_medians[col] = median
        kept.append(col)

    imputed = numeric[kept].copy()
    for col in kept:
        imputed[col] = imputed[col].fillna(state.imputation_medians[col])

    final: list[str] = []
    for col in kept:
        mean = float(imputed[col].mean())
        std = float(imputed[col].std(ddof=0))
        if not np.isfinite(mean) or not np.isfinite(std):
            state.dropped_columns[col] = "nonfinite_standardization_state"
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
