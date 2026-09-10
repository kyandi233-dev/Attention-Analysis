from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attention_pipeline.supervised_learning.preprocessing import (
    PreprocessingContractError,
    apply_preprocessing,
    fit_preprocessing,
)


def _training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_group_id": [f"P-{i:02d}" for i in range(10)],
            "sparse_but_variable": [1.0, 3.0] + [np.nan] * 8,
            "dense": np.arange(10, dtype=float),
            "constant": [5.0] * 10,
            "all_missing": [np.nan] * 10,
        }
    )


def test_low_coverage_feature_is_not_dropped_by_a_global_percentage_rule() -> None:
    train = _training_frame()
    state = fit_preprocessing(
        train,
        columns=["sparse_but_variable", "dense", "constant", "all_missing"],
    )

    assert "sparse_but_variable" in state.output_columns
    assert state.imputation_medians["sparse_but_variable"] == 2.0
    assert state.dropped_columns["constant"] == "zero_variance_in_training"
    assert state.dropped_columns["all_missing"] == "all_missing_in_training"


def test_validation_values_cannot_change_already_fitted_training_state() -> None:
    train = _training_frame()
    state = fit_preprocessing(train, columns=["sparse_but_variable", "dense"])
    before = state.audit_dict()

    validation = pd.DataFrame(
        {
            "participant_group_id": ["V-01", "V-01"],
            "sparse_but_variable": [1e9, np.nan],
            "dense": [-1e12, 1e12],
        }
    )
    transformed = apply_preprocessing(validation, state)

    assert state.audit_dict() == before
    assert transformed.shape == (2, 2)
    assert np.isfinite(transformed.to_numpy()).all()
    assert "V-01" not in state.fit_group_ids


def test_participant_ids_are_audit_only_not_a_feature_transform() -> None:
    train = _training_frame()[["participant_group_id", "sparse_but_variable", "dense"]].copy()
    relabeled = train.copy()
    relabeled["participant_group_id"] = [f"OTHER-{i:02d}" for i in range(len(relabeled))]

    state_a = fit_preprocessing(train, columns=["sparse_but_variable", "dense"])
    state_b = fit_preprocessing(relabeled, columns=["sparse_but_variable", "dense"])

    assert state_a.imputation_medians == state_b.imputation_medians
    assert state_a.standardization_mean == state_b.standardization_mean
    assert state_a.standardization_std == state_b.standardization_std
    np.testing.assert_allclose(
        apply_preprocessing(train, state_a).to_numpy(),
        apply_preprocessing(relabeled, state_b).to_numpy(),
    )


def test_missing_or_non_numeric_feature_fails_closed() -> None:
    train = _training_frame()
    with pytest.raises(PreprocessingContractError, match="missing required preprocessing columns"):
        fit_preprocessing(train, columns=["not_here"])

    bad = train[["participant_group_id", "dense"]].copy()
    bad["dense"] = bad["dense"].astype(object)
    bad.loc[0, "dense"] = "bad-value"
    with pytest.raises(PreprocessingContractError, match="non-numeric values in feature dense"):
        fit_preprocessing(bad, columns=["dense"])
