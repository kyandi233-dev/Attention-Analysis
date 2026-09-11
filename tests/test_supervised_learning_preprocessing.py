from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attention_pipeline.supervised_learning.preprocessing import (
    PreprocessingContractError,
    apply_preprocessing,
    fit_preprocessing,
    participant_equal_row_weights,
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
    assert state.imputation_observed_participant_count["sparse_but_variable"] == 2
    assert state.imputation_valid_observation_count["sparse_but_variable"] == 2
    assert state.dropped_columns["constant"] == "zero_variance_in_training"
    assert state.dropped_columns["all_missing"] == "all_missing_in_training"
    assert state.audit_dict()["participant_equal_preprocessing"] is True


def test_featurewise_weighted_median_gives_each_observed_participant_equal_mass() -> None:
    train = pd.DataFrame(
        {
            "participant_group_id": ["A", "A", "A", "A", "B"],
            "signal": [0.0, 0.0, 0.0, 0.0, 10.0],
        }
    )
    state = fit_preprocessing(train, columns=["signal"])

    # Participant A contributes total mass 0.5 and B contributes total mass 0.5.
    # The cumulative mass lands exactly at 0.5 between distinct values 0 and 10,
    # so the frozen D8 boundary rule returns their midpoint.
    assert state.imputation_medians["signal"] == 5.0
    assert state.imputation_observed_participant_count["signal"] == 2
    assert state.imputation_valid_observation_count["signal"] == 5


def test_participant_equal_standardization_is_not_probe_count_weighted() -> None:
    train = pd.DataFrame(
        {
            "participant_group_id": ["A", "A", "A", "A", "B"],
            "signal": [0.0, 0.0, 0.0, 0.0, 10.0],
        }
    )
    state = fit_preprocessing(train, columns=["signal"])

    assert state.standardization_mean["signal"] == pytest.approx(5.0)
    assert state.standardization_std["signal"] == pytest.approx(5.0)
    assert state.standardization_mean["signal"] != pytest.approx(train["signal"].mean())


def test_participant_equal_training_weights_have_equal_group_totals_and_mean_one() -> None:
    train = pd.DataFrame(
        {
            "participant_group_id": ["A", "A", "A", "A", "B", "C", "C"],
            "signal": np.arange(7, dtype=float),
        }
    )
    weights = participant_equal_row_weights(train)
    grouped = pd.DataFrame({"group": train["participant_group_id"], "weight": weights}).groupby("group")["weight"].sum()

    assert float(np.mean(weights)) == pytest.approx(1.0)
    assert grouped.nunique() == 1


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


def test_participant_identity_affects_training_distribution_not_test_specific_transform() -> None:
    train = pd.DataFrame(
        {
            "participant_group_id": ["A", "A", "A", "A", "B"],
            "signal": [0.0, 0.0, 0.0, 0.0, 10.0],
        }
    )
    row_relabelled = train.copy()
    row_relabelled["participant_group_id"] = [f"ROW-{i}" for i in range(len(row_relabelled))]

    state_equal_participants = fit_preprocessing(train, columns=["signal"])
    state_equal_rows = fit_preprocessing(row_relabelled, columns=["signal"])
    assert state_equal_participants.standardization_mean["signal"] == pytest.approx(5.0)
    assert state_equal_rows.standardization_mean["signal"] == pytest.approx(2.0)

    held_out = pd.DataFrame({"participant_group_id": ["NEW", "NEW"], "signal": [0.0, 10.0]})
    transformed = apply_preprocessing(held_out, state_equal_participants)
    np.testing.assert_allclose(transformed["signal"].to_numpy(), [-1.0, 1.0])


def test_missing_or_non_numeric_feature_fails_closed() -> None:
    train = _training_frame()
    with pytest.raises(PreprocessingContractError, match="missing required preprocessing columns"):
        fit_preprocessing(train, columns=["not_here"])

    bad = train[["participant_group_id", "dense"]].copy()
    bad["dense"] = bad["dense"].astype(object)
    bad.loc[0, "dense"] = "bad-value"
    with pytest.raises(PreprocessingContractError, match="non-numeric values in feature dense"):
        fit_preprocessing(bad, columns=["dense"])
