from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import attention_pipeline.supervised_learning.models as supervised_models
from attention_pipeline.supervised_learning.feature_schemes import FeatureScheme
from attention_pipeline.supervised_learning.models import (
    ModelSelectionError,
    refit_logistic_and_predict,
    select_logistic_model,
)
from attention_pipeline.supervised_learning.task import SupervisedLearningContractError


def _frame(seed: int = 7) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    labels: list[int] = []
    for group_index in range(8):
        for row_index in range(8):
            y = row_index % 2
            rows.append(
                {
                    "participant_group_id": f"P-{group_index:02d}",
                    "signal": 4.0 * y + rng.normal(0, 0.1),
                    "noise": rng.normal(0, 1.0),
                    "sparse": (float(y) if row_index < 2 else np.nan),
                }
            )
            labels.append(y)
    return pd.DataFrame(rows), np.asarray(labels, dtype=int)


def _unequal_frame(seed: int = 19) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    labels: list[int] = []
    for group_index, n_rows in enumerate((2, 4, 6, 8, 10, 12)):
        for row_index in range(n_rows):
            y = row_index % 2
            rows.append(
                {
                    "participant_group_id": f"U-{group_index:02d}",
                    "signal": 2.5 * y + rng.normal(0, 0.2),
                    "noise": rng.normal(0, 1.0),
                }
            )
            labels.append(y)
    return pd.DataFrame(rows), np.asarray(labels, dtype=int)


def test_nested_selection_prefers_predeclared_predictive_scheme() -> None:
    frame, y = _frame()
    result = select_logistic_model(
        frame,
        y,
        feature_schemes=[FeatureScheme("signal", ("signal",)), FeatureScheme("noise", ("noise",))],
        c_candidates=[0.1, 1.0],
        n_splits=4,
        seed=11,
    )
    assert result.feature_scheme.feature_set_id == "signal"
    assert result.selected_c in {0.1, 1.0}
    assert len(result.inner_fold_audits) == 8
    assert result.audit_dict()["selection_metric"] == "participant_macro_log_loss"


def test_inner_selection_aggregates_participant_losses_not_equal_fold_means() -> None:
    frame, y = _frame()
    result = select_logistic_model(
        frame,
        y,
        feature_schemes=[FeatureScheme("signal", ("signal",))],
        c_candidates=[1.0],
        n_splits=3,
        seed=17,
    )

    participant_losses: list[float] = []
    fold_macro: list[float] = []
    fold_n_participants: list[int] = []
    for audit in result.inner_fold_audits:
        losses = audit["participant_loss_by_c"]["1.0"]
        participant_losses.extend(float(value) for value in losses.values())
        fold_macro.append(float(audit["loss_by_c"]["1.0"]))
        fold_n_participants.append(len(audit["validation_group_ids"]))

    assert sorted(fold_n_participants) == [2, 3, 3]
    expected_participant_macro = float(np.mean(participant_losses))
    expected_from_weighted_fold_macro = float(np.average(fold_macro, weights=fold_n_participants))
    score = result.candidate_participant_macro_log_loss["signal|C=1"]
    assert score == pytest.approx(expected_participant_macro)
    assert score == pytest.approx(expected_from_weighted_fold_macro)


def test_inner_training_weights_are_recomputed_and_participant_equal() -> None:
    frame, y = _unequal_frame()
    result = select_logistic_model(
        frame,
        y,
        feature_schemes=[FeatureScheme("signal", ("signal",))],
        c_candidates=[1.0],
        n_splits=3,
    )

    for audit in result.inner_fold_audits:
        weights = audit["training_weights"]
        assert weights["mean_row_weight"] == pytest.approx(1.0)
        assert weights["participant_total_weight_min"] == pytest.approx(
            weights["participant_total_weight_max"]
        )
        assert set(weights["participant_total_weight"]) == set(audit["train_group_ids"])


def test_inner_validation_participant_cannot_change_its_inner_training_preprocessing() -> None:
    frame, y = _frame()
    schemes = [FeatureScheme("signal", ("signal", "sparse"))]
    base = select_logistic_model(frame, y, feature_schemes=schemes, c_candidates=[1.0], n_splits=4)

    target_group = "P-00"
    altered = frame.copy()
    mask = altered["participant_group_id"].eq(target_group)
    altered.loc[mask, "signal"] = 1e12
    altered.loc[mask, "sparse"] = -1e12
    changed = select_logistic_model(altered, y, feature_schemes=schemes, c_candidates=[1.0], n_splits=4)

    base_audit = next(row for row in base.inner_fold_audits if target_group in row["validation_group_ids"])
    changed_audit = next(row for row in changed.inner_fold_audits if target_group in row["validation_group_ids"])
    assert target_group not in base_audit["train_group_ids"]
    assert base_audit["train_group_ids"] == changed_audit["train_group_ids"]
    assert base_audit["preprocessing"] == changed_audit["preprocessing"]
    assert base_audit["training_weights"] == changed_audit["training_weights"]


def test_complete_outer_training_refit_uses_all_training_groups_and_participant_equal_weights() -> None:
    frame, y = _unequal_frame()
    outer_test = pd.DataFrame(
        {
            "participant_group_id": ["HELD-OUT", "HELD-OUT"],
            "signal": [0.0, 2.5],
            "noise": [0.0, 0.0],
        }
    )
    scheme = FeatureScheme("signal", ("signal",))
    result = refit_logistic_and_predict(frame, y, outer_test, feature_scheme=scheme, selected_c=1.0)

    assert set(result["train_group_ids"]) == set(frame["participant_group_id"].unique())
    assert result["test_group_ids"] == ["HELD-OUT"]
    assert "HELD-OUT" not in result["train_group_ids"]
    assert result["preprocessing"]["fit_group_ids"] == result["train_group_ids"]
    assert result["coefficient_scale"] == "post_imputation_participant_equal_standardized_predictors"
    assert set(result["standardized_coefficients"]) == set(result["preprocessing"]["output_columns"])
    assert result["standardized_coefficients"]["signal"] > 0
    assert np.isfinite(result["intercept"])
    assert len(result["p_positive"]) == len(outer_test)
    assert len(result["predicted_label"]) == len(outer_test)
    weights = result["training_weights"]
    assert weights["mean_row_weight"] == pytest.approx(1.0)
    assert weights["participant_total_weight_min"] == pytest.approx(weights["participant_total_weight_max"])


def test_final_predict_interface_rejects_test_outcome_columns() -> None:
    frame, y = _frame()
    outer_test = pd.DataFrame(
        {
            "participant_group_id": ["HELD-OUT"],
            "signal": [1.0],
            "q1_nominal_4class": [1],
        }
    )
    with pytest.raises(SupervisedLearningContractError, match="outcome-free"):
        refit_logistic_and_predict(
            frame,
            y,
            outer_test,
            feature_scheme=FeatureScheme("signal", ("signal",)),
            selected_c=1.0,
        )


def test_inner_group_count_is_not_silently_reduced() -> None:
    frame, y = _frame()
    subset = frame[frame["participant_group_id"].isin(["P-00", "P-01", "P-02"])].copy()
    y_subset = y[subset.index.to_numpy()]
    with pytest.raises(ModelSelectionError, match="needs 5 participant groups"):
        select_logistic_model(
            subset.reset_index(drop=True),
            y_subset,
            feature_schemes=[FeatureScheme("signal", ("signal",))],
            c_candidates=[1.0],
            n_splits=5,
        )


def test_unexpected_programming_error_in_model_selection_propagates(monkeypatch) -> None:
    frame, y = _frame()

    def _programmer_defect(*args, **kwargs):
        raise TypeError("simulated programmer defect")

    monkeypatch.setattr(supervised_models, "_fit_logistic", _programmer_defect)
    with pytest.raises(TypeError, match="simulated programmer defect"):
        select_logistic_model(
            frame,
            y,
            feature_schemes=[FeatureScheme("signal", ("signal",))],
            c_candidates=[1.0],
            n_splits=4,
        )
