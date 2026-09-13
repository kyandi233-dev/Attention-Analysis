"""Four-class model-selection tests: inner-fold disjointness, train-only preprocessing,
explicit missing-class failure, route-B rules, empty-set fallback, leakage guard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import attention_pipeline.supervised_learning.models_multiclass as multiclass_models
from attention_pipeline.supervised_learning.feature_schemes import FeatureScheme
from attention_pipeline.supervised_learning.models_multiclass import (
    PRIOR_FALLBACK_FEATURE_SET_ID,
    SELECTION_METRIC,
    MulticlassModelSelectionError,
    forward_select_multiclass,
    refit_multiclass_and_predict,
    select_multiclass_logistic,
)
from attention_pipeline.supervised_learning.task import SupervisedLearningContractError


def _balanced_frame(
    *, n_groups: int = 12, rows_per_group: int = 12, seed: int = 3
) -> tuple[pd.DataFrame, np.ndarray]:
    """构造四类齐全、参与者分组的可学习帧（signal 四聚类，noise 无信息）。"""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    labels: list[int] = []
    for group_index in range(n_groups):
        for row_index in range(rows_per_group):
            y = int(row_index % 4) + 1
            rows.append(
                {
                    "participant_group_id": f"P-{group_index:02d}",
                    "signal": 6.0 * (y - 1) + rng.normal(0, 0.2),
                    "noise": rng.normal(0, 1.0),
                }
            )
            labels.append(y)
    return pd.DataFrame(rows), np.asarray(labels, dtype=int)


def test_selection_prefers_the_predeclared_predictive_scheme() -> None:
    frame, y = _balanced_frame()
    result = select_multiclass_logistic(
        frame,
        y,
        feature_schemes=[FeatureScheme("signal", ("signal",)), FeatureScheme("noise", ("noise",))],
        c_candidates=[0.1, 1.0],
        n_splits=4,
        seed=11,
    )
    assert result.feature_scheme.feature_set_id == "signal"
    assert result.selected_c in {0.1, 1.0}
    assert result.audit_dict()["selection_metric"] == SELECTION_METRIC


def test_inner_folds_are_participant_disjoint_and_preprocessing_is_train_only() -> None:
    frame, y = _balanced_frame()
    schemes = [FeatureScheme("signal", ("signal", "noise"))]
    result = select_multiclass_logistic(
        frame, y, feature_schemes=schemes, c_candidates=[1.0], n_splits=4
    )
    assert len(result.inner_fold_audits) == 4
    for audit in result.inner_fold_audits:
        train_groups = set(audit["train_group_ids"])
        valid_groups = set(audit["validation_group_ids"])
        # 内层训练/验证参与者不得重叠——否则验证参与者会参与自身预处理的拟合。
        assert not (train_groups & valid_groups)
        state = audit["preprocessing"]
        assert set(state["fit_group_ids"]) == train_groups
        assert valid_groups.isdisjoint(set(state["fit_group_ids"]))
        assert state["participant_specific_transform"] is False
        assert state["participant_equal_preprocessing"] is True
        # 每个内层训练折都必须四类齐全，并逐折重新拟合（fit_group_ids 随折变化）。
        assert audit["inner_train_missing_classes"] == []


def test_validation_participant_cannot_change_its_own_inner_preprocessing() -> None:
    frame, y = _balanced_frame()
    schemes = [FeatureScheme("signal", ("signal", "noise"))]
    base = select_multiclass_logistic(frame, y, feature_schemes=schemes, c_candidates=[1.0], n_splits=4)

    target_group = "P-00"
    altered = frame.copy()
    mask = altered["participant_group_id"].eq(target_group)
    altered.loc[mask, "signal"] = 1e12
    altered.loc[mask, "noise"] = -1e12
    changed = select_multiclass_logistic(altered, y, feature_schemes=schemes, c_candidates=[1.0], n_splits=4)

    base_audit = next(r for r in base.inner_fold_audits if target_group in r["validation_group_ids"])
    changed_audit = next(r for r in changed.inner_fold_audits if target_group in r["validation_group_ids"])
    assert target_group not in base_audit["train_group_ids"]
    assert base_audit["train_group_ids"] == changed_audit["train_group_ids"]
    assert base_audit["preprocessing"] == changed_audit["preprocessing"]
    assert base_audit["training_weights"] == changed_audit["training_weights"]


def test_training_weights_are_participant_equal_and_mean_one() -> None:
    frame, y = _balanced_frame()
    result = select_multiclass_logistic(
        frame, y, feature_schemes=[FeatureScheme("signal", ("signal",))], c_candidates=[1.0], n_splits=4
    )
    for audit in result.inner_fold_audits:
        weights = audit["training_weights"]
        assert weights["mean_row_weight"] == pytest.approx(1.0)
        assert weights["participant_total_weight_min"] == pytest.approx(
            weights["participant_total_weight_max"]
        )


def test_missing_class_in_a_training_split_is_a_recorded_failure_not_a_three_class_model() -> None:
    """三分类数据必须产生显式失败记录，绝不静默拟合成 3 类模型。"""
    frame, y = _balanced_frame()
    depleted = frame.loc[y != 3].reset_index(drop=True)
    y_depleted = y[y != 3]

    with pytest.raises(MulticlassModelSelectionError, match="candidates failed inner grouped CV"):
        select_multiclass_logistic(
            depleted,
            y_depleted,
            feature_schemes=[FeatureScheme("signal", ("signal",))],
            c_candidates=[1.0],
            n_splits=4,
        )

    # 直接检查内层实现记录的失败原因，确认缺类被写成显式原因而不是被丢弃。
    failures: dict[float, list[str]] = {1.0: []}
    audits: list[dict[str, object]] = []
    multiclass_models._inner_grouped_cv_participant_losses(
        depleted,
        y_depleted,
        scheme=FeatureScheme("signal", ("signal",)),
        candidates=[1.0],
        group_col="participant_group_id",
        n_splits=4,
        max_iter=2000,
        seed=1,
        scheme_index=0,
        fold_audits=audits,
        failures=failures,
        spec=multiclass_models.Q1_MULTICLASS_SPEC,
    )
    assert failures[1.0], "expected an explicit recorded failure for the missing class"
    assert any("does not contain all declared classes" in reason for reason in failures[1.0])
    assert 3 in audits[0]["inner_train_missing_classes"]


def test_final_refit_rejects_outer_test_outcome_columns() -> None:
    frame, y = _balanced_frame()
    for leaked_column in ("q1_nominal_4class", "q1_binary"):
        outer_test = pd.DataFrame(
            {"participant_group_id": ["HELD-OUT"], "signal": [1.0], leaked_column: [1]}
        )
        with pytest.raises(SupervisedLearningContractError, match="outcome-free"):
            refit_multiclass_and_predict(
                frame,
                y,
                outer_test,
                feature_scheme=FeatureScheme("signal", ("signal",)),
                selected_c=1.0,
            )


def test_final_refit_reports_four_class_audit_keys() -> None:
    frame, y = _balanced_frame()
    outer_test = pd.DataFrame(
        {"participant_group_id": ["HELD-OUT", "HELD-OUT"], "signal": [0.0, 18.0], "noise": [0.0, 0.0]}
    )
    fitted = refit_multiclass_and_predict(
        frame,
        y,
        outer_test,
        feature_scheme=FeatureScheme("signal", ("signal",)),
        selected_c=1.0,
    )
    assert set(fitted["model_classes"]) == {1, 2, 3, 4}
    assert fitted["coefficient_scale"] == "post_imputation_participant_equal_standardized_predictors"
    assert fitted["class_probabilities"].shape == (2, 4)
    assert np.allclose(fitted["class_probabilities"].sum(axis=1), 1.0)
    assert set(fitted["standardized_coefficients"]) == {"1", "2", "3", "4"}
    assert set(fitted["intercepts"]) == {"1", "2", "3", "4"}
    assert len(fitted["predicted_class"]) == 2
    assert "HELD-OUT" not in fitted["train_group_ids"]
    assert fitted["uses_features"] is True


def test_prior_fallback_refit_uses_no_features() -> None:
    frame, y = _balanced_frame()
    outer_test = pd.DataFrame({"participant_group_id": ["H-O"] * 3, "signal": [1.0, 2.0, 3.0]})
    fitted = refit_multiclass_and_predict(
        frame,
        y,
        outer_test,
        feature_scheme=None,
        selected_c=None,
    )
    assert fitted["feature_set_id"] == PRIOR_FALLBACK_FEATURE_SET_ID
    assert fitted["uses_features"] is False
    assert fitted["preprocessing"] is None
    assert fitted["class_probabilities"].shape == (3, 4)
    # 三行必须拿到完全相同的常数概率（这正是「无信息基线」的含义）。
    assert np.allclose(fitted["class_probabilities"][0], fitted["class_probabilities"][2])


def test_prior_fallback_refuses_a_selected_c() -> None:
    frame, y = _balanced_frame()
    outer_test = pd.DataFrame({"participant_group_id": ["H-O"], "signal": [1.0]})
    with pytest.raises(SupervisedLearningContractError, match="cannot carry a selected C"):
        refit_multiclass_and_predict(
            frame, y, outer_test, feature_scheme=None, selected_c=1.0
        )


# --------------------------------------------------------------------------- route B
def _forward(monkeypatch, frame, y, candidate_columns, losses, *, c_candidates=(1.0,)):
    """用受控损失替换内层 CV，单独测试 §5.2 的选择规则。

    ``losses`` 的键为 ``(加入候选特征后的特征 ID 元组, C)``；缺失的组合视为不可估
    （返回空损失，从而该候选在该折失败）。
    """
    def _fake_inner(
        outer_train,
        labels,
        *,
        scheme,
        candidates,
        group_col,
        n_splits,
        max_iter,
        seed,
        scheme_index,
        fold_audits,
        failures,
        spec,
    ):
        selected_ids = tuple(
            feature_id
            for feature_id, column in sorted(candidate_columns.items())
            if column in scheme.columns
        )
        # 按列的实际顺序还原特征集合（前向选择是按字典序递增加入的）。
        ordered_ids = tuple(
            fid for fid in sorted(candidate_columns) if candidate_columns[fid] in scheme.columns
        )
        assert set(selected_ids) == set(ordered_ids)
        result: dict[float, dict[str, float]] = {}
        for c in candidates:
            key = (ordered_ids, float(c))
            if key not in losses:
                failures[float(c)].append(f"inner_fold=0: controlled_unestimable for {key}")
                result[float(c)] = {}
                continue
            value = float(losses[key])
            result[float(c)] = {
                f"P-{index:02d}": value + index * 1e-9 for index in range(12)
            }
        return result

    monkeypatch.setattr(multiclass_models, "_inner_grouped_cv_participant_losses", _fake_inner)
    return forward_select_multiclass(
        frame,
        y,
        candidate_columns=candidate_columns,
        c_candidates=c_candidates,
        n_splits=4,
        seed=1,
    )


def test_route_b_accepts_only_strict_improvement(monkeypatch) -> None:
    frame, y = _balanced_frame()
    candidates = {"feature.one": "signal", "feature.two": "noise"}
    # 第一步（从空集出发）必然接受 feature.one；第二步再加入 feature.two 时损失完全相同，
    # 不是严格改善，因此必须停止。feature.one 的字典序更小，所以第一步就选中它。
    losses = {
        (("feature.one",), 1.0): 1.20,
        (("feature.one", "feature.two"), 1.0): 1.20,
    }
    audit = _forward(monkeypatch, frame, y, candidates, losses)
    assert audit["selected_feature_set"] == ["feature.one"]
    assert audit["stop_reason"] == "no_strict_improvement"
    assert audit["steps"][0]["accepted"] is True
    assert audit["steps"][0]["current_loss_before"] is None
    assert audit["steps"][-1]["strict_improvement"] is False
    assert audit["steps"][-1]["accepted"] is False
    assert audit["steps"][-1]["current_loss_before"] == pytest.approx(1.20)
    assert audit["steps"][-1]["best_candidate_loss"] == pytest.approx(1.20)


def test_route_b_accepts_a_strictly_better_second_feature(monkeypatch) -> None:
    frame, y = _balanced_frame()
    candidates = {"feature.one": "signal", "feature.two": "noise"}
    losses = {
        (("feature.one",), 1.0): 1.20,
        (("feature.one", "feature.two"), 1.0): 1.10,
    }
    audit = _forward(monkeypatch, frame, y, candidates, losses)
    assert audit["selected_feature_set"] == ["feature.one", "feature.two"]
    assert audit["steps"][1]["strict_improvement"] is True
    assert audit["steps"][1]["accepted"] is True


def test_route_b_empty_set_is_explicit_when_no_candidate_is_estimable(monkeypatch) -> None:
    """候选全部不可估（例如某折缺类 / 预处理失败）时必须显式终止于空集。"""
    frame, y = _balanced_frame()
    # D 网格中没有任何 C 的得分 -> 每个候选都在该折失败。
    audit = _forward(monkeypatch, frame, y, {"noise": "noise"}, {}, c_candidates=(0.01, 1.0))
    assert audit["selected_feature_set"] == []
    assert audit["empty_feature_set"] is True
    assert audit["selected_c"] is None
    assert audit["selected_inner_participant_macro_multiclass_log_loss"] is None
    assert audit["empty_feature_set_fallback"] == (
        "outer_training_fold_class_prior_constant_prediction"
    )
    assert audit["stop_reason"] == "no_estimable_candidate_in_this_outer_training_fold"


def test_route_b_empty_set_falls_back_to_the_outer_training_class_priors() -> None:
    """空集退化路径必须真的产出训练折类别先验常数预测，并显式标记不使用特征。"""
    frame, y = _balanced_frame()
    outer_test = pd.DataFrame({"participant_group_id": ["H-O"] * 2, "noise": [0.5, -0.5]})
    fitted = refit_multiclass_and_predict(
        frame, y, outer_test, feature_scheme=None, selected_c=None
    )
    assert fitted["feature_set_id"] == PRIOR_FALLBACK_FEATURE_SET_ID
    assert fitted["uses_features"] is False
    assert fitted["preprocessing"] is None
    assert fitted["model_classes"] == [1, 2, 3, 4]
    expected_prior = 0.25
    assert fitted["train_class_priors"] == {
        "1": pytest.approx(expected_prior),
        "2": pytest.approx(expected_prior),
        "3": pytest.approx(expected_prior),
        "4": pytest.approx(expected_prior),
    }
    assert np.allclose(fitted["class_probabilities"], expected_prior)
    # 两行必须拿到完全相同的常数概率——这正是「无信息」的含义。
    assert np.allclose(fitted["class_probabilities"][0], fitted["class_probabilities"][1])


def test_route_b_tie_break_prefers_smaller_c_then_lexicographic_feature_id(monkeypatch) -> None:
    frame, y = _balanced_frame()
    candidates = {"zzz.feature": "signal", "aaa.feature": "noise"}
    # 两个候选在同一 C 下并列 -> 必须选 feature_id 字典序更小的 aaa.feature。
    losses = {
        (("aaa.feature",), 1.0): 1.00,
        (("zzz.feature",), 1.0): 1.00,
    }
    audit = _forward(monkeypatch, frame, y, candidates, losses)
    assert audit["selected_feature_set"] == ["aaa.feature"]
    assert audit["selected_c"] == pytest.approx(1.0)

    # 同一特征的多个 C 并列 -> 必须选更小的 C（0.01）。
    monkeypatch.undo()
    losses_same_feature = {
        (("only.feature",), 0.01): 0.90,
        (("only.feature",), 1.0): 0.90,
    }
    audit_two = _forward(
        monkeypatch,
        frame,
        y,
        {"only.feature": "signal"},
        losses_same_feature,
        c_candidates=(0.01, 1.0),
    )
    assert audit_two["selected_feature_set"] == ["only.feature"]
    assert audit_two["selected_c"] == pytest.approx(0.01)


def test_route_b_records_unavailable_candidate_columns(monkeypatch) -> None:
    frame, y = _balanced_frame()
    # 整列缺失的候选特征不得被选入，但必须记入审计。
    frame = frame.copy()
    frame["dead"] = np.nan
    candidates = {"dead.feature": "dead", "live.feature": "signal"}
    losses = {
        (("live.feature",), 1.0): 0.50,
        (("live.feature", "dead.feature"), 1.0): 0.40,
    }
    audit = _forward(monkeypatch, frame, y, candidates, losses)
    assert audit["unavailable_candidate_columns"] == {
        "dead.feature": "column_all_missing_in_outer_training_fold:dead"
    }
    assert audit["selected_feature_set"] == ["live.feature"]


def test_route_b_missing_column_is_reported_not_ignored(monkeypatch) -> None:
    frame, y = _balanced_frame()
    candidates = {"absent.feature": "not_in_frame", "live.feature": "signal"}
    losses = {(("live.feature",), 1.0): 0.50}
    audit = _forward(monkeypatch, frame, y, candidates, losses)
    assert "absent.feature" in audit["unavailable_candidate_columns"]
    assert audit["available_candidate_feature_ids"] == ["live.feature"]
    assert audit["selected_feature_set"] == ["live.feature"]
