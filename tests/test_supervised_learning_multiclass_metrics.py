"""Four-class metric tests: participant-equal aggregation, per-fold prior baseline,
supplementary discrimination with an absent class, fixed-OOF bootstrap, paired increment."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attention_pipeline.supervised_learning.evaluation import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    EvaluationContractError,
    fixed_oof_participant_bootstrap,
)
from attention_pipeline.supervised_learning.metrics_multiclass import (
    PRIMARY_METRIC_NAME,
    multiclass_bootstrap_summary,
    multiclass_discrimination,
    multiclass_probe_log_loss,
    outer_fold_class_priors,
    paired_multiclass_increment,
    participant_multiclass_log_loss,
    prior_baseline_participant_macro_log_loss,
)
from attention_pipeline.supervised_learning.task import SupervisedLearningContractError
from attention_pipeline.supervised_learning.task_multiclass import Q1_MULTICLASS_SPEC

PROBABILITY_COLUMNS = Q1_MULTICLASS_SPEC.probability_columns


def _oof_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """把 (participant, probe_index, class, probabilities) 规格展开成折外预测存档。"""
    records: list[dict[str, object]] = []
    for row in rows:
        participant = str(row["participant_group_id"])
        probabilities = [float(v) for v in row["probabilities"]]
        record: dict[str, object] = {
            "participant_group_id": participant,
            "session_id": f"{participant}-S1",
            "block_id": "B1",
            "probe_event_id": f"{participant}-{row['probe']}",
            "analysis_set_id": "AS.test",
            "membership_type": "included_missing_aware",
            "model_id": "M1",
            "outer_fold_group": participant,
            "q1_nominal_4class": int(row["q1_nominal_4class"]),
            "model_failed": False,
        }
        for column, value in zip(PROBABILITY_COLUMNS, probabilities, strict=True):
            record[column] = value
        records.append(record)
    return pd.DataFrame(records)


def test_multiclass_probe_log_loss_matches_definition() -> None:
    y = np.array([1, 2, 3, 4])
    proba = np.array(
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.6, 0.2, 0.1],
            [0.1, 0.1, 0.5, 0.3],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )
    loss = multiclass_probe_log_loss(y, proba)
    expected = -np.log([0.7, 0.6, 0.5, 0.25])
    assert np.allclose(loss, expected)


def test_multiclass_probe_log_loss_is_fail_closed() -> None:
    good = np.array([[0.25, 0.25, 0.25, 0.25]])
    with pytest.raises(SupervisedLearningContractError, match="unexpected multiclass labels"):
        multiclass_probe_log_loss(np.array([5]), good)
    with pytest.raises(SupervisedLearningContractError, match="non-finite"):
        multiclass_probe_log_loss(np.array([1]), np.array([[np.nan, 0.0, 0.0, 1.0]]))
    with pytest.raises(SupervisedLearningContractError, match="within \\[0, 1\\]"):
        multiclass_probe_log_loss(np.array([1]), np.array([[1.2, -0.2, 0.0, 0.0]]))
    with pytest.raises(SupervisedLearningContractError, match="shape"):
        multiclass_probe_log_loss(np.array([1]), np.array([[0.2, 0.2, 0.6]]))


def test_participant_aggregation_is_participant_equal_not_probe_equal() -> None:
    # P-A 一个 probe 完全正确；P-B 三个 probe 各 1/3 概率（损失 ln 3）。
    frame = _oof_frame(
        [
            {"participant_group_id": "P-A", "probe": 1, "q1_nominal_4class": 1,
             "probabilities": [1.0, 0.0, 0.0, 0.0]},
            {"participant_group_id": "P-B", "probe": 2, "q1_nominal_4class": 1,
             "probabilities": [1 / 3, 1 / 3, 1 / 3, 0.0]},
            {"participant_group_id": "P-B", "probe": 3, "q1_nominal_4class": 2,
             "probabilities": [1 / 3, 1 / 3, 1 / 3, 0.0]},
            {"participant_group_id": "P-B", "probe": 4, "q1_nominal_4class": 3,
             "probabilities": [1 / 3, 1 / 3, 1 / 3, 0.0]},
        ]
    )
    participant, summary = participant_multiclass_log_loss(frame)
    assert participant["n_probes"].tolist() == [1, 3]
    loss_a, loss_b = participant["mean_multiclass_log_loss"].tolist()
    assert loss_a == pytest.approx(0.0, abs=1e-12)
    assert loss_b == pytest.approx(float(np.log(3.0)))
    # 参与者等权：主指标是两个参与者均值的平均，而不是 4 个 probe 的平均。
    assert summary[PRIMARY_METRIC_NAME] == pytest.approx((loss_a + loss_b) / 2.0)
    probe_equal = -np.log([1.0, 1 / 3, 1 / 3, 1 / 3]).mean()
    assert summary[PRIMARY_METRIC_NAME] != pytest.approx(float(probe_equal))
    assert summary["n_participants"] == 2
    assert summary["aggregation"] == "participant_equal_within_participant_probe_equal"


def test_participant_aggregation_refuses_failed_rows() -> None:
    frame = _oof_frame(
        [{"participant_group_id": "P-A", "probe": 1, "q1_nominal_4class": 1,
          "probabilities": [0.7, 0.1, 0.1, 0.1]}]
    )
    frame["model_failed"] = True
    with pytest.raises(EvaluationContractError, match="failed OOF rows"):
        participant_multiclass_log_loss(frame)


# --------------------------------------------------------------------------- prior baseline
def _prior_train_frame(
    counts: dict[int, int], *, group_col: str = "participant_group_id"
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    index = 0
    for label, count in counts.items():
        for _ in range(count):
            rows.append({group_col: f"T-{index:03d}", "q1_nominal_4class": int(label)})
            index += 1
    return pd.DataFrame(rows)


def test_outer_fold_class_priors_include_absent_classes_as_zero() -> None:
    train = _prior_train_frame({1: 6, 2: 2, 3: 2})
    priors = outer_fold_class_priors(train)
    assert priors[1] == pytest.approx(0.6)
    assert priors[2] == pytest.approx(0.2)
    assert priors[3] == pytest.approx(0.2)
    assert priors[4] == pytest.approx(0.0)


def test_prior_baseline_uses_training_fold_priors_per_fold() -> None:
    """基线必须逐折重算：同一测试折在不同训练先验下给出不同基线值。"""
    test_flat = _oof_frame(
        [
            {"participant_group_id": "H1", "probe": 1, "q1_nominal_4class": 1,
             "probabilities": [0.25, 0.25, 0.25, 0.25]},
            {"participant_group_id": "H2", "probe": 2, "q1_nominal_4class": 2,
             "probabilities": [0.25, 0.25, 0.25, 0.25]},
        ]
    )
    uniform_train = _prior_train_frame({1: 25, 2: 25, 3: 25, 4: 25})
    uniform = prior_baseline_participant_macro_log_loss(uniform_train, test_flat)
    assert uniform["prior_baseline_participant_macro_multiclass_log_loss"] == pytest.approx(
        float(np.log(4.0))
    )

    skewed_train = _prior_train_frame({1: 97, 2: 1, 3: 1, 4: 1})
    skewed = prior_baseline_participant_macro_log_loss(skewed_train, test_flat)
    expected = float(np.mean([-np.log(0.97), -np.log(0.01)]))
    assert skewed["prior_baseline_participant_macro_multiclass_log_loss"] == pytest.approx(expected)
    assert skewed["class_priors"] == {
        "1": pytest.approx(0.97),
        "2": pytest.approx(0.01),
        "3": pytest.approx(0.01),
        "4": pytest.approx(0.01),
    }
    # 两个折结构完全相同的测试集，只因为训练先验不同就得到不同基线——证明基线逐折可算。
    assert uniform["prior_baseline_participant_macro_multiclass_log_loss"] != pytest.approx(
        skewed["prior_baseline_participant_macro_multiclass_log_loss"]
    )


def test_prior_baseline_reports_absent_test_classes_and_floors_zero_priors() -> None:
    train = _prior_train_frame({1: 8, 2: 1, 3: 1})
    test_frame = _oof_frame(
        [{"participant_group_id": "H1", "probe": 1, "q1_nominal_4class": 3,
          "probabilities": [0.25, 0.25, 0.25, 0.25]}]
    )
    baseline = prior_baseline_participant_macro_log_loss(train, test_frame)
    assert baseline["absent_test_classes"] == [1, 2, 4]
    assert baseline["observed_test_classes"] == [3]
    assert baseline["prior_probability_floor_applied"] is True
    assert np.isfinite(
        baseline["prior_baseline_participant_macro_multiclass_log_loss"]
    )


# --------------------------------------------------------------------------- discrimination
def test_discrimination_reports_absent_class_without_crashing() -> None:
    frame = _oof_frame(
        [
            {"participant_group_id": "P-A", "probe": 1, "q1_nominal_4class": 1,
             "probabilities": [0.8, 0.1, 0.05, 0.05]},
            {"participant_group_id": "P-B", "probe": 2, "q1_nominal_4class": 2,
             "probabilities": [0.2, 0.7, 0.05, 0.05]},
            {"participant_group_id": "P-C", "probe": 3, "q1_nominal_4class": 1,
             "probabilities": [0.6, 0.3, 0.05, 0.05]},
            {"participant_group_id": "P-D", "probe": 4, "q1_nominal_4class": 2,
             "probabilities": [0.3, 0.6, 0.05, 0.05]},
        ]
    )
    result = multiclass_discrimination(frame)
    assert result["absent_classes"] == [3, 4]
    assert result["class_absent_from_evaluation_fold"] is True
    assert result["class_absence_note"]
    assert result["class_support"] == {"1": 2, "2": 2, "3": 0, "4": 0}
    # 类别 1/2 可估；类别 3/4 必须显式报告为不可估，而不是被删掉或给 0。
    assert result["per_class_ovr_auroc"]["3"] is None
    assert result["per_class_ovr_auroc"]["4"] is None
    assert result["per_class_log_loss"]["3"] is None
    assert result["per_class_log_loss"]["4"] is None
    assert result["ovr_macro_auroc_estimable_classes"] == [1, 2]
    assert result["ovr_macro_auroc_not_estimable_classes"] == ["3", "4"]
    assert result["ovr_macro_auroc"] == pytest.approx(1.0)
    assert result["balanced_accuracy"] == pytest.approx(1.0)
    # macro-F1 覆盖全部四个声明类别；类别 3/4 在折内缺失，其 F1 按 0 计入宏平均，
    # 因此 (1 + 1 + 0 + 0) / 4 = 0.5，而不是把缺失类别从分母里删掉得到 1.0。
    assert result["macro_f1"] == pytest.approx(0.5)
    matrix = result["confusion_matrix"]
    assert len(matrix) == 4 and all(len(row) == 4 for row in matrix)
    assert result["confusion_matrix_labels"] == [1, 2, 3, 4]
    # 混淆矩阵第 3、4 行（类别 3/4 的真值）全为 0，且没有被抹掉。
    assert matrix[2] == [0, 0, 0, 0]
    assert matrix[3] == [0, 0, 0, 0]
    assert result["n_participants"] == 4


def test_discrimination_handles_a_single_class_evaluation_fold() -> None:
    frame = _oof_frame(
        [
            {"participant_group_id": "P-A", "probe": 1, "q1_nominal_4class": 1,
             "probabilities": [0.9, 0.05, 0.03, 0.02]},
            {"participant_group_id": "P-B", "probe": 2, "q1_nominal_4class": 1,
             "probabilities": [0.8, 0.1, 0.05, 0.05]},
        ]
    )
    result = multiclass_discrimination(frame)
    assert result["absent_classes"] == [2, 3, 4]
    assert result["ovr_macro_auroc"] is None
    assert result["balanced_accuracy"] == pytest.approx(1.0)
    assert result["per_class_log_loss"]["1"] is not None


def test_discrimination_confusion_matrix_is_pooled_and_transposed_correctly() -> None:
    frame = _oof_frame(
        [
            {"participant_group_id": "P-A", "probe": 1, "q1_nominal_4class": 1,
             "probabilities": [0.1, 0.7, 0.1, 0.1]},
            {"participant_group_id": "P-B", "probe": 2, "q1_nominal_4class": 2,
             "probabilities": [0.1, 0.7, 0.1, 0.1]},
        ]
    )
    result = multiclass_discrimination(frame)
    matrix = result["confusion_matrix"]
    # 第 1 行（真值 1）应全部落在第 2 列（预测 2）。
    assert matrix[0] == [0, 1, 0, 0]
    assert matrix[1] == [0, 1, 0, 0]


# --------------------------------------------------------------------------- bootstrap / paired
def test_bootstrap_summary_reuses_the_frozen_fixed_oof_implementation() -> None:
    values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    summary = multiclass_bootstrap_summary(values)
    assert summary["replicates"] == DEFAULT_BOOTSTRAP_REPLICATES
    assert summary["seed"] == DEFAULT_BOOTSTRAP_SEED
    assert summary["confidence_level"] == 0.95
    assert summary["retrain_within_bootstrap"] is False
    assert summary["resampling_unit"] == "participant"
    assert summary["reused_from"] == "evaluation.fixed_oof_participant_bootstrap"
    assert summary["metric"] == PRIMARY_METRIC_NAME
    reference = fixed_oof_participant_bootstrap(values)
    assert summary["point_estimate"] == pytest.approx(reference["point_estimate"])
    assert summary["ci_lower"] == pytest.approx(reference["ci_lower"])
    assert summary["ci_upper"] == pytest.approx(reference["ci_upper"])


def _paired_archive(participant: str, probe: int, label: int, probabilities: list[float]):
    frame = _oof_frame(
        [{"participant_group_id": participant, "probe": probe,
          "q1_nominal_4class": label, "probabilities": probabilities}]
    )
    return frame


def test_paired_increment_definition_and_bootstrap() -> None:
    baseline = pd.concat(
        [
            _paired_archive("P-A", 1, 1, [0.5, 0.5, 0.0, 0.0]),
            _paired_archive("P-B", 2, 2, [0.5, 0.5, 0.0, 0.0]),
        ],
        ignore_index=True,
    ).assign(model_id="route_A")
    added = pd.concat(
        [
            _paired_archive("P-A", 1, 1, [0.9, 0.1, 0.0, 0.0]),
            _paired_archive("P-B", 2, 2, [0.1, 0.9, 0.0, 0.0]),
        ],
        ignore_index=True,
    ).assign(model_id="route_B")
    result = paired_multiclass_increment(baseline, added)
    # 增量 = 基准损失 - 对照损失；对照更好 -> 正增量。
    expected = (-np.log(0.5)) - (-np.log(0.9))
    assert result["overall_increment"] == pytest.approx(float(expected))
    assert result["increment_definition"] == "baseline_log_loss_minus_added_log_loss"
    assert result["bootstrap"]["paired_model_resampling"] is True
    assert result["bootstrap"]["retrain_within_bootstrap"] is False
    assert set(result["participant_increments"]["participant_group_id"]) == {"P-A", "P-B"}
    # 解释边界必须写明这是两种程序的比较，不是单特征的四分类贡献。
    assert "not the increment of any single feature" in result["interpretation_boundary"]


def test_paired_increment_rejects_mismatched_probe_membership() -> None:
    baseline = _paired_archive("P-A", 1, 1, [0.5, 0.5, 0.0, 0.0])
    added = _paired_archive("P-B", 1, 1, [0.5, 0.5, 0.0, 0.0])
    with pytest.raises(EvaluationContractError, match="mismatch"):
        paired_multiclass_increment(baseline, added)


def test_paired_increment_rejects_different_analysis_sets() -> None:
    baseline = _paired_archive("P-A", 1, 1, [0.5, 0.5, 0.0, 0.0])
    added = _paired_archive("P-A", 1, 1, [0.5, 0.5, 0.0, 0.0])
    added["analysis_set_id"] = "AS.other"
    with pytest.raises(EvaluationContractError, match="same analysis_set_id"):
        paired_multiclass_increment(baseline, added)
