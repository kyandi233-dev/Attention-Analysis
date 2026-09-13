"""多类概率诊断层的契约测试（布里尔分数与校准诊断）。

文件：test_probability_diagnostics_multiclass.py
版本：1.0.0
功能：以小型合成折外预测为夹具，机械核验
      ``attention_pipeline.supervised_learning.probability_diagnostics_multiclass``
      的口径与治理声明：布里尔分数定义、无信息基线重建与 fail-closed 对账、
      参与者等权聚合、顶端标签校准总体偏差、可靠性分箱计数闭合、校准斜率可解读性
      守卫、run 身份进入分组键、以及全部非选择性标志。
用法：``python -m pytest tests/test_probability_diagnostics_multiclass.py -q``
依赖：pytest、numpy、pandas
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from attention_pipeline.supervised_learning import probability_diagnostics_multiclass as m
from attention_pipeline.supervised_learning.metrics_multiclass import _safe_macro_auroc
from attention_pipeline.supervised_learning.task_multiclass import Q1_MULTICLASS_SPEC

PROB_COLUMNS = list(Q1_MULTICLASS_SPEC.probability_columns)


def _frame(
    *,
    rows: list[dict[str, object]],
    model_id: str = "full",
    run_id: str = "run-a",
    analysis_set_id: str = "AS.full",
    membership_type: str = "included_missing_aware",
    route: str = "A",
) -> pd.DataFrame:
    """把 ``(participant, fold, label, probability)`` 行扩成诊断层要求的列结构。"""
    records: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        record = {
            "run_id": run_id,
            "analysis_set_id": analysis_set_id,
            "membership_type": membership_type,
            "model_id": model_id,
            "route": route,
            "outer_fold_group": row["fold"],
            "participant_group_id": row["participant"],
            "q1_nominal_4class": row["label"],
            "probe_event_id": f"p{index}",
            "model_failed": False,
        }
        for column, value in zip(PROB_COLUMNS, row["probability"], strict=True):
            record[column] = float(value)
        records.append(record)
    return pd.DataFrame(records)


def _onehot_frame(*, participants: int = 4, probes: int = 3, accuracy: float = 1.0) -> pd.DataFrame:
    """构造一个明确可判别的合成集合：`probability` 在目标类别上取高值。

    高值随 probe 序号在 0.60–0.88 之间变化，因此顶端标签**置信度不是常数**；
    否则校准回归只有一个常数自变量，斜率会退化为 0。
    """
    rows: list[dict[str, object]] = []
    for p in range(participants):
        participant = f"P{p:03d}"
        for i in range(probes):
            label = (i % 4) + 1
            # ``accuracy`` 控制正确率：不正确的 probe 把高概率放到相邻类别上。
            correct = ((p + i) % 10) < int(round(accuracy * 10))
            target = label if correct else (label % 4) + 1
            high = 0.60 + 0.04 * (i % 8)
            probability = [(1.0 - high) / 3.0] * 4
            probability[target - 1] = high
            rows.append(
                {
                    "participant": participant,
                    "fold": participant,
                    "label": label,
                    "probability": probability,
                }
            )
    return _frame(rows=rows)


# ---------------------------------------------------------------------------
# 布里尔分数定义
# ---------------------------------------------------------------------------


def test_probe_brier_matches_hand_computation() -> None:
    """逐 probe 布里尔分数必须等于逐类平方差之和。"""
    labels = np.array([1, 3])
    probability = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.2, 0.3, 0.4]])
    values = m.multiclass_probe_brier(labels, probability)
    expected_first = (0.7 - 1) ** 2 + 0.1**2 + 0.1**2 + 0.1**2
    expected_second = 0.1**2 + 0.2**2 + (0.3 - 1) ** 2 + 0.4**2
    assert values[0] == pytest.approx(expected_first)
    assert values[1] == pytest.approx(expected_second)


def test_probe_brier_bounds() -> None:
    """完美预测为 0；四类均匀预测为 0.75（不是 2）。"""
    labels = np.array([1, 2])
    perfect = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    assert m.multiclass_probe_brier(labels, perfect).tolist() == [0.0, 0.0]
    uniform = np.full((2, 4), 0.25)
    assert m.multiclass_probe_brier(labels, uniform).tolist() == pytest.approx([0.75, 0.75])


def test_participant_brier_per_class_sums_to_overall() -> None:
    """逐类布里尔分数之和必须等于总体布里尔分数（参与者的先均值后等权不改变这一点）。"""
    frame = _onehot_frame()
    values = m.participant_multiclass_brier_values(frame)
    total = sum(values[f"class_{k}"] for k in (1, 2, 3, 4))
    assert np.allclose(total, values["overall"], atol=1e-12)


# ---------------------------------------------------------------------------
# 无信息基线：训练侧先验重建与 fail-closed 对账
# ---------------------------------------------------------------------------


def test_reconstructed_priors_exclude_the_held_out_participant() -> None:
    """外层折 g 的先验必须只由**其它**参与者构成，接触留出参与者即为泄漏。"""
    rows = [
        {"participant": "P0", "fold": "P0", "label": 1, "probability": [0.25] * 4},
        {"participant": "P1", "fold": "P1", "label": 2, "probability": [0.25] * 4},
        {"participant": "P2", "fold": "P2", "label": 2, "probability": [0.25] * 4},
    ]
    priors = m.reconstruct_outer_fold_priors(_frame(rows=rows))
    # 折 P0 的训练侧只有 P1 与 P2，两类各半。
    assert priors["P0"].tolist() == pytest.approx([0.0, 1.0, 0.0, 0.0])
    # 折 P1 的训练侧是 P0（类别 1）与 P2（类别 2）。
    assert priors["P1"].tolist() == pytest.approx([0.5, 0.5, 0.0, 0.0])


def test_prior_constant_log_loss_matches_frozen_metric_definition() -> None:
    """先验常数预测器的对数损失必须与冻结 `multiclass_probe_log_loss` 在同一输入上一致。"""
    from attention_pipeline.supervised_learning.metrics_multiclass import (
        multiclass_probe_log_loss,
    )

    frame = _frame(
        rows=[
            {"participant": "P0", "fold": "P0", "label": 1, "probability": [0.25] * 4},
            {"participant": "P1", "fold": "P1", "label": 2, "probability": [0.25] * 4},
            {"participant": "P2", "fold": "P2", "label": 4, "probability": [0.25] * 4},
        ]
    )
    priors = m.reconstruct_outer_fold_priors(frame)
    diagnostic = m.prior_constant_diagnostics(frame, priors)
    labels = frame["q1_nominal_4class"].to_numpy(dtype=int)
    constant = m._expand_priors(frame, priors)
    direct = multiclass_probe_log_loss(labels, constant)
    assert diagnostic["pooled_probe_log_loss_descriptive"] == pytest.approx(float(np.mean(direct)))
    assert diagnostic["pooled_probe_brier_descriptive"] == pytest.approx(
        float(np.mean(m.multiclass_probe_brier(labels, constant)))
    )


def test_prior_reconstruction_mismatch_fails_closed() -> None:
    """重建先验与冻结基线不一致时必须抛错，而不是静默写出一行看起来合理的数字。"""
    frame = _onehot_frame()
    with pytest.raises(m.MulticlassProbabilityDiagnosticsError, match="do not reproduce"):
        m.build_multiclass_probability_diagnostics(
            frame,
            replicates=20,
            frozen_prior_log_loss={("AS.full", "full"): 0.123456},
        )


def test_prior_reconstruction_match_is_recorded() -> None:
    """给出一致的冻结基线时，行内必须记录对账结果为真。"""
    frame = _onehot_frame()
    priors = m.reconstruct_outer_fold_priors(frame)
    baseline = float(m.prior_constant_diagnostics(frame, priors)["log_loss_macro"])
    diagnostics, _, audit = m.build_multiclass_probability_diagnostics(
        frame,
        replicates=20,
        frozen_prior_log_loss={("AS.full", "full"): baseline},
    )
    assert bool(diagnostics.loc[0, "prior_baseline_log_loss_matches_frozen"]) is True
    assert audit["n_prior_rows_checked_against_frozen"] == 1
    assert audit["n_prior_rows_mismatching_frozen"] == 0


# ---------------------------------------------------------------------------
# 聚合口径与主指标对账
# ---------------------------------------------------------------------------


def test_aggregation_is_participant_equal_not_pooled() -> None:
    """主指标必须是「参与者内 probe 等权 → 参与者间等权」，不得等于合并 probe 均值。"""
    rows: list[dict[str, object]] = []
    # P0 只有一个 probe 且损失极高；P1 有 4 个 probe 且损失为 0。
    rows.append({"participant": "P0", "fold": "P0", "label": 1, "probability": [0.01, 0.33, 0.33, 0.33]})
    for _ in range(4):
        rows.append({"participant": "P1", "fold": "P1", "label": 2, "probability": [0.0, 1.0, 0.0, 0.0]})
    rows.append({"participant": "P2", "fold": "P2", "label": 2, "probability": [0.0, 1.0, 0.0, 0.0]})
    frame = _frame(rows=rows)
    diagnostics, _, _ = m.build_multiclass_probability_diagnostics(frame, replicates=20)
    row = diagnostics.iloc[0]
    pooled = row["pooled_probe_multiclass_brier_descriptive"]
    macro = row["participant_macro_multiclass_brier"]
    assert macro != pytest.approx(pooled)
    # 手工口径：三名参与者各自 probe 均值，再等权。
    # P0：标签 1、概率 [0.01, 0.33, 0.33, 0.33] → 0.99² + 3 × 0.33²；P1/P2 完美 → 0。
    manual = np.mean([0.99**2 + 3 * 0.33**2, 0.0, 0.0])
    assert macro == pytest.approx(manual)


def test_run_id_is_part_of_the_grouping_identity() -> None:
    """两个 run 共用 analysis_set_id / model_id 时必须产出两行，不得被静默合并。"""
    frame_a = _onehot_frame()
    frame_b = _onehot_frame()
    for column in ("run_id",):
        frame_b[column] = "run-b"
    combined = pd.concat([frame_a, frame_b], ignore_index=True)
    diagnostics, _, _ = m.build_multiclass_probability_diagnostics(combined, replicates=20)
    assert len(diagnostics) == 2
    assert sorted(diagnostics["run_id"].tolist()) == ["run-a", "run-b"]


def test_failed_rows_are_excluded() -> None:
    """``model_failed`` 为真的行不得进入任何诊断。"""
    frame = _onehot_frame()
    frame["model_failed"] = [True] + [False] * (len(frame) - 1)
    diagnostics, _, _ = m.build_multiclass_probability_diagnostics(frame, replicates=20)
    assert int(diagnostics.loc[0, "n_probes"]) == len(frame) - 1


def test_requires_label_and_probability_columns() -> None:
    """缺少必需列时必须 fail-closed。"""
    frame = _onehot_frame().drop(columns=["p_q1_multiclass_3"])
    with pytest.raises(m.MulticlassProbabilityDiagnosticsError, match="missing required columns"):
        m.build_multiclass_probability_diagnostics(frame, replicates=20)


def test_rejects_too_few_bins() -> None:
    """分箱数小于 2 时必须拒绝。"""
    with pytest.raises(m.MulticlassProbabilityDiagnosticsError, match="n_bins"):
        m.build_multiclass_probability_diagnostics(_onehot_frame(), n_bins=1, replicates=20)


# ---------------------------------------------------------------------------
# 顶端标签校准与守卫
# ---------------------------------------------------------------------------


def test_top_label_confidence_uses_argmax_class() -> None:
    """顶端标签必须是最大概率对应的**原始类别值**，不是列序号。"""
    frame = _frame(
        rows=[
            {"participant": "P0", "fold": "P0", "label": 4, "probability": [0.1, 0.2, 0.3, 0.4]},
            {"participant": "P1", "fold": "P1", "label": 2, "probability": [0.1, 0.6, 0.2, 0.1]},
        ]
    )
    predicted, confidence, correct = m.top_label_confidence(frame)
    assert predicted.tolist() == [4, 2]
    assert confidence.tolist() == pytest.approx([0.4, 0.6])
    assert correct.tolist() == [1.0, 1.0]


def test_calibration_in_the_large_is_confidence_minus_accuracy() -> None:
    """校准总体偏差 = 参与者内（平均置信度 − 正确率）后再参与者等权。"""
    frame = _frame(
        rows=[
            {"participant": "P0", "fold": "P0", "label": 1, "probability": [0.9, 0.05, 0.03, 0.02]},
            {"participant": "P0", "fold": "P0", "label": 2, "probability": [0.9, 0.05, 0.03, 0.02]},
            {"participant": "P1", "fold": "P1", "label": 3, "probability": [0.1, 0.1, 0.7, 0.1]},
        ]
    )
    values = m.participant_top_label_calibration_in_the_large_values(frame)
    # P0：平均置信度 0.9，正确率 0.5 → +0.4；P1：0.7 − 1.0 → −0.3。
    assert values.tolist() == pytest.approx([0.4, -0.3])


@pytest.mark.parametrize(
    ("status", "low", "high", "expected"),
    [
        ("estimable", 0.62, 0.78, m.CALIBRATION_SLOPE_REPORTABLE),
        ("estimable", 0.41, 0.58, m.CALIBRATION_SLOPE_NOT_REPORTABLE_LOW_DISCRIMINATION),
        ("estimable", float("nan"), float("nan"), m.CALIBRATION_SLOPE_NOT_REPORTABLE_LOW_DISCRIMINATION),
        ("not_estimable_single_outcome", 0.62, 0.78, m.CALIBRATION_SLOPE_NOT_ESTIMABLE),
    ],
)
def test_calibration_slope_guard(
    status: str, low: float, high: float, expected: str
) -> None:
    """守卫只依赖判别力区间，与斜率估计本身无关。"""
    actual, note = m._calibration_reporting_status(
        calibration_status=status, auroc_ci_lower=low, auroc_ci_upper=high
    )
    assert actual == expected
    assert isinstance(note, str) and note


def test_guard_never_hides_the_raw_slope_and_interval() -> None:
    """守卫只加标记：无论判定为哪一种状态，原始斜率/截距列与区间列都必须保留。"""
    diagnostics, _, _ = m.build_multiclass_probability_diagnostics(
        _onehot_frame(participants=6, probes=6, accuracy=0.5), replicates=20
    )
    row = diagnostics.iloc[0]
    assert row["calibration_slope_reporting"] in {
        m.CALIBRATION_SLOPE_REPORTABLE,
        m.CALIBRATION_SLOPE_NOT_REPORTABLE_LOW_DISCRIMINATION,
        m.CALIBRATION_SLOPE_NOT_ESTIMABLE,
    }
    for column in (
        "calibration_intercept",
        "calibration_intercept_ci_lower",
        "calibration_intercept_ci_upper",
        "calibration_slope",
        "calibration_slope_ci_lower",
        "calibration_slope_ci_upper",
    ):
        assert column in diagnostics.columns
    if row["calibration_status"] == "estimable":
        assert math.isfinite(float(row["calibration_slope"]))
        assert math.isfinite(float(row["calibration_slope_ci_lower"]))
        assert math.isfinite(float(row["calibration_slope_ci_upper"]))


def test_discriminating_frame_is_reportable() -> None:
    """真正可判别的合成集合必须被判为可解读，证明守卫不是恒为「不可解读」。"""
    diagnostics, _, _ = m.build_multiclass_probability_diagnostics(
        _onehot_frame(participants=8, probes=8, accuracy=0.9), replicates=40
    )
    row = diagnostics.iloc[0]
    assert row["pooled_probe_ovr_macro_auroc"] > 0.9
    assert row["calibration_slope_reporting"] == m.CALIBRATION_SLOPE_REPORTABLE


def test_pooled_macro_auroc_matches_frozen_helper() -> None:
    """折外合并宏平均 AUROC 必须与冻结多类指标层的同一实现一致。"""
    frame = _onehot_frame(accuracy=0.8)
    labels = frame["q1_nominal_4class"].to_numpy(dtype=int)
    probability = frame[PROB_COLUMNS].to_numpy(dtype=float)
    frozen = _safe_macro_auroc(labels, probability, spec=Q1_MULTICLASS_SPEC)["ovr_macro_auroc"]
    assert m.pooled_ovr_macro_auroc(frame) == pytest.approx(float(frozen))


# ---------------------------------------------------------------------------
# 可靠性分箱与治理声明
# ---------------------------------------------------------------------------


def test_calibration_bin_counts_close() -> None:
    """每个模型的可靠性分箱探针数之和必须等于该模型的探针数。"""
    frame = _onehot_frame(participants=6, probes=6, accuracy=0.7)
    diagnostics, bins, _ = m.build_multiclass_probability_diagnostics(frame, replicates=20)
    total = int(bins["n_probes"].sum())
    assert total == int(diagnostics["n_probes"].sum())


def test_expected_calibration_error_recomputes_from_bins() -> None:
    """期望校准误差必须能由分箱表逐箱还原。"""
    frame = _onehot_frame(participants=6, probes=6, accuracy=0.7)
    diagnostics, bins, _ = m.build_multiclass_probability_diagnostics(frame, n_bins=4, replicates=20)
    total = bins["n_probes"].sum()
    recomputed = float(((bins["n_probes"] / total) * bins["abs_gap"]).sum())
    assert float(diagnostics.loc[0, "expected_calibration_error"]) == pytest.approx(recomputed, abs=1e-12)


def test_brier_skill_score_definition() -> None:
    """布里尔技能分数必须等于 1 − 模型布里尔 / 先验布里尔。"""
    diagnostics, _, _ = m.build_multiclass_probability_diagnostics(
        _onehot_frame(accuracy=0.6), replicates=20
    )
    row = diagnostics.iloc[0]
    expected = 1.0 - row["participant_macro_multiclass_brier"] / row[
        "prior_baseline_participant_macro_multiclass_brier"
    ]
    assert row["brier_skill_score"] == pytest.approx(expected)


def test_paired_brier_difference_uses_participant_clusters() -> None:
    """配对布里尔差必须是先验减模型，且区间包住点估计。"""
    diagnostics, _, _ = m.build_multiclass_probability_diagnostics(
        _onehot_frame(participants=6, probes=5, accuracy=0.6), replicates=40
    )
    row = diagnostics.iloc[0]
    point = row["participant_macro_brier_minus_prior"]
    assert point == pytest.approx(
        row["prior_baseline_participant_macro_multiclass_brier"]
        - row["participant_macro_multiclass_brier"]
    )
    assert row["participant_macro_brier_minus_prior_ci_lower"] <= point
    assert row["participant_macro_brier_minus_prior_ci_upper"] >= point


def test_governance_flags_are_non_selective() -> None:
    """每一行都必须声明未被用于模型/C/特征选择，且预测模型未在自助中重训。"""
    diagnostics, _, _ = m.build_multiclass_probability_diagnostics(
        _onehot_frame(), replicates=20
    )
    assert bool(diagnostics["used_for_model_or_c_selection"].any()) is False
    assert bool(diagnostics["used_for_feature_selection"].any()) is False
    assert bool(diagnostics["prediction_model_retrained_in_bootstrap"].any()) is False
    assert bool(diagnostics["calibration_model_refitted_in_bootstrap"].all()) is True
    assert set(diagnostics["role"]) == {"supplementary_descriptive_diagnostic"}
    assert bool(diagnostics["participant_macro_multiclass_brier"].notna().all()) is True


def test_determinism_within_a_single_process() -> None:
    """同一输入连续两次必须给出逐值相同的诊断表。"""
    frame = _onehot_frame(participants=5, probes=4, accuracy=0.7)
    first, _, _ = m.build_multiclass_probability_diagnostics(frame, replicates=30)
    second, _, _ = m.build_multiclass_probability_diagnostics(frame, replicates=30)
    pd.testing.assert_frame_equal(first, second)


def test_summary_reports_the_guard_distribution() -> None:
    """摘要必须同时给出可解读、低判别力不可解读与不可估三种计数。"""
    diagnostics, _, _ = m.build_multiclass_probability_diagnostics(
        _onehot_frame(), replicates=20
    )
    summary = m.multiclass_diagnostic_summary(diagnostics)
    total = (
        summary["n_calibration_slope_reportable"]
        + summary["n_calibration_slope_not_reportable_low_discrimination"]
        + summary["n_calibration_slope_not_estimable"]
    )
    assert total == summary["n_diagnostic_rows"]
    assert summary["participant_level_auroc_reported"] is False
    assert summary["used_for_model_or_c_selection"] is False
    assert summary["used_for_feature_selection"] is False
