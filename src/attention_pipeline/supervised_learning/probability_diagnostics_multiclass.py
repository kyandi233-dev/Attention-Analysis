"""多类折外概率诊断层：多类布里尔分数与校准诊断（只读后处理）。

文件：probability_diagnostics_multiclass.py
版本：1.0.0
功能：从**已归档**的四分类折外预测（``probe_predictions.csv``）计算预注册
      ``1.16.24`` §4.2 所列的最后一项补充指标——多类布里尔分数（Brier score）
      与校准诊断。具体包括：

      - 逐 probe 多类布里尔分数（平方和形式，取值 [0, 2]），参与者内 probe 等权、
        参与者间等权；
      - 与**逐外层折类别先验常数预测器**（同一无信息基线）在**同一批** probe 上的
        配对布里尔差与布里尔技能分数；
      - 每类布里尔分数与逐类先验参照；
      - 折外合并的 one-vs-rest 宏平均 AUROC 及其参与者簇区间（用于校准斜率的
        **可解读性守卫**，不用于模型选择）；
      - 以「最大概率」为横轴的可靠性分箱表（top-label reliability）与期望校准误差；
      - 顶端标签校准模型 ``logit(1{argmax = y}) = intercept + slope * logit(max p)``
        及其参与者簇区间。

      本模块**只读**：不重训预测模型、不重跑任何 LOSO 折、不参与 C 或候选选择。
      冻结主指标（``participant_macro_multiclass_log_loss``）由冻结评价层产出，
      本层只是重新计算一次用于**对账**，不改变它。

用法：
    from attention_pipeline.supervised_learning.probability_diagnostics_multiclass import (
        build_multiclass_probability_diagnostics,
        multiclass_diagnostic_summary,
    )

依赖：numpy、pandas、scikit-learn

口径与治理（必须与任何本层数字一起报告）：
1. **指标范围属事前预注册**：``1.16.24`` §4.2 明确列入「Brier score（多类）与校准
   诊断」，因此本层不是看到性能之后才新增的评价维度。
2. **多类校准的具体估计量属事后登记的操作化**：``1.16.24`` 未冻结「多类校准」的
   估计量，本模块把二分类线 ``1.16.22`` §3 的同一套做法按类别数推广（顶端标签
   置信度 + 无惩罚逻辑回归），并在 ``1.16.27`` 中如实登记为
   ``POST_HOC_REPORTING_RULE``。该操作化**统一适用于所有行**，不做个案豁免。
3. **校准斜率可解读性守卫沿用 ``1.16.22`` §4.2**：仅当折外合并宏平均 AUROC 的
   95% 区间**排除 0.5** 时才把斜率标为 ``reportable``，否则标为
   ``not_reportable_low_discrimination``。原始斜率与区间始终保留，守卫不隐藏数字。
4. **参与者级 AUROC 不报告**：``1.16.24`` §4.2 已规定；本模块只用折外合并 AUROC。
5. 四类为**无序类别**，本层任何数字都不得写成「注意水平预测」或「四类注意识别」。

术语（首次出现展开，APA 7）：
- 布里尔分数（Brier score）：预测概率与 0/1 结果之差的平方，多类下对全部类别求和；
- 折外（out-of-fold [OOF]）：预测来自未参与该折训练的参与者；
- 置信区间（confidence interval [CI]）。

陷阱（本仓库既有环境事实）：
- 分析虚拟环境内 ``attention-analysis`` 的 editable 安装指向**另一个 worktree**，
  因此裸跑 CLI 会静默 import 错误的代码树。任何 CLI 运行必须先设
  ``$env:PYTHONPATH = "<worktree>\\src"``；pytest 不受影响。
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .evaluation import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE_LEVEL,
    fixed_oof_participant_bootstrap,
)
# ``_safe_macro_auroc`` 是私有名，但本层必须与冻结汇总使用**同一个**宏平均定义，
# 因此刻意复用而不是复制一份实现（见 ``pooled_ovr_macro_auroc`` 的说明）。
from .metrics_multiclass import _safe_macro_auroc, multiclass_probe_log_loss
from .task_multiclass import Q1_MULTICLASS_SPEC, MulticlassTaskSpec

# ---------------------------------------------------------------------------
# 常量：全部集中声明，便于审阅与修改。
# ---------------------------------------------------------------------------

#: 诊断表的结构版本；产物 manifest 与本表同时写入，便于事后核对。
MULTICLASS_DIAGNOSTIC_SCHEMA_VERSION = "1.16.24-multiclass-probability-diagnostics-v1"

#: 分组身份列。``run_id`` **必须**进入分组键：探索性冒烟目录会与正式目录共用
#: ``analysis_set_id`` / ``model_id``，只用后两者分组会把两个不同的 probe 集合
#: 静默合并成一行，所有计数都会错但外观完全合理（``1.16.22`` §4.1 的同类陷阱）。
PARTICIPANT_COLUMN = "participant_group_id"
MODEL_COLUMN = "model_id"
RUN_COLUMN = "run_id"
LABEL_COLUMN = Q1_MULTICLASS_SPEC.source_column
PROBABILITY_COLUMNS: tuple[str, ...] = Q1_MULTICLASS_SPEC.probability_columns
FOLD_COLUMN = "outer_fold_group"
ROUTE_COLUMN = "route"
FAILED_COLUMN = "model_failed"
MEMBERSHIP_COLUMN = "membership_type"

#: 可靠性表默认分箱数（与二分类诊断层保持一致）。默认值说明：10 个概率分位箱；
#: 可调范围 2–20；箱数越少区间越稳但越粗，报告必须同时给出实际生效箱数。
DEFAULT_CALIBRATION_BINS = 10

#: logit 变换的概率截断，避免 log(0)。
_LOGIT_EPSILON = 1e-6
#: 先验概率的加性下限，与 ``metrics_multiclass`` 的无信息基线口径一致。
_PROBABILITY_FLOOR = 1e-12
#: 校准斜率可解读性状态（与 ``probability_diagnostics`` 同名同义）。
CALIBRATION_SLOPE_REPORTABLE = "reportable"
CALIBRATION_SLOPE_NOT_REPORTABLE_LOW_DISCRIMINATION = "not_reportable_low_discrimination"
CALIBRATION_SLOPE_NOT_ESTIMABLE = "not_estimable"


class MulticlassProbabilityDiagnosticsError(ValueError):
    """折外预测无法产出一致诊断时抛出（fail-closed，不做静默修补）。"""


# ---------------------------------------------------------------------------
# 输入清洗与基础量
# ---------------------------------------------------------------------------


def _valid_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """只保留成功、有限、按合同编码的折外行。

    参数：
        predictions: 拼接后的 ``probe_predictions.csv`` 内容。
    返回：
        过滤后的副本；标签转为整数，四个概率列转为浮点。
    异常：
        MulticlassProbabilityDiagnosticsError: 必需列缺失，或过滤后无任何可用行。
    """
    required = [
        MODEL_COLUMN,
        PARTICIPANT_COLUMN,
        LABEL_COLUMN,
        FOLD_COLUMN,
        *PROBABILITY_COLUMNS,
    ]
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise MulticlassProbabilityDiagnosticsError(
            f"predictions missing required columns: {missing}"
        )

    frame = predictions.copy()
    if FAILED_COLUMN in frame.columns:
        # CSV 往返可能把布尔写成字符串，两种都接受。
        failed = frame[FAILED_COLUMN]
        if failed.dtype == object:
            failed = failed.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        frame = frame[~failed.astype(bool)]
    frame[LABEL_COLUMN] = pd.to_numeric(frame[LABEL_COLUMN], errors="coerce")
    for column in PROBABILITY_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=[LABEL_COLUMN, PARTICIPANT_COLUMN, FOLD_COLUMN])
    frame = frame[frame[LABEL_COLUMN].isin(list(Q1_MULTICLASS_SPEC.sorted_classes))]
    frame = frame.dropna(subset=list(PROBABILITY_COLUMNS))
    probability = frame.loc[:, list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    keep = np.isfinite(probability).all(axis=1) & (probability >= 0.0).all(axis=1)
    keep &= (probability <= 1.0).all(axis=1)
    frame = frame[keep]
    if frame.empty:
        raise MulticlassProbabilityDiagnosticsError("no usable out-of-fold predictions remain")
    frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype(int)
    frame[PARTICIPANT_COLUMN] = frame[PARTICIPANT_COLUMN].astype(str)
    frame[FOLD_COLUMN] = frame[FOLD_COLUMN].astype(str)
    return frame.reset_index(drop=True)


def _labels_and_probabilities(
    frame: pd.DataFrame,
    *,
    spec: MulticlassTaskSpec = Q1_MULTICLASS_SPEC,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 ``(整数标签, (n, 4) 概率矩阵)``，列顺序为 ``spec.sorted_classes``。"""
    labels = frame[LABEL_COLUMN].to_numpy(dtype=int)
    probability = frame.loc[:, list(spec.probability_columns)].to_numpy(dtype=float)
    return labels, probability


def multiclass_probe_brier(y_true: np.ndarray, proba_matrix: np.ndarray) -> np.ndarray:
    """逐 probe 多类布里尔分数：``sum_k (p_k - 1{y = k})^2``。

    参数：
        y_true: ``(n,)`` 整数类别标签。
        proba_matrix: ``(n, k)`` 概率矩阵，列顺序与 ``sorted_classes`` 一致。
    返回：``(n,)`` 浮点数组，取值在 [0, 2]（k 类时上界为 2）。
    说明：与对数损失不同，布里尔分数对**任何**参与者都可估，不存在单类不可估问题。
    """
    truth = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba_matrix, dtype=float)
    if proba.ndim != 2:
        raise MulticlassProbabilityDiagnosticsError("class probabilities must be a 2D array")
    if proba.shape[0] != truth.shape[0]:
        raise MulticlassProbabilityDiagnosticsError(
            "label count and probability row count do not match"
        )
    onehot = np.zeros_like(proba)
    for index, label in enumerate(Q1_MULTICLASS_SPEC.sorted_classes):
        onehot[:, index] = (truth == int(label)).astype(float)
    return ((proba - onehot) ** 2).sum(axis=1)


def _participant_mean(frame: pd.DataFrame, values: np.ndarray, name: str) -> np.ndarray:
    """把逐 probe 数值先按参与者求均值，再按参与者名排序返回数组。"""
    working = frame[[PARTICIPANT_COLUMN]].copy()
    working[name] = np.asarray(values, dtype=float)
    return working.groupby(PARTICIPANT_COLUMN, sort=True)[name].mean().to_numpy(dtype=float)


def participant_multiclass_brier_values(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """每位参与者的多类布里尔分数：总体与逐类。

    参数：
        frame: 单模型（单 run、单分析集合）的折外行。
    返回：
        ``{"overall": (n_participants,), "class_1": ..., ..., "class_4": ...}``，
        每一列都是「参与者内 probe 等权」的均值。
    """
    labels, probability = _labels_and_probabilities(frame)
    per_probe = multiclass_probe_brier(labels, probability)
    result = {"overall": _participant_mean(frame, per_probe, "__brier")}
    for index, label in enumerate(Q1_MULTICLASS_SPEC.sorted_classes):
        onehot = (labels == int(label)).astype(float)
        per_class = (probability[:, index] - onehot) ** 2
        result[f"class_{int(label)}"] = _participant_mean(frame, per_class, "__brier_class")
    return result


# ---------------------------------------------------------------------------
# 无信息基线：逐外层折类别先验常数预测器（与主指标完全同一批 probe）
# ---------------------------------------------------------------------------


def reconstruct_outer_fold_priors(
    frame: pd.DataFrame,
    *,
    spec: MulticlassTaskSpec = Q1_MULTICLASS_SPEC,
) -> dict[str, np.ndarray]:
    """从归档折外预测重建每个外层折的**训练侧**类别先验向量。

    参数：
        frame: 该运行（同一分析集合、同一批 probe）的折外行，需含外层折列与真实标签。
        spec: 四分类任务合同。
    返回：``{外层折: (4,) 先验向量}``，先验已做加性下限截断并重新归一化。

    方法说明（为什么这样重建是精确的）：
        四分类线沿用二分类线的**留一参与者**外层划分，每名参与者恰好被留出一次，
        因此外层折 ``g`` 的训练集就是「所有外层折不等于 ``g`` 的参与者」的全部合法
        probe——这些行正好也完整出现在归档里。于是训练侧先验可以逐字节精确重建，
        不依赖任何特征、也不接触被留出的那一名参与者。
    异常：
        MulticlassProbabilityDiagnosticsError: 某个外层折剔除自身后没有训练行。
    """
    labels = frame[LABEL_COLUMN].to_numpy(dtype=int)
    folds = frame[FOLD_COLUMN].astype(str).to_numpy()
    classes = list(spec.sorted_classes)
    priors: dict[str, np.ndarray] = {}
    for fold in sorted(set(folds.tolist())):
        train_labels = labels[folds != fold]
        if train_labels.size == 0:
            raise MulticlassProbabilityDiagnosticsError(
                f"outer fold {fold!r} has no training rows outside itself"
            )
        counts = np.array(
            [float((train_labels == int(label)).sum()) for label in classes], dtype=float
        )
        vector = counts / counts.sum()
        vector = np.maximum(vector, _PROBABILITY_FLOOR)
        priors[fold] = vector / vector.sum()
    return priors


def _expand_priors(
    frame: pd.DataFrame,
    priors: Mapping[str, np.ndarray],
    *,
    spec: MulticlassTaskSpec = Q1_MULTICLASS_SPEC,
) -> np.ndarray:
    """把逐折先验展开成与 ``frame`` 行对齐的常数概率矩阵。"""
    folds = frame[FOLD_COLUMN].astype(str).to_numpy()
    missing = sorted(set(folds.tolist()) - set(priors.keys()))
    if missing:
        raise MulticlassProbabilityDiagnosticsError(
            f"outer folds without a reconstructed prior: {missing}"
        )
    matrix = np.vstack([np.asarray(priors[fold], dtype=float) for fold in folds])
    if matrix.shape[1] != len(spec.sorted_classes):
        raise MulticlassProbabilityDiagnosticsError("prior vector width does not match classes")
    return matrix


def prior_constant_diagnostics(
    frame: pd.DataFrame,
    priors: Mapping[str, np.ndarray],
    *,
    spec: MulticlassTaskSpec = Q1_MULTICLASS_SPEC,
) -> dict[str, object]:
    """在**同一批** probe 上评估逐折类别先验常数预测器的对数损失与布里尔分数。

    参数：
        frame: 单模型折外行。
        priors: ``reconstruct_outer_fold_priors`` 的输出。
        spec: 四分类任务合同。
    返回：含参与者级数值、macro 点估计与逐类布里尔参照的字典。
    """
    labels, _ = _labels_and_probabilities(frame, spec=spec)
    constant = _expand_priors(frame, priors, spec=spec)
    log_loss = multiclass_probe_log_loss(labels, constant, spec=spec)
    brier = multiclass_probe_brier(labels, constant)
    log_loss_values = _participant_mean(frame, log_loss, "__prior_ll")
    brier_values = _participant_mean(frame, brier, "__prior_bs")
    result = {
        "log_loss_participant_values": log_loss_values,
        "log_loss_macro": float(log_loss_values.mean()),
        "brier_participant_values": brier_values,
        "brier_macro": float(brier_values.mean()),
        "pooled_probe_log_loss_descriptive": float(np.mean(log_loss)),
        "pooled_probe_brier_descriptive": float(np.mean(brier)),
    }
    for index, label in enumerate(spec.sorted_classes):
        onehot = (labels == int(label)).astype(float)
        per_class = (constant[:, index] - onehot) ** 2
        result[f"class_{int(label)}_brier_macro"] = float(
            _participant_mean(frame, per_class, "__prior_bs_class").mean()
        )
    return result


# ---------------------------------------------------------------------------
# 校准诊断
# ---------------------------------------------------------------------------


def _logit(probabilities: np.ndarray) -> np.ndarray:
    """带截断的 logit 变换，保证有限。"""
    clipped = np.clip(np.asarray(probabilities, dtype=float), _LOGIT_EPSILON, 1.0 - _LOGIT_EPSILON)
    return np.log(clipped / (1.0 - clipped))


def top_label_confidence(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 ``(预测类别, 最大概率, 是否正确)``。

    参数：
        frame: 单模型折外行。
    返回：三个等长数组；``预测类别`` 为 argmax 对应的原始类别值。
    """
    labels, probability = _labels_and_probabilities(frame)
    index = probability.argmax(axis=1)
    predicted = np.array(
        [Q1_MULTICLASS_SPEC.sorted_classes[i] for i in index.tolist()], dtype=int
    )
    confidence = probability.max(axis=1)
    return predicted, confidence, (predicted == labels).astype(float)


def participant_top_label_calibration_in_the_large_values(frame: pd.DataFrame) -> np.ndarray:
    """每位参与者的「平均置信度 − 实际正确率」（顶端标签口径的校准总体偏差）。"""
    _, confidence, correct = top_label_confidence(frame)
    working = frame[[PARTICIPANT_COLUMN]].copy()
    working["__confidence"] = confidence
    working["__correct"] = correct
    grouped = working.groupby(PARTICIPANT_COLUMN, sort=True)
    return (grouped["__confidence"].mean() - grouped["__correct"].mean()).to_numpy(dtype=float)


def calibration_model_top_label(frame: pd.DataFrame) -> tuple[float, float, str]:
    """拟合 ``logit(1{argmax = y}) = intercept + slope * logit(max p)``。

    参数：
        frame: 单模型折外行。
    返回：``(截距, 斜率, 状态)``；不可估时返回 NaN 与状态字符串。
    说明：
        与二分类诊断层一致，使用**准无惩罚**（``C = 1e6``）逻辑回归：把
        ``penalty=None`` 换成超大 ``C`` 以兼容本仓库支持的多个 scikit-learn 版本。
        无判别力时该拟合数值不稳定，因此斜率是否可解读由 AUROC 守卫单独决定。
    """
    _, confidence, correct = top_label_confidence(frame)
    if len(np.unique(correct)) < 2:
        return math.nan, math.nan, "not_estimable_single_outcome"
    features = _logit(confidence).reshape(-1, 1)
    if not np.isfinite(features).all():
        return math.nan, math.nan, "not_estimable_nonfinite_logit"
    try:
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        model.fit(features, correct.astype(int))
    except Exception:  # pragma: no cover - 防御性：收敛病态
        return math.nan, math.nan, "not_estimable_calibration_fit_failed"
    return float(model.intercept_[0]), float(model.coef_[0][0]), "estimable"


def _cluster_bootstrap_calibration(
    frame: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, float | int]:
    """校准模型的参与者簇百分位自助。

    预测模型**从不**重拟合；只有这个小型校准诊断在每次重抽样上重新拟合，
    该事实随每一行输出（``calibration_model_refitted_in_bootstrap``）。
    """
    participants = np.sort(frame[PARTICIPANT_COLUMN].astype(str).unique())
    empty: dict[str, float | int] = {
        "intercept_ci_lower": math.nan,
        "intercept_ci_upper": math.nan,
        "slope_ci_lower": math.nan,
        "slope_ci_upper": math.nan,
        "n_valid_replicates": 0,
    }
    if len(participants) < 2:
        return empty
    blocks = {name: group for name, group in frame.groupby(frame[PARTICIPANT_COLUMN].astype(str))}
    rng = np.random.default_rng(int(seed))
    intercepts: list[float] = []
    slopes: list[float] = []
    for _ in range(int(replicates)):
        draw = rng.integers(0, len(participants), size=len(participants))
        resampled = pd.concat([blocks[participants[i]] for i in draw], ignore_index=True)
        intercept, slope, status = calibration_model_top_label(resampled)
        if status == "estimable" and math.isfinite(intercept) and math.isfinite(slope):
            intercepts.append(intercept)
            slopes.append(slope)
    if not intercepts:
        return empty
    alpha = (1.0 - float(confidence_level)) / 2.0
    intercept_low, intercept_high = np.quantile(intercepts, [alpha, 1.0 - alpha])
    slope_low, slope_high = np.quantile(slopes, [alpha, 1.0 - alpha])
    return {
        "intercept_ci_lower": float(intercept_low),
        "intercept_ci_upper": float(intercept_high),
        "slope_ci_lower": float(slope_low),
        "slope_ci_upper": float(slope_high),
        "n_valid_replicates": len(intercepts),
    }


def pooled_ovr_macro_auroc(frame: pd.DataFrame) -> float:
    """折外合并的 one-vs-rest 宏平均 AUROC。

    说明：这里**刻意**直接复用冻结多类指标层的同一实现
    （``metrics_multiclass._safe_macro_auroc``），而不是另写一份：该函数定义了
    「某类在样本内缺失时不计入宏平均、全部缺失才返回不可估」的规则，本层的
    折外合并 AUROC 必须与冻结汇总逐位一致，才能作为校准斜率守卫的依据。
    返回 NaN 仅当没有任何类别可估。
    """
    labels, probability = _labels_and_probabilities(frame)
    result = _safe_macro_auroc(labels, probability, spec=Q1_MULTICLASS_SPEC)
    value = result.get("ovr_macro_auroc")
    if value is None:
        return math.nan
    return float(value)


def _bootstrap_pooled_auroc(
    frame: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, float | int]:
    """折外合并宏平均 AUROC 的参与者簇自助区间。

    说明：本区间**只**用于校准斜率的可解读性守卫与描述性报告；``1.16.24`` §4.2
    已禁止报告参与者级 AUROC，因此这里不产生任何参与者级数值。
    """
    participants = np.sort(frame[PARTICIPANT_COLUMN].astype(str).unique())
    point = pooled_ovr_macro_auroc(frame)
    if len(participants) < 2 or not math.isfinite(point):
        return {
            "point_estimate": point,
            "ci_lower": math.nan,
            "ci_upper": math.nan,
            "n_valid_replicates": 0,
        }
    blocks = {name: group for name, group in frame.groupby(frame[PARTICIPANT_COLUMN].astype(str))}
    rng = np.random.default_rng(int(seed))
    draws: list[float] = []
    for _ in range(int(replicates)):
        index = rng.integers(0, len(participants), size=len(participants))
        resampled = pd.concat([blocks[participants[i]] for i in index], ignore_index=True)
        value = pooled_ovr_macro_auroc(resampled)
        if math.isfinite(value):
            draws.append(value)
    if not draws:
        return {
            "point_estimate": point,
            "ci_lower": math.nan,
            "ci_upper": math.nan,
            "n_valid_replicates": 0,
        }
    alpha = (1.0 - float(confidence_level)) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    return {
        "point_estimate": point,
        "ci_lower": float(low),
        "ci_upper": float(high),
        "n_valid_replicates": len(draws),
    }


def _calibration_reporting_status(
    *,
    calibration_status: str,
    auroc_ci_lower: float,
    auroc_ci_upper: float,
) -> tuple[str, str]:
    """判断校准斜率是否可解读，并给出理由文本（沿用 ``1.16.22`` §4.2）。

    参数：
        calibration_status: 校准模型自身的可估状态。
        auroc_ci_lower: 折外合并宏平均 AUROC 的区间下界。
        auroc_ci_upper: 折外合并宏平均 AUROC 的区间上界。
    返回：``(状态, 说明)``。
    说明：规则**不依赖斜率估计本身**，只依赖判别力区间，因此无法被调成想要的
        斜率符号或大小；规则对全部行统一适用，且作用方向是限制解释。
    """
    if calibration_status != "estimable":
        return (
            CALIBRATION_SLOPE_NOT_ESTIMABLE,
            f"the calibration model itself was not estimable ({calibration_status})",
        )
    if not (math.isfinite(auroc_ci_lower) and math.isfinite(auroc_ci_upper)):
        return (
            CALIBRATION_SLOPE_NOT_REPORTABLE_LOW_DISCRIMINATION,
            "the pooled out-of-fold macro AUROC interval could not be estimated, so "
            "discrimination is unestablished",
        )
    if auroc_ci_lower > 0.5:
        return (
            CALIBRATION_SLOPE_REPORTABLE,
            "the pooled out-of-fold macro AUROC 95% interval excludes 0.5, so discrimination "
            "is established and the slope may be read against perfect calibration "
            "(slope 1, intercept 0); the slope interval must still be consulted",
        )
    return (
        CALIBRATION_SLOPE_NOT_REPORTABLE_LOW_DISCRIMINATION,
        "the pooled out-of-fold macro AUROC 95% interval includes 0.5; the unpenalised "
        "calibration slope is then numerically unstable and must not be read as "
        "over-confidence or as inverted probabilities",
    )


def _reliability_bins(
    frame: pd.DataFrame,
    *,
    n_bins: int,
) -> tuple[list[dict[str, object]], float, int]:
    """顶端标签可靠性分箱表与期望校准误差。

    参数：
        frame: 单模型折外行。
        n_bins: 请求的分位箱数（可调范围 2–20）。
    返回：``(行列表, 期望校准误差, 实际生效箱数)``。
    说明：分箱按**置信度的概率分位**切分；置信度高度集中时实际箱数会少于请求箱数，
        因此实际生效箱数必须随表报告。
    """
    _, confidence, correct = top_label_confidence(frame)
    per_probe_brier = multiclass_probe_brier(*_labels_and_probabilities(frame))
    edges = np.quantile(confidence, np.linspace(0.0, 1.0, int(n_bins) + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        return [], math.nan, 0
    working = frame[[PARTICIPANT_COLUMN]].copy()
    working["__confidence"] = confidence
    working["__correct"] = correct
    working["__brier"] = per_probe_brier
    binned = pd.cut(working["__confidence"], bins=edges, include_lowest=True, duplicates="drop")
    rows: list[dict[str, object]] = []
    total = float(len(working))
    ece = 0.0
    categories = list(binned.cat.categories)
    for interval, block in working.groupby(binned, observed=True):
        n_probes = int(len(block))
        mean_confidence = float(block["__confidence"].mean())
        observed_accuracy = float(block["__correct"].mean())
        gap = observed_accuracy - mean_confidence
        ece += (n_probes / total) * abs(gap)
        rows.append(
            {
                "bin_index": int(categories.index(interval)),
                "bin_lower": float(interval.left),
                "bin_upper": float(interval.right),
                "n_probes": n_probes,
                "mean_confidence": mean_confidence,
                "observed_accuracy": observed_accuracy,
                "observed_minus_confidence": gap,
                "abs_gap": abs(gap),
                "mean_multiclass_brier": float(block["__brier"].mean()),
            }
        )
    return rows, float(ece), len(rows)


def _bootstrap_row(values: np.ndarray, *, replicates: int, seed: int, confidence_level: float):
    """对参与者级数值做固定 OOF 参与者簇百分位自助（与冻结评价层同一实现）。"""
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        return {
            "point_estimate": math.nan,
            "ci_lower": math.nan,
            "ci_upper": math.nan,
            "n_participants": 0,
            "n_valid_replicates": 0,
        }
    return fixed_oof_participant_bootstrap(
        values, replicates=replicates, seed=seed, confidence_level=confidence_level
    )


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def build_multiclass_probability_diagnostics(
    predictions: pd.DataFrame,
    *,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    frozen_prior_log_loss: Mapping[tuple[str, str], float] | None = None,
    prior_log_loss_tolerance: float = 1e-9,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """从归档折外预测产出多类布里尔分数与校准诊断。

    参数：
        predictions: 已拼接的折外预测行（可含多个运行目录）。
        n_bins: 可靠性分箱数，默认 10；可调范围 2–20。
        replicates: 自助重抽样次数，默认复用冻结值 1000。
        seed: 自助随机种子，默认复用冻结值 20260830。
        confidence_level: 置信水平，默认 0.95。
        frozen_prior_log_loss: 冻结汇总里 ``(analysis_set_id, model_id) ->
            prior_baseline_participant_macro_multiclass_log_loss``；给出时会对账
            重建先验，不一致则 fail-closed。
        prior_log_loss_tolerance: 对账容差，默认 1e-9。
    返回：
        ``(diagnostics, calibration_bins, audit)``：
        diagnostics 每行一个 ``(run, 分析集合, 模型, 隶属类型, 路线)``；
        calibration_bins 每行一个模型的一个置信度分箱；
        audit 记录重建先验的对账结果与拒绝原因。
    异常：
        MulticlassProbabilityDiagnosticsError: 分箱数越界、缺少外层折列、
            或先验重建与冻结值不一致。
    """
    if int(n_bins) < 2:
        raise MulticlassProbabilityDiagnosticsError("n_bins must be at least 2")
    frame = _valid_predictions(predictions)

    group_columns = [MODEL_COLUMN]
    for optional in (RUN_COLUMN, "analysis_set_id", MEMBERSHIP_COLUMN, ROUTE_COLUMN):
        if optional in frame.columns:
            group_columns.append(optional)

    diagnostic_rows: list[dict[str, object]] = []
    bin_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identity = dict(zip(group_columns, keys))
        group = group.reset_index(drop=True)

        labels, probability = _labels_and_probabilities(group)

        # --- 主指标对账 + 无信息基线（同一批 probe） ---
        probe_log_loss = multiclass_probe_log_loss(labels, probability)
        log_loss_values = _participant_mean(group, probe_log_loss, "__ll")
        priors = reconstruct_outer_fold_priors(group)
        prior = prior_constant_diagnostics(group, priors)
        frozen_value = None
        matches_frozen = None
        if frozen_prior_log_loss is not None:
            key = (
                str(identity.get("analysis_set_id")),
                str(identity.get(MODEL_COLUMN)),
            )
            if key in frozen_prior_log_loss:
                frozen_value = float(frozen_prior_log_loss[key])
                matches_frozen = bool(
                    abs(float(prior["log_loss_macro"]) - frozen_value)
                    <= float(prior_log_loss_tolerance)
                )
                if not matches_frozen:
                    raise MulticlassProbabilityDiagnosticsError(
                        "reconstructed outer-fold class priors do not reproduce the frozen "
                        f"baseline for {key}: reconstructed={prior['log_loss_macro']!r} "
                        f"frozen={frozen_value!r}"
                    )
        audit_rows.append(
            {
                **identity,
                "n_outer_folds": int(group[FOLD_COLUMN].nunique()),
                "reconstructed_prior_log_loss": float(prior["log_loss_macro"]),
                "frozen_prior_log_loss": frozen_value,
                "matches_frozen": matches_frozen,
            }
        )

        # --- 布里尔分数 ---
        brier = participant_multiclass_brier_values(group)
        brier_boot = _bootstrap_row(
            brier["overall"], replicates=replicates, seed=seed, confidence_level=confidence_level
        )
        prior_brier_boot = _bootstrap_row(
            prior["brier_participant_values"],
            replicates=replicates,
            seed=seed,
            confidence_level=confidence_level,
        )
        # 配对差：先验布里尔 − 模型布里尔，正值表示模型优于无信息基线。
        improvement = np.asarray(prior["brier_participant_values"], dtype=float) - np.asarray(
            brier["overall"], dtype=float
        )
        improvement_boot = _bootstrap_row(
            improvement, replicates=replicates, seed=seed, confidence_level=confidence_level
        )
        skill = (
            1.0 - float(brier_boot["point_estimate"]) / float(prior_brier_boot["point_estimate"])
            if math.isfinite(float(prior_brier_boot["point_estimate"]))
            and float(prior_brier_boot["point_estimate"]) != 0.0
            else math.nan
        )

        # --- 校准 ---
        auroc_boot = _bootstrap_pooled_auroc(
            group, replicates=replicates, seed=seed, confidence_level=confidence_level
        )
        citl_boot = _bootstrap_row(
            participant_top_label_calibration_in_the_large_values(group),
            replicates=replicates,
            seed=seed,
            confidence_level=confidence_level,
        )
        intercept, slope, calibration_status = calibration_model_top_label(group)
        calibration_boot = _cluster_bootstrap_calibration(
            group, replicates=replicates, seed=seed, confidence_level=confidence_level
        )
        slope_reporting, slope_note = _calibration_reporting_status(
            calibration_status=calibration_status,
            auroc_ci_lower=float(auroc_boot["ci_lower"]),
            auroc_ci_upper=float(auroc_boot["ci_upper"]),
        )
        predicted, confidence, correct = top_label_confidence(group)
        bin_frame_rows, ece, n_bins_effective = _reliability_bins(group, n_bins=int(n_bins))

        row: dict[str, object] = {
            **identity,
            "schema_version": MULTICLASS_DIAGNOSTIC_SCHEMA_VERSION,
            "n_participants": int(group[PARTICIPANT_COLUMN].nunique()),
            "n_probes": int(len(group)),
            "n_outer_folds": int(group[FOLD_COLUMN].nunique()),
            # 主指标（重新计算，仅用于与冻结汇总对账，不取代它）
            "participant_macro_multiclass_log_loss_recomputed": float(np.mean(log_loss_values)),
            "prior_baseline_participant_macro_multiclass_log_loss_recomputed": float(
                prior["log_loss_macro"]
            ),
            "frozen_prior_baseline_participant_macro_multiclass_log_loss": frozen_value,
            "prior_baseline_log_loss_matches_frozen": matches_frozen,
            # 布里尔分数
            "participant_macro_multiclass_brier": brier_boot["point_estimate"],
            "participant_macro_multiclass_brier_ci_lower": brier_boot["ci_lower"],
            "participant_macro_multiclass_brier_ci_upper": brier_boot["ci_upper"],
            "pooled_probe_multiclass_brier_descriptive": float(np.mean(
                multiclass_probe_brier(labels, probability)
            )),
            "prior_baseline_participant_macro_multiclass_brier": prior_brier_boot["point_estimate"],
            "prior_baseline_participant_macro_multiclass_brier_ci_lower": prior_brier_boot[
                "ci_lower"
            ],
            "prior_baseline_participant_macro_multiclass_brier_ci_upper": prior_brier_boot[
                "ci_upper"
            ],
            "brier_skill_score": skill,
            "participant_macro_brier_minus_prior": improvement_boot["point_estimate"],
            "participant_macro_brier_minus_prior_ci_lower": improvement_boot["ci_lower"],
            "participant_macro_brier_minus_prior_ci_upper": improvement_boot["ci_upper"],
            # 逐类布里尔分数
            "participant_macro_brier_class_1": float(brier["class_1"].mean()),
            "participant_macro_brier_class_2": float(brier["class_2"].mean()),
            "participant_macro_brier_class_3": float(brier["class_3"].mean()),
            "participant_macro_brier_class_4": float(brier["class_4"].mean()),
            "prior_baseline_brier_class_1": prior["class_1_brier_macro"],
            "prior_baseline_brier_class_2": prior["class_2_brier_macro"],
            "prior_baseline_brier_class_3": prior["class_3_brier_macro"],
            "prior_baseline_brier_class_4": prior["class_4_brier_macro"],
            # 类别构成与预测分布（无序类别，仅用于解释模型在预测哪一类）
            **{
                f"class_support_{int(label)}": int((labels == int(label)).sum())
                for label in Q1_MULTICLASS_SPEC.sorted_classes
            },
            **{
                f"predicted_share_class_{int(label)}": float((predicted == int(label)).mean())
                for label in Q1_MULTICLASS_SPEC.sorted_classes
            },
            # 判别力（只用于斜率守卫与描述，不报告参与者级 AUROC）
            "pooled_probe_ovr_macro_auroc": auroc_boot["point_estimate"],
            "pooled_probe_ovr_macro_auroc_ci_lower": auroc_boot["ci_lower"],
            "pooled_probe_ovr_macro_auroc_ci_upper": auroc_boot["ci_upper"],
            "pooled_probe_ovr_macro_auroc_n_valid_replicates": int(
                auroc_boot["n_valid_replicates"]
            ),
            # 校准
            "participant_macro_top_label_confidence": float(np.mean(confidence)),
            "pooled_top_label_accuracy_descriptive": float(np.mean(correct)),
            "participant_macro_top_label_accuracy": float(
                _participant_mean(group, correct, "__correct").mean()
            ),
            "participant_macro_top_label_calibration_in_the_large": citl_boot["point_estimate"],
            "participant_macro_top_label_calibration_in_the_large_ci_lower": citl_boot["ci_lower"],
            "participant_macro_top_label_calibration_in_the_large_ci_upper": citl_boot["ci_upper"],
            "expected_calibration_error": ece,
            "calibration_bins_requested": int(n_bins),
            "calibration_bins_effective": int(n_bins_effective),
            "calibration_intercept": intercept,
            "calibration_intercept_ci_lower": calibration_boot["intercept_ci_lower"],
            "calibration_intercept_ci_upper": calibration_boot["intercept_ci_upper"],
            "calibration_slope": slope,
            "calibration_slope_ci_lower": calibration_boot["slope_ci_lower"],
            "calibration_slope_ci_upper": calibration_boot["slope_ci_upper"],
            "calibration_status": calibration_status,
            "calibration_slope_reporting": slope_reporting,
            "calibration_slope_reporting_note": slope_note,
            "calibration_n_valid_replicates": int(calibration_boot["n_valid_replicates"]),
            # 治理
            "bootstrap_method": "fixed_oof_participant_cluster_percentile",
            "bootstrap_replicates": int(replicates),
            "bootstrap_seed": int(seed),
            "bootstrap_confidence_level": float(confidence_level),
            "prediction_model_retrained_in_bootstrap": False,
            "calibration_model_refitted_in_bootstrap": True,
            "used_for_model_or_c_selection": False,
            "used_for_feature_selection": False,
            "aggregation": "participant_equal_within_participant_probe_equal",
            "role": "supplementary_descriptive_diagnostic",
        }
        diagnostic_rows.append(row)

        for bin_row in bin_frame_rows:
            bin_rows.append({**identity, "schema_version": MULTICLASS_DIAGNOSTIC_SCHEMA_VERSION, **bin_row})

    diagnostics = pd.DataFrame(diagnostic_rows)
    calibration_bins = pd.DataFrame(bin_rows)
    audit = {
        "schema_version": MULTICLASS_DIAGNOSTIC_SCHEMA_VERSION,
        "prior_reconstruction": audit_rows,
        "n_prior_reconstruction_rows": len(audit_rows),
        "n_prior_rows_checked_against_frozen": int(
            sum(1 for row in audit_rows if row["matches_frozen"] is not None)
        ),
        "n_prior_rows_mismatching_frozen": int(
            sum(1 for row in audit_rows if row["matches_frozen"] is False)
        ),
        "prior_reconstruction_method": (
            "for outer fold g the training side is rebuilt from every archived row whose "
            "outer_fold_group is not g; the four-class line holds out exactly one "
            "participant per outer fold, so this is exact and never touches the held-out "
            "participant"
        ),
    }
    return diagnostics, calibration_bins, audit


def multiclass_diagnostic_summary(diagnostics: pd.DataFrame) -> dict[str, object]:
    """紧凑的机器可读摘要，写入 manifest 与交接记录。"""
    if "calibration_slope_reporting" in diagnostics.columns:
        n_reportable = int(
            (diagnostics["calibration_slope_reporting"] == CALIBRATION_SLOPE_REPORTABLE).sum()
        )
        n_low = int(
            (
                diagnostics["calibration_slope_reporting"]
                == CALIBRATION_SLOPE_NOT_REPORTABLE_LOW_DISCRIMINATION
            ).sum()
        )
        n_not_estimable = int(
            (diagnostics["calibration_slope_reporting"] == CALIBRATION_SLOPE_NOT_ESTIMABLE).sum()
        )
    else:  # pragma: no cover - 防御性：仅当上游删列
        n_reportable = n_low = n_not_estimable = None
    return {
        "schema_version": MULTICLASS_DIAGNOSTIC_SCHEMA_VERSION,
        "n_diagnostic_rows": int(len(diagnostics)),
        "n_unique_models": int(diagnostics[MODEL_COLUMN].nunique()),
        "n_runs": int(diagnostics[RUN_COLUMN].nunique()) if RUN_COLUMN in diagnostics.columns else None,
        "n_calibration_slope_reportable": n_reportable,
        "n_calibration_slope_not_reportable_low_discrimination": n_low,
        "n_calibration_slope_not_estimable": n_not_estimable,
        "metrics": [
            "participant_macro_multiclass_brier",
            "brier_skill_score",
            "participant_macro_brier_minus_prior",
            "participant_macro_top_label_calibration_in_the_large",
            "expected_calibration_error",
            "calibration_intercept",
            "calibration_slope",
        ],
        "aggregation": "participant_equal_within_participant_probe_equal",
        "brier_definition": "sum_over_classes (p_k - 1{y = k})^2, pooled per participant then equal across participants",
        "no_information_comparator": (
            "per-outer-fold class-prior constant predictor on exactly the same probes; "
            "the reconstruction is checked against the frozen baseline before any row is written"
        ),
        "calibration_operationalisation": (
            "top-label: logit(1{argmax = y}) = intercept + slope * logit(max p); this is a "
            "post-hoc operationalisation of the metric range registered in 1.16.24 §4.2 and "
            "is applied uniformly to every row"
        ),
        "calibration_slope_rule": (
            "the calibration slope is marked interpretable only when the pooled out-of-fold "
            "macro AUROC 95% interval excludes 0.5; raw estimates are always kept"
        ),
        "participant_level_auroc_reported": False,
        "used_for_model_or_c_selection": False,
        "used_for_feature_selection": False,
        "prediction_model_retrained_in_bootstrap": False,
        "calibration_model_refitted_in_bootstrap": True,
    }


__all__ = [
    "CALIBRATION_SLOPE_NOT_ESTIMABLE",
    "CALIBRATION_SLOPE_NOT_REPORTABLE_LOW_DISCRIMINATION",
    "CALIBRATION_SLOPE_REPORTABLE",
    "DEFAULT_CALIBRATION_BINS",
    "MULTICLASS_DIAGNOSTIC_SCHEMA_VERSION",
    "MulticlassProbabilityDiagnosticsError",
    "build_multiclass_probability_diagnostics",
    "calibration_model_top_label",
    "multiclass_diagnostic_summary",
    "multiclass_probe_brier",
    "participant_multiclass_brier_values",
    "participant_top_label_calibration_in_the_large_values",
    "pooled_ovr_macro_auroc",
    "prior_constant_diagnostics",
    "reconstruct_outer_fold_priors",
    "top_label_confidence",
]
