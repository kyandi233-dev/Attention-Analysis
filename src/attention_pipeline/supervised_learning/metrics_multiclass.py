"""Participant-equal multiclass evaluation for the Q1 four-class line.

文件：metrics_multiclass.py
版本：1.0.0
功能：实现预注册 `1.16.24` §4 的四分类指标与无信息基线：
      - 主指标 participant_macro_multiclass_log_loss（参与者等权、参与者内 probe 等权）；
      - 逐外层折的类别先验常数预测基线（§4.3）；
      - 折外合并补充指标（§4.2）：one-vs-rest 宏平均 AUROC、balanced accuracy、
        macro-F1、每类对数损失、每类 AUROC、4x4 混淆矩阵；
      - 复用二分类线的固定 OOF 参与者簇 bootstrap（1000 / 20260830 / 0.95 / 不重训）；
      - 路线 A 与路线 B 的配对参与者增量比较。
      本模块只新增；不修改 evaluation.py、models.py、runner.py。
用法：
    from attention_pipeline.supervised_learning.metrics_multiclass import (
        participant_multiclass_log_loss, prior_baseline_participant_macro_log_loss,
    )
依赖：numpy、pandas、scikit-learn

术语（首次出现展开，APA 7）：
- 对数损失（log loss），数值越低越好；
- 受试者工作特征曲线下面积（area under the receiver operating characteristic curve [AUROC]）；
- 宏平均（macro average）= 先按类别求值再等权平均，不按样本量加权。
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)

from .evaluation import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE_LEVEL,
    EvaluationContractError,
    fixed_oof_participant_bootstrap,
)
from .task import SupervisedLearningContractError
from .task_multiclass import (
    Q1_MULTICLASS_SPEC,
    MulticlassTaskSpec,
    aligned_class_probabilities,
)

MEMBERSHIP_COLUMN = "membership_type"
PAIR_KEY_COLUMNS = (
    "participant_group_id",
    "session_id",
    "block_id",
    "probe_event_id",
)
#: 主指标名称：参与者宏平均多类对数损失。
PRIMARY_METRIC_NAME = "participant_macro_multiclass_log_loss"
#: 无信息基线名称：逐外层折类别先验常数预测器。
PRIOR_BASELINE_NAME = "outer_fold_class_prior_constant"
_INCREMENT_DEFINITION = "baseline_log_loss_minus_added_log_loss"
#: 类别概率被加性下限截断时的概率地板，保证 log 有限。
_PROBABILITY_FLOOR = 1e-12


def multiclass_probe_log_loss(
    y_true: Sequence[int] | np.ndarray,
    proba_matrix: np.ndarray,
    *,
    spec: MulticlassTaskSpec = Q1_MULTICLASS_SPEC,
) -> np.ndarray:
    """逐 probe 多类对数损失，使用显式 ``labels`` 顺序。

    参数：
        y_true: 真实类别序列，取值必须在 ``spec.sorted_classes`` 内。
        proba_matrix: ``(n, 4)`` 概率矩阵，列顺序等于 ``spec.sorted_classes``。
        spec: 四分类任务合同。
    返回：长度 n 的 float 数组，元素为 ``-log p(y_i)``。
    异常：
        SupervisedLearningContractError: 输入不对齐、标签越界、概率非有限或越界。
    说明：
        类别概率被截断到 ``[eps, 1-eps]``（与二分类线 ``binary_probe_log_loss`` 的做法一致），
        这样即使模型给出 0 概率也不会产生无穷大损失而掩盖真实差异。
    """
    labels = list(spec.sorted_classes)
    y = np.asarray(y_true)
    proba = np.asarray(proba_matrix, dtype=float)
    if y.ndim != 1 or len(y) == 0:
        raise SupervisedLearningContractError("y_true must be a non-empty 1D array")
    if proba.shape != (len(y), len(labels)):
        raise SupervisedLearningContractError(
            f"probability matrix must have shape ({len(y)}, {len(labels)}); got {proba.shape}"
        )
    if not np.isfinite(proba).all():
        raise SupervisedLearningContractError("class probabilities contain non-finite values")
    if np.any((proba < 0.0) | (proba > 1.0)):
        raise SupervisedLearningContractError("class probabilities must lie within [0, 1]")

    numeric = pd.to_numeric(pd.Series(y), errors="coerce")
    if numeric.isna().any():
        raise SupervisedLearningContractError("multiclass labels contain missing/non-numeric values")
    integer_labels = numeric.astype(int).to_numpy()
    if np.asarray(y).dtype.kind == "f" and not np.array_equal(numeric.to_numpy(dtype=float), integer_labels):
        raise SupervisedLearningContractError("multiclass labels must be integer-valued")
    unexpected = sorted(set(integer_labels.tolist()) - set(labels))
    if unexpected:
        raise SupervisedLearningContractError(f"unexpected multiclass labels: {unexpected}")

    # 反向散射到 proba 的第几列由 spec 顺序决定；因为这正是本模块定义的列顺序，
    # 直接用 searchsorted 即可，但仍显式校验一次，避免 spec 被改成非排序元组。
    class_index = {int(label): index for index, label in enumerate(labels)}
    column = np.asarray([class_index[int(value)] for value in integer_labels], dtype=int)
    observed = proba[np.arange(len(integer_labels)), column]
    eps = np.finfo(float).eps
    clipped = np.clip(observed, eps, 1.0 - eps)
    return -np.log(clipped)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, context: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise EvaluationContractError(f"{context} missing required columns: {missing}")


def _require_single_value(frame: pd.DataFrame, column: str, *, context: str) -> str:
    """要求某列在整帧内只有唯一非空非空白取值。"""
    if column not in frame.columns:
        raise EvaluationContractError(f"{context} missing required column: {column}")
    raw = frame[column]
    if raw.isna().any():
        raise EvaluationContractError(f"{context} {column} contains missing values")
    values = raw.astype(str).str.strip()
    if values.eq("").any():
        raise EvaluationContractError(f"{context} {column} contains blank values")
    unique = values.drop_duplicates().tolist()
    if len(unique) != 1:
        raise EvaluationContractError(f"{context} requires one {column}; got {unique}")
    return str(unique[0])


def _validate_oof_frame(
    frame: pd.DataFrame,
    *,
    context: str,
    spec: MulticlassTaskSpec,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """校验一份多分类折外预测存档并返回 (frame, 整数标签, 对齐概率矩阵)。"""
    _require_columns(
        frame,
        [
            *PAIR_KEY_COLUMNS,
            "analysis_set_id",
            MEMBERSHIP_COLUMN,
            "model_id",
            "outer_fold_group",
            spec.source_column,
            *spec.probability_columns,
            "model_failed",
        ],
        context=context,
    )
    if frame.empty:
        raise EvaluationContractError(f"{context} is empty")
    if frame.duplicated(list(PAIR_KEY_COLUMNS)).any():
        raise EvaluationContractError(
            f"{context} contains duplicate probe keys: {list(PAIR_KEY_COLUMNS)}"
        )
    if frame["participant_group_id"].isna().any():
        raise EvaluationContractError(f"{context} participant_group_id contains missing values")

    # 折外预测行的外层折必须就是被留出的参与者本人，否则参与者等权聚合会把
    # 别人的 probe 算进该参与者的均值。
    participants = frame["participant_group_id"].astype(str)
    outer_groups = frame["outer_fold_group"].astype(str)
    if not participants.equals(outer_groups):
        raise EvaluationContractError(
            f"{context} outer_fold_group must equal the held-out participant_group_id on every OOF row"
        )

    failed_raw = frame["model_failed"]
    if pd.api.types.is_bool_dtype(failed_raw.dtype):
        failed = failed_raw.astype(bool)
    else:
        normalized = failed_raw.astype(str).str.strip().str.lower()
        mapped = normalized.map({"true": True, "false": False, "1": True, "0": False})
        if mapped.isna().any():
            bad = sorted(normalized[mapped.isna()].drop_duplicates().tolist())
            raise EvaluationContractError(f"{context} model_failed contains invalid values: {bad}")
        failed = mapped.astype(bool)
    if failed.any():
        failed_groups = sorted(frame.loc[failed, "participant_group_id"].astype(str).unique().tolist())
        raise EvaluationContractError(
            f"{context} has failed OOF rows; participant-equal evaluation cannot silently drop them: {failed_groups}"
        )

    labels_series = pd.to_numeric(frame[spec.source_column], errors="coerce")
    if labels_series.isna().any():
        raise EvaluationContractError(f"{context} {spec.source_column} contains missing/non-numeric values")
    integer_labels = labels_series.astype(int).to_numpy()
    unexpected = sorted(set(integer_labels.tolist()) - set(spec.sorted_classes))
    if unexpected:
        raise EvaluationContractError(f"{context} {spec.source_column} contains unexpected classes: {unexpected}")

    proba_raw = frame.loc[:, list(spec.probability_columns)].to_numpy(dtype=float)
    if not np.isfinite(proba_raw).all() or np.any((proba_raw < 0.0) | (proba_raw > 1.0)):
        raise EvaluationContractError(
            f"{context} probability columns must contain finite values within [0, 1]"
        )
    return frame.copy(), integer_labels, proba_raw


def participant_multiclass_log_loss(
    frame: pd.DataFrame,
    *,
    spec: MulticlassTaskSpec = Q1_MULTICLASS_SPEC,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """计算参与者内 probe 等权、参与者间等权的多类对数损失主指标。

    参数：
        frame: 单模型折外预测存档，需含 probe 键、analysis_set_id、membership_type、
            model_id、outer_fold_group、原始四类标签、四个概率列、model_failed。
        spec: 四分类任务合同。
    返回：
        (participant_frame, summary)：
        participant_frame 每行一个参与者（n_probes、mean_multiclass_log_loss）；
        summary 的键与 ``evaluation.participant_log_loss`` 对应，
        但 ``metric`` 为 ``multiclass_log_loss``，主指标键为 ``participant_macro_multiclass_log_loss``。
    """
    data, integer_labels, proba = _validate_oof_frame(frame, context="OOF model archive", spec=spec)
    model_id = _require_single_value(data, "model_id", context="OOF model archive")
    analysis_set_id = _require_single_value(data, "analysis_set_id", context="OOF model archive")
    membership_type = _require_single_value(data, MEMBERSHIP_COLUMN, context="OOF model archive")

    data = data.copy()
    data["probe_multiclass_log_loss"] = multiclass_probe_log_loss(
        integer_labels, proba, spec=spec
    )
    participant = (
        data.groupby("participant_group_id", sort=True, as_index=False)
        .agg(
            n_probes=("probe_multiclass_log_loss", "size"),
            mean_multiclass_log_loss=("probe_multiclass_log_loss", "mean"),
        )
    )
    participant.insert(0, MEMBERSHIP_COLUMN, membership_type)
    participant.insert(0, "analysis_set_id", analysis_set_id)
    participant.insert(0, "model_id", model_id)
    overall = float(participant["mean_multiclass_log_loss"].mean())
    pooled_probe = float(data["probe_multiclass_log_loss"].mean())
    summary: dict[str, object] = {
        "model_id": model_id,
        "analysis_set_id": analysis_set_id,
        MEMBERSHIP_COLUMN: membership_type,
        "metric": "multiclass_log_loss",
        "aggregation": "participant_equal_within_participant_probe_equal",
        "n_participants": int(len(participant)),
        "n_probes": int(len(data)),
        PRIMARY_METRIC_NAME: overall,
        "pooled_probe_multiclass_log_loss_descriptive": pooled_probe,
    }
    return participant, summary


def outer_fold_class_priors(
    outer_train: pd.DataFrame,
    *,
    spec: MulticlassTaskSpec = Q1_MULTICLASS_SPEC,
) -> dict[int, float]:
    """返回某一个外层训练折的四类先验概率（不含任何特征信息）。

    参数：
        outer_train: 该外层折的训练集（必须含原始四类标签列）。
        spec: 四分类任务合同。
    返回：``{类别: 先验概率}``，四个类别都在，缺失类别的先验为 0.0。
    异常：
        SupervisedLearningContractError: 训练集为空、标签缺失或出现越界类别。
    """
    if spec.source_column not in outer_train.columns:
        raise SupervisedLearningContractError(
            f"prior baseline requires the training-fold label column {spec.source_column}"
        )
    if outer_train.empty:
        raise SupervisedLearningContractError("prior baseline requires a non-empty training fold")
    numeric = pd.to_numeric(outer_train[spec.source_column], errors="coerce")
    if numeric.isna().any():
        raise SupervisedLearningContractError(
            "prior baseline training fold contains missing/non-numeric labels"
        )
    integer_labels = numeric.astype(int).to_numpy()
    unexpected = sorted(set(integer_labels.tolist()) - set(spec.sorted_classes))
    if unexpected:
        raise SupervisedLearningContractError(
            f"prior baseline training fold contains unexpected classes: {unexpected}"
        )
    counts = pd.Series(integer_labels).value_counts()
    total = float(len(integer_labels))
    return {
        int(label): float(counts.get(int(label), 0)) / total for label in spec.sorted_classes
    }


def prior_baseline_participant_macro_log_loss(
    outer_train: pd.DataFrame,
    outer_test_predictions: pd.DataFrame,
    *,
    spec: MulticlassTaskSpec = Q1_MULTICLASS_SPEC,
) -> dict[str, object]:
    """无信息基线（预注册 §4.3）：逐外层折类别先验常数预测器。

    用**该折外层训练集**的四类先验概率对测试 probe 做常数预测，再按主指标
    （参与者内 probe 等权、参与者间等权）计算同一个多类对数损失。
    该基线不使用任何特征，是本轮「是否超过无信息」的判定依据。

    参数：
        outer_train: 该外层折的训练集。
        outer_test_predictions: 该折的折外预测行（至少含 participant_group_id 与真实标签列）。
        spec: 四分类任务合同。
    返回：
        含 ``prior_baseline_participant_macro_multiclass_log_loss``、逐参与者数值、
        逐类先验、观测类别、缺失类别与样本量的审计字典。
    说明：
        常数预测器对每一类都给相同概率，因此逐 probe 损失只取决于真实类别与先验，
        但为了口径完全一致，这里仍然走 ``multiclass_probe_log_loss`` 同一实现。
    """
    priors = outer_fold_class_priors(outer_train, spec=spec)
    if spec.source_column not in outer_test_predictions.columns:
        raise SupervisedLearningContractError(
            f"prior baseline requires the test-fold label column {spec.source_column}"
        )
    if "participant_group_id" not in outer_test_predictions.columns:
        raise SupervisedLearningContractError(
            "prior baseline requires participant_group_id on the test-fold rows"
        )
    if outer_test_predictions.empty:
        raise SupervisedLearningContractError("prior baseline requires a non-empty test fold")

    numeric = pd.to_numeric(outer_test_predictions[spec.source_column], errors="coerce")
    if numeric.isna().any():
        raise SupervisedLearningContractError(
            "prior baseline test fold contains missing/non-numeric labels"
        )
    integer_labels = numeric.astype(int).to_numpy()
    unexpected = sorted(set(integer_labels.tolist()) - set(spec.sorted_classes))
    if unexpected:
        raise SupervisedLearningContractError(
            f"prior baseline test fold contains unexpected classes: {unexpected}"
        )

    observed_classes = sorted(set(integer_labels.tolist()))
    absent_classes = sorted(set(spec.sorted_classes) - set(observed_classes))
    # 训练折若缺失某类，其先验为 0；直接取对数会得到无穷大，掩盖真实的基线水平。
    # 因此把先验做加性下限截断到 _PROBABILITY_FLOOR 并显式记录这一事实。
    floor_applied = any(priors[int(label)] <= 0.0 for label in spec.sorted_classes)
    prior_vector = np.asarray(
        [max(priors[int(label)], _PROBABILITY_FLOOR) for label in spec.sorted_classes],
        dtype=float,
    )
    prior_vector = prior_vector / float(prior_vector.sum())
    constant = np.tile(prior_vector, (len(integer_labels), 1))

    probe_losses = multiclass_probe_log_loss(integer_labels, constant, spec=spec)
    scored = pd.DataFrame(
        {
            "participant_group_id": outer_test_predictions["participant_group_id"]
            .astype(str)
            .reset_index(drop=True),
            "probe_multiclass_log_loss": probe_losses,
            spec.source_column: integer_labels,
        }
    )
    participant = (
        scored.groupby("participant_group_id", sort=True, as_index=False)
        .agg(
            n_probes=("probe_multiclass_log_loss", "size"),
            mean_multiclass_log_loss=("probe_multiclass_log_loss", "mean"),
        )
    )
    participant_values = participant["mean_multiclass_log_loss"].to_numpy(dtype=float)
    return {
        "baseline": PRIOR_BASELINE_NAME,
        "definition": (
            "constant prediction from this outer fold's training-set class priors, "
            "evaluated with the same participant-equal aggregation as the primary metric"
        ),
        "uses_features": False,
        "n_train_rows": int(len(outer_train)),
        "n_test_rows": int(len(outer_test_predictions)),
        "n_test_participants": int(len(participant)),
        "class_priors": {str(label): float(priors[int(label)]) for label in spec.sorted_classes},
        "observed_test_classes": observed_classes,
        "absent_test_classes": absent_classes,
        "prior_probability_floor_applied": floor_applied,
        "prior_probability_floor": _PROBABILITY_FLOOR if floor_applied else None,
        "participant_values": {
            str(group): float(value)
            for group, value in zip(
                participant["participant_group_id"].tolist(), participant_values.tolist(), strict=True
            )
        },
        "participant_baseline_frame": participant,
        "prior_baseline_participant_macro_multiclass_log_loss": float(
            np.mean(participant_values)
        ),
        "pooled_probe_multiclass_log_loss_descriptive": float(np.mean(probe_losses)),
    }


def _safe_macro_auroc(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    spec: MulticlassTaskSpec,
) -> dict[str, object]:
    """one-vs-rest 宏平均 AUROC；某类在评估折完全缺失时返回 not_estimable 而非崩溃。"""
    n_classes = len(spec.sorted_classes)
    per_class: dict[str, float | None] = {}
    absent: list[int] = []
    estimable_classes: list[int] = []
    for index, label in enumerate(spec.sorted_classes):
        label = int(label)
        binary_truth = (y_true == label).astype(int)
        if binary_truth.min() == binary_truth.max():
            per_class[str(label)] = None
            absent.append(label)
            continue
        per_class[str(label)] = float(roc_auc_score(binary_truth, proba[:, index]))
        estimable_classes.append(label)
    if not estimable_classes:
        return {
            "ovr_macro_auroc": None,
            "ovr_macro_auroc_estimable_classes": [],
            "ovr_macro_auroc_not_estimable_classes": [str(v) for v in absent],
            "per_class_ovr_auroc": per_class,
            "reason": "no class has both positive and negative out-of-fold probes",
        }
    values = [float(per_class[str(label)]) for label in estimable_classes]
    return {
        "ovr_macro_auroc": float(np.mean(values)),
        "ovr_macro_auroc_estimable_classes": estimable_classes,
        "ovr_macro_auroc_not_estimable_classes": [str(v) for v in absent],
        "per_class_ovr_auroc": per_class,
        "reason": "",
        "n_classes_expected": n_classes,
    }


def multiclass_discrimination(
    frame: pd.DataFrame,
    *,
    spec: MulticlassTaskSpec = Q1_MULTICLASS_SPEC,
) -> dict[str, object]:
    """折外合并的补充指标（预注册 §4.2），不替代主指标。

    参数：
        frame: 单模型折外预测存档（同 ``participant_multiclass_log_loss``）。
        spec: 四分类任务合同。
    返回：含宏平均 AUROC、balanced accuracy、macro-F1、每类对数损失、
        每类 one-vs-rest AUROC、4x4 混淆矩阵的审计字典。
    说明：
        某类别在评估折中完全缺失时，依赖该类正负样本的指标不可估；本函数**不删除**
        参与者或类别来凑齐分母，而是在 ``absent_classes`` 与
        ``not_estimable`` 字段中显式报告该事实。
    """
    data, integer_labels, proba = _validate_oof_frame(
        frame, context="discrimination frame", spec=spec
    )
    labels = list(spec.sorted_classes)
    observed_classes = sorted(set(integer_labels.tolist()))
    absent_classes = sorted(set(labels) - set(observed_classes))

    result: dict[str, object] = {
        "metric_role": "supplementary_not_a_replacement_for_the_primary_metric",
        "n_probes": int(len(data)),
        "n_participants": int(data["participant_group_id"].nunique()),
        "observed_classes": observed_classes,
        "absent_classes": absent_classes,
        "class_absent_from_evaluation_fold": bool(absent_classes),
        "class_absence_note": (
            "one or more declared classes have no out-of-fold probes; "
            "class-specific metrics for those classes are not estimable and no "
            "participant or class was dropped to fill the denominator"
            if absent_classes
            else ""
        ),
        "class_support": {
            str(label): int((integer_labels == int(label)).sum()) for label in labels
        },
    }

    # 逐 probe 多类对数损失 -> 折外合并平均值（描述性；主指标仍是参与者等权）。
    probe_losses = multiclass_probe_log_loss(integer_labels, proba, spec=spec)
    result["pooled_probe_multiclass_log_loss"] = float(np.mean(probe_losses))

    # 每类对数损失：对该类的探针单独取损失平均。类别缺失时返回 None 并记录。
    per_class_log_loss: dict[str, float | None] = {}
    for label in labels:
        mask = integer_labels == int(label)
        per_class_log_loss[str(label)] = (
            float(np.mean(probe_losses[mask])) if bool(mask.any()) else None
        )
    result["per_class_log_loss"] = per_class_log_loss

    result.update(_safe_macro_auroc(integer_labels, proba, spec=spec))

    # 预测类别：取概率最大列；平局时取 spec 顺序中靠前的类别，保证确定性。
    predicted = np.asarray(
        [spec.sorted_classes[int(index)] for index in np.argmax(proba, axis=1)], dtype=int
    )
    result["predicted_class_distribution"] = {
        str(label): int((predicted == int(label)).sum()) for label in labels
    }

    try:
        # 注意：balanced_accuracy_score 在本环境（scikit-learn 1.9）不支持 labels 形参，
        # 其类别集合由 y_true 与 y_pred 的并集决定。因此某类在折内完全缺失时，
        # 该类不会进入分母；这一事实由上面的 absent_classes / class_absence_note
        # 显式报告，而不是被隐藏。
        result["balanced_accuracy"] = float(
            balanced_accuracy_score(integer_labels, predicted)
        )
        result["balanced_accuracy_class_scope"] = "union_of_observed_and_predicted_classes"
    except ValueError as exc:  # 例如评估折只含单一类别
        result["balanced_accuracy"] = None
        result["balanced_accuracy_reason"] = f"{type(exc).__name__}: {exc}"
    try:
        # labels 显式声明四个类别：这样某类在折内完全缺失时，该类的 precision/recall
        # 记为 0（zero_division=0）并计入宏平均分母，而不是被静默从类别集合里删掉。
        result["macro_f1"] = float(
            f1_score(integer_labels, predicted, labels=labels, average="macro", zero_division=0)
        )
    except ValueError as exc:
        result["macro_f1"] = None
        result["macro_f1_reason"] = f"{type(exc).__name__}: {exc}"

    matrix = confusion_matrix(integer_labels, predicted, labels=labels)
    result["confusion_matrix_labels"] = labels
    result["confusion_matrix"] = [[int(value) for value in row] for row in matrix.tolist()]
    result["confusion_matrix_note"] = (
        "rows = observed class in the fixed order "
        f"{labels}; columns = predicted class in the same order; out-of-fold pooled"
    )

    row_sums = proba.sum(axis=1)
    result["probability_row_sum_max_abs_deviation"] = float(np.max(np.abs(row_sums - 1.0)))
    return result


def multiclass_bootstrap_summary(
    participant_values: Sequence[float] | np.ndarray,
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> dict[str, object]:
    """固定 OOF 参与者整簇百分位 bootstrap（复用二分类线实现，不重写）。

    参数：
        participant_values: 每个参与者的主指标数值（参与者等权平均的对象）。
        replicates: 默认 1000（预注册 §3 冻结）。
        seed: 默认 20260830（同一冻结种子）。
        confidence_level: 默认 0.95。
    返回：``evaluation.fixed_oof_participant_bootstrap`` 的审计字典，
        外加多分类指标标签，且明确标记 bootstrap 内不重训模型。
    """
    summary = dict(
        fixed_oof_participant_bootstrap(
            participant_values,
            replicates=replicates,
            seed=seed,
            confidence_level=confidence_level,
        )
    )
    summary["metric"] = PRIMARY_METRIC_NAME
    summary["retrain_within_bootstrap"] = False
    summary["reused_from"] = "evaluation.fixed_oof_participant_bootstrap"
    return summary


def paired_multiclass_increment(
    baseline: pd.DataFrame,
    added: pd.DataFrame,
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    spec: MulticlassTaskSpec = Q1_MULTICLASS_SPEC,
) -> dict[str, object]:
    """路线 A 与路线 B 的配对参与者增量（两种分析程序的比较，不是单特征增量）。

    增量定义与二分类线一致：``baseline_log_loss_minus_added_log_loss``；
    正值表示 added 程序的对数损失更低。两个输入必须共享同一 analysis_set_id、
    同一 membership、同一 probe 成员与同一外层折指派，否则判定为不可比并报错。

    参数：
        baseline: 基准程序（路线 A 或 B）的折外预测存档。
        added: 对照程序（另一条路线）的折外预测存档。
        replicates / seed / confidence_level: 固定 OOF 配对参与者重抽参数。
        spec: 四分类任务合同。
    返回：含逐参与者增量、总体增量、配对 bootstrap 与「两种程序」解释边界的字典。
    """
    base, base_labels, base_proba = _validate_oof_frame(baseline, context="baseline OOF archive", spec=spec)
    aug, aug_labels, aug_proba = _validate_oof_frame(added, context="added OOF archive", spec=spec)
    base_set = _require_single_value(base, "analysis_set_id", context="baseline OOF archive")
    aug_set = _require_single_value(aug, "analysis_set_id", context="added OOF archive")
    base_membership = _require_single_value(base, MEMBERSHIP_COLUMN, context="baseline OOF archive")
    aug_membership = _require_single_value(aug, MEMBERSHIP_COLUMN, context="added OOF archive")
    base_model = _require_single_value(base, "model_id", context="baseline OOF archive")
    aug_model = _require_single_value(aug, "model_id", context="added OOF archive")
    if base_set != aug_set:
        raise EvaluationContractError(
            f"paired comparison requires the same analysis_set_id; got {base_set!r} vs {aug_set!r}"
        )
    if base_membership != aug_membership:
        raise EvaluationContractError(
            f"paired comparison requires the same {MEMBERSHIP_COLUMN}; "
            f"got {base_membership!r} vs {aug_membership!r}"
        )

    base_sorted = base.sort_values(list(PAIR_KEY_COLUMNS)).reset_index(drop=True)
    aug_sorted = aug.sort_values(list(PAIR_KEY_COLUMNS)).reset_index(drop=True)
    if len(base_sorted) != len(aug_sorted):
        raise EvaluationContractError(
            f"paired comparison probe count mismatch: baseline={len(base_sorted)}, added={len(aug_sorted)}"
        )
    for column in [*PAIR_KEY_COLUMNS, "outer_fold_group", spec.source_column]:
        left = base_sorted[column].astype(str).reset_index(drop=True)
        right = aug_sorted[column].astype(str).reset_index(drop=True)
        if not left.equals(right):
            raise EvaluationContractError(
                f"paired comparison mismatch in {column}; direct increment is not comparable"
            )

    # 两份存档按同一 probe 键排序后，行位置一一对应；标签与概率矩阵都按同一顺序重排，
    # 因此这里的逐 probe 增量是真正的配对量。
    base_loss = multiclass_probe_log_loss(base_labels, base_proba, spec=spec)
    aug_loss = multiclass_probe_log_loss(aug_labels, aug_proba, spec=spec)
    paired = pd.DataFrame(
        {
            "participant_group_id": base_sorted["participant_group_id"].astype(str),
            "probe_increment": base_loss - aug_loss,
        }
    )
    participant = (
        paired.groupby("participant_group_id", sort=True, as_index=False)
        .agg(
            n_probes=("probe_increment", "size"),
            mean_multiclass_log_loss_increment=("probe_increment", "mean"),
        )
    )
    participant.insert(0, MEMBERSHIP_COLUMN, base_membership)
    participant.insert(0, "analysis_set_id", base_set)
    participant.insert(0, "added_model_id", aug_model)
    participant.insert(0, "baseline_model_id", base_model)
    values = participant["mean_multiclass_log_loss_increment"].to_numpy(dtype=float)
    bootstrap = multiclass_bootstrap_summary(
        values,
        replicates=replicates,
        seed=seed,
        confidence_level=confidence_level,
    )
    bootstrap["paired_model_resampling"] = True
    bootstrap["increment_definition"] = _INCREMENT_DEFINITION
    return {
        "baseline_model_id": base_model,
        "added_model_id": aug_model,
        "analysis_set_id": base_set,
        MEMBERSHIP_COLUMN: base_membership,
        "participant_increments": participant,
        "overall_increment": float(np.mean(values)),
        "increment_definition": _INCREMENT_DEFINITION,
        "positive_increment_interpretation": "added_procedure_has_lower_multiclass_log_loss",
        "interpretation_boundary": (
            "route A versus route B compares two analysis procedures, not the increment "
            "of any single feature; this difference must not be read as a feature's "
            "four-class contribution nor compared with the binary line's B -> B+x increments"
        ),
        "bootstrap": bootstrap,
    }


__all__ = [
    "MEMBERSHIP_COLUMN",
    "PAIR_KEY_COLUMNS",
    "PRIMARY_METRIC_NAME",
    "PRIOR_BASELINE_NAME",
    "multiclass_bootstrap_summary",
    "multiclass_discrimination",
    "multiclass_probe_log_loss",
    "outer_fold_class_priors",
    "paired_multiclass_increment",
    "participant_multiclass_log_loss",
    "prior_baseline_participant_macro_log_loss",
]
