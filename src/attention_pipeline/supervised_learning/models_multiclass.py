"""Nested participant-equal **multiclass** selection and outer refit (route A / route B).

文件：models_multiclass.py
版本：1.0.0
功能：为预注册 `1.16.24` 提供四分类（名义四类）嵌套选择与外层重拟合：
      - `select_multiclass_logistic`：路线 A 的内层 GroupKFold + 逐折重拟合预处理 + C 选择；
      - `forward_select_multiclass`：路线 B 的 §5.2 前向逐步选择（严格改进 + 固定平局规则）；
      - `refit_multiclass_and_predict`：外层训练集全量重拟合与无标签外层测试预测。
      与二分类线共用同一预处理、参与者等权权重与分组实现；本模块只新增，
      不修改 models.py / evaluation.py / runner.py。
用法：
    from attention_pipeline.supervised_learning.models_multiclass import (
        select_multiclass_logistic, forward_select_multiclass, refit_multiclass_and_predict,
    )
依赖：numpy、pandas、scikit-learn

关键冻结约束：
- 外层测试行的任何结局列都不得进入本模块的预测接口（fail-closed）；
- 内层每个折都必须重新拟合预处理，验证参与者不得影响其自身预处理的拟合；
- 训练折缺少四个类别中的任何一个时，该 (方案, C) / (特征, C) 候选必须被显式记为失败，
  绝不静默拟合成三类模型。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

from .feature_schemes import FeatureScheme, require_scheme_columns
from .preprocessing import (
    PreprocessingContractError,
    apply_preprocessing,
    fit_preprocessing,
    participant_equal_row_weights,
    participant_weight_audit,
)
from .task import Q1_BINARY_SPEC, SupervisedLearningContractError
from .task_multiclass import (
    Q1_MULTICLASS_SPEC,
    MulticlassTaskSpec,
    aligned_class_probabilities,
)


DEFAULT_C_CANDIDATES = (0.01, 0.1, 1.0, 10.0)
DEFAULT_INNER_SPLITS = 5
DEFAULT_MAX_ITER = 2000
#: 选择准则名称（预注册 §3 冻结：内层以四分类主指标选择）。
SELECTION_METRIC = "participant_macro_multiclass_log_loss"
#: 路线 B 终止于空集时使用的退化方案标识。
PRIOR_FALLBACK_FEATURE_SET_ID = "outer_train_class_prior_constant"
#: 路线 B 空集退化预测器的模型类别顺序（与 spec 一致）。
_CROSS_VALIDATED_OUTCOME_COLUMNS = frozenset(
    {
        Q1_MULTICLASS_SPEC.source_column,
        Q1_BINARY_SPEC.source_column,
        "q1_binary",
        Q1_BINARY_SPEC.positive_probability_name,
        "predicted_q1_binary",
    }
)


class MulticlassModelSelectionError(RuntimeError):
    """Raised when multiclass nested selection cannot produce a valid candidate."""


#: 预期内的模型失败：按折记录并继续，而不是让整轮运行崩溃。
_EXPECTED_MODEL_FAILURES = (
    MulticlassModelSelectionError,
    SupervisedLearningContractError,
    PreprocessingContractError,
    ValueError,
    FloatingPointError,
    np.linalg.LinAlgError,
)


@dataclass
class MulticlassSelectionResult:
    """路线 A 的胜者及其完整内层开发审计。"""

    feature_scheme: FeatureScheme
    selected_c: float
    candidate_participant_macro_log_loss: dict[str, float]
    inner_fold_audits: list[dict[str, object]] = field(default_factory=list)
    failed_candidates: dict[str, list[str]] = field(default_factory=dict)

    def audit_dict(self) -> dict[str, object]:
        return {
            "feature_scheme": self.feature_scheme.audit_dict(),
            "selected_c": float(self.selected_c),
            "candidate_participant_macro_log_loss": dict(
                self.candidate_participant_macro_log_loss
            ),
            "inner_fold_audits": list(self.inner_fold_audits),
            "failed_candidates": dict(self.failed_candidates),
            "selection_metric": SELECTION_METRIC,
            "selection_route": "A_registered_feature_schemes",
        }


def multiclass_labels(
    y: Sequence[object] | np.ndarray,
    expected_n: int,
    *,
    spec: MulticlassTaskSpec = Q1_MULTICLASS_SPEC,
) -> np.ndarray:
    """把外层训练标签校验成 1D 整数数组，越界或缺缺失即报错。"""
    arr = np.asarray(y)
    if arr.ndim != 1 or len(arr) != expected_n:
        raise SupervisedLearningContractError(
            "multiclass labels must be a 1D array aligned with frame rows"
        )
    numeric = pd.to_numeric(pd.Series(arr), errors="coerce")
    if numeric.isna().any():
        raise SupervisedLearningContractError("multiclass labels contain missing/non-numeric values")
    labels = numeric.astype(int).to_numpy()
    unexpected = sorted(set(labels.tolist()) - set(spec.sorted_classes))
    if unexpected:
        raise SupervisedLearningContractError(f"unexpected multiclass labels: {unexpected}")
    return labels


def _class_counts(labels: np.ndarray, *, spec: MulticlassTaskSpec) -> dict[str, int]:
    return {
        str(label): int((labels == int(label)).sum()) for label in spec.sorted_classes
    }


def _fit_multinomial_logistic(
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    c: float,
    max_iter: int,
    seed: int,
    sample_weight: Sequence[float] | np.ndarray | None = None,
    spec: MulticlassTaskSpec = Q1_MULTICLASS_SPEC,
) -> LogisticRegression:
    """拟合 L2 正则化多项逻辑回归，并要求训练折四类齐全。

    参数：
        x: 已用**训练折**状态预处理后的预测矩阵。
        y: 整数类别标签（1..4）。
        c: 正则强度倒数的候选值。
        max_iter: 最大迭代次数（冻结 2000）。
        seed: 随机种子（冻结 20260910 起算）。
        sample_weight: 参与者等权行权重。
        spec: 四分类任务合同。
    返回：已拟合的 ``LogisticRegression``。
    异常：
        MulticlassModelSelectionError: 训练折缺失四个类别中的任何一个，或权重非法。
    说明：
        scikit-learn 在 1.9 已移除 ``multi_class`` 形参；多类问题在该版本中本身就是
        多项式（multinomial）目标。因此这里只在旧版本上显式传入
        ``multi_class="multinomial"``，而在新版本上依赖其唯一的多类行为，
        使「多项式」语义在两种版本下都成立，不需要哨兵式的 in-place 改造。
    """
    expected_classes = len(spec.sorted_classes)
    present = sorted(set(np.asarray(y).astype(int).tolist()))
    missing = sorted(set(spec.sorted_classes) - set(present))
    if len(present) < expected_classes:
        raise MulticlassModelSelectionError(
            "training split does not contain all declared classes; "
            f"missing={missing}, observed={present}"
        )
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if weights is not None:
        if weights.ndim != 1 or len(weights) != len(y):
            raise MulticlassModelSelectionError(
                "sample_weight must be 1D and aligned with training labels"
            )
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise MulticlassModelSelectionError(
                "sample_weight must contain finite positive values"
            )
    kwargs: dict[str, object] = {}
    if "multi_class" in LogisticRegression().get_params():
        kwargs["multi_class"] = "multinomial"
    model = LogisticRegression(
        C=float(c),
        solver="lbfgs",
        max_iter=int(max_iter),
        random_state=int(seed),
        **kwargs,
    )
    model.fit(x, y, sample_weight=weights)
    return model


def participant_multiclass_validation_losses(
    y_valid: np.ndarray,
    proba_valid: np.ndarray,
    valid_groups: Sequence[object] | np.ndarray,
    *,
    spec: MulticlassTaskSpec = Q1_MULTICLASS_SPEC,
) -> dict[str, float]:
    """返回每个验证参与者一个 probe 等权多类对数损失（参与者宏平均的分子项）。"""
    groups = np.asarray(valid_groups).astype(str)
    probabilities = np.asarray(proba_valid, dtype=float)
    if groups.ndim != 1:
        raise MulticlassModelSelectionError("validation groups must be 1D")
    if probabilities.ndim != 2 or probabilities.shape != (len(y_valid), len(spec.sorted_classes)):
        raise MulticlassModelSelectionError(
            "validation probabilities must be an (n, 4) matrix aligned with labels"
        )
    if len(groups) != len(y_valid):
        raise MulticlassModelSelectionError("validation groups must align with validation labels")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise MulticlassModelSelectionError(
            "validation probabilities must be finite values in [0, 1]"
        )

    participant_losses: dict[str, float] = {}
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        loss = float(
            log_loss(
                y_valid[mask],
                probabilities[mask],
                labels=list(spec.sorted_classes),
            )
        )
        if not np.isfinite(loss):
            raise MulticlassModelSelectionError(
                f"validation multiclass log loss is non-finite for participant {group}"
            )
        participant_losses[str(group)] = loss
    return participant_losses


def _inner_grouped_cv_participant_losses(
    outer_train: pd.DataFrame,
    labels: np.ndarray,
    *,
    scheme: FeatureScheme,
    candidates: Sequence[float],
    group_col: str,
    n_splits: int,
    max_iter: int,
    seed: int,
    scheme_index: int,
    fold_audits: list[dict[str, object]],
    failures: dict[float, list[str]],
    spec: MulticlassTaskSpec,
) -> dict[float, dict[str, float]]:
    """在一个方案上跑内层分组 5 折，返回 {C: {参与者: 多类对数损失}}。

    该辅助函数被路线 A（多候选方案）与路线 B（单特征方案）共用，确保两条路线
    的内层折结构、预处理重拟合与参与者等权权重完全一致。
    """
    inner_losses: dict[float, dict[str, float]] = {float(c): {} for c in candidates}
    require_scheme_columns(outer_train, scheme)
    splitter = GroupKFold(n_splits=int(n_splits))
    splits = list(splitter.split(outer_train, labels, outer_train[group_col].astype(str).to_numpy()))

    for fold_index, (train_idx, valid_idx) in enumerate(splits):
        inner_train = outer_train.iloc[train_idx].copy()
        inner_valid = outer_train.iloc[valid_idx].copy()
        y_train = labels[train_idx]
        y_valid = labels[valid_idx]
        train_groups = sorted(inner_train[group_col].astype(str).unique().tolist())
        valid_groups = sorted(inner_valid[group_col].astype(str).unique().tolist())
        if set(train_groups) & set(valid_groups):
            raise MulticlassModelSelectionError(
                "inner training/validation participant overlap detected"
            )

        train_weights = participant_equal_row_weights(
            inner_train,
            group_col=group_col,
            normalize_mean_one=True,
        )
        audit: dict[str, object] = {
            "scheme_index": int(scheme_index),
            "feature_set_id": scheme.feature_set_id,
            "inner_fold": int(fold_index),
            "train_group_ids": train_groups,
            "validation_group_ids": valid_groups,
            "training_weights": participant_weight_audit(
                inner_train, train_weights, group_col=group_col
            ),
            "preprocessing": None,
            "loss_by_c": {},
            "participant_loss_by_c": {},
            "failure_by_c": {},
            "inner_train_class_support": _class_counts(y_train, spec=spec),
            "inner_validation_class_support": _class_counts(y_valid, spec=spec),
            "inner_train_missing_classes": sorted(
                set(spec.sorted_classes) - set(np.unique(y_train).tolist())
            ),
            "selection_metric": SELECTION_METRIC,
        }
        try:
            # 每个内层折都用**自己的**内层训练参与者重新拟合预处理；
            # 验证参与者的取值不参与中位数/标准化参数的估计。
            state = fit_preprocessing(inner_train, columns=scheme.columns, group_col=group_col)
            x_train = apply_preprocessing(inner_train, state, group_col=group_col)
            x_valid = apply_preprocessing(inner_valid, state, group_col=group_col)
            audit["preprocessing"] = state.audit_dict()
        except (PreprocessingContractError, SupervisedLearningContractError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            for c in candidates:
                failures[float(c)].append(f"inner_fold={fold_index}: {reason}")
                audit["failure_by_c"][str(c)] = reason
            fold_audits.append(audit)
            continue

        validation_group_rows = inner_valid[group_col].astype(str).to_numpy()
        for c_index, c in enumerate(candidates):
            c_value = float(c)
            try:
                model = _fit_multinomial_logistic(
                    x_train,
                    y_train,
                    c=c_value,
                    max_iter=max_iter,
                    seed=seed + scheme_index * 1000 + fold_index * 100 + c_index,
                    sample_weight=train_weights,
                    spec=spec,
                )
                aligned = aligned_class_probabilities(
                    model.predict_proba(x_valid), model.classes_, spec=spec
                )
                per_participant = participant_multiclass_validation_losses(
                    y_valid, aligned, validation_group_rows, spec=spec
                )
                overlap = set(inner_losses[c_value]) & set(per_participant)
                if overlap:
                    raise MulticlassModelSelectionError(
                        "validation participants appeared in more than one inner fold: "
                        f"{sorted(overlap)}"
                    )
                inner_losses[c_value].update(per_participant)
                fold_macro = float(np.mean(list(per_participant.values())))
                audit["loss_by_c"][str(c_value)] = fold_macro
                audit["participant_loss_by_c"][str(c_value)] = dict(per_participant)
            except _EXPECTED_MODEL_FAILURES as exc:
                reason = f"{type(exc).__name__}: {exc}"
                failures[c_value].append(f"inner_fold={fold_index}: {reason}")
                audit["failure_by_c"][str(c_value)] = reason
        fold_audits.append(audit)
    return inner_losses


def _finalize_inner_scores(
    inner_losses: Mapping[float, Mapping[str, float]],
    failures: Mapping[float, Sequence[str]],
    *,
    scheme_id: str,
    candidates: Sequence[float],
    required_validation_groups: Sequence[str],
) -> tuple[dict[str, float], list[tuple[float, float]]]:
    """把逐折参与者损失汇总为 (审计用分数字典, 可用候选列表)。"""
    required = set(str(group) for group in required_validation_groups)
    scores: dict[str, float] = {}
    eligible: list[tuple[float, float]] = []
    for c in candidates:
        c_value = float(c)
        score_key = f"{scheme_id}|C={c_value:g}"
        observed = set(str(group) for group in inner_losses[c_value])
        reasons = list(failures[c_value])
        if reasons or observed != required:
            if observed != required and not reasons:
                reasons.append(
                    "incomplete_validation_participants: "
                    f"missing={sorted(required - observed)}, extra={sorted(observed - required)}"
                )
            continue
        macro_loss = float(np.mean([inner_losses[c_value][group] for group in sorted(required)]))
        if not np.isfinite(macro_loss):
            continue
        scores[score_key] = macro_loss
        eligible.append((macro_loss, c_value))
    return scores, eligible


def select_multiclass_logistic(
    outer_train: pd.DataFrame,
    y_outer_train: Sequence[object] | np.ndarray,
    *,
    feature_schemes: Sequence[FeatureScheme],
    group_col: str = "participant_group_id",
    c_candidates: Sequence[float] = DEFAULT_C_CANDIDATES,
    n_splits: int = DEFAULT_INNER_SPLITS,
    max_iter: int = DEFAULT_MAX_ITER,
    seed: int = 20260910,
    spec: MulticlassTaskSpec = Q1_MULTICLASS_SPEC,
) -> MulticlassSelectionResult:
    """路线 A：按内层参与者分组 5 折的四分类主指标选择特征方案与 C。

    参数：
        outer_train: 某一外层折的训练集（只含外层训练参与者）。
        y_outer_train: 与 ``outer_train`` 行对齐的四分类整数标签。
        feature_schemes: 冻结的比较特异特征方案（来自注册表，不是按性能挑选的赢家）。
        group_col: 参与者分组列。
        c_candidates: 正则强度候选（冻结 ``(0.01, 0.1, 1.0, 10.0)``，不得扩展）。
        n_splits: 内层分组折数（冻结 5）。
        max_iter: 最大迭代次数（冻结 2000）。
        seed: 随机种子基底（冻结 20260910）。
        spec: 四分类任务合同。
    返回：``MulticlassSelectionResult``，含胜者、选择准则名与完整内层折审计。
    异常：
        SupervisedLearningContractError: 列/分组缺失或标签非法。
        MulticlassModelSelectionError: 所有候选都失败，或参与分组数不足以做内层 CV。
    """
    if group_col not in outer_train.columns:
        raise SupervisedLearningContractError(f"missing grouping column: {group_col}")
    if outer_train[group_col].isna().any():
        raise SupervisedLearningContractError(f"{group_col} contains missing participant IDs")
    if not feature_schemes:
        raise MulticlassModelSelectionError("no feature schemes were supplied for nested selection")
    candidates = tuple(float(c) for c in c_candidates)
    if not candidates or any((not np.isfinite(c) or c <= 0) for c in candidates):
        raise MulticlassModelSelectionError("Logistic C candidates must be finite positive values")
    if int(n_splits) < 2:
        raise MulticlassModelSelectionError("inner grouped CV requires at least 2 splits")

    labels = multiclass_labels(y_outer_train, len(outer_train), spec=spec)
    groups = outer_train[group_col].astype(str).to_numpy()
    unique_groups = sorted(set(groups.tolist()))
    if len(unique_groups) < int(n_splits):
        raise MulticlassModelSelectionError(
            f"inner grouped CV needs {n_splits} participant groups; got {len(unique_groups)}"
        )

    participant_losses: dict[str, dict[str, float]] = {}
    failures: dict[tuple[str, float], list[str]] = {
        (scheme.feature_set_id, float(c)): [] for scheme in feature_schemes for c in candidates
    }
    fold_audits: list[dict[str, object]] = []
    scores: dict[str, float] = {}
    eligible: list[tuple[float, int, int, FeatureScheme, float]] = []

    for scheme_index, scheme in enumerate(feature_schemes):
        scheme_failures: dict[float, list[str]] = {float(c): [] for c in candidates}
        inner_losses = _inner_grouped_cv_participant_losses(
            outer_train,
            labels,
            scheme=scheme,
            candidates=candidates,
            group_col=group_col,
            n_splits=int(n_splits),
            max_iter=int(max_iter),
            seed=int(seed),
            scheme_index=scheme_index,
            fold_audits=fold_audits,
            failures=scheme_failures,
            spec=spec,
        )
        scheme_scores, scheme_eligible = _finalize_inner_scores(
            inner_losses,
            scheme_failures,
            scheme_id=scheme.feature_set_id,
            candidates=candidates,
            required_validation_groups=unique_groups,
        )
        scores.update(scheme_scores)
        for loss, c_value in scheme_eligible:
            c_index = candidates.index(c_value)
            eligible.append((loss, scheme_index, c_index, scheme, c_value))
        for c in candidates:
            c_value = float(c)
            reasons = scheme_failures[c_value]
            observed = set(str(group) for group in inner_losses[c_value])
            if observed != set(unique_groups) and not reasons:
                reasons.append(
                    "incomplete_validation_participants: "
                    f"missing={sorted(set(unique_groups) - observed)}, "
                    f"extra={sorted(observed - set(unique_groups))}"
                )
            participant_losses[f"{scheme.feature_set_id}|C={c_value:g}"] = dict(inner_losses[c_value])
            failures[(scheme.feature_set_id, c_value)] = reasons

    failed_candidates = {
        f"{scheme_id}|C={c:g}": reasons
        for (scheme_id, c), reasons in failures.items()
        if reasons
    }
    if not eligible:
        raise MulticlassModelSelectionError(
            "all multiclass feature-scheme/C candidates failed inner grouped CV; "
            f"failures={failed_candidates}"
        )

    _, _, _, winner_scheme, winner_c = min(eligible, key=lambda item: (item[0], item[1], item[2]))
    return MulticlassSelectionResult(
        feature_scheme=winner_scheme,
        selected_c=float(winner_c),
        candidate_participant_macro_log_loss=scores,
        inner_fold_audits=fold_audits,
        failed_candidates=failed_candidates,
    )


def _single_feature_scheme(feature_id: str, column: str) -> FeatureScheme:
    """把注册特征池中的一个候选特征包装成单列方案。"""
    return FeatureScheme(feature_set_id=feature_id, columns=(column,), description=feature_id)


def forward_select_multiclass(
    outer_train: pd.DataFrame,
    y_outer_train: Sequence[object] | np.ndarray,
    *,
    candidate_columns: Mapping[str, str],
    group_col: str = "participant_group_id",
    c_candidates: Sequence[float] = DEFAULT_C_CANDIDATES,
    n_splits: int = DEFAULT_INNER_SPLITS,
    max_iter: int = DEFAULT_MAX_ITER,
    seed: int = 20260910,
    spec: MulticlassTaskSpec = Q1_MULTICLASS_SPEC,
) -> dict[str, object]:
    """路线 B：外层训练折内部的前向逐步特征选择（预注册 §5.2）。

    参数：
        outer_train: 某一外层折的训练集。
        y_outer_train: 对齐的四分类整数标签。
        candidate_columns: ``{feature_id: predictor_column}``；feature_id 只使用冻结注册
            特征 ID，predictor_column 是注册表中该特征唯一的预测列。
        group_col: 参与者分组列。
        c_candidates: 每个候选特征都要试的 C 网格（冻结 (0.01, 0.1, 1.0, 10.0)）。
        n_splits: 内层分组折数（冻结 5）。
        max_iter: 最大迭代次数（冻结 2000）。
        seed: 随机种子基底（冻结 20260910）。
        spec: 四分类任务合同。
    返回：
        审计字典，含 ``selected_feature_set``（空集是合法结果且必须显式记录）、
        ``selected_feature_ids``、``selected_columns``、``selected_c``、
        ``selected_inner_participant_macro_multiclass_log_loss``、逐步 ``steps`` 审计
        以及 ``unavailable_candidate_columns``。
    规则（逐字实现 §5.2，不得改动）：
        1. 起点为空特征集；
        2. 每一步对集合外每个**该折内实际可用**的候选特征配以每个 C 运行内层分组 5 折；
        3. 若最低损失严格低于当前集合的对应损失则加入并固定该 C，否则停止；
        4. 平局按：更少特征 -> 更小 C -> feature_id 字典序；
        5. 允许终止于空集，此时外层预测退化为训练折类别先验常数预测；
        6. 某特征整列不可用（模态缺失）时不得选入，并记入折审计。
    """
    if group_col not in outer_train.columns:
        raise SupervisedLearningContractError(f"missing grouping column: {group_col}")
    if not candidate_columns:
        raise SupervisedLearningContractError("route B requires a non-empty registered feature pool")
    candidates = tuple(float(c) for c in c_candidates)
    if not candidates or any((not np.isfinite(c) or c <= 0) for c in candidates):
        raise MulticlassModelSelectionError("Logistic C candidates must be finite positive values")
    labels = multiclass_labels(y_outer_train, len(outer_train), spec=spec)
    unique_groups = sorted(outer_train[group_col].astype(str).unique().tolist())
    if len(unique_groups) < int(n_splits):
        raise MulticlassModelSelectionError(
            f"inner grouped CV needs {n_splits} participant groups; got {len(unique_groups)}"
        )

    available: dict[str, str] = {}
    unavailable: dict[str, str] = {}
    for feature_id in sorted(candidate_columns):
        column = str(candidate_columns[feature_id])
        if column not in outer_train.columns:
            unavailable[str(feature_id)] = f"column_absent_from_outer_training_fold:{column}"
            continue
        numeric = pd.to_numeric(outer_train[column], errors="coerce")
        if numeric.notna().sum() == 0:
            # 整列不可用（模态在该折缺失）：不得选入，但必须记录。
            unavailable[str(feature_id)] = f"column_all_missing_in_outer_training_fold:{column}"
            continue
        available[str(feature_id)] = column

    selected_ids: list[str] = []
    selected_columns: list[str] = []
    selected_c: float | None = None
    selected_loss: float | None = None
    steps: list[dict[str, object]] = []
    stop_reason = "no_strict_improvement"

    step_index = 0
    while True:
        remaining = [fid for fid in sorted(available) if fid not in selected_ids]
        if not remaining:
            stop_reason = "candidate_pool_exhausted"
            break

        step_evaluations: list[dict[str, object]] = []
        best: tuple[float, float, str, FeatureScheme] | None = None
        for candidate_feature_id in remaining:
            column = available[candidate_feature_id]
            trial_columns = tuple(selected_columns + [column])
            trial_scheme = FeatureScheme(
                feature_set_id="forward::" + "+".join(
                    selected_ids + [candidate_feature_id]
                ),
                columns=trial_columns,
                description="route B forward selection trial feature set",
            )
            trial_failures: dict[float, list[str]] = {float(c): [] for c in candidates}
            trial_audits: list[dict[str, object]] = []
            inner_losses = _inner_grouped_cv_participant_losses(
                outer_train,
                labels,
                scheme=trial_scheme,
                candidates=candidates,
                group_col=group_col,
                n_splits=int(n_splits),
                max_iter=int(max_iter),
                seed=int(seed),
                scheme_index=step_index,
                fold_audits=trial_audits,
                failures=trial_failures,
                spec=spec,
            )
            scheme_scores, scheme_eligible = _finalize_inner_scores(
                inner_losses,
                trial_failures,
                scheme_id=trial_scheme.feature_set_id,
                candidates=candidates,
                required_validation_groups=unique_groups,
            )
            step_evaluations.append(
                {
                    "candidate_feature_id": candidate_feature_id,
                    "candidate_column": column,
                    "trial_feature_set_id": trial_scheme.feature_set_id,
                    "trial_columns": list(trial_columns),
                    "loss_by_c": scheme_scores,
                    "failure_by_c": {
                        f"C={float(c):g}": list(trial_failures[float(c)]) for c in candidates
                    },
                    "inner_fold_audits": trial_audits,
                }
            )
            if not scheme_eligible:
                continue
            # 每个特征先只保留它自己最好的那个 C；跨特征的比较键就是
            # （损失, C, feature_id），这正是 §5.2 第 4 条的平局顺序。
            best_for_feature = min(
                scheme_eligible, key=lambda item: (item[0], item[1], candidate_feature_id)
            )
            # 注意：此处比较的集合大小始终相同（都是「当前集合 + 1 个候选特征」），
            # 因此 §5.2 第 4 条中「更少特征优先」只会在跨步比较时生效，而跨步比较
            # 由下面的 strict_improvement 规则处理（不改进就不接受）。
            candidate_key = (
                best_for_feature[0],
                best_for_feature[1],
                candidate_feature_id,
                trial_scheme,
            )
            if best is None or candidate_key[:3] < best[:3]:
                best = candidate_key

        if best is None:
            steps.append(
                {
                    "step": step_index,
                    "feature_set_before": list(selected_ids),
                    "candidate_evaluations": step_evaluations,
                    "accepted": False,
                    "stop_reason": "no_estimable_candidate_in_this_outer_training_fold",
                }
            )
            stop_reason = "no_estimable_candidate_in_this_outer_training_fold"
            break

        best_loss, best_c, best_feature_id, best_scheme = best
        strict_improvement = selected_loss is None or best_loss < selected_loss
        step_audit: dict[str, object] = {
            "step": step_index,
            "feature_set_before": list(selected_ids),
            "current_loss_before": None if selected_loss is None else float(selected_loss),
            "best_candidate_feature_id": best_feature_id,
            "best_candidate_c": float(best_c),
            "best_candidate_loss": float(best_loss),
            "strict_improvement": bool(strict_improvement),
            "candidate_evaluations": step_evaluations,
        }
        if not strict_improvement:
            step_audit["accepted"] = False
            step_audit["stop_reason"] = "no_strict_improvement"
            steps.append(step_audit)
            stop_reason = "no_strict_improvement"
            break

        step_audit["accepted"] = True
        steps.append(step_audit)
        selected_ids.append(best_feature_id)
        selected_columns.append(available[best_feature_id])
        selected_c = float(best_c)
        selected_loss = float(best_loss)
        step_index += 1

    return {
        "route": "B_forward_selection_within_outer_training_fold",
        "selection_metric": SELECTION_METRIC,
        "candidate_feature_pool": {str(k): str(v) for k, v in sorted(candidate_columns.items())},
        "available_candidate_feature_ids": sorted(available),
        "unavailable_candidate_columns": unavailable,
        "selected_feature_set": list(selected_ids),
        "selected_feature_ids": list(selected_ids),
        "selected_columns": list(selected_columns),
        "selected_c": None if selected_c is None else float(selected_c),
        "selected_inner_participant_macro_multiclass_log_loss": (
            None if selected_loss is None else float(selected_loss)
        ),
        "empty_feature_set": not selected_ids,
        "empty_feature_set_fallback": (
            "outer_training_fold_class_prior_constant_prediction"
            if not selected_ids
            else ""
        ),
        "stop_reason": stop_reason,
        "n_steps": int(len(steps)),
        "steps": steps,
    }


def refit_multiclass_and_predict(
    outer_train: pd.DataFrame,
    y_outer_train: Sequence[object] | np.ndarray,
    outer_test_features: pd.DataFrame,
    *,
    feature_scheme: FeatureScheme | None,
    selected_c: float | None,
    group_col: str = "participant_group_id",
    max_iter: int = DEFAULT_MAX_ITER,
    seed: int = 20260910,
    spec: MulticlassTaskSpec = Q1_MULTICLASS_SPEC,
) -> dict[str, object]:
    """在外层训练集上全量重拟合并预测无标签外层测试行。

    参数：
        outer_train: 该外层折的训练集。
        y_outer_train: 对齐的四分类标签。
        outer_test_features: 外层测试行的**无结局**特征表（含分组列与方案列）。
        feature_scheme: 选中的特征方案；``None`` 表示路线 B 终止于空集。
        selected_c: 选中的正则强度；``None`` 时必须与空集方案同时出现。
        group_col: 参与者分组列。
        max_iter: 最大迭代次数（冻结 2000）。
        seed: 随机种子。
        spec: 四分类任务合同。
    返回：
        与 ``models.refit_logistic_and_predict`` 对应的四分类审计字典：
        ``feature_set_id``、``selected_c``、``n_train_rows``、``n_test_rows``、
        ``train_group_ids``、``test_group_ids``、``training_weights``、``preprocessing``、
        ``model_classes``、``coefficient_scale``、逐类 ``standardized_coefficients``、
        ``intercepts``、``(n,4)`` 的 ``class_probabilities``、``predicted_class``。
    异常：
        SupervisedLearningContractError: 外层测试表含任何结局列（泄漏），
            或空集方案与 ``selected_c`` 的组合自相矛盾。
    """
    leaked = sorted(_CROSS_VALIDATED_OUTCOME_COLUMNS & set(outer_test_features.columns))
    if leaked:
        raise SupervisedLearningContractError(
            f"outer_test_features must be outcome-free; leaked columns: {leaked}"
        )
    labels = multiclass_labels(y_outer_train, len(outer_train), spec=spec)
    test_group_ids = sorted(outer_test_features[group_col].astype(str).unique().tolist())

    if feature_scheme is None:
        # 路线 B 空集退化路径：只能用外层训练折的类别先验做常数预测。
        # 这条路径不使用任何特征，因此也没有预处理状态可拟合。
        if selected_c is not None:
            raise SupervisedLearningContractError(
                "an empty forward-selected feature set cannot carry a selected C"
            )
        priors = np.asarray(
            [
                float((labels == int(label)).sum()) / float(len(labels))
                for label in spec.sorted_classes
            ],
            dtype=float,
        )
        proba = np.tile(priors, (len(outer_test_features), 1))
        predicted = np.asarray(
            [spec.sorted_classes[int(index)] for index in np.argmax(proba, axis=1)], dtype=int
        )
        return {
            "feature_set_id": PRIOR_FALLBACK_FEATURE_SET_ID,
            "selected_c": None,
            "fallback": "outer_training_fold_class_prior_constant",
            "n_train_rows": int(len(outer_train)),
            "n_test_rows": int(len(outer_test_features)),
            "train_group_ids": sorted(outer_train[group_col].astype(str).unique().tolist()),
            "test_group_ids": test_group_ids,
            "training_weights": participant_weight_audit(
                outer_train,
                participant_equal_row_weights(
                    outer_train, group_col=group_col, normalize_mean_one=True
                ),
                group_col=group_col,
            ),
            "preprocessing": None,
            "model_classes": list(spec.sorted_classes),
            "coefficient_scale": "not_applicable_prior_constant_predictor",
            "standardized_coefficients": {str(label): {} for label in spec.sorted_classes},
            "intercepts": {str(label): 0.0 for label in spec.sorted_classes},
            "train_class_priors": {
                str(label): float(priors[index]) for index, label in enumerate(spec.sorted_classes)
            },
            "class_probabilities": proba,
            "predicted_class": predicted,
            "uses_features": False,
        }

    if selected_c is None:
        raise SupervisedLearningContractError(
            "a non-empty forward-selected feature set requires a selected C"
        )
    require_scheme_columns(outer_train, feature_scheme)
    require_scheme_columns(outer_test_features, feature_scheme)
    state = fit_preprocessing(outer_train, columns=feature_scheme.columns, group_col=group_col)
    x_train = apply_preprocessing(outer_train, state, group_col=group_col)
    x_test = apply_preprocessing(outer_test_features, state, group_col=group_col)
    train_weights = participant_equal_row_weights(
        outer_train, group_col=group_col, normalize_mean_one=True
    )
    model = _fit_multinomial_logistic(
        x_train,
        labels,
        c=float(selected_c),
        max_iter=max_iter,
        seed=seed,
        sample_weight=train_weights,
        spec=spec,
    )
    proba = aligned_class_probabilities(model.predict_proba(x_test), model.classes_, spec=spec)
    predicted = model.predict(x_test).astype(int)

    n_classes = len(spec.sorted_classes)
    if model.coef_.shape != (n_classes, x_train.shape[1]) or model.intercept_.shape != (n_classes,):
        raise MulticlassModelSelectionError(
            "unexpected multiclass logistic coefficient shape "
            f"coef={model.coef_.shape}, intercept={model.intercept_.shape}"
        )
    # 系数用训练集预处理后的列名标注：这才是被正则化、被解释的对象。
    # 外层测试行从不参与标准化参数的估计，因此不能用测试列来命名系数。
    standardized_coefficients = {
        str(label): {
            str(column): float(value)
            for column, value in zip(x_train.columns.tolist(), model.coef_[index].tolist(), strict=True)
        }
        for index, label in enumerate(model.classes_.tolist())
    }
    intercepts = {
        str(label): float(value)
        for label, value in zip(model.classes_.tolist(), model.intercept_.tolist(), strict=True)
    }

    return {
        "feature_set_id": feature_scheme.feature_set_id,
        "selected_c": float(selected_c),
        "fallback": "",
        "n_train_rows": int(len(outer_train)),
        "n_test_rows": int(len(outer_test_features)),
        "train_group_ids": list(state.fit_group_ids),
        "test_group_ids": test_group_ids,
        "training_weights": participant_weight_audit(
            outer_train, train_weights, group_col=group_col
        ),
        "preprocessing": state.audit_dict(),
        "model_classes": [int(value) for value in model.classes_.tolist()],
        "coefficient_scale": "post_imputation_participant_equal_standardized_predictors",
        "standardized_coefficients": standardized_coefficients,
        "intercepts": intercepts,
        "class_probabilities": proba,
        "predicted_class": predicted,
        "uses_features": True,
    }


__all__ = [
    "DEFAULT_C_CANDIDATES",
    "DEFAULT_INNER_SPLITS",
    "DEFAULT_MAX_ITER",
    "MulticlassModelSelectionError",
    "MulticlassSelectionResult",
    "PRIOR_FALLBACK_FEATURE_SET_ID",
    "SELECTION_METRIC",
    "forward_select_multiclass",
    "multiclass_labels",
    "participant_multiclass_validation_losses",
    "refit_multiclass_and_predict",
    "select_multiclass_logistic",
]
