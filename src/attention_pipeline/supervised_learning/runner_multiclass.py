"""Participant-disjoint outer LOSO orchestration for the Q1 four-class line.

文件：runner_multiclass.py
版本：1.0.0
功能：为预注册 `1.16.24` 提供外层参与者互斥 LOSO 编排，支持两条公平对照路线：
      - `route="A"`：复用二分类线已冻结的比较特异特征方案，只把标签换成四分类；
      - `route="B"`：在每个外层训练折内部做 §5.2 前向逐步特征选择。
      折结构、预处理、参与者等权权重与失败保留规则与二分类 `runner.run_nested_loso` 一致。
      本模块只新增：不修改 runner.py / models.py / evaluation.py。
用法：
    from attention_pipeline.supervised_learning.runner_multiclass import run_nested_loso_multiclass
依赖：numpy、pandas

结构性保证：
- 外层测试参与者的标签从不进入选择或预测接口（预测接口只接收无结局列的特征表）；
- 失败折与四类不全的折都写成显式记录，并同时进入 predictions / failures / fold_audits，
  不做静默删除或补类。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .feature_schemes import FeatureScheme, validate_mainline_feature_scheme
from .models_multiclass import (
    MulticlassModelSelectionError,
    PRIOR_FALLBACK_FEATURE_SET_ID,
    forward_select_multiclass,
    multiclass_labels,
    refit_multiclass_and_predict,
    select_multiclass_logistic,
)
from .preprocessing import (
    PreprocessingContractError,
    participant_equal_row_weights,
    participant_weight_audit,
)
from .runner import (
    ALLOWED_MEMBERSHIP_TYPES,
    DEFAULT_GROUP_COLUMN,
    MEMBERSHIP_COLUMN,
    OPTIONAL_PROBE_LOCATORS,
    OPTIONAL_REPORT_CONTEXT,
    REQUIRED_PROBE_LOCATORS,
    _resolve_analysis_set_id,
    _resolve_membership_type,
)
from .task import Q1_BINARY_SPEC, SupervisedLearningContractError
from .task_multiclass import (
    Q1_MULTICLASS_SPEC,
    MulticlassTaskSpec,
    encode_q1_multiclass,
)

#: 允许的路线标识。
ALLOWED_ROUTES = frozenset({"A", "B"})
#: 路线 B 的模型标识（单模型，因为特征选择在折内完成）。
ROUTE_B_MODEL_ID = "forward_selected_4class"
#: 任何结局列都不得出现在外层测试特征表中。
_OUTCOME_COLUMNS = frozenset(
    {
        Q1_MULTICLASS_SPEC.source_column,
        "q1_binary",
        Q1_BINARY_SPEC.positive_probability_name,
        "predicted_q1_binary",
    }
)
_EXPECTED_FOLD_FAILURES = (
    MulticlassModelSelectionError,
    PreprocessingContractError,
    SupervisedLearningContractError,
    ValueError,
    FloatingPointError,
    np.linalg.LinAlgError,
)


@dataclass
class MulticlassSupervisedRunResult:
    """四分类外层运行结果：逐 probe 预测、逐折审计、失败表与 metadata。"""

    predictions: pd.DataFrame
    fold_audits: list[dict[str, object]] = field(default_factory=list)
    failures: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict[str, object] = field(default_factory=dict)


def outer_training_prior_fallback_predict(
    outer_train: pd.DataFrame,
    outer_test_features: pd.DataFrame,
    *,
    group_col: str = DEFAULT_GROUP_COLUMN,
    spec: MulticlassTaskSpec = Q1_MULTICLASS_SPEC,
) -> dict[str, object]:
    """路线 B 空集退化预测器（预注册 §5.2 第 5 条）。

    参数：
        outer_train: 外层训练集（只用其四类标签估计先验）。
        outer_test_features: 外层测试特征表（无结局列）。
        group_col: 参与者分组列。
        spec: 四分类任务合同。
    返回：与 ``refit_multiclass_and_predict`` 兼容的审计字典，且显式标记
        ``uses_features=False``。
    """
    leaked = sorted(_OUTCOME_COLUMNS & set(outer_test_features.columns))
    if leaked:
        raise SupervisedLearningContractError(
            f"outer_test_features must be outcome-free; leaked columns: {leaked}"
        )
    labels = multiclass_labels(
        encode_q1_multiclass(outer_train[spec.source_column], spec=spec).to_numpy(dtype=int),
        len(outer_train),
        spec=spec,
    )
    counts = np.asarray(
        [int((labels == int(label)).sum()) for label in spec.sorted_classes], dtype=float
    )
    priors = counts / float(counts.sum())
    proba = np.tile(priors, (len(outer_test_features), 1))
    predicted = np.asarray(
        [spec.sorted_classes[int(index)] for index in np.argmax(proba, axis=1)], dtype=int
    )
    weights = participant_equal_row_weights(
        outer_train, group_col=group_col, normalize_mean_one=True
    )
    return {
        "feature_set_id": PRIOR_FALLBACK_FEATURE_SET_ID,
        "selected_c": None,
        "fallback": "outer_training_fold_class_prior_constant",
        "uses_features": False,
        "n_train_rows": int(len(outer_train)),
        "n_test_rows": int(len(outer_test_features)),
        "train_group_ids": sorted(outer_train[group_col].astype(str).unique().tolist()),
        "test_group_ids": sorted(outer_test_features[group_col].astype(str).unique().tolist()),
        "training_weights": participant_weight_audit(
            outer_train, weights, group_col=group_col
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
    }


def _declared_feature_columns(
    model_feature_schemes: Mapping[str, Sequence[FeatureScheme]],
) -> tuple[str, ...]:
    columns: list[str] = []
    seen: set[str] = set()
    for schemes in model_feature_schemes.values():
        for scheme in schemes:
            for column in scheme.columns:
                if column not in seen:
                    seen.add(column)
                    columns.append(column)
    return tuple(columns)


def _validate_complete_membership(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    membership_type: str,
) -> None:
    """``included_complete`` 必须在四分类线内同样语义完整，否则报错。"""
    if membership_type != "included_complete":
        return
    missing_counts: dict[str, int] = {}
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        invalid_text = frame[column].notna() & numeric.isna()
        if invalid_text.any():
            bad = frame.loc[invalid_text, column].astype(str).drop_duplicates().tolist()
            raise SupervisedLearningContractError(
                f"included_complete contains non-numeric values in feature {column}: {bad}"
            )
        missing_n = int(numeric.isna().sum())
        if missing_n:
            missing_counts[column] = missing_n
    if missing_counts:
        raise SupervisedLearningContractError(
            "included_complete cannot contain missing predictor values; "
            f"missing_counts={missing_counts}"
        )


def _compact_forward_selection_audit(audit: Mapping[str, object]) -> dict[str, object]:
    """精简路线 B 的逐步审计：保留每个 (特征, C) 的分数与失败，去掉逐内层折大数组。

    为什么需要：前向选择每一步都会为每个候选 (特征, C) 跑一遍内层 5 折，逐折审计
    数组会随「步数 x 候选数 x C 数 x 折数」膨胀，而折审计是每折一份的 JSON；真正需要
    长期留存的是每个候选的参与者宏平均损失、失败原因与最终选择，逐折细节只在需要
    复核时才重新运行时产出。
    """
    compact = dict(audit)
    steps: list[dict[str, object]] = []
    for raw_step in audit.get("steps", []) or []:
        step = dict(raw_step)
        evaluations = []
        for raw_evaluation in step.get("candidate_evaluations", []) or []:
            evaluation = dict(raw_evaluation)
            inner_audits = evaluation.pop("inner_fold_audits", [])
            evaluation["n_inner_fold_audits"] = int(len(inner_audits))
            evaluations.append(evaluation)
        step["candidate_evaluations"] = evaluations
        steps.append(step)
    compact["steps"] = steps
    return compact


def _registered_feature_columns(registered_features: object) -> dict[str, str]:
    """把冻结注册特征池转成 ``{feature_id: 唯一预测列}``。

    参数：
        registered_features: ``RegisteredFeature`` 序列，或含 ``feature_id`` /
            ``columns`` 键的映射序列。
    返回：``{feature_id: predictor_column}``（按 feature_id 排序）。
    异常：
        SupervisedLearningContractError: 池为空、条目格式非法，或某特征不是单列表示。
    说明：
        路线 B 按 feature_id 字典序打破平局，因此 feature_id 必须来自冻结注册表；
        一个注册特征若对应多列，其「前向一步」就不再是单维候选，本模块直接拒绝，
        而不是自行挑一列。
    """
    if registered_features is None:
        raise SupervisedLearningContractError(
            "route B requires the frozen registered feature pool"
        )
    if isinstance(registered_features, (str, bytes)):
        raise SupervisedLearningContractError("registered_features must be a sequence of feature records")
    if isinstance(registered_features, Mapping):
        items = list(registered_features.items())
    else:
        items = list(registered_features)
    if not items:
        raise SupervisedLearningContractError("route B registered feature pool is empty")

    mapping: dict[str, str] = {}
    for item in items:
        if isinstance(item, Mapping):
            feature_id = str(item.get("feature_id", "")).strip()
            columns = item.get("columns")
        elif isinstance(item, tuple) and len(item) == 2:
            # 允许直接传入 ``{feature_id: predictor_column}`` 的条目对，
            # 这是本模块公开文档里推荐的紧凑写法。
            feature_id = str(item[0]).strip()
            columns = (item[1],)
        else:
            feature_id = str(getattr(item, "feature_id", "")).strip()
            columns = getattr(item, "columns", None)
        if not feature_id:
            raise SupervisedLearningContractError("registered feature record has a blank feature_id")
        if not isinstance(columns, Sequence) or isinstance(columns, (str, bytes)):
            raise SupervisedLearningContractError(
                f"registered feature {feature_id} must declare a columns sequence"
            )
        column_list = [str(value).strip() for value in columns]
        if len(column_list) != 1:
            raise SupervisedLearningContractError(
                f"registered feature {feature_id} must map to exactly one predictor column for "
                f"forward selection; got {column_list}"
            )
        if feature_id in mapping:
            raise SupervisedLearningContractError(
                f"duplicate registered feature_id in route B pool: {feature_id}"
            )
        mapping[feature_id] = column_list[0]
    return {key: mapping[key] for key in sorted(mapping)}


def _validate_frame(
    frame: pd.DataFrame,
    *,
    route: str,
    model_feature_schemes: Mapping[str, Sequence[FeatureScheme]] | None,
    candidate_columns: Mapping[str, str] | None,
    group_col: str,
    spec: MulticlassTaskSpec,
) -> tuple[pd.DataFrame, list[str]]:
    """校验四分类分析帧，返回 (归一化帧, 需要保留的定位列)。"""
    required = {group_col, spec.source_column, *REQUIRED_PROBE_LOCATORS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SupervisedLearningContractError(
            f"four-class supervised analysis frame missing required columns: {missing}"
        )
    if frame.empty:
        raise SupervisedLearningContractError("four-class supervised analysis frame is empty")
    if frame[group_col].isna().any():
        raise SupervisedLearningContractError(f"{group_col} contains missing participant IDs")
    if frame.duplicated(list(REQUIRED_PROBE_LOCATORS)).any():
        raise SupervisedLearningContractError(
            f"duplicate probe locator rows for {list(REQUIRED_PROBE_LOCATORS)}"
        )

    if route == "A":
        if not model_feature_schemes:
            raise SupervisedLearningContractError(
                "route A requires the frozen model/feature-scheme families"
            )
        for model_id, schemes in model_feature_schemes.items():
            name = str(model_id).strip()
            if not name:
                raise SupervisedLearningContractError("model_id must be non-empty")
            if not schemes:
                raise SupervisedLearningContractError(f"model {name} has no feature schemes")
            for scheme in schemes:
                validate_mainline_feature_scheme(scheme)
                scheme_missing = sorted(set(scheme.columns) - set(frame.columns))
                if scheme_missing:
                    raise SupervisedLearningContractError(
                        f"feature scheme {scheme.feature_set_id} missing upstream columns: {scheme_missing}"
                    )
    else:
        if not candidate_columns:
            raise SupervisedLearningContractError(
                "route B requires the frozen registered feature pool"
            )

    encoded = encode_q1_multiclass(frame[spec.source_column], spec=spec)
    if encoded.isna().any():
        raise SupervisedLearningContractError(
            "analysis frame contains probes with missing Q1 four-class labels; "
            "the B-layer analysis set must resolve this before the four-class run"
        )
    out = frame.copy().reset_index(drop=True)
    # 保留原始整数值（1..4），不做 0..3 重映射；额外存一份整数列供模型使用。
    out[spec.source_column] = encoded.reset_index(drop=True).astype(int)
    archive_columns = (
        list(REQUIRED_PROBE_LOCATORS)
        + [c for c in OPTIONAL_PROBE_LOCATORS if c in out.columns]
        + [c for c in OPTIONAL_REPORT_CONTEXT if c in out.columns]
    )
    return out, archive_columns


def _route_a_fold_selection(
    outer_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    schemes: Sequence[FeatureScheme],
    group_col: str,
    c_candidates: Sequence[float],
    inner_splits: int,
    max_iter: int,
    fold_seed: int,
    spec: MulticlassTaskSpec,
) -> tuple[FeatureScheme, float, dict[str, object]]:
    """路线 A 单折选择：返回 (选中方案, 选中 C, 选择审计)。"""
    selection = select_multiclass_logistic(
        outer_train,
        y_train,
        feature_schemes=schemes,
        group_col=group_col,
        c_candidates=c_candidates,
        n_splits=int(inner_splits),
        max_iter=int(max_iter),
        seed=fold_seed,
        spec=spec,
    )
    return selection.feature_scheme, float(selection.selected_c), selection.audit_dict()


def run_nested_loso_multiclass(
    frame: pd.DataFrame,
    *,
    model_feature_schemes: Mapping[str, Sequence[FeatureScheme]] | None = None,
    registered_features: object = None,
    route: str,
    group_col: str = DEFAULT_GROUP_COLUMN,
    c_candidates: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    inner_splits: int = 5,
    max_iter: int = 2000,
    seed: int = 20260910,
    run_id: str = "q1-4class-in-memory",
    analysis_set_id: str | None = None,
    membership_type: str | None = None,
    spec: MulticlassTaskSpec = Q1_MULTICLASS_SPEC,
) -> MulticlassSupervisedRunResult:
    """运行一次完整的四分类外层参与者互斥 LOSO 分析。

    参数：
        frame: 已对齐、已准入的 probe 级分析帧。
        model_feature_schemes: 路线 A 必须提供，映射 model_id -> FeatureScheme 列表。
        registered_features: 路线 B 必须提供，冻结注册特征池（11 个正式特征）。
        route: ``"A"`` 或 ``"B"``。
        group_col: 外层/内层参与者分组列。
        c_candidates: 正则强度候选（冻结 (0.01, 0.1, 1.0, 10.0)）。
        inner_splits: 内层分组折数（冻结 5）。
        max_iter: 最大迭代次数（冻结 2000）。
        seed: 随机种子基底（冻结 20260910）。
        run_id / analysis_set_id / membership_type: 运行与样本标识。
        spec: 四分类任务合同。
    返回：``MulticlassSupervisedRunResult``。
    异常：
        SupervisedLearningContractError: 帧、路线参数或标识不符合冻结合同。
    """
    resolved_route = str(route).strip().upper()
    if resolved_route not in ALLOWED_ROUTES:
        raise SupervisedLearningContractError(
            f"unsupported multiclass route {route!r}; expected one of {sorted(ALLOWED_ROUTES)}"
        )
    candidate_columns = (
        _registered_feature_columns(registered_features) if resolved_route == "B" else None
    )
    data, archive_columns = _validate_frame(
        frame,
        route=resolved_route,
        model_feature_schemes=model_feature_schemes,
        candidate_columns=candidate_columns,
        group_col=group_col,
        spec=spec,
    )
    resolved_analysis_set = _resolve_analysis_set_id(data, analysis_set_id)
    resolved_membership = _resolve_membership_type(data, membership_type)
    declared_columns = (
        _declared_feature_columns(model_feature_schemes or {})
        if resolved_route == "A"
        else tuple(sorted(candidate_columns.values()))
    )
    _validate_complete_membership(
        data, declared_columns, membership_type=resolved_membership
    )

    groups = sorted(data[group_col].astype(str).unique().tolist())
    if len(groups) < 2:
        raise SupervisedLearningContractError(
            "outer LOSO requires at least two participant groups"
        )
    if len(groups) - 1 < int(inner_splits):
        raise SupervisedLearningContractError(
            f"after holding out one participant, inner CV needs {inner_splits} training groups; "
            f"only {len(groups) - 1} remain"
        )

    if resolved_route == "A":
        model_items: list[tuple[str, list[FeatureScheme] | None]] = [
            (str(name), list(schemes)) for name, schemes in (model_feature_schemes or {}).items()
        ]
    else:
        model_items = [(ROUTE_B_MODEL_ID, None)]

    prediction_frames: list[pd.DataFrame] = []
    fold_audits: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []

    for outer_index, held_out_group in enumerate(groups):
        test_mask = data[group_col].astype(str).eq(held_out_group)
        outer_train = data.loc[~test_mask].copy()
        outer_test = data.loc[test_mask].copy()
        train_groups = sorted(outer_train[group_col].astype(str).unique().tolist())
        test_groups = sorted(outer_test[group_col].astype(str).unique().tolist())
        if set(train_groups) & set(test_groups):
            raise SupervisedLearningContractError("outer train/test participant overlap detected")
        if test_groups != [held_out_group]:
            raise SupervisedLearningContractError(
                "outer test fold does not contain exactly the held-out participant"
            )

        y_train = multiclass_labels(
            outer_train[spec.source_column].to_numpy(dtype=int), len(outer_train), spec=spec
        )
        train_class_support = {
            str(label): int((y_train == int(label)).sum()) for label in spec.sorted_classes
        }
        missing_classes = sorted(
            set(spec.sorted_classes) - set(np.unique(y_train).tolist())
        )
        outer_train_class_complete = not missing_classes

        for model_index, (model_id, schemes) in enumerate(model_items):
            fold_seed = int(seed) + outer_index * 10000 + model_index * 1000
            base_prediction = outer_test[
                archive_columns + [group_col, spec.source_column]
            ].copy()
            base_prediction["run_id"] = str(run_id)
            base_prediction["analysis_set_id"] = resolved_analysis_set
            base_prediction[MEMBERSHIP_COLUMN] = resolved_membership
            base_prediction["model_id"] = model_id
            base_prediction["outer_fold_group"] = held_out_group
            base_prediction["route"] = resolved_route

            fold_audit: dict[str, object] = {
                "run_id": str(run_id),
                "analysis_set_id": resolved_analysis_set,
                MEMBERSHIP_COLUMN: resolved_membership,
                "model_id": model_id,
                "route": resolved_route,
                "outer_fold_group": held_out_group,
                "outer_train_group_ids": train_groups,
                "outer_test_group_ids": test_groups,
                "n_outer_train_rows": int(len(outer_train)),
                "n_outer_test_rows": int(len(outer_test)),
                # 外层训练折的四类覆盖必须显式记录；缺失时该折不可估，绝不补类。
                "outer_train_class_support": train_class_support,
                "outer_train_missing_classes": missing_classes,
                "outer_train_class_complete": bool(outer_train_class_complete),
                "selection": None,
                "final_refit": None,
                "failed": False,
                "reason": "",
            }

            try:
                if not outer_train_class_complete:
                    raise MulticlassModelSelectionError(
                        "outer training fold is not estimable: missing declared classes "
                        f"{missing_classes} (observed support {train_class_support})"
                    )

                feature_scheme: FeatureScheme | None
                selected_c: float | None
                if resolved_route == "A":
                    feature_scheme, selected_c, selection_audit = _route_a_fold_selection(
                        outer_train,
                        y_train,
                        schemes=schemes or [],
                        group_col=group_col,
                        c_candidates=c_candidates,
                        inner_splits=int(inner_splits),
                        max_iter=int(max_iter),
                        fold_seed=fold_seed,
                        spec=spec,
                    )
                    fold_audit["selection"] = selection_audit
                else:
                    selection_audit = forward_select_multiclass(
                        outer_train,
                        y_train,
                        candidate_columns=candidate_columns or {},
                        group_col=group_col,
                        c_candidates=c_candidates,
                        n_splits=int(inner_splits),
                        max_iter=int(max_iter),
                        seed=fold_seed,
                        spec=spec,
                    )
                    fold_audit["selection"] = _compact_forward_selection_audit(selection_audit)
                    selected_columns = list(selection_audit["selected_columns"])
                    if selected_columns:
                        feature_scheme = FeatureScheme(
                            feature_set_id="forward::" + "+".join(
                                selection_audit["selected_feature_ids"]
                            ),
                            columns=tuple(selected_columns),
                            description="route B forward-selected feature set",
                        )
                        selected_c = float(selection_audit["selected_c"])
                    else:
                        # §5.2 第 5 条：允许终止于空集，此时退化为训练折类别先验常数预测。
                        feature_scheme = None
                        selected_c = None

                if feature_scheme is None:
                    fitted = outer_training_prior_fallback_predict(
                        outer_train, outer_test[list(dict.fromkeys([group_col]))].copy(),
                        group_col=group_col, spec=spec,
                    )
                else:
                    model_test_columns = list(
                        dict.fromkeys([group_col, *feature_scheme.columns])
                    )
                    outer_test_features = outer_test[model_test_columns].copy()
                    fitted = refit_multiclass_and_predict(
                        outer_train,
                        y_train,
                        outer_test_features,
                        feature_scheme=feature_scheme,
                        selected_c=selected_c,
                        group_col=group_col,
                        max_iter=int(max_iter),
                        seed=fold_seed + 777,
                        spec=spec,
                    )
                if set(fitted["train_group_ids"]) & set(fitted["test_group_ids"]):
                    raise SupervisedLearningContractError(
                        "final refit participant overlap detected"
                    )

                class_probabilities = np.asarray(fitted["class_probabilities"], dtype=float)
                if class_probabilities.shape != (len(outer_test), len(spec.sorted_classes)):
                    raise SupervisedLearningContractError(
                        "final refit produced an unexpected class-probability shape: "
                        f"{class_probabilities.shape}"
                    )
                base_prediction["feature_set_id"] = str(fitted["feature_set_id"])
                base_prediction["selected_c"] = (
                    np.nan if fitted["selected_c"] is None else float(fitted["selected_c"])
                )
                for index, column in enumerate(spec.probability_columns):
                    base_prediction[column] = class_probabilities[:, index]
                base_prediction[spec.predicted_column_name] = np.asarray(
                    fitted["predicted_class"], dtype=int
                )
                base_prediction["model_failed"] = False
                base_prediction["failure_reason"] = ""
                prediction_frames.append(base_prediction)

                final_audit = dict(fitted)
                # 原始概率/标签数组只进逐 probe 预测表，不进折审计（与二分类一致）。
                final_audit.pop("class_probabilities", None)
                final_audit.pop("predicted_class", None)
                fold_audit["final_refit"] = final_audit
                fold_audit["failed"] = False
                fold_audit["reason"] = ""
                fold_audits.append(fold_audit)
            except _EXPECTED_FOLD_FAILURES as exc:
                reason = f"{type(exc).__name__}: {exc}"
                base_prediction["feature_set_id"] = None
                base_prediction["selected_c"] = np.nan
                for column in spec.probability_columns:
                    base_prediction[column] = np.nan
                base_prediction[spec.predicted_column_name] = pd.NA
                base_prediction["model_failed"] = True
                base_prediction["failure_reason"] = reason
                prediction_frames.append(base_prediction)
                failure_rows.append(
                    {
                        "run_id": str(run_id),
                        "analysis_set_id": resolved_analysis_set,
                        MEMBERSHIP_COLUMN: resolved_membership,
                        "model_id": model_id,
                        "route": resolved_route,
                        "outer_fold_group": held_out_group,
                        "n_outer_train_rows": int(len(outer_train)),
                        "n_outer_test_rows": int(len(outer_test)),
                        "reason": reason,
                    }
                )
                fold_audit["final_refit"] = None
                fold_audit["failed"] = True
                fold_audit["reason"] = reason
                fold_audits.append(fold_audit)

    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    failures = pd.DataFrame(failure_rows)
    metadata = {
        "run_id": str(run_id),
        "analysis_set_id": resolved_analysis_set,
        MEMBERSHIP_COLUMN: resolved_membership,
        "task": spec.name,
        "route": resolved_route,
        "class_labels": list(spec.sorted_classes),
        "class_semantics": "nominal_unordered",
        "probability_columns": list(spec.probability_columns),
        "predicted_class_column": spec.predicted_column_name,
        "n_input_rows": int(len(data)),
        "n_participant_groups": int(len(groups)),
        "participant_groups": groups,
        "n_models": int(len(model_items)),
        "model_ids": [name for name, _ in model_items],
        "inner_splits": int(inner_splits),
        "c_candidates": [float(v) for v in c_candidates],
        "outer_method": "leave_one_participant_out",
        "zero_individual_calibration": True,
        "outer_test_outcomes_passed_to_model": False,
        "q2_retained_for_reporting_only": "q2_ordinal_4level" in data.columns,
        "upstream_analysis_set_generation_in_runner": False,
        "registered_feature_pool": (
            {str(k): str(v) for k, v in (candidate_columns or {}).items()}
            if resolved_route == "B"
            else None
        ),
    }
    return MulticlassSupervisedRunResult(
        predictions=predictions, fold_audits=fold_audits, failures=failures, metadata=metadata
    )


__all__ = [
    "ALLOWED_ROUTES",
    "ROUTE_B_MODEL_ID",
    "MulticlassSupervisedRunResult",
    "outer_training_prior_fallback_predict",
    "run_nested_loso_multiclass",
]
