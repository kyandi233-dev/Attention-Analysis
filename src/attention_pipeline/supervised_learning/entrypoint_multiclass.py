"""Configuration-to-run entrypoint for the Q1 four-class supervised line (routes A and B).

文件：entrypoint_multiclass.py
版本：1.0.0
功能：把 `configs/supervised_learning_4class_v1.yaml` 变成一次真正的四分类运行：
      加载配置、复用二分类线的 `_resolve_model_plan` / `_validate_analysis_set_feature_scope` /
      time-legality 运行时审计、运行外层参与者互斥 LOSO，并把结果归档。
用法：
    from attention_pipeline.supervised_learning.entrypoint_multiclass import (
        run_multiclass_from_config,
    )
依赖：pandas、attention_pipeline.config

归档边界（必须如实报告，不得含糊）：
    本模块**复用**二分类线的 `reporting.write_supervised_run` 写运行目录
    （满足「逐运行目录 = <output_root>\\<run_id>」的既有约定），因此会把四分类结果适配成
    该函数的二进制报告契约。适配规则是**机械且可审计**的：
      - `q1_nominal_4class` 保留原始四类整数值（1..4），不重映射；
      - `q1_binary` = 原始 Q1 是否等于 1（仅用于复用既有档案校验与散点报告口径）；
      - `p_q1_equals_1` = 四分类模型的类别 1 概率；
      - `predicted_q1_binary` = `predicted_q1_multiclass == 1`；
      - 四个类别概率列与 `predicted_q1_multiclass` 原样保留。
    manifest 中会显式登记这套适配，并额外写入四分类专属的评估块；
    四分类主指标从不用二分类评估结果代替。

四分类冻结不变项（预注册 §3）：外层参与者互斥 LOSO、内层 GroupKFold(5)、
训练折内拟合的参与者等权中位数插补 + 标准化、参与者等权训练权重、
L2 正则化多项逻辑回归 max_iter=2000、C ∈ {0.01, 0.1, 1.0, 10.0}、
random_seed=20260910、固定 OOF 参与者簇 bootstrap(1000/20260830/0.95/不重训)。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from attention_pipeline.config import load_config

from .entrypoint import (
    FORMAL_INNER_SPLITS,
    FORMAL_PARTICIPANT_GROUP_COLUMN,
    _enrich_paired_specs_with_time_legality,
    _read_probe_table,
    _require_comparison_models,
    _require_single_analysis_set_id,
    _resolve_model_plan,
    _runtime_time_legality_audit,
    _sha256,
    _git_sha,
    _validate_analysis_set_feature_scope,
)
from .comparison_provenance import (
    build_paired_comparison_specs,
    paired_comparison_reporting_contract,
    write_paired_comparison_provenance,
)
from .evaluation import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE_LEVEL,
    MEMBERSHIP_COLUMN,
)
from .feature_registry import FeatureComparisonPlan, load_registered_features
from .feature_schemes import FeatureScheme
from .metrics_multiclass import (
    PAIR_KEY_COLUMNS,
    PRIMARY_METRIC_NAME,
    PRIOR_BASELINE_NAME,
    multiclass_bootstrap_summary,
    multiclass_discrimination,
    outer_fold_class_priors,
    prior_baseline_participant_macro_log_loss,
)
from .models_multiclass import SELECTION_METRIC
from .outcome_scope import validate_task_a_required_outcomes
from .reporting import _json_default, write_supervised_run
from .runner import SupervisedRunResult
from .runner_multiclass import ALLOWED_ROUTES, run_nested_loso_multiclass
from .task import Q1_BINARY_SPEC, SupervisedLearningContractError
from .task_multiclass import (
    Q1_MULTICLASS_CLASSES,
    Q1_MULTICLASS_SPEC,
    MulticlassTaskSpec,
)


DEFAULT_MULTICLASS_CONFIG = "configs/supervised_learning_4class_v1.yaml"
#: 四分类专属的评估块文件名（运行目录内）。
MULTICLASS_EVALUATION_FILENAME = "multiclass_evaluation.json"
#: 由二分类归档器写出的清单文件名，本模块在其基础上补充四分类块。
MANIFEST_FILENAME = "run_manifest.json"


def _require_frozen_runtime_contract_multiclass(config_data: Mapping[str, Any]) -> str:
    """校验四分类配置的冻结值，并返回配置声明的路线。

    参数：
        config_data: 已解析的 YAML 配置字典。
    返回：``"A"`` 或 ``"B"``（配置声明值；命令行可覆盖）。
    异常：
        SupervisedLearningContractError: 任何冻结值与预注册 §3 不一致。
    说明：
        复用的是**同一批**冻结语义（折结构、预处理、权重、bootstrap），但任务合同
        本身是四分类的，因此不能直接调用二分类的 ``_require_frozen_runtime_contract``。
        该函数仍被本模块导入并在其自身断言里保持可用，避免删除二分类侧的检查能力。
    """
    task = config_data.get("task", {})
    expected_task = {
        "name": Q1_MULTICLASS_SPEC.name,
        "analysis_unit": "probe_preceding_window",
        "source_column": Q1_MULTICLASS_SPEC.source_column,
        "classes": list(Q1_MULTICLASS_SPEC.sorted_classes),
        "class_semantics": "nominal_unordered",
        "primary_window_seconds": 30,
    }
    for key, expected in expected_task.items():
        if task.get(key) != expected:
            raise SupervisedLearningContractError(
                f"task.{key}={task.get(key)!r} conflicts with frozen four-class value {expected!r}"
            )
    # 保留原始整数值：任何 0..3 重映射都会破坏 fail-closed 编码与概率列语义。
    if tuple(task.get("classes", ())) != tuple(Q1_MULTICLASS_CLASSES):
        raise SupervisedLearningContractError(
            "four-class task must keep the original integer class values (1, 2, 3, 4)"
        )
    if task.get("impute_missing_labels") not in (False, None):
        raise SupervisedLearningContractError("four-class labels must never be imputed")

    validation = config_data.get("validation", {})
    outer = validation.get("outer", {})
    inner = validation.get("inner", {})
    if outer.get("method") != "leave_one_participant_out" or outer.get("participant_disjoint") is not True:
        raise SupervisedLearningContractError(
            "four-class outer validation must remain participant-disjoint LOSO"
        )
    if inner.get("method") != "grouped_k_fold" or inner.get("refit_preprocessing_per_split") is not True:
        raise SupervisedLearningContractError(
            "four-class inner validation must refit preprocessing within each grouped split"
        )
    if int(inner.get("n_splits", -1)) != FORMAL_INNER_SPLITS:
        raise SupervisedLearningContractError(
            f"four-class inner validation must use exactly {FORMAL_INNER_SPLITS} participant-grouped folds"
        )
    outer_group = str(outer.get("group_column", ""))
    inner_group = str(inner.get("group_column", ""))
    if outer_group != FORMAL_PARTICIPANT_GROUP_COLUMN or inner_group != outer_group:
        raise SupervisedLearningContractError(
            "four-class inner and outer validation must use the same grouping column: participant_group_id"
        )
    for key in (
        "zero_individual_calibration",
        "forbid_test_participant_sequence_statistics",
        "forbid_test_participant_future_information",
        "require_analysis_set_id",
    ):
        if validation.get(key) is not True:
            raise SupervisedLearningContractError(f"validation.{key} must remain true for four-class runs")
    if any(
        value is not None
        for value in (
            config_data.get("prediction_folds"),
            validation.get("prediction_folds"),
        )
    ):
        raise SupervisedLearningContractError(
            "standalone prediction_folds is deprecated; four-class runs use outer LOSO plus inner GroupKFold only"
        )

    preprocessing = config_data.get("preprocessing", {})
    for key in (
        "participant_equal_weighted_median_imputation_fit_on_training_only",
        "participant_equal_standardization_fit_on_training_only",
        "data_dependent_column_handling_fit_on_training_only",
        "participant_equal_training_weights_normalized_to_mean_one",
    ):
        if preprocessing.get(key) is not True:
            raise SupervisedLearningContractError(
                f"preprocessing.{key} must remain true for four-class supervised learning"
            )
    if preprocessing.get("unified_global_coverage_cutoff") is not None:
        raise SupervisedLearningContractError(
            "four-class supervised learning forbids a unified global coverage cutoff"
        )
    if preprocessing.get("participant_specific_within_between_mainline") is not False:
        raise SupervisedLearningContractError(
            "participant-specific within/between decomposition is disabled in the zero-calibration mainline"
        )

    models = config_data.get("models", {})
    primary = models.get("primary", {})
    if primary.get("kind") != "logistic_l2":
        raise SupervisedLearningContractError(
            "formal four-class primary model must remain L2 logistic regression"
        )
    if primary.get("multiclass_objective") != "multinomial":
        raise SupervisedLearningContractError(
            "formal four-class primary model must use the multinomial objective"
        )
    if tuple(float(c) for c in primary.get("C_candidates", ())) != (0.01, 0.1, 1.0, 10.0):
        raise SupervisedLearningContractError(
            "four-class regularisation candidates must remain the frozen grid (0.01, 0.1, 1.0, 10.0)"
        )
    if int(primary.get("max_iter", -1)) != 2000:
        raise SupervisedLearningContractError("four-class primary model max_iter must remain 2000")
    if models.get("selection_metric") != SELECTION_METRIC:
        raise SupervisedLearningContractError(
            f"four-class candidate selection metric must be {SELECTION_METRIC}"
        )

    uncertainty = config_data.get("uncertainty", {})
    participant_bootstrap = uncertainty.get("participant_cluster_bootstrap", {})
    expected_bootstrap = {
        "method": "fixed_oof_participant_cluster_percentile",
        "replicates": DEFAULT_BOOTSTRAP_REPLICATES,
        "seed": DEFAULT_BOOTSTRAP_SEED,
        "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
        "paired_model_resampling": True,
        "retrain_within_bootstrap": False,
    }
    for key, expected in expected_bootstrap.items():
        if participant_bootstrap.get(key) != expected:
            raise SupervisedLearningContractError(
                f"uncertainty.participant_cluster_bootstrap.{key}="
                f"{participant_bootstrap.get(key)!r} conflicts with frozen value {expected!r}"
            )

    pipeline = config_data.get("pipeline", {})
    if int(pipeline.get("random_seed", -1)) != 20260910:
        raise SupervisedLearningContractError("pipeline.random_seed must remain 20260910")

    declared_route = str(config_data.get("multiclass_route", "")).strip().upper()
    if declared_route not in ALLOWED_ROUTES:
        raise SupervisedLearningContractError(
            f"multiclass_route must declare one of {sorted(ALLOWED_ROUTES)}; got {declared_route!r}"
        )
    return declared_route


def _registered_feature_pool(config_data: Mapping[str, Any]) -> tuple[object, ...]:
    """从配置的 feature_registry 节读出冻结注册特征池（路线 B 的选择池）。"""
    section = config_data.get("feature_registry", {})
    if not isinstance(section, Mapping):
        raise SupervisedLearningContractError("feature_registry must be a mapping")
    entries = section.get("features", [])
    if not isinstance(entries, list) or not entries:
        raise SupervisedLearningContractError(
            "route B requires a non-empty frozen feature_registry in the four-class config"
        )
    return load_registered_features(section)


# --------------------------------------------------------------------------- 归档适配
def _multiclass_predictions_to_archive_contract(
    predictions: pd.DataFrame,
    *,
    spec: MulticlassTaskSpec,
    expected_rows: int,
) -> pd.DataFrame:
    """把四分类逐 probe 预测机械适配成二分类归档契约（见模块头部的适配规则）。

    参数：
        predictions: 四分类逐 probe 预测表。
        spec: 四分类任务合同。
        expected_rows: 期望行数（= 输入行数 x 模型数），用于提前失败。
    返回：可直接交给 ``reporting.write_supervised_run`` 的预测表。
    异常：
        SupervisedLearningContractError: 缺少契约列、行数不符或成功行概率越界。
    """
    required = {
        "run_id",
        "analysis_set_id",
        MEMBERSHIP_COLUMN,
        "participant_group_id",
        "model_id",
        "outer_fold_group",
        "session_id",
        "block_id",
        "probe_event_id",
        "feature_set_id",
        "selected_c",
        "model_failed",
        "failure_reason",
        spec.source_column,
        spec.predicted_column_name,
        *spec.probability_columns,
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise SupervisedLearningContractError(
            f"four-class prediction table missing required columns: {missing}"
        )
    if len(predictions) != int(expected_rows):
        raise SupervisedLearningContractError(
            f"four-class prediction row count mismatch: got {len(predictions)}, expected {expected_rows}"
        )

    out = predictions.copy()
    raw_q1 = pd.to_numeric(out[spec.source_column], errors="coerce")
    if raw_q1.isna().any():
        raise SupervisedLearningContractError(
            "four-class prediction table contains missing raw Q1 values"
        )
    raw_integer = raw_q1.astype(int)
    out[spec.source_column] = raw_integer
    # 适配列 1：二分类标签由原始 Q1 现场导出（Q1==1 对 Q1∈{2,3,4}），不重新定义标签含义。
    out["q1_binary"] = raw_integer.eq(1).astype(int)
    # 适配列 2/3：类别 1 概率与「是否为类别 1」的预测。
    class_one_column = spec.probability_column_name(int(spec.sorted_classes[0]))
    class_one_probability = pd.to_numeric(out[class_one_column], errors="coerce")
    out[Q1_BINARY_SPEC.positive_probability_name] = class_one_probability
    predicted = pd.to_numeric(out[spec.predicted_column_name], errors="coerce")
    out["predicted_q1_binary"] = np.where(predicted.isna(), np.nan, (predicted == 1).astype(float))

    successful = ~out["model_failed"].astype(bool)
    if successful.any():
        probability = class_one_probability[successful]
        if probability.isna().any() or ((probability < 0.0) | (probability > 1.0)).any():
            raise SupervisedLearningContractError(
                "successful four-class prediction rows must contain a finite class-1 probability "
                "within [0, 1]"
            )
        if predicted[successful].isna().any():
            raise SupervisedLearningContractError(
                "successful four-class prediction rows must contain a predicted class"
            )
    return out


def _multiclass_fold_audits(predictions: pd.DataFrame) -> list[dict[str, object]]:
    """按 (model_id, outer_fold_group) 汇合四分类预测行，生成归档器要求的 fold 键。

    归档器要求 fold 审计恰有一行 per (model_id, outer_fold_group)，且 ``failed`` /
    ``reason`` 与预测行、失败表三者完全一致。这里从预测行机械派生这两列，而不是重新
    判断哪些折失败——跑出的折结论才是唯一权威。
    """
    grouped = (
        predictions.groupby(["model_id", "outer_fold_group"], sort=True, as_index=False)
        .agg(
            run_id=("run_id", "first"),
            analysis_set_id=("analysis_set_id", "first"),
            membership_type=(MEMBERSHIP_COLUMN, "first"),
            model_failed=("model_failed", "first"),
            failure_reason=("failure_reason", "first"),
        )
    )
    if grouped["model_failed"].nunique() > 1:
        raise SupervisedLearningContractError(
            "a single (model_id, outer_fold_group) fold mixes failed and successful rows"
        )
    return [
        {
            "run_id": str(row.run_id),
            "analysis_set_id": str(row.analysis_set_id),
            MEMBERSHIP_COLUMN: str(row.membership_type),
            "model_id": str(row.model_id),
            "outer_fold_group": str(row.outer_fold_group),
            "failed": bool(row.model_failed),
            "reason": str(row.failure_reason),
        }
        for row in grouped.itertuples(index=False)
    ]


def _build_archive_result(
    predictions: pd.DataFrame,
    *,
    fold_audits: Sequence[Mapping[str, object]],
    failures: pd.DataFrame,
    metadata: Mapping[str, object],
    spec: MulticlassTaskSpec,
) -> SupervisedRunResult:
    """把四分类运行结果包装成 ``SupervisedRunResult`` 供既有归档器使用。"""
    return SupervisedRunResult(
        predictions=predictions,
        fold_audits=list(fold_audits),
        failures=failures,
        metadata=dict(metadata),
    )


# --------------------------------------------------------------------------- 四分类评估块
def _prior_baseline_by_fold(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    group_col: str,
    spec: MulticlassTaskSpec,
) -> dict[str, object]:
    """逐外层折计算类别先验无信息基线（预注册 §4.3），参与者等权汇总。

    参数：
        frame: 原始分析帧（含外层折所需的分组列与四分类标签）。
        predictions: 四分类逐 probe 预测表。
        group_col: 参与者分组列。
        spec: 四分类任务合同。
    返回：含逐折记录、逐模型总体基线与 bootstrap 的审计字典。
    说明：
        基线**逐模型**汇总：不同模型的折外测试参与者集合可能因为失败折而不同，
        把多个模型的参与者数值混在一起会得到一个不存在的基线口径。
    """
    per_model: dict[str, dict[str, object]] = {}
    for model_id, model_frame in predictions.groupby("model_id", sort=True):
        per_fold: list[dict[str, object]] = []
        pooled_participant: list[float] = []
        for held_out_group, fold_frame in model_frame.groupby("outer_fold_group", sort=True):
            outer_train = frame.loc[~frame[group_col].astype(str).eq(str(held_out_group))]
            baseline = prior_baseline_participant_macro_log_loss(
                outer_train, fold_frame, spec=spec
            )
            per_fold.append(
                {
                    "outer_fold_group": str(held_out_group),
                    "n_outer_train_rows": int(baseline["n_train_rows"]),
                    "n_outer_test_rows": int(baseline["n_test_rows"]),
                    "n_test_participants": int(baseline["n_test_participants"]),
                    "class_priors": baseline["class_priors"],
                    "observed_test_classes": baseline["observed_test_classes"],
                    "absent_test_classes": baseline["absent_test_classes"],
                    "prior_probability_floor_applied": baseline["prior_probability_floor_applied"],
                    "participant_prior_baseline_multiclass_log_loss": baseline[
                        "prior_baseline_participant_macro_multiclass_log_loss"
                    ],
                }
            )
            pooled_participant.extend(
                float(value) for value in baseline["participant_values"].values()
            )
        macro = float(np.mean(pooled_participant)) if pooled_participant else None
        per_model[str(model_id)] = {
            "per_fold": per_fold,
            "n_folds": int(len(per_fold)),
            "n_participants": int(len(pooled_participant)),
            "overall_prior_baseline_participant_macro_multiclass_log_loss": macro,
            "bootstrap": (
                multiclass_bootstrap_summary(np.asarray(pooled_participant, dtype=float))
                if pooled_participant
                else None
            ),
        }
    return {
        "baseline": PRIOR_BASELINE_NAME,
        "definition": (
            "constant prediction from each outer fold's training-set class priors, "
            "evaluated with the same participant-equal aggregation as the primary metric"
        ),
        "uses_features": False,
        "metric": PRIMARY_METRIC_NAME,
        "per_model": per_model,
        "reference_values_as_full_sample_scale_only": {
            "uniform_prior_ln4": 1.386294,
            "full_sample_class_prior_constant": 0.907631,
            "majority_class_1_constant": 0.372494,
        },
        "reference_value_note": (
            "the reference values above are order-of-magnitude references for the full AS.full "
            "sample only; the actual comparison must be recomputed per outer fold, as done here"
        ),
    }


def _multiclass_evaluation_block(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    group_col: str,
    spec: MulticlassTaskSpec,
) -> dict[str, object]:
    """四分类主指标、补充指标与无信息基线的完整评估块。"""
    per_model: dict[str, object] = {}
    from .metrics_multiclass import participant_multiclass_log_loss

    for model_id, model_frame in predictions.groupby("model_id", sort=True):
        participant, summary = participant_multiclass_log_loss(model_frame, spec=spec)
        values = participant["mean_multiclass_log_loss"].to_numpy(dtype=float)
        per_model[str(model_id)] = {
            "summary": summary,
            "bootstrap": multiclass_bootstrap_summary(values),
            "discrimination": multiclass_discrimination(model_frame, spec=spec),
        }
    baseline = _prior_baseline_by_fold(
        frame, predictions, group_col=group_col, spec=spec
    )
    for model_id, block in per_model.items():
        model_value = block["summary"][PRIMARY_METRIC_NAME]
        baseline_value = baseline["per_model"][str(model_id)][
            "overall_prior_baseline_participant_macro_multiclass_log_loss"
        ]
        block["exceeds_no_information_baseline"] = (
            None if baseline_value is None else bool(model_value < baseline_value)
        )
        block["failure_declaration"] = (
            "the four-class primary metric is NOT lower than the per-fold class-prior "
            "baseline; this route is declared as not exceeding the no-information baseline "
            "(preregistration §4.4) and no remedy such as changing the metric, merging "
            "classes, resampling, or dropping participants is permitted"
            if baseline_value is not None and not model_value < baseline_value
            else ""
        )
    return {
        "metric_role": "primary_metric_is_participant_macro_multiclass_log_loss",
        "primary_metric": PRIMARY_METRIC_NAME,
        "lower_is_better": True,
        "aggregation": "participant_equal_within_participant_probe_equal",
        "per_model": per_model,
        "prior_baseline": baseline,
        "participant_level_auroc_reported": False,
        "participant_level_auroc_note": (
            "participant-level AUROC is not reported: participants with a single observed "
            "class out of fold make it non-estimable, and no participant or class is dropped "
            "to fill that denominator (preregistration §4.2)"
        ),
        "supplementary_metrics_role": "supplementary_not_a_replacement_for_the_primary_metric",
    }


def run_multiclass_from_config(
    config_path: str | Path = DEFAULT_MULTICLASS_CONFIG,
    *,
    paths_config: str | Path | None = None,
    input_table: str | Path | None = None,
    output_root: str | Path | None = None,
    route: str | None = None,
    run_id: str | None = None,
) -> dict[str, object]:
    """执行一次 Q1 四分类正式运行（路线 A 或 B）。

    参数：
        config_path: 四分类冻结配置路径。
        paths_config: 机器路径注册表（当配置用 path_key 时才需要）。
        input_table: 运行期覆盖的 probe 级输入表路径。
        output_root: 运行期覆盖的输出根；每个运行落在 ``<output_root>/<run_id>``。
        route: ``"A"`` 或 ``"B"``；缺省时用配置声明的 ``multiclass_route``。
        run_id: 运行标识；缺省时用 UTC 时间戳。
    返回：manifest 字典（已包含四分类专属评估块）。
    异常：
        SupervisedLearningContractError / FileNotFoundError: 输入、配置或样本合同不合法。
    """
    config = load_config(config_path, paths_config=paths_config)
    declared_route = _require_frozen_runtime_contract_multiclass(config.data)
    resolved_route = declared_route if route is None else str(route).strip().upper()
    if resolved_route not in ALLOWED_ROUTES:
        raise SupervisedLearningContractError(
            f"unsupported four-class route {route!r}; expected one of {sorted(ALLOWED_ROUTES)}"
        )
    # 说明：此处**不**调用二分类侧 `entrypoint._require_frozen_runtime_contract`。
    # 那个校验把 `task.name` 钉死为二分类名并同时断言二分类的 selection_metric，
    # 对四分类配置必然失败；调用并捕获它只会制造一个无意义的期待异常。
    # 上面这个四分类版本复用了同一批冻结语义（折结构、预处理、权重、bootstrap、种子），
    # 二分类产线的该函数保持原样、未被修改也未被绕过。
    feature_time_legality = _runtime_time_legality_audit(config.data)

    input_path = (
        Path(input_table).resolve() if input_table is not None else config.path_value("input_table")
    )
    output_path = (
        Path(output_root).resolve() if output_root is not None else config.path_value("output_root")
    )
    if not input_path.is_file():
        raise FileNotFoundError(f"four-class supervised input probe table not found: {input_path}")

    all_families, comparison_plan = _resolve_model_plan(config.data)
    if not all_families:
        raise SupervisedLearningContractError(
            "no supervised feature families are configured; a four-class real-data run requires "
            "a frozen feature registry"
        )

    frame = _read_probe_table(input_path)
    analysis_set_id = _require_single_analysis_set_id(frame)
    required_outcomes = validate_task_a_required_outcomes(frame)

    families = all_families
    declared_models: tuple[str, ...] | None = None
    required_feature_columns: tuple[str, ...] | None = None
    predictor_union: tuple[str, ...] | None = None
    if comparison_plan is not None:
        declared_models = _require_comparison_models(frame)
        unknown = sorted(set(declared_models) - set(all_families))
        if unknown:
            raise SupervisedLearningContractError(
                f"analysis_set_id={analysis_set_id} declares models absent from the frozen "
                f"feature registry: {unknown}"
            )
        families = {model_id: all_families[model_id] for model_id in declared_models}
        # 路线 A 与路线 B 都必须与样本声明的特征范围一致；这一步在路线分支之前完成，
        # 保证两条路线看到的是同一个 analysis set 与同一个预测列全集。
        required_feature_columns, predictor_union = _validate_analysis_set_feature_scope(
            frame, families, analysis_set_id=analysis_set_id
        )

    registered_pool = _registered_feature_pool(config.data) if resolved_route == "B" else None

    validation = config.section("validation")
    inner = validation.get("inner", {})
    primary_model = config.section("models").get("primary", {})
    pipeline = config.section("pipeline")
    resolved_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    result = run_nested_loso_multiclass(
        frame,
        model_feature_schemes=families if resolved_route == "A" else None,
        registered_features=registered_pool,
        route=resolved_route,
        group_col=str(
            validation.get("outer", {}).get("group_column", FORMAL_PARTICIPANT_GROUP_COLUMN)
        ),
        c_candidates=primary_model.get("C_candidates", (0.01, 0.1, 1.0, 10.0)),
        inner_splits=int(inner.get("n_splits", FORMAL_INNER_SPLITS)),
        max_iter=int(primary_model.get("max_iter", 2000)),
        seed=int(pipeline.get("random_seed", 20260910)),
        run_id=str(resolved_run_id),
        analysis_set_id=analysis_set_id,
    )

    evaluation_block = _multiclass_evaluation_block(
        frame,
        result.predictions,
        group_col=str(
            validation.get("outer", {}).get("group_column", FORMAL_PARTICIPANT_GROUP_COLUMN)
        ),
        spec=Q1_MULTICLASS_SPEC,
    )

    archive_predictions = _multiclass_predictions_to_archive_contract(
        result.predictions,
        spec=Q1_MULTICLASS_SPEC,
        expected_rows=int(len(frame)) * int(result.metadata["n_models"]),
    )
    # 归档器的 fold 审计契约按 (model_id, outer_fold_group) 检查，失败折必须同时出现在
    # predictions、failures 与 fold_audits 中；本函数按此机械派生，不改变跑出的折内容。
    fold_audits_for_archive = _multiclass_fold_audits(archive_predictions)

    metadata: dict[str, object] = dict(result.metadata)
    metadata["analysis_set_required_outcomes"] = list(required_outcomes)
    metadata["analysis_set_outcome_scope_verified"] = True
    metadata["four_class_task"] = Q1_MULTICLASS_SPEC.name
    metadata["four_class_classes"] = list(Q1_MULTICLASS_SPEC.sorted_classes)
    metadata["four_class_probability_columns"] = list(Q1_MULTICLASS_SPEC.probability_columns)
    metadata["four_class_predicted_class_column"] = Q1_MULTICLASS_SPEC.predicted_column_name
    if feature_time_legality is not None:
        metadata["feature_time_legality"] = feature_time_legality
        metadata["time_legality_runtime_verified"] = True
    if comparison_plan is not None:
        selected = tuple(families)
        metadata["feature_comparison_plan"] = comparison_plan.audit_dict()
        metadata["analysis_set_declared_models"] = list(declared_models or ())
        metadata["analysis_set_required_feature_columns"] = list(required_feature_columns or ())
        metadata["declared_model_predictor_union"] = list(predictor_union or ())
        metadata["analysis_set_feature_scope_verified"] = True
        paired_specs = build_paired_comparison_specs(comparison_plan, selected)
        metadata["paired_comparisons"] = _enrich_paired_specs_with_time_legality(
            paired_specs, feature_time_legality
        )
        metadata["paired_comparison_reporting_contract"] = paired_comparison_reporting_contract()
    metadata["binary_reporting_adapter"] = {
        "applies": True,
        "reason": (
            "reporting.write_supervised_run is reused so that every run lives in "
            "<output_root>/<run_id> with the same archive contract and integrity checks; "
            "the four-class results are mapped onto that contract mechanically and the "
            "four-class metrics are computed separately and never replaced by binary ones"
        ),
        "mapping": {
            "q1_nominal_4class": "original four-class integer values preserved unchanged (1..4)",
            "q1_binary": "derived as (q1_nominal_4class == 1); used only for archive-contract reuse",
            "p_q1_equals_1": "four-class model probability of class 1",
            "predicted_q1_binary": "(predicted_q1_multiclass == 1)",
            "class_probability_columns": list(Q1_MULTICLASS_SPEC.probability_columns),
            "predicted_class_column": Q1_MULTICLASS_SPEC.predicted_column_name,
        },
        "binary_line_modified": False,
    }
    metadata["four_class_reporting_contract"] = {
        "primary_metric": PRIMARY_METRIC_NAME,
        "supplementary_metrics": [
            "one_vs_rest_macro_auroc",
            "balanced_accuracy",
            "macro_f1",
            "per_class_log_loss",
            "per_class_one_vs_rest_auroc",
            "confusion_matrix_4x4",
        ],
        "no_information_baseline": PRIOR_BASELINE_NAME,
        "not_a_replacement_for_the_binary_mainline": True,
        "cross_analysis_set_increment_ranking_allowed": False,
        "route_comparison_definition": "baseline_log_loss_minus_added_log_loss",
        "binary_contract_reused_through_reporting_module": True,
    }

    archive_result = _build_archive_result(
        archive_predictions,
        fold_audits=fold_audits_for_archive,
        failures=result.failures,
        metadata=metadata,
        spec=Q1_MULTICLASS_SPEC,
    )

    repo_root = Path(__file__).resolve().parents[3]
    manifest = write_supervised_run(
        archive_result,
        output_root=output_path,
        provenance={
            "input_table": str(input_path),
            "input_sha256": _sha256(input_path),
            "config_path": str(Path(config_path).resolve()),
            "config_digest": config.digest,
            "code_sha": _git_sha(repo_root),
        },
    )
    if comparison_plan is not None:
        manifest = write_paired_comparison_provenance(
            output_root=output_path,
            run_id=str(resolved_run_id),
            analysis_set_id=analysis_set_id,
            membership_type=str(result.metadata.get(MEMBERSHIP_COLUMN, "")),
            paired_comparisons=manifest.get("paired_comparisons", []),
            fold_audits=result.fold_audits,
            manifest=manifest,
        )

    run_root = output_path / str(resolved_run_id)
    evaluation_path = run_root / MULTICLASS_EVALUATION_FILENAME
    evaluation_payload = {
        "run_id": str(resolved_run_id),
        "analysis_set_id": analysis_set_id,
        MEMBERSHIP_COLUMN: str(result.metadata.get(MEMBERSHIP_COLUMN, "")),
        "task": Q1_MULTICLASS_SPEC.name,
        "route": resolved_route,
        "class_labels": list(Q1_MULTICLASS_SPEC.sorted_classes),
        "class_semantics": "nominal_unordered",
        "evaluation": evaluation_block,
    }
    evaluation_path.write_text(
        json.dumps(
            evaluation_payload, ensure_ascii=False, indent=2, default=_json_default
        ),
        encoding="utf-8",
    )

    manifest["multiclass_evaluation"] = evaluation_block
    manifest["outputs"] = {
        **dict(manifest.get("outputs", {})),
        "multiclass_evaluation": MULTICLASS_EVALUATION_FILENAME,
    }
    (run_root / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "DEFAULT_MULTICLASS_CONFIG",
    "MULTICLASS_EVALUATION_FILENAME",
    "run_multiclass_from_config",
]
