"""Classification metrics and participant-level bootstrap paired differences.

File: metrics.py
Version: multimodal-fusion-v1.0.0
Purpose:
    四个共同报告指标：ROC-AUC（one-vs-rest 宏平均，只对有定义分量取平均）、
    平衡准确率、宏 F1、对数损失；以及 M1-M7 相对 M0 的逐折配对差异与
    bootstrap CI（按参与者重抽样，与 LOSO 折结构一致，不做跨折泄漏）。

Contract:
    - Q1/Q2 类别 1..4；测试折未出现的类别其 ovr 分量无定义，跳过并在
      折记录中标记，宏平均按有定义分量计算；
    - bootstrap 重抽样单位 = 参与者（= LOSO 折），每个 bootstrap 样本内
      重新聚合逐折差异，不跨折借用统计量；
    - 聚合默认逐折等权（每个参与者贡献相等）。

Usage:
    >>> from attention_pipeline.multimodal_formal.metrics import score_fold, bootstrap_paired_difference
    >>> scores = score_fold(y_true, y_proba)
    >>> ci = bootstrap_paired_difference(diff_by_fold, n=2000, seed=42)

Dependencies:
    numpy, pandas, scikit-learn
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)

# ---- 冻结参数 ----
METRIC_NAMES = ("roc_auc_ovr_macro", "balanced_accuracy", "macro_f1", "log_loss")
# log_loss 的类别列表（全 4 类，缺失类别概率置 0 用于该指标）
_LOG_LOSS_LABELS = (1, 2, 3, 4)


def roc_auc_ovr_macro(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, dict[str, Any]]:
    """one-vs-rest 宏平均 ROC-AUC，只对有定义分量取平均。

    Parameters
    ----------
    y_true : 真实标签（1..4 整数）。
    y_proba : n x 4 概率矩阵（缺失类别列可为 NaN）。

    Returns
    -------
    (auc, detail)：detail 含各分量 AUC、跳过分量与有定义分量数。
    """
    n_outcomes = y_proba.shape[1]
    components: dict[str, float] = {}
    skipped: list[str] = []
    for label in range(1, n_outcomes + 1):
        binary_true = (y_true == label).astype(int)
        if binary_true.sum() == 0 or binary_true.sum() == len(y_true):
            # 测试折内该类无样本或全为该类：ovr 分量无定义
            skipped.append(str(label))
            continue
        column = y_proba[:, label - 1]
        if not np.isfinite(column).all():
            skipped.append(f"{label}:nonfinite_proba")
            continue
        try:
            components[str(label)] = float(roc_auc_score(binary_true, column))
        except ValueError:
            # 概率退化等极端情形：分量无定义，跳过而非整折失败
            skipped.append(f"{label}:undefinable")
            continue
    if not components:
        return np.nan, {"components": {}, "skipped": skipped, "defined": 0}
    return float(np.mean(list(components.values()))), {
        "components": components, "skipped": skipped, "defined": len(components),
    }


def score_fold(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """计算单个测试折的四个指标。

    Parameters
    ----------
    y_true : 真实标签。
    y_proba : 全类别概率矩阵（缺失类别列 NaN）。

    Returns
    -------
    {metric: value}；roc_auc 无定义分量为 NaN。
    """
    auc, _ = roc_auc_ovr_macro(y_true, y_proba)
    proba_safe = np.where(np.isfinite(y_proba), y_proba, 0.0)
    # log_loss 概率按行归一（缺失类别置 0 后重新归一）
    row_sums = proba_safe.sum(axis=1, keepdims=True)
    proba_norm = proba_safe / np.where(row_sums == 0, 1.0, row_sums)
    return {
        "roc_auc_ovr_macro": float(auc) if np.isfinite(auc) else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, proba_norm.argmax(axis=1) + 1)),
        "macro_f1": float(f1_score(y_true, proba_norm.argmax(axis=1) + 1, average="macro", zero_division=0.0)),
        "log_loss": float(log_loss(y_true, proba_norm, labels=list(_LOG_LOSS_LABELS))),
    }


def fold_metrics_frame(
    records: list[dict[str, Any]],
    *,
    combination: str,
    outcome: str,
    model_kind: str,
) -> pd.DataFrame:
    """把一组合×结局×模型的全折记录整理为逐折指标表（含 coverage）。

    Parameters
    ----------
    records : fit_predict_fold 记录列表（每折一条）。
    combination/outcome/model_kind : 标识列。

    Returns
    -------
    逐折表：fold（参与者）、各指标、coverage 列、类别覆盖、失败标记。
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        y_true = np.asarray(record["y_true"], dtype=int)
        y_proba = np.asarray(record["y_proba"], dtype=float)
        row: dict[str, Any] = {
            "combination": combination,
            "outcome": outcome,
            "model": model_kind,
            "fold_group": record["fold_group"],
            "n_train_rows": record["n_train_rows"],
            "n_test_rows": record["n_test_rows"],
            "test_categories": ",".join(str(v) for v in sorted(set(y_true.tolist()))),
            "model_failed": bool(record["failed"]),
            "model_reason": record.get("reason", ""),
        }
        if record["failed"] or len(y_true) == 0:
            for metric in METRIC_NAMES:
                row[metric] = np.nan
        else:
            scores = score_fold(y_true, y_proba)
            row.update(scores)
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_paired_difference(
    diff_by_fold: np.ndarray,
    *,
    n: int = 2000,
    seed: int = 42,
    ci_percentiles: tuple[float, float] = (2.5, 97.5),
) -> dict[str, float]:
    """按参与者（= LOSO 折）重抽样的配对差异 bootstrap CI。

    Parameters
    ----------
    diff_by_fold : 逐折配对差异向量（每参与者一值，可含 NaN，按有定义折聚合）。
    n : bootstrap 次数。
    seed : 随机种子。
    ci_percentiles : CI 百分位。

    Returns
    -------
    {mean_diff, ci_low, ci_high, n_folds_defined, n_folds_total}。
    """
    defined = diff_by_fold[np.isfinite(diff_by_fold)]
    total = len(diff_by_fold)
    if len(defined) == 0:
        return {
            "mean_diff": np.nan, "ci_low": np.nan, "ci_high": np.nan,
            "n_folds_defined": 0, "n_folds_total": total,
        }
    rng = np.random.default_rng(seed)
    # 重抽样单位 = 折索引（参与者），每个 bootstrap 样本内重新聚合均值
    bootstrap_means = np.empty(n, dtype=float)
    for draw in range(n):
        indices = rng.integers(0, len(defined), size=len(defined))
        bootstrap_means[draw] = float(np.mean(defined[indices]))
    low, high = np.percentile(bootstrap_means, ci_percentiles)
    return {
        "mean_diff": float(np.mean(defined)),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_folds_defined": int(len(defined)),
        "n_folds_total": total,
    }


def summarize_fold_metric(by_fold: pd.DataFrame, metric: str) -> dict[str, float]:
    """逐折等权汇总一个指标（每个参与者贡献相等）。

    Parameters
    ----------
    by_fold : 逐折指标表（含 metric 列）。
    metric : 指标名。

    Returns
    -------
    {mean, sd, n_folds_defined, n_folds_total}。
    """
    values = pd.to_numeric(by_fold[metric], errors="coerce")
    defined = values.dropna()
    return {
        "mean": float(defined.mean()) if len(defined) else np.nan,
        "sd": float(defined.std(ddof=1)) if len(defined) > 1 else np.nan,
        "n_folds_defined": int(len(defined)),
        "n_folds_total": int(len(values)),
    }
