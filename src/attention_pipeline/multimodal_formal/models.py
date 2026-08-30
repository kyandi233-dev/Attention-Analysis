"""Per-fold model fitting: regularized multinomial logistic and Random Forest.

File: models.py
Version: multimodal-fusion-v1.0.0
Purpose:
    单个 LOSO 折内的训练与预测：
    - 主模型：LogisticRegression(multi_class="multinomial", L2)，C 在训练折内
      按参与者 GroupKFold 选择（候选 C 来自科学配置）；内层 CV 失败回退
      fallback C 并写入折记录；
    - 补充模型：RandomForestClassifier 固定超参（声明于科学配置，无折外选参）；
    - 两种模型使用同一折内预处理矩阵与同一测试折。

Contract:
    - 参数选择、标准化、插补全部只在训练折内（见 preprocessing.py）；
    - 测试折预测概率按模型 classes_ 对齐为全类别（1..K）概率矩阵，
      缺失类别概率置 NaN 并在折记录中标记，指标层按有定义分量处理。

Usage:
    >>> from attention_pipeline.multimodal_formal.models import fit_predict_fold
    >>> record = fit_predict_fold(train, test, model_cfg, state, ...)

Dependencies:
    numpy, pandas, scikit-learn
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss

# ---- 冻结参数（默认值；科学配置可覆盖） ----
DEFAULT_LR_C_CANDIDATES = (0.01, 0.1, 1.0, 10.0)
DEFAULT_LR_FALLBACK_C = 1.0
DEFAULT_LR_MAX_ITER = 2000
DEFAULT_RF_N_ESTIMATORS = 300
DEFAULT_RF_MIN_SAMPLES_LEAF = 5
DEFAULT_INNER_CV_SPLITS = 5


def _select_c_lr(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    candidates: tuple[float, ...],
    n_splits: int,
    fallback_c: float,
    seed: int,
) -> tuple[float, dict[str, Any]]:
    """在训练折内用 GroupKFold 选择 LogisticRegression 的 C。

    Parameters
    ----------
    x_train / y_train / groups_train : 训练折特征、标签、参与者分组码。
    candidates : C 候选值。
    n_splits : 内层 GroupKFold 折数。
    fallback_c : 内层选择失败时的回退 C。
    seed : 内层折划分随机种子。

    Returns
    -------
    (best_c, detail)：detail 含各候选的内层平均负对数损失与失败记录。
    """
    detail: dict[str, Any] = {"candidate_scores": {}, "fallback_used": False, "reason": ""}
    scores: dict[float, float] = {}
    try:
        splitter = GroupKFold(n_splits=n_splits)
        for c in candidates:
            fold_losses: list[float] = []
            for train_idx, valid_idx in splitter.split(x_train, y_train, groups_train):
                model = LogisticRegression(
                    C=c, solver="lbfgs",
                    max_iter=DEFAULT_LR_MAX_ITER, random_state=seed,
                )
                model.fit(x_train.iloc[train_idx], y_train[train_idx])
                proba = model.predict_proba(x_train.iloc[valid_idx])
                fold_losses.append(log_loss(
                    y_train[valid_idx], proba,
                    labels=list(model.classes_),
                ))
            scores[c] = float(np.mean(fold_losses))
        detail["candidate_scores"] = {str(k): v for k, v in scores.items()}
        best_c = min(scores, key=scores.get)
    except Exception as exc:
        # 内层 CV 失败（如某内层折缺类别导致不收敛）：回退并记录，不中断全量
        best_c = fallback_c
        detail["fallback_used"] = True
        detail["reason"] = f"{type(exc).__name__}: {exc}"
    return best_c, detail


def _align_proba(proba: np.ndarray, classes: np.ndarray, n_outcomes: int) -> np.ndarray:
    """把模型预测概率对齐为全类别（1..n_outcomes）概率矩阵。

    Parameters
    ----------
    proba : 模型 predict_proba 输出。
    classes : 模型 classes_。
    n_outcomes : 结局类别总数（Q1/Q2 均为 4）。

    Returns
    -------
    n 行 x n_outcomes 列矩阵；模型未见的类别列全为 NaN。
    """
    aligned = np.full((proba.shape[0], n_outcomes), np.nan, dtype=float)
    for idx, label in enumerate(classes):
        position = int(label) - 1
        if 0 <= position < n_outcomes:
            aligned[:, position] = proba[:, idx]
    return aligned


def fit_predict_fold(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    groups_train: np.ndarray,
    *,
    model_kind: str,
    model_cfg: dict[str, Any],
    n_outcomes: int,
    seed: int,
) -> dict[str, Any]:
    """单个折内训练一个模型并预测测试折，返回预测记录。

    Parameters
    ----------
    x_train/y_train/groups_train : 训练折特征、标签、参与者分组码。
    x_test/y_test : 测试折特征与标签。
    model_kind : logistic 或 random_forest。
    model_cfg : 科学配置 models.<kind> 段。
    n_outcomes : 结局类别数（4）。
    seed : 随机种子（折序数派生，保证可复现）。

    Returns
    -------
    记录 dict：预测概率（n_test x n_outcomes）、真实标签、训练行数、
    测试行数、模型细节（C 值/内层分数/失败原因）。
    """
    if model_kind == "logistic":
        candidates = tuple(float(v) for v in model_cfg.get("C_candidates", DEFAULT_LR_C_CANDIDATES))
        fallback = float(model_cfg.get("inner_cv", {}).get("fallback_C", DEFAULT_LR_FALLBACK_C))
        n_splits = int(model_cfg.get("inner_cv", {}).get("n_splits", DEFAULT_INNER_CV_SPLITS))
        best_c, detail = _select_c_lr(x_train, y_train, groups_train, candidates, n_splits, fallback, seed)
        model = LogisticRegression(
            C=best_c, solver="lbfgs",
            max_iter=int(model_cfg.get("max_iter", DEFAULT_LR_MAX_ITER)),
            random_state=seed,
        )
        model.fit(x_train, y_train)
        proba = _align_proba(model.predict_proba(x_test), np.asarray(model.classes_), n_outcomes)
        return {
            "model_kind": model_kind,
            "selected_C": float(best_c),
            "c_selection_detail": detail,
            "n_train_rows": int(len(x_train)),
            "n_test_rows": int(len(x_test)),
            "y_true": np.asarray(y_test, dtype=int),
            "y_proba": proba,
            "n_classes_seen": int(len(model.classes_)),
            "failed": False,
            "reason": "",
        }

    # Random Forest：固定超参，无折外参数选择
    model = RandomForestClassifier(
        n_estimators=int(model_cfg.get("n_estimators", DEFAULT_RF_N_ESTIMATORS)),
        max_features=model_cfg.get("max_features", "sqrt"),
        min_samples_leaf=int(model_cfg.get("min_samples_leaf", DEFAULT_RF_MIN_SAMPLES_LEAF)),
        random_state=seed,
        n_jobs=1,
    )
    try:
        model.fit(x_train, y_train)
        proba = _align_proba(model.predict_proba(x_test), np.asarray(model.classes_), n_outcomes)
        return {
            "model_kind": model_kind,
            "selected_C": None,
            "c_selection_detail": {},
            "n_train_rows": int(len(x_train)),
            "n_test_rows": int(len(x_test)),
            "y_true": np.asarray(y_test, dtype=int),
            "y_proba": proba,
            "n_classes_seen": int(len(model.classes_)),
            "failed": False,
            "reason": "",
        }
    except Exception as exc:
        # 训练失败（如训练折缺类别导致 RF 无法拟合）：记录失败，不中断全量
        return {
            "model_kind": model_kind,
            "selected_C": None,
            "c_selection_detail": {},
            "n_train_rows": int(len(x_train)),
            "n_test_rows": int(len(x_test)),
            "y_true": np.asarray(y_test, dtype=int),
            "y_proba": np.full((len(x_test), n_outcomes), np.nan, dtype=float),
            "n_classes_seen": 0,
            "failed": True,
            "reason": f"{type(exc).__name__}: {exc}",
        }
