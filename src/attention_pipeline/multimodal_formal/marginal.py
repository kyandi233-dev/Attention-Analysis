"""Shapley-style average marginal contribution of each sensing modality.

File: marginal.py
Version: multimodal-fusion-v1.0.0
Purpose:
    基于八组合 LOSO 性能，计算 NIR/毫米波/RGB 三模态在所有 6 种加入顺序
    上的平均边际贡献（Shapley 式平均）。仅描述，不作因果解释；
    解释产物与预测产物分轨输出（见 runner 输出目录结构）。

Definition:
    对每种排列 pi（模态加入顺序）与模态 m：
        marginal(m; pi) = v(S_pi(m) ∪ {m}) - v(S_pi(m))
    其中 v(组合) 为该组合的逐折等权聚合指标，基值 v(M0) 为行为基准。
    Shapley 式平均 = 6 种排列的均值。
    log_loss 方向取负（值越小越好，边际贡献 = 基值 - 新值）。

Usage:
    >>> from attention_pipeline.multimodal_formal.marginal import marginal_contributions
    >>> table = marginal_contributions(performance_summary, modalities=("nir","mmwave","rgb"))

Dependencies:
    numpy, pandas, itertools
"""
from __future__ import annotations

from itertools import permutations
from typing import Any

import numpy as np
import pandas as pd

# ---- 冻结参数 ----
# 组合名 -> 模态集合的映射（与 1.10 规格第 4 节八组合一致）
COMBINATION_BLOCKS = {
    "M0": ("behavior",),
    "M1": ("behavior", "nir"),
    "M2": ("behavior", "mmwave"),
    "M3": ("behavior", "rgb"),
    "M4": ("behavior", "nir", "mmwave"),
    "M5": ("behavior", "nir", "rgb"),
    "M6": ("behavior", "mmwave", "rgb"),
    "M7": ("behavior", "nir", "mmwave", "rgb"),
}
# 指标方向：log_loss 越小越好（贡献取负）
_SMALLER_IS_BETTER = {"log_loss"}


def _combination_lookup(blocks: tuple[str, ...]) -> str:
    """由模态集合反查组合名；未知集合报错（fail closed）。"""
    for name, value in COMBINATION_BLOCKS.items():
        if set(value) == set(blocks):
            return name
    raise ValueError(f"unknown combination blocks: {blocks}")


def marginal_contributions(
    performance: pd.DataFrame,
    *,
    modalities: tuple[str, ...] = ("nir", "mmwave", "rgb"),
    metric: str = "roc_auc_ovr_macro",
) -> pd.DataFrame:
    """计算三模态 Shapley 式平均边际贡献（全 6 种加入顺序平均）。

    Parameters
    ----------
    performance : performance_summary 表（列含 combination、outcome、model、metric_mean）。
    modalities : 传感模态顺序无关集合。
    metric : 性能指标名（列名 metric_mean 对应值）。

    Returns
    -------
    一行 = (outcome, model, modality)：mean_marginal 与 6 种排列的逐排列明细
    （permutation_1..6 与 base_value、full_value）。
    """
    rows: list[dict[str, Any]] = []
    for (outcome, model), group in performance.groupby(["outcome", "model"], sort=True):
        value = dict(zip(group["combination"], group["metric_mean"]))
        base_value = value.get("M0", np.nan)
        full_value = value.get("M7", np.nan)
        for modality in modalities:
            perms = list(permutations(modalities))
            marginals: list[float] = []
            perm_detail: dict[str, float] = {}
            for idx, perm in enumerate(perms, start=1):
                # S_pi(m) = perm 中 m 之前的模态集合
                before = perm[:perm.index(modality)]
                v_without = value.get(_combination_lookup(("behavior",) + tuple(before)), np.nan)
                v_with = value.get(_combination_lookup(("behavior",) + tuple(before) + (modality,)), np.nan)
                if metric in _SMALLER_IS_BETTER:
                    diff = (v_without - v_with) if np.isfinite(v_without) and np.isfinite(v_with) else np.nan
                else:
                    diff = (v_with - v_without) if np.isfinite(v_without) and np.isfinite(v_with) else np.nan
                marginals.append(diff)
                perm_detail[f"permutation_{idx}"] = float(diff) if np.isfinite(diff) else np.nan
            finite = [m for m in marginals if np.isfinite(m)]
            rows.append({
                "outcome": outcome,
                "model": model,
                "modality": modality,
                "metric": metric,
                "mean_marginal": float(np.mean(finite)) if finite else np.nan,
                "base_value_M0": float(base_value) if np.isfinite(base_value) else np.nan,
                "full_value_M7": float(full_value) if np.isfinite(full_value) else np.nan,
                **perm_detail,
                "n_permutations_defined": int(len(finite)),
            })
    return pd.DataFrame(rows)
