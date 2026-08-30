"""Training-fold-only preprocessing: imputation, standardization, NIR within/between.

File: preprocessing.py
Version: multimodal-fusion-v1.0.0
Purpose:
    所有预处理统计量（组均值、中位数、均值、标准差、待删列）只在训练折内
    拟合，测试折只应用。禁止跨折信息泄漏与身份泄漏：
    - 中位数插补值只由训练折决定；
    - 标准化 mean/std 只由训练折决定；
    - NIR 主指标 within/between 分解的组均值只由训练折拟合；测试参与者
      的 between 分量用训练折组均值的均值（先验），within = x - 先验，
      不使用测试折自身统计量；
    - 特征筛选（全缺失列、零方差列）只由训练折决定。

Usage:
    >>> state = fit_preprocessing(train_df, columns, nir_metrics=("pupil_geom_mean_diameter", "hard_pupil_fraction"))
    >>> X_train = apply_preprocessing(train_df, state, is_train=True)
    >>> X_test = apply_preprocessing(test_df, state, is_train=False)

Dependencies:
    numpy, pandas
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# ---- 冻结参数 ----
# 组均值分解时，组内有效样本数低于该阈值仍允许分解（组均值即观测值本身）；
# 仅当组内无任何非缺失值时，组均值才为 NaN 并交插补处理。
_NIR_WITHIN_SUFFIX = "_within"
_NIR_BETWEEN_SUFFIX = "_between"


@dataclass
class PreprocessingState:
    """训练折内拟合的预处理状态（可序列化，测试折只应用）。"""

    # 特征列 -> 训练折中位数（插补用）
    imputation_medians: dict[str, float] = field(default_factory=dict)
    # 特征列 -> 训练折均值/标准差（标准化用）
    standardization_mean: dict[str, float] = field(default_factory=dict)
    standardization_std: dict[str, float] = field(default_factory=dict)
    # NIR 组均值（within/between 分解用）：participant_group_id -> metric -> mean
    nir_group_means: dict[str, dict[str, float]] = field(default_factory=dict)
    # 训练折组均值的均值（测试参与者 between 先验）
    nir_group_mean_priors: dict[str, float] = field(default_factory=dict)
    # 训练折内决定的丢弃列（全 NaN 或零方差）
    dropped_columns: list[str] = field(default_factory=list)
    # 输入特征列（含 NIR 分解前原始指标名）
    input_columns: list[str] = field(default_factory=list)
    # 分解后特征列顺序（模型输入顺序）
    output_columns: list[str] = field(default_factory=list)


def _group_means(frame: pd.DataFrame, group_col: str, metrics: tuple[str, ...]) -> dict[str, dict[str, float]]:
    """计算每组的 metric 均值（跳过缺失值）；组内全缺失时为 NaN。

    Parameters
    ----------
    frame : 训练折数据。
    group_col : 分组列（participant_group_id）。
    metrics : 需组内中心化的指标列。

    Returns
    -------
    {group: {metric: mean}}。
    """
    result: dict[str, dict[str, float]] = {}
    for group, sub in frame.groupby(group_col, sort=True):
        result[str(group)] = {
            metric: float(pd.to_numeric(sub[metric], errors="coerce").mean())
            for metric in metrics
        }
    return result


def decompose_within_between(
    frame: pd.DataFrame,
    group_col: str,
    metrics: tuple[str, ...],
    group_means: dict[str, dict[str, float]] | None = None,
    *,
    is_train: bool = True,
    priors: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict[str, float]]:
    """把 NIR 主指标分解为 within/between 两列。

    Parameters
    ----------
    frame : 含原始指标列的数据。
    group_col : 分组列。
    metrics : 指标列。
    group_means : 训练折组均值（is_train=False 时必须传入）。
    is_train : True 时由本折拟合组均值；False 时只用传入的 group_means/priors。
    priors : 训练折组均值的均值（测试折 between 先验）。

    Returns
    -------
    (out, fitted_group_means, fitted_priors)：out 为分解后表；训练时返回拟合值。
    """
    if is_train:
        group_means = _group_means(frame, group_col, metrics)
        priors = {
            metric: float(np.nanmean([values[metric] for values in group_means.values()]))
            for metric in metrics
        }
    else:
        assert group_means is not None and priors is not None, "test fold requires fitted group means"
    out = frame.copy()
    groups = out[group_col].astype(str)
    for metric in metrics:
        values = pd.to_numeric(out[metric], errors="coerce")
        if is_train:
            between = groups.map(lambda g: group_means.get(g, {}).get(metric, np.nan))
        else:
            # 测试参与者无训练组均值：between 取先验，within = x - 先验
            between = pd.Series(priors[metric], index=out.index, dtype=float)
        out[f"{metric}{_NIR_WITHIN_SUFFIX}"] = values - between
        out[f"{metric}{_NIR_BETWEEN_SUFFIX}"] = between
    return out, group_means or {}, priors or {}


def fit_preprocessing(
    train: pd.DataFrame,
    *,
    columns: list[str],
    group_col: str,
    nir_metrics: tuple[str, ...] = (),
    drop_all_nan: bool = True,
    drop_zero_variance: bool = True,
) -> PreprocessingState:
    """在训练折内拟合全部预处理统计量。

    Parameters
    ----------
    train : 训练折数据（含 group_col 与全部输入特征列）。
    columns : 模型输入特征列（含 NIR 原始指标名）。
    group_col : 分组列。
    nir_metrics : 需 within/between 分解的 NIR 指标。
    drop_all_nan / drop_zero_variance : 特征筛选开关。

    Returns
    -------
    拟合后的 PreprocessingState。
    """
    state = PreprocessingState()
    state.input_columns = list(columns)

    # 1) NIR within/between 分解（训练折内拟合组均值）
    decomposed = train
    state.output_columns = list(columns)
    if nir_metrics:
        decomposed, state.nir_group_means, state.nir_group_mean_priors = decompose_within_between(
            train, group_col, tuple(nir_metrics), is_train=True
        )
        expanded: list[str] = []
        for col in columns:
            if col in nir_metrics:
                expanded.append(f"{col}{_NIR_WITHIN_SUFFIX}")
                expanded.append(f"{col}{_NIR_BETWEEN_SUFFIX}")
            else:
                expanded.append(col)
        state.output_columns = expanded

    # 2) 特征筛选：全缺失列、零方差列（只看训练折）
    dropped: list[str] = []
    numeric = decomposed[state.output_columns].apply(pd.to_numeric, errors="coerce")
    for col in state.output_columns:
        series = numeric[col]
        if drop_all_nan and series.isna().all():
            dropped.append(col)
            continue
        if drop_zero_variance and series.nunique(dropna=True) < 2:
            dropped.append(col)
            continue
    state.dropped_columns = dropped
    kept = [c for c in state.output_columns if c not in dropped]

    # 3) 中位数插补统计量（训练折内拟合）
    for col in kept:
        series = numeric[col].dropna()
        state.imputation_medians[col] = float(series.median()) if len(series) else np.nan

    # 4) 标准化统计量（训练折内拟合；按插补后的值计算）
    imputed = numeric[kept].copy()
    for col in kept:
        imputed[col] = imputed[col].fillna(state.imputation_medians[col])
    for col in kept:
        series = imputed[col]
        mean = float(series.mean())
        std = float(series.std(ddof=0))
        # 零方差保护：插补后仍恒定的列按已丢弃处理
        if not np.isfinite(std) or std <= 1e-12:
            dropped.append(col)
            continue
        state.standardization_mean[col] = mean
        state.standardization_std[col] = std
    state.dropped_columns = sorted(set(dropped))
    state.output_columns = [c for c in state.output_columns if c not in state.dropped_columns]
    return state


def apply_preprocessing(
    frame: pd.DataFrame,
    state: PreprocessingState,
    *,
    group_col: str,
    nir_metrics: tuple[str, ...] = (),
    is_train: bool = False,
) -> pd.DataFrame:
    """应用训练折拟合的预处理状态到任意折数据。

    Parameters
    ----------
    frame : 待转换数据（含输入特征列）。
    state : fit_preprocessing 的产物。
    group_col : 分组列。
    nir_metrics : NIR 指标名（与拟合时一致）。
    is_train : True 时按训练路径重算 within/between（测试用，契约测试断言用）；
        False 时测试折用训练组均值先验。

    Returns
    -------
    模型输入矩阵（列序 = state.output_columns，无缺失、已标准化）。
    """
    working = frame.copy()
    columns: list[str] = []
    if nir_metrics and is_train:
        working, _, _ = decompose_within_between(
            working, group_col, tuple(nir_metrics), is_train=True
        )
        expanded: list[str] = []
        for col in state.input_columns:
            if col in nir_metrics:
                expanded += [f"{col}{_NIR_WITHIN_SUFFIX}", f"{col}{_NIR_BETWEEN_SUFFIX}"]
            else:
                expanded.append(col)
        columns = [c for c in expanded if c not in state.dropped_columns]
    else:
        if nir_metrics:
            working, _, _ = decompose_within_between(
                working, group_col, tuple(nir_metrics),
                group_means=state.nir_group_means, is_train=False,
                priors=state.nir_group_mean_priors,
            )
            expanded = []
            for col in state.input_columns:
                if col in nir_metrics:
                    expanded += [f"{col}{_NIR_WITHIN_SUFFIX}", f"{col}{_NIR_BETWEEN_SUFFIX}"]
                else:
                    expanded.append(col)
            columns = [c for c in expanded if c not in state.dropped_columns]
        else:
            columns = [c for c in state.input_columns if c not in state.dropped_columns]

    numeric = working[columns].apply(pd.to_numeric, errors="coerce")
    # 插补（只应用训练折中位数）
    for col in columns:
        if col in state.imputation_medians:
            numeric[col] = numeric[col].fillna(state.imputation_medians[col])
    # 标准化（只应用训练折 mean/std）
    out = pd.DataFrame(index=numeric.index)
    for col in columns:
        mean = state.standardization_mean.get(col)
        std = state.standardization_std.get(col)
        if mean is None or std is None:
            continue  # 训练折已丢弃的列
        out[col] = (numeric[col] - mean) / std
    # 输出列序与训练一致
    final_columns = [c for c in state.output_columns if c in out.columns]
    return out[final_columns].astype(float)
