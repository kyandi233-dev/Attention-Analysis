from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .figure_style import (
    LINESTYLES,
    PALETTE,
    clean_axis,
    finalize_layout,
    make_figure,
    panel_label,
    save_figure,
)


def _empty(ax: plt.Axes, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(plt.Rectangle((0.02, 0.05), 0.96, 0.9, transform=ax.transAxes,
                               fill=True, facecolor="#F4F4F4", edgecolor="#BBBBBB", linewidth=0.6, zorder=0))


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# 图例/刻度条件标签：数据取值保持英文标识（不改变统计口径），仅显示层翻译为中文。
CONDITION_LABELS_ZH = {
    "go_correct": "Go 正确",
    "go_omission_program": "Go 程序遗漏",
    "nogo_correct": "No-Go 正确",
    "nogo_commission": "No-Go 误按",
    "correct_inhibition": "正确抑制",
    "commission": "误按",
    "clean_omission": "无歧义遗漏",
    "prestimulus_associated_omission": "前刺激关联遗漏",
    "carryover_associated_omission": "携带关联遗漏",
    "prestimulus_and_carryover_associated_omission": "前刺激+携带关联遗漏",
    "omission": "遗漏",
}


def _label_zh(value) -> str:
    """条件取值映射为中文图例标签；未覆盖的取值原样保留英文。"""
    return CONDITION_LABELS_ZH.get(str(value), str(value))


def _add_zero(ax: plt.Axes, *, vertical: bool = False) -> None:
    if vertical:
        ax.axvline(0, color="#777777", linewidth=0.65, linestyle=":", zorder=0)
    else:
        ax.axhline(0, color="#777777", linewidth=0.65, linestyle=":", zorder=0)


def _heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "coolwarm",
    cbar_label: str = "",
    annotate: bool = False,
    cbar_ax: list | None = None,
) -> None:
    if matrix.empty:
        _empty(ax, "无数据")
        return
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    x_labels = [str(x) for x in matrix.columns]
    n_cols = len(matrix.columns)
    if n_cols > 12:
        # 稀疏化刻度标签，避免密集列标签糊成黑带
        step = int(np.ceil(n_cols / 10))
        shown = {i * step for i in range(n_cols)} | {n_cols - 1}
        x_labels = [lab if i in shown else "" for i, lab in enumerate(x_labels)]
    y_labels = [textwrap.fill(str(x).replace("_", " "), width=18, break_long_words=False) for x in matrix.index]
    ax.set_xticks(np.arange(len(matrix.columns)), x_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), y_labels)
    if annotate and matrix.shape[0] <= 8 and matrix.shape[1] <= 8:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.8)
    if cbar_ax is None:
        cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=7)
        cbar.ax.tick_params(labelsize=6)
    else:
        cbar = ax.figure.colorbar(image, ax=cbar_ax, fraction=0.030, pad=0.02)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=7)
        cbar.ax.tick_params(labelsize=6)


def _subject_block_box(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    category_col: str,
    value_col: str,
    order: list[str],
    ylabel: str,
    tick_labels: list[str] | None = None,
    tick_rotation: float = 22,
) -> None:
    if frame.empty or value_col not in frame.columns:
        _empty(ax, "无条件行")
        return
    subject = (
        frame.assign(**{value_col: _numeric(frame[value_col])})
        .groupby(["subject", "block_num", category_col], as_index=False)[value_col]
        .median()
    )
    offsets = {1: -0.17, 2: 0.17}
    width = 0.27
    legend_handles = []
    legend_labels = []
    for block_num in (1, 2):
        color = PALETTE[f"block{block_num}"]
        for idx, category in enumerate(order):
            values = subject.loc[
                subject["block_num"].eq(block_num)
                & subject[category_col].astype(str).eq(str(category)),
                value_col,
            ].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            pos = idx + 1 + offsets[block_num]
            bp = ax.boxplot(
                [values],
                positions=[pos],
                widths=width,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                medianprops={"color": "black", "linewidth": 0.8},
                whiskerprops={"linewidth": 0.65},
                capprops={"linewidth": 0.65},
                boxprops={"linewidth": 0.65, "edgecolor": color},
            )
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.30)
            rng = np.random.default_rng(1000 + block_num * 100 + idx)
            x = pos + rng.normal(0, 0.025, size=len(values))
            ax.scatter(x, values, s=7, alpha=0.48, color=color, linewidths=0, zorder=3)
        handle = plt.Line2D([0], [0], color=color, linewidth=2)
        legend_handles.append(handle)
        legend_labels.append(f"B{block_num}")
    labels = tick_labels if tick_labels is not None else order
    ax.set_xticks(
        np.arange(1, len(order) + 1),
        labels,
        rotation=tick_rotation,
        ha="right" if tick_rotation else "center",
    )
    ax.set_ylabel(ylabel)
    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc="best", ncol=2)
    _add_zero(ax)
    clean_axis(ax, grid_y=True)


def _trajectory_by_condition(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    condition_col: str = "event_condition",
    value_col: str = "pupil_median",
    block_num: int | None = None,
    title: str = "",
    ylabel: str = "中心化 PIR",
) -> None:
    if frame.empty or value_col not in frame.columns:
        _empty(ax, "无连续轨迹")
        return
    df = frame.copy()
    if block_num is not None:
        df = df[_numeric(df["block_num"]).eq(block_num)]
    if df.empty:
        _empty(ax, f"无 B{block_num} 轨迹")
        return
    for idx, (condition, current) in enumerate(df.groupby(condition_col, sort=True)):
        subject = current.groupby(["subject", "time_bin_mid_sec"], as_index=False)[value_col].median()
        summary = subject.groupby("time_bin_mid_sec")[value_col].agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        ).sort_index()
        color = PALETTE.get(str(condition), plt.get_cmap("tab10")(idx % 10))
        style = LINESTYLES.get(str(condition), "-")
        ax.plot(summary.index, summary["median"], color=color, linestyle=style, label=_label_zh(condition))
        ax.fill_between(summary.index, summary["q25"], summary["q75"], color=color, alpha=0.10, linewidth=0)
    _add_zero(ax, vertical=True)
    _add_zero(ax)
    ax.set_xlabel("相对事件时间（s）")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    clean_axis(ax, grid_y=True)


def figure01_global_landscape(
    global_detail: pd.DataFrame,
    global_summary: pd.DataFrame,
    distribution: pd.DataFrame,
    transition: pd.DataFrame,
    *,
    base: Path,
    formats: Iterable[str],
    raster_dpi: int,
) -> list[str]:
    fig, axes = make_figure(width="full", height_cm=15.5, nrows=2, ncols=2)
    ax = axes[0, 0]
    if global_detail.empty:
        _empty(ax, "无全局轨迹")
    else:
        for _, subject in global_detail.groupby("subject", sort=True):
            ax.plot(subject["global_time_sec"] / 60.0, subject["pupil_median"], color="#BDBDBD", linewidth=0.35, alpha=0.22)
        for block_num in (1, 2):
            current = global_summary[_numeric(global_summary["block_num"]).eq(block_num)]
            if current.empty:
                continue
            color = PALETTE[f"block{block_num}"]
            ax.plot(current["global_bin_sec"] / 60.0, current["median"], color=color, linewidth=1.7, label=f"B{block_num}")
            ax.fill_between(current["global_bin_sec"] / 60.0, current["q25"], current["q75"], color=color, alpha=0.14, linewidth=0)
        b1_end = float(global_detail["block1_display_end_sec"].median()) / 60.0
        b2_start = float(global_detail["block2_display_start_sec"].median()) / 60.0
        ax.axvspan(b1_end, b2_start, color="#E6E6E6", alpha=0.75, linewidth=0)
        ax.text((b1_end + b2_start) / 2.0, ax.get_ylim()[1], "区块间间隔", ha="center", va="top", fontsize=6)
        ax.set_xlabel("实验全局时间（min）")
        ax.set_ylabel("中心化 PIR")
        ax.legend(loc="best", ncol=2)
        _add_zero(ax)
        clean_axis(ax, grid_y=True)
    ax.set_title("全实验 PIR 全景")
    panel_label(ax, "A")

    ax = axes[0, 1]
    if global_detail.empty:
        _empty(ax, "无对齐区块轨迹")
    else:
        work = global_detail.copy()
        work["aligned_bin"] = np.floor(_numeric(work["time_in_block_sec"]) / 30.0) * 30.0 + 15.0
        for block_num in (1, 2):
            current = work[_numeric(work["block_num"]).eq(block_num)]
            subject = current.groupby(["subject", "aligned_bin"], as_index=False)["pupil_median"].median()
            summary = subject.groupby("aligned_bin")["pupil_median"].agg(
                median="median",
                q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75),
            ).sort_index()
            color = PALETTE[f"block{block_num}"]
            ax.plot(summary.index / 60.0, summary["median"], color=color, linestyle=LINESTYLES[f"block{block_num}"], label=f"B{block_num}")
            ax.fill_between(summary.index / 60.0, summary["q25"], summary["q75"], color=color, alpha=0.12, linewidth=0)
        ax.set_xlabel("区块内时间（min）")
        ax.set_ylabel("中心化 PIR")
        ax.legend(loc="best")
        _add_zero(ax)
        clean_axis(ax, grid_y=True)
    ax.set_title("按区块对齐的时间效应")
    panel_label(ax, "B")

    ax = axes[1, 0]
    if distribution.empty:
        _empty(ax, "无区块分布")
    else:
        groups = [distribution.loc[_numeric(distribution["block_num"]).eq(block), "pupil_median"].dropna().to_numpy(dtype=float) for block in (1, 2)]
        vp = ax.violinplot(groups, positions=[1, 2], showmedians=True, showextrema=False, widths=0.72)
        for idx, body in enumerate(vp["bodies"], start=1):
            body.set_facecolor(PALETTE[f"block{idx}"])
            body.set_edgecolor(PALETTE[f"block{idx}"])
            body.set_alpha(0.28)
        for _, row in distribution.pivot(index="subject", columns="block_num", values="pupil_median").iterrows():
            if 1 in row.index and 2 in row.index and pd.notna(row[1]) and pd.notna(row[2]):
                ax.plot([1, 2], [row[1], row[2]], color="#9B9B9B", linewidth=0.45, alpha=0.45)
        ax.set_xticks([1, 2], ["B1", "B2"])
        ax.set_ylabel("被试中位数中心化 PIR")
        _add_zero(ax)
        clean_axis(ax, grid_y=True)
    ax.set_title("全局区块分布")
    panel_label(ax, "C")

    ax = axes[1, 1]
    if transition.empty:
        _empty(ax, "无区块转换轨迹")
    else:
        for block_num, label in ((1, "B1 结束"), (2, "B2 开始")):
            current = transition[_numeric(transition["block_num"]).eq(block_num)].copy()
            current["bin"] = np.floor(_numeric(current["transition_time_sec"]) / 10.0) * 10.0 + 5.0
            subject = current.groupby(["subject", "bin"], as_index=False)["pupil_median"].median()
            summary = subject.groupby("bin")["pupil_median"].agg(
                median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)
            ).sort_index()
            color = PALETTE[f"block{block_num}"]
            ax.plot(summary.index, summary["median"], color=color, label=label)
            ax.fill_between(summary.index, summary["q25"], summary["q75"], color=color, alpha=0.12, linewidth=0)
        _add_zero(ax, vertical=True)
        _add_zero(ax)
        ax.set_xlabel("相对区块边界时间（s）")
        ax.set_ylabel("中心化 PIR")
        ax.legend(loc="best")
        clean_axis(ax, grid_y=True)
    ax.set_title("区块结束/开始恢复")
    panel_label(ax, "D")

    finalize_layout(fig, wspace=0.28, hspace=0.42)
    return save_figure(fig, base, formats, raster_dpi=raster_dpi)


def figure02_block_time_on_task(
    block_summary: pd.DataFrame,
    heterogeneity: pd.DataFrame,
    recovery_summary: pd.DataFrame,
    global_detail: pd.DataFrame,
    *,
    base: Path,
    formats: Iterable[str],
    raster_dpi: int,
) -> list[str]:
    fig, axes = make_figure(width="full", height_cm=14.5, nrows=2, ncols=2)
    ax = axes[0, 0]
    if block_summary.empty:
        _empty(ax, "无配对区块汇总")
    else:
        pivot = block_summary.pivot(index="subject", columns="block_num", values="pupil_median")
        for _, row in pivot.iterrows():
            if 1 in row.index and 2 in row.index and pd.notna(row[1]) and pd.notna(row[2]):
                ax.plot([1, 2], [row[1], row[2]], marker="o", markersize=2.6, color="#8E8E8E", linewidth=0.55, alpha=0.55)
        med = block_summary.groupby("block_num")["pupil_median"].median()
        ax.plot(med.index, med.values, marker="o", markersize=4.5, color="black", linewidth=1.8, label="队列中位数")
        ax.set_xticks([1, 2], ["B1", "B2"])
        ax.set_ylabel("中心化 PIR 中位数")
        ax.legend(loc="best")
        _add_zero(ax)
        clean_axis(ax, grid_y=True)
    ax.set_title("被试内区块变化")
    panel_label(ax, "A")

    ax = axes[0, 1]
    slope_cols = [col for col in ("block1_time_slope_per_sec", "block2_time_slope_per_sec") if col in heterogeneity.columns]
    if heterogeneity.empty or len(slope_cols) < 2:
        _empty(ax, "无被试时间效应斜率")
    else:
        x = _numeric(heterogeneity[slope_cols[0]])
        y = _numeric(heterogeneity[slope_cols[1]])
        ok = x.notna() & y.notna()
        ax.scatter(x[ok], y[ok], s=14, facecolors="none", edgecolors="#555555", linewidths=0.75)
        lim = np.nanmax(np.abs(np.concatenate([x[ok].to_numpy(), y[ok].to_numpy()]))) if int(ok.sum()) else 1.0
        lim = max(float(lim), 1e-6)
        ax.plot([-lim, lim], [-lim, lim], color="#999999", linestyle=":", linewidth=0.7)
        ax.axhline(0, color="#BBBBBB", linewidth=0.6)
        ax.axvline(0, color="#BBBBBB", linewidth=0.6)
        ax.set_xlabel("B1 PIR 斜率（/s）")
        ax.set_ylabel("B2 PIR 斜率（/s）")
        clean_axis(ax, grid_y=False)
    ax.set_title("个体时间效应斜率")
    panel_label(ax, "B")

    ax = axes[1, 0]
    if recovery_summary.empty or "recovery_delta_block2_minus_block1" not in recovery_summary.columns:
        _empty(ax, "无恢复差值")
    else:
        values = _numeric(recovery_summary["recovery_delta_block2_minus_block1"]).dropna().sort_values().to_numpy(dtype=float)
        ax.axhline(0, color="#777777", linestyle=":", linewidth=0.7)
        ax.scatter(np.arange(1, len(values) + 1), values, s=12, color=PALETTE["block2"], alpha=0.75, linewidths=0)
        ax.set_xlabel("按恢复差值排序的被试")
        ax.set_ylabel("B2 前 60 s - B1 末 60 s")
        clean_axis(ax, grid_y=True)
    ax.set_title("区块转换恢复异质性")
    panel_label(ax, "C")

    ax = axes[1, 1]
    if global_detail.empty:
        _empty(ax, "无前后半段对比")
    else:
        work = global_detail.copy()
        work["half"] = work.groupby(["subject", "block_num"])["time_in_block_sec"].transform(
            lambda x: np.where(x <= x.median(), "first half", "second half")
        )
        subject = work.groupby(["subject", "block_num", "half"], as_index=False)["pupil_median"].median()
        for block_num in (1, 2):
            pivot = subject[_numeric(subject["block_num"]).eq(block_num)].pivot(index="subject", columns="half", values="pupil_median")
            if {"first half", "second half"}.issubset(pivot.columns):
                delta = (pivot["second half"] - pivot["first half"]).dropna()
                pos = block_num
                bp = ax.boxplot([delta], positions=[pos], widths=0.5, patch_artist=True, showfliers=False, manage_ticks=False)
                bp["boxes"][0].set_facecolor(PALETTE[f"block{block_num}"])
                bp["boxes"][0].set_alpha(0.28)
                bp["boxes"][0].set_edgecolor(PALETTE[f"block{block_num}"])
                rng = np.random.default_rng(230 + block_num)
                ax.scatter(pos + rng.normal(0, 0.035, len(delta)), delta, s=8, color=PALETTE[f"block{block_num}"], alpha=0.55, linewidths=0)
        ax.set_xticks([1, 2], ["B1", "B2"])
        ax.set_ylabel("后半段 - 前半段 PIR")
        _add_zero(ax)
        clean_axis(ax, grid_y=True)
    ax.set_title("区块内衰减对比")
    panel_label(ax, "D")

    finalize_layout(fig, wspace=0.30, hspace=0.42)
    return save_figure(fig, base, formats, raster_dpi=raster_dpi)


def figure03_trial_states(
    trial_conditions: pd.DataFrame,
    advanced_behavior: pd.DataFrame,
    *,
    base: Path,
    formats: Iterable[str],
    raster_dpi: int,
) -> list[str]:
    fig, axes = make_figure(width="full", height_cm=15.0, nrows=2, ncols=2)
    order = ["go_correct", "go_omission_program", "nogo_correct", "nogo_commission"]
    _subject_block_box(
        axes[0, 0], trial_conditions, category_col="outcome", value_col="pupil_median", order=order,
        ylabel="试次前中心化 PIR",
        tick_labels=["Go 正确", "Go 程序遗漏", "No-Go 正确", "No-Go 误按"],
    )
    axes[0, 0].set_title("程序评分结果 × 区块")
    panel_label(axes[0, 0], "A")

    omission_order = [
        "clean_omission",
        "prestimulus_associated_omission",
        "carryover_associated_omission",
        "prestimulus_and_carryover_associated_omission",
    ]
    omission = trial_conditions[trial_conditions.get("omission_qc_type", pd.Series(index=trial_conditions.index, dtype=str)).astype(str).isin(omission_order)].copy()
    _subject_block_box(
        axes[0, 1], omission, category_col="omission_qc_type", value_col="pupil_median", order=omission_order,
        ylabel="试次前中心化 PIR",
        tick_labels=[
            "无歧义",
            "前刺激",
            "携带",
            "前+携带",
        ],
        tick_rotation=0,
    )
    axes[0, 1].set_xlim(0.5, 4.7)
    axes[0, 1].set_title("遗漏动作时序亚型 × 区块")
    panel_label(axes[0, 1], "B")

    ax = axes[1, 0]
    if trial_conditions.empty or "rt" not in trial_conditions.columns:
        _empty(ax, "无 Go RT/PIR 行")
    else:
        df = trial_conditions[trial_conditions["outcome"].astype(str).eq("go_correct")].copy()
        for block_num in (1, 2):
            current = df[_numeric(df["block_num"]).eq(block_num)].copy()
            current["rt"] = _numeric(current["rt"])
            current["pupil_median"] = _numeric(current["pupil_median"])
            ok = current["rt"].notna() & current["pupil_median"].notna()
            current = current.loc[ok]
            if current.empty:
                continue
            bins = pd.qcut(current["pupil_median"], q=min(8, max(2, current["pupil_median"].nunique())), duplicates="drop")
            current["pupil_bin"] = bins
            summary = current.groupby("pupil_bin", observed=True).agg(pupil=("pupil_median", "median"), rt=("rt", "median")).sort_values("pupil")
            ax.plot(summary["pupil"], summary["rt"], marker="o", color=PALETTE[f"block{block_num}"], linestyle=LINESTYLES[f"block{block_num}"], label=f"B{block_num}")
        ax.set_xlabel("分箱试次前 PIR")
        ax.set_ylabel("正确 Go RT（ms）")
        ax.legend(loc="best")
        clean_axis(ax, grid_y=True)
    ax.set_title("PIR 状态下的正确 Go 行为")
    panel_label(ax, "C")

    ax = axes[1, 1]
    metrics = [col for col in ("commission_rate", "program_omission_rate", "clean_omission_rate", "ambiguous_omission_rate") if col in advanced_behavior.columns]
    if advanced_behavior.empty or not metrics:
        _empty(ax, "无错误率汇总")
    else:
        x = np.arange(len(metrics))
        width = 0.34
        for block_num, offset in ((1, -width / 2), (2, width / 2)):
            frame = advanced_behavior[_numeric(advanced_behavior["block_num"]).eq(block_num)]
            means = [float(_numeric(frame[m]).median()) for m in metrics]
            ax.bar(x + offset, means, width=width, color=PALETTE[f"block{block_num}"], alpha=0.65, label=f"B{block_num}")
        rate_tick_zh = {"commission": "误按", "program_omission": "程序遗漏", "clean_omission": "无歧义遗漏", "ambiguous_omission": "歧义遗漏"}
        ax.set_xticks(x, [rate_tick_zh.get(m.replace("_rate", ""), m.replace("_rate", "")) for m in metrics], rotation=25, ha="right")
        ax.set_ylabel("被试中位数比率")
        ax.set_ylim(bottom=0)
        ax.legend(loc="best")
        clean_axis(ax, grid_y=True)
    ax.set_title("行为错误/QC 概况")
    panel_label(ax, "D")

    finalize_layout(fig, wspace=0.34, hspace=0.46)
    return save_figure(fig, base, formats, raster_dpi=raster_dpi)


def figure04_error_precursors(
    nogo_continuous: pd.DataFrame,
    omission_continuous: pd.DataFrame,
    nogo_trial_lag: pd.DataFrame,
    *,
    base: Path,
    formats: Iterable[str],
    raster_dpi: int,
) -> list[str]:
    fig, axes = make_figure(width="full", height_cm=15.0, nrows=2, ncols=2)
    _trajectory_by_condition(axes[0, 0], nogo_continuous, block_num=1, title="No-Go 前兆 — B1")
    panel_label(axes[0, 0], "A")
    _trajectory_by_condition(axes[0, 1], nogo_continuous, block_num=2, title="No-Go 前兆 — B2")
    panel_label(axes[0, 1], "B")
    _trajectory_by_condition(axes[1, 0], omission_continuous, title="Go 遗漏前兆 — 两区块")
    panel_label(axes[1, 0], "C")

    ax = axes[1, 1]
    if nogo_trial_lag.empty:
        _empty(ax, "无试次滞后前兆")
    else:
        for condition, current in nogo_trial_lag.groupby("event_outcome", sort=True):
            subject = current.groupby(["subject", "lag"], as_index=False).agg(rt=("go_rt_ms", "median"), pir=("pupil_median", "median"))
            rt = subject.groupby("lag")["rt"].median().sort_index()
            color = PALETTE.get(str(condition), "#555555")
            style = LINESTYLES.get(str(condition), "-")
            ax.plot(rt.index, rt.values, marker="o", color=color, linestyle=style, label=_label_zh(condition))
        ax.axvline(0, color="#777777", linestyle=":", linewidth=0.65)
        ax.set_xlabel("No-Go 前正确 Go 试次滞后")
        ax.set_ylabel("正确 Go RT（ms）")
        ax.legend(loc="best")
        clean_axis(ax, grid_y=True)
    ax.set_title("离散试次滞后行为前兆")
    panel_label(ax, "D")

    finalize_layout(fig, wspace=0.30, hspace=0.42)
    return save_figure(fig, base, formats, raster_dpi=raster_dpi)


def figure05_probe_states(
    probe_events: pd.DataFrame,
    probe_continuous: pd.DataFrame,
    probe_transitions: pd.DataFrame,
    *,
    base: Path,
    formats: Iterable[str],
    raster_dpi: int,
) -> list[str]:
    fig, axes = make_figure(width="full", height_cm=15.0, nrows=2, ncols=2)
    ax = axes[0, 0]
    if probe_events.empty or "probe_response" not in probe_events.columns:
        _empty(ax, "无 probe_response")
    else:
        counts = probe_events.groupby(["block_num", "probe_response"]).size().rename("n").reset_index()
        totals = counts.groupby("block_num")["n"].transform("sum")
        counts["fraction"] = counts["n"] / totals
        options = sorted(counts["probe_response"].dropna().unique())
        x = np.arange(len(options))
        width = 0.34
        for block_num, offset in ((1, -width / 2), (2, width / 2)):
            current = counts[_numeric(counts["block_num"]).eq(block_num)].set_index("probe_response")
            vals = [current["fraction"].get(opt, 0.0) for opt in options]
            ax.bar(x + offset, vals, width=width, color=PALETTE[f"block{block_num}"], alpha=0.68, label=f"B{block_num}")
        ax.set_xticks(x, [str(int(v)) if float(v).is_integer() else str(v) for v in options])
        ax.set_xlabel("probe_response 原始编码")
        ax.set_ylabel("探针占比")
        ax.legend(loc="best")
        clean_axis(ax, grid_y=True)
    ax.set_title("探针作答分布 × 区块")
    panel_label(ax, "A")

    ax = axes[0, 1]
    if probe_events.empty or not {"probe_response", "probe_vigilance"}.issubset(probe_events.columns):
        _empty(ax, "无作答 × 警觉")
    else:
        df = probe_events.copy()
        df["probe_vigilance"] = _numeric(df["probe_vigilance"])
        for block_num in (1, 2):
            current = df[_numeric(df["block_num"]).eq(block_num)]
            summary = current.groupby("probe_response")["probe_vigilance"].median().sort_index()
            ax.plot(summary.index, summary.values, marker="o", color=PALETTE[f"block{block_num}"], linestyle=LINESTYLES[f"block{block_num}"], label=f"B{block_num}")
        ax.set_xlabel("probe_response 原始编码")
        ax.set_ylabel("probe_vigilance 中位数")
        ax.legend(loc="best")
        clean_axis(ax, grid_y=True)
    ax.set_title("探针作答 × 警觉结构")
    panel_label(ax, "B")

    _trajectory_by_condition(axes[1, 0], probe_continuous, title="探针前连续 PIR 轨迹")
    panel_label(axes[1, 0], "C")

    ax = axes[1, 1]
    if probe_transitions.empty or "response_transition" not in probe_transitions.columns:
        _empty(ax, "无探针序列转换")
    else:
        counts = probe_transitions.groupby("response_transition").size().rename("n").sort_values(ascending=False)
        top = counts.head(12).sort_values()
        ax.barh(np.arange(len(top)), top.values, color="#777777", alpha=0.72)
        ax.set_yticks(np.arange(len(top)), top.index)
        ax.set_xlabel("转换计数")
        clean_axis(ax, grid_y=False)
    ax.set_title("探针状态序列转换")
    panel_label(ax, "D")

    finalize_layout(fig, wspace=0.32, hspace=0.44)
    return save_figure(fig, base, formats, raster_dpi=raster_dpi)


def _binned_scatter(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x_col: str,
    *,
    title: str,
    legend_loc: str = "best",
) -> None:
    if frame.empty or x_col not in frame.columns or "pupil_median" not in frame.columns:
        _empty(ax, "视觉协变量不可用")
        return
    for block_num in (1, 2):
        current = frame[_numeric(frame["block_num"]).eq(block_num)].copy()
        current[x_col] = _numeric(current[x_col])
        current["pupil_median"] = _numeric(current["pupil_median"])
        current = current.dropna(subset=[x_col, "pupil_median"])
        if current.empty:
            continue
        q = min(8, max(2, current[x_col].nunique()))
        current["bin"] = pd.qcut(current[x_col], q=q, duplicates="drop")
        summary = current.groupby("bin", observed=True).agg(x=(x_col, "median"), y=("pupil_median", "median")).sort_values("x")
        ax.plot(summary["x"], summary["y"], marker="o", color=PALETTE[f"block{block_num}"], linestyle=LINESTYLES[f"block{block_num}"], label=f"B{block_num}")
    ax.set_xlabel(x_col.replace("current_", ""))
    ax.set_ylabel("试次前中心化 PIR")
    ax.set_title(title)
    ax.legend(loc=legend_loc, fontsize=6.5)
    _add_zero(ax)
    clean_axis(ax, grid_y=True)


def figure06_visual_controls(
    visual_trial: pd.DataFrame,
    stimulus_summary: pd.DataFrame,
    visual_correlations: pd.DataFrame,
    *,
    base: Path,
    formats: Iterable[str],
    raster_dpi: int,
) -> list[str]:
    fig, axes = make_figure(width="full", height_cm=15.0, nrows=2, ncols=2)
    ax = axes[0, 0]
    if stimulus_summary.empty or not {"stimulus_size", "pupil_median"}.issubset(stimulus_summary.columns):
        _empty(ax, "无刺激大小汇总")
    else:
        for block_num in (1, 2):
            for is_nogo, marker in ((0, "o"), (1, "s")):
                current = stimulus_summary[
                    _numeric(stimulus_summary["block_num"]).eq(block_num)
                    & _numeric(stimulus_summary["is_no_go"]).eq(is_nogo)
                ]
                summary = current.groupby("stimulus_size")["pupil_median"].median().sort_index()
                if summary.empty:
                    continue
                ax.plot(
                    summary.index,
                    summary.values,
                    marker=marker,
                    color=PALETTE[f"block{block_num}"],
                    linestyle="-" if is_nogo == 0 else ":",
                    label=f"B{block_num} {'No-Go' if is_nogo else 'Go'}",
                )
        ax.set_xlabel("刺激大小（%）")
        ax.set_ylabel("试次前中心化 PIR")
        ax.legend(loc="best", ncol=2)
        _add_zero(ax)
        clean_axis(ax, grid_y=True)
    ax.set_title("刺激大小/任务类型控制")
    panel_label(ax, "A")

    _binned_scatter(axes[0, 1], visual_trial, "current_central_rel_lum_mean", title="当前刺激亮度")
    panel_label(axes[0, 1], "B")
    _binned_scatter(
        axes[1, 0],
        visual_trial,
        "current_central_rms_contrast",
        title="当前刺激对比度",
        legend_loc="upper left",
    )
    panel_label(axes[1, 0], "C")

    ax = axes[1, 1]
    if visual_correlations.empty:
        _empty(ax, "无视觉协变量相关")
    else:
        df = visual_correlations.sort_values("spearman_rho_with_pir")
        ax.barh(np.arange(len(df)), df["spearman_rho_with_pir"], color="#777777", alpha=0.72)
        ax.set_yticks(
            np.arange(len(df)),
            [
                (
                    ("当前" if str(x).startswith("current_") else "先前")
                    + ": "
                    + ("水果" if "fruit_" in str(x) else "中央")
                    + "\n"
                    + (
                        "可见面积"
                        if "visible_area" in str(x)
                        else "对比度"
                        if "rms_contrast" in str(x)
                        else "亮度变化"
                        if "delta_" in str(x) and "rel_lum" in str(x)
                        else "亮度"
                    )
                )
                for x in df["covariate"]
            ],
        )
        ax.tick_params(axis="y", labelsize=5.5)
        ax.axvline(0, color="#777777", linestyle=":", linewidth=0.65)
        ax.set_xlabel("与 PIR 的 Spearman ρ（仅验证）")
        clean_axis(ax, grid_y=False)
    ax.set_title("当前与先前视觉协变量")
    panel_label(ax, "D")

    finalize_layout(fig, left=0.16, wspace=0.42, hspace=0.44)
    return save_figure(fig, base, formats, raster_dpi=raster_dpi)


def figure07_individual_differences(
    heterogeneity: pd.DataFrame,
    raw_pir: pd.DataFrame,
    *,
    base: Path,
    formats: Iterable[str],
    raster_dpi: int,
) -> list[str]:
    fig, axes = make_figure(width="full", height_cm=14.5, nrows=2, ncols=2)
    ax = axes[0, 0]
    if heterogeneity.empty or not {"block1_time_slope_per_sec", "block2_time_slope_per_sec"}.issubset(heterogeneity.columns):
        _empty(ax, "无被试斜率数据")
    else:
        x = _numeric(heterogeneity["block1_time_slope_per_sec"])
        y = _numeric(heterogeneity["block2_time_slope_per_sec"])
        ok = x.notna() & y.notna()
        ax.scatter(x[ok], y[ok], s=18, color="#555555", alpha=0.72, linewidths=0)
        for subject, xv, yv in zip(heterogeneity.loc[ok, "subject"], x[ok], y[ok]):
            ax.annotate(str(subject).replace("sub-", ""), (xv, yv), xytext=(2, 2), textcoords="offset points", fontsize=5.2, color="#666666")
        ax.axhline(0, color="#BBBBBB", linewidth=0.6)
        ax.axvline(0, color="#BBBBBB", linewidth=0.6)
        ax.set_xlabel("B1 斜率（/s）")
        ax.set_ylabel("B2 斜率（/s）")
        clean_axis(ax)
    ax.set_title("时间效应斜率异质性")
    panel_label(ax, "A")

    ax = axes[0, 1]
    if raw_pir.empty or "raw_pupil_subject_median" not in raw_pir.columns:
        _empty(ax, "无个体间原始 pupil 数据")
    else:
        df = raw_pir.sort_values("raw_pupil_subject_median")
        ax.scatter(np.arange(1, len(df) + 1), df["raw_pupil_subject_median"], s=15, color="#555555", alpha=0.78, linewidths=0)
        ax.set_xlabel("按原始 pupil 基线排序的被试")
        ax.set_ylabel("原始双眼 pupil 中位数")
        clean_axis(ax, grid_y=True)
    ax.set_title("个体间原始 pupil 特征")
    panel_label(ax, "B")

    ax = axes[1, 0]
    if heterogeneity.empty or "recovery_delta_block2_minus_block1" not in heterogeneity.columns:
        _empty(ax, "无恢复异质性")
    else:
        df = heterogeneity[["subject", "recovery_delta_block2_minus_block1"]].dropna().sort_values("recovery_delta_block2_minus_block1")
        ax.bar(np.arange(len(df)), df["recovery_delta_block2_minus_block1"], color=np.where(df["recovery_delta_block2_minus_block1"].ge(0), PALETTE["block2"], PALETTE["block1"]), alpha=0.70)
        ax.axhline(0, color="#777777", linewidth=0.65)
        ax.set_xlabel("按区块转换变化排序的被试")
        ax.set_ylabel("恢复差值")
        clean_axis(ax, grid_y=True)
    ax.set_title("区块转换异质性")
    panel_label(ax, "C")

    ax = axes[1, 1]
    effect_cols = [col for col in heterogeneity.columns if col.startswith("median_effect__")]
    if heterogeneity.empty or not effect_cols:
        _empty(ax, "无被试事件效应异质性")
    else:
        long = heterogeneity[["subject", *effect_cols]].melt(id_vars="subject", var_name="effect", value_name="value")
        long["effect"] = long["effect"].str.replace("median_effect__", "", regex=False)
        effects = list(dict.fromkeys(long["effect"]))
        for idx, effect in enumerate(effects):
            values = _numeric(long.loc[long["effect"].eq(effect), "value"]).dropna().to_numpy(dtype=float)
            if len(values):
                pos = idx + 1
                ax.boxplot([values], positions=[pos], widths=0.52, showfliers=False, manage_ticks=False)
                rng = np.random.default_rng(500 + idx)
                ax.scatter(pos + rng.normal(0, 0.035, len(values)), values, s=8, color="#666666", alpha=0.55, linewidths=0)
        short_effects = [
            textwrap.fill(
                effect.replace("_minus_", " - ").replace("_", " "),
                width=20,
                break_long_words=False,
            )
            for effect in effects
        ]
        ax.set_xticks(np.arange(1, len(effects) + 1), short_effects, rotation=0, ha="center")
        ax.set_ylabel("被试级中位数对比")
        _add_zero(ax)
        clean_axis(ax, grid_y=True)
    ax.set_title("事件关联异质性")
    panel_label(ax, "D")

    finalize_layout(fig, wspace=0.32, hspace=0.44)
    return save_figure(fig, base, formats, raster_dpi=raster_dpi)


def _corr_matrix(long: pd.DataFrame, row: str, col: str, value: str) -> pd.DataFrame:
    return long.pivot(index=row, columns=col, values=value) if not long.empty else pd.DataFrame()


def figure08_feature_structure(
    feature_within: pd.DataFrame,
    within_metrics: pd.DataFrame,
    between_metrics: pd.DataFrame,
    window_stability: pd.DataFrame,
    *,
    base: Path,
    formats: Iterable[str],
    raster_dpi: int,
) -> list[str]:
    fig, axes = make_figure(width="full", height_cm=17.0, nrows=2, ncols=2)
    matrix = _corr_matrix(feature_within, "feature_a", "feature_b", "r")
    _heatmap(axes[0, 0], matrix, vmin=-1, vmax=1, cmap="coolwarm", cbar_label="个体内 r")
    axes[0, 0].set_title("PIR 特征冗余")
    panel_label(axes[0, 0], "A")

    matrix = _corr_matrix(within_metrics, "metric_a", "metric_b", "r")
    _heatmap(axes[0, 1], matrix, vmin=-1, vmax=1, cmap="coolwarm", cbar_label="个体内 r", annotate=True)
    axes[0, 1].set_title("个体内 NIR–行为结构")
    panel_label(axes[0, 1], "B")

    matrix = _corr_matrix(between_metrics, "metric_a", "metric_b", "r")
    _heatmap(axes[1, 0], matrix, vmin=-1, vmax=1, cmap="coolwarm", cbar_label="个体间 r")
    axes[1, 0].set_title("个体间原始 PIR 结构")
    panel_label(axes[1, 0], "C")

    ax = axes[1, 1]
    if window_stability.empty:
        _empty(ax, "无多尺度稳定性汇总")
    else:
        for idx, (contrast, current) in enumerate(window_stability.groupby("contrast", sort=True)):
            current = current.sort_values("window_sec")
            color = plt.get_cmap("tab10")(idx % 10)
            ax.plot(current["window_sec"], current["median"], marker="o", color=color, label=str(contrast))
            ax.fill_between(current["window_sec"], current["q25"], current["q75"], color=color, alpha=0.12, linewidth=0)
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 5, 10, 20, 30, 60], ["1", "3", "5", "10", "20", "30", "60"])
        ax.set_xlabel("事件前窗口（s；对数轴）")
        ax.set_ylabel("被试级效应/对比")
        ax.legend(loc="best")
        _add_zero(ax)
        clean_axis(ax, grid_y=True)
    ax.set_title("预设窗口间效应稳定性")
    panel_label(ax, "D")

    finalize_layout(fig, left=0.20, bottom=0.15, wspace=0.50, hspace=0.50)
    return save_figure(fig, base, formats, raster_dpi=raster_dpi)


def _normalize_coverage_wide(coverage: pd.DataFrame) -> pd.DataFrame:
    """Adapt new wide coverage tables to the legacy long contract.

    11_analysis_tables publishes one median column per coverage dimension;
    the figure suite expects a long table with coverage_metric and
    coverage_value. A table that is already long passes through unchanged.
    """
    if coverage.empty or "coverage_metric" in coverage.columns:
        return coverage
    value_cols = [
        col
        for col in (
            "pupil_valid_fraction_median",
            "available_duration_fraction_median",
            "internal_coverage_fraction_median",
            "boundary_truncated_fraction",
            "max_temporal_gap_sec_p95",
        )
        if col in coverage.columns
    ]
    id_vars = [
        col
        for col in ("session_id", "analysis_group_token", "block_num", "track", "level")
        if col in coverage.columns
    ]
    if not value_cols:
        return coverage
    long = coverage.melt(
        id_vars=id_vars,
        value_vars=value_cols,
        var_name="coverage_metric",
        value_name="coverage_value",
    )
    if "session_id" in long.columns:
        long["subject"] = long["session_id"].astype(str)
    if "level" in long.columns:
        long["analysis_level"] = long["level"].astype(str)
    return long


def _coverage_fraction_matrix(
    coverage: pd.DataFrame,
    *,
    level: str,
    track: str,
) -> pd.DataFrame:
    if coverage.empty:
        return pd.DataFrame()
    fraction_metrics = [
        "pupil_valid_fraction_median",
        "available_duration_fraction_median",
        "internal_coverage_fraction_median",
        "boundary_truncated_fraction",
    ]
    df = coverage[
        coverage["analysis_level"].astype(str).eq(level)
        & coverage["track"].astype(str).eq(track)
        & coverage["coverage_metric"].astype(str).isin(fraction_metrics)
    ].copy()
    if df.empty:
        return pd.DataFrame()
    df["subject_block"] = df["subject"].astype(str).str.replace("sub-", "", regex=False) + "/B" + _numeric(df["block_num"]).astype("Int64").astype(str)
    return df.pivot_table(index="coverage_metric", columns="subject_block", values="coverage_value", aggfunc="median")


def figure09_quality_control(
    coverage: pd.DataFrame,
    source_mode: pd.DataFrame,
    *,
    track: str,
    base: Path,
    formats: Iterable[str],
    raster_dpi: int,
) -> list[str]:
    coverage = _normalize_coverage_wide(coverage)
    coverage_metric_zh = {
        "pupil_valid_fraction_median": "有效瞳孔占比中位数",
        "available_duration_fraction_median": "可用时长占比中位数",
        "internal_coverage_fraction_median": "窗内覆盖占比中位数",
        "boundary_truncated_fraction": "边界截断占比",
    }
    fig, axes = make_figure(width="full", height_cm=8.0, nrows=1, ncols=3)
    trial_matrix = _coverage_fraction_matrix(coverage, level="trial", track=track)
    if not trial_matrix.empty:
        trial_matrix.index = [coverage_metric_zh.get(str(name), str(name)) for name in trial_matrix.index]
    _heatmap(axes[0], trial_matrix, vmin=0, vmax=1, cmap="viridis", cbar_label="占比")
    axes[0].set_title("试次窗口覆盖维度")
    panel_label(axes[0], "A")

    probe_matrix = _coverage_fraction_matrix(coverage, level="probe", track=track)
    if not probe_matrix.empty:
        probe_matrix.index = [coverage_metric_zh.get(str(name), str(name)) for name in probe_matrix.index]
    _heatmap(axes[1], probe_matrix, vmin=0, vmax=1, cmap="viridis", cbar_label="占比")
    axes[1].set_title("探针窗口覆盖维度")
    panel_label(axes[1], "B")

    ax = axes[2]
    if source_mode.empty:
        _empty(ax, "无双眼来源模式 QC")
    else:
        df = source_mode[source_mode["track"].astype(str).eq(track)].copy() if "track" in source_mode.columns else source_mode.copy()
        cols = [col for col in ("source_mode_binocular_fraction", "source_mode_left_only_fraction", "source_mode_right_only_fraction", "source_mode_missing_fraction") if col in df.columns]
        summary = df.groupby("block_num")[cols].mean().sort_index()
        x = np.arange(len(summary.index))
        bottom = np.zeros(len(x))
        colors = ["#4C78A8", "#59A14F", "#F28E2B", "#B8B8B8"]
        source_zh = {"binocular": "双眼", "left_only": "仅左眼", "right_only": "仅右眼", "missing": "缺失"}
        for col, color in zip(cols, colors):
            values = summary[col].to_numpy(dtype=float)
            base_name = col.replace("source_mode_", "").replace("_fraction", "")
            ax.bar(x, values, bottom=bottom, color=color, alpha=0.78, label=source_zh.get(base_name, base_name))
            bottom += np.nan_to_num(values)
        ax.set_xticks(x, [f"B{int(v)}" for v in summary.index])
        ax.set_ylim(0, 1)
        ax.set_ylabel("平均窗口占比")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False, fontsize=7)
        clean_axis(ax, grid_y=True)
    ax.set_title("双眼/单眼/缺失构成")
    panel_label(ax, "C")



    finalize_layout(fig, left=0.10, bottom=0.30, right=0.985, wspace=1.25, hspace=None)
    return save_figure(fig, base, formats, raster_dpi=raster_dpi)


def figure10_robustness(
    track_correlations: pd.DataFrame,
    track_agreement: pd.DataFrame,
    model_results: pd.DataFrame,
    *,
    base: Path,
    formats: Iterable[str],
    raster_dpi: int,
) -> list[str]:
    fig, axes = make_figure(width="full", height_cm=8.0, nrows=1, ncols=3)
    matrix = _corr_matrix(track_correlations, "track_a", "track_b", "correlation")
    _heatmap(axes[0], matrix, vmin=-1, vmax=1, cmap="coolwarm", cbar_label="Pearson r", annotate=True)
    axes[0].set_title("六轨道一致性")
    panel_label(axes[0], "A")

    ax = axes[1]
    if track_agreement.empty:
        _empty(ax, "无轨道一致性汇总")
    else:
        df = track_agreement.sort_values("pearson_r")
        ax.barh(np.arange(len(df)), df["pearson_r"], color="#777777", alpha=0.72)
        ax.set_yticks(np.arange(len(df)), df["comparison_track"])
        ax.set_xlim(-1, 1)
        ax.axvline(0, color="#777777", linewidth=0.65)
        ax.set_xlabel("与主轨道的相关")
        clean_axis(ax)
    ax.set_title("与双眼主轨道一致性")
    panel_label(ax, "B")

    ax = axes[2]
    if track_agreement.empty:
        _empty(ax, "无绝对差值汇总")
    else:
        df = track_agreement.sort_values("median_absolute_difference")
        ax.barh(np.arange(len(df)), df["median_absolute_difference"], color="#999999", alpha=0.72)
        ax.set_yticks(np.arange(len(df)), df["comparison_track"])
        ax.set_xlabel("PIR 绝对差值中位数")
        clean_axis(ax)
    ax.set_title("轨道分歧程度")
    panel_label(ax, "C")



    finalize_layout(fig, left=0.12, bottom=0.28, right=0.97, wspace=0.95, hspace=None)
    return save_figure(fig, base, formats, raster_dpi=raster_dpi)


def write_publication_suite(
    *,
    output_dir: Path,
    formats: Iterable[str],
    raster_dpi: int,
    main_track: str,
    global_detail: pd.DataFrame,
    global_summary: pd.DataFrame,
    global_distribution: pd.DataFrame,
    block_transition: pd.DataFrame,
    block_summary: pd.DataFrame,
    heterogeneity: pd.DataFrame,
    recovery_summary: pd.DataFrame,
    trial_conditions: pd.DataFrame,
    advanced_behavior: pd.DataFrame,
    nogo_continuous: pd.DataFrame,
    omission_continuous: pd.DataFrame,
    nogo_trial_lag: pd.DataFrame,
    probe_events: pd.DataFrame,
    probe_continuous: pd.DataFrame,
    probe_transitions: pd.DataFrame,
    visual_trial: pd.DataFrame,
    stimulus_summary: pd.DataFrame,
    visual_correlations: pd.DataFrame,
    raw_pir: pd.DataFrame,
    feature_within: pd.DataFrame,
    within_metrics: pd.DataFrame,
    between_metrics: pd.DataFrame,
    window_stability: pd.DataFrame,
    coverage: pd.DataFrame,
    source_mode: pd.DataFrame,
    track_correlations: pd.DataFrame,
    track_agreement: pd.DataFrame,
    model_results: pd.DataFrame,
) -> dict[str, list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = {
        "Figure01_global_PIR_landscape": lambda: figure01_global_landscape(
            global_detail, global_summary, global_distribution, block_transition,
            base=output_dir / "Figure01_global_PIR_landscape", formats=formats, raster_dpi=raster_dpi,
        ),
        "Figure02_Block_time_on_task": lambda: figure02_block_time_on_task(
            block_summary, heterogeneity, recovery_summary, global_detail,
            base=output_dir / "Figure02_Block_time_on_task", formats=formats, raster_dpi=raster_dpi,
        ),
        "Figure03_trial_behavior_states": lambda: figure03_trial_states(
            trial_conditions, advanced_behavior,
            base=output_dir / "Figure03_trial_behavior_states", formats=formats, raster_dpi=raster_dpi,
        ),
        "Figure04_error_precursor_trajectories": lambda: figure04_error_precursors(
            nogo_continuous, omission_continuous, nogo_trial_lag,
            base=output_dir / "Figure04_error_precursor_trajectories", formats=formats, raster_dpi=raster_dpi,
        ),
        "Figure05_probe_states_trajectories": lambda: figure05_probe_states(
            probe_events, probe_continuous, probe_transitions,
            base=output_dir / "Figure05_probe_states_trajectories", formats=formats, raster_dpi=raster_dpi,
        ),
        "Figure06_visual_PLR_controls": lambda: figure06_visual_controls(
            visual_trial, stimulus_summary, visual_correlations,
            base=output_dir / "Figure06_visual_PLR_controls", formats=formats, raster_dpi=raster_dpi,
        ),
        "Figure07_individual_differences": lambda: figure07_individual_differences(
            heterogeneity, raw_pir,
            base=output_dir / "Figure07_individual_differences", formats=formats, raster_dpi=raster_dpi,
        ),
        "Figure08_feature_structure_multiscale": lambda: figure08_feature_structure(
            feature_within, within_metrics, between_metrics, window_stability,
            base=output_dir / "Figure08_feature_structure_multiscale", formats=formats, raster_dpi=raster_dpi,
        ),
        "Figure09_data_quality_coverage": lambda: figure09_quality_control(
            coverage, source_mode, track=main_track,
            base=output_dir / "Figure09_data_quality_coverage", formats=formats, raster_dpi=raster_dpi,
        ),
        "Figure10_robustness_models": lambda: figure10_robustness(
            track_correlations, track_agreement, model_results,
            base=output_dir / "Figure10_robustness_models", formats=formats, raster_dpi=raster_dpi,
        ),
    }
    outputs: dict[str, list[str]] = {}
    for name, job in jobs.items():
        outputs[name] = job()
    return outputs
