from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .figure_style import PALETTE, clean_axis, finalize_layout, make_figure, panel_label, save_figure
from .publication_figures import _label_zh


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _empty(ax: plt.Axes, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _condition_trajectory(ax: plt.Axes, frame: pd.DataFrame, value_col: str, ylabel: str) -> None:
    if frame.empty or value_col not in frame.columns:
        _empty(ax, f"无 {value_col} 轨迹")
        return
    for idx, (condition, current) in enumerate(frame.groupby("event_condition", sort=True)):
        subject = current.groupby(["subject", "time_bin_mid_sec"], as_index=False)[value_col].median()
        summary = subject.groupby("time_bin_mid_sec")[value_col].agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        ).sort_index()
        color = PALETTE.get(str(condition), plt.get_cmap("tab10")(idx % 10))
        ax.plot(summary.index, summary["median"], color=color, label=_label_zh(condition))
        ax.fill_between(summary.index, summary["q25"], summary["q75"], color=color, alpha=0.10, linewidth=0)
    ax.axvline(0, color="#777777", linestyle=":", linewidth=0.65)
    ax.set_xlabel("相对事件时间（s）")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    clean_axis(ax, grid_y=True)


def supplementary01_error_dynamics(
    nogo_continuous: pd.DataFrame,
    omission_continuous: pd.DataFrame,
    nogo_trial_lag: pd.DataFrame,
    *,
    base: Path,
    formats: Iterable[str],
    raster_dpi: int,
) -> list[str]:
    fig, axes = make_figure(width="full", height_cm=15.0, nrows=2, ncols=2)
    _condition_trajectory(axes[0, 0], nogo_continuous, "pupil_sd", "分箱内 PIR SD")
    axes[0, 0].set_title("No-Go 前兆 PIR 变异性")
    panel_label(axes[0, 0], "A")

    _condition_trajectory(axes[0, 1], omission_continuous, "pupil_sd", "分箱内 PIR SD")
    axes[0, 1].set_title("遗漏前兆 PIR 变异性")
    panel_label(axes[0, 1], "B")

    ax = axes[1, 0]
    if nogo_trial_lag.empty or "go_rt_ms" not in nogo_trial_lag.columns:
        _empty(ax, "无试次滞后 RT")
    else:
        for condition, current in nogo_trial_lag.groupby("event_outcome", sort=True):
            subject = current.groupby(["subject", "lag"], as_index=False)["go_rt_ms"].median()
            summary = subject.groupby("lag")["go_rt_ms"].agg(median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)).sort_index()
            color = PALETTE.get(str(condition), "#666666")
            ax.plot(summary.index, summary["median"], marker="o", color=color, label=_label_zh(condition))
            ax.fill_between(summary.index, summary["q25"], summary["q75"], color=color, alpha=0.10, linewidth=0)
        ax.axvline(0, color="#777777", linestyle=":", linewidth=0.65)
        ax.set_xlabel("相对 No-Go 的试次滞后")
        ax.set_ylabel("正确 Go RT（ms）")
        ax.legend(loc="best")
        clean_axis(ax, grid_y=True)
    ax.set_title("行为速度前兆")
    panel_label(ax, "C")

    ax = axes[1, 1]
    if nogo_trial_lag.empty or "pupil_mad" not in nogo_trial_lag.columns:
        _empty(ax, "无试次滞后 PIR 变异性")
    else:
        for condition, current in nogo_trial_lag.groupby("event_outcome", sort=True):
            subject = current.groupby(["subject", "lag"], as_index=False)["pupil_mad"].median()
            summary = subject.groupby("lag")["pupil_mad"].median().sort_index()
            color = PALETTE.get(str(condition), "#666666")
            ax.plot(summary.index, summary.values, marker="o", color=color, label=_label_zh(condition))
        ax.axvline(0, color="#777777", linestyle=":", linewidth=0.65)
        ax.set_xlabel("相对 No-Go 的试次滞后")
        ax.set_ylabel("前导试次 PIR MAD")
        ax.legend(loc="best")
        clean_axis(ax, grid_y=True)
    ax.set_title("离散试次级 PIR 变异性前兆")
    panel_label(ax, "D")

    finalize_layout(fig, wspace=0.32, hspace=0.44)
    return save_figure(fig, base, formats, raster_dpi=raster_dpi)


def _probe_metric_panel(ax: plt.Axes, frame: pd.DataFrame, metric: str, ylabel: str) -> None:
    if frame.empty or metric not in frame.columns or "probe_response" not in frame.columns:
        _empty(ax, f"无 {metric}")
        return
    df = frame.copy()
    df["window_sec"] = df["window_name"].astype(str).str.extract(r"pre_(\d+(?:\.\d+)?)s", expand=False)
    df["window_sec"] = _numeric(df["window_sec"])
    df[metric] = _numeric(df[metric])
    df = df.dropna(subset=["window_sec", metric, "probe_response"])
    for idx, ((block_num, response), current) in enumerate(df.groupby(["block_num", "probe_response"], sort=True)):
        subject = current.groupby(["subject", "window_sec"], as_index=False)[metric].median()
        summary = subject.groupby("window_sec")[metric].median().sort_index()
        color = plt.get_cmap("tab10")(idx % 10)
        style = "-" if int(block_num) == 1 else "--"
        ax.plot(summary.index, summary.values, marker="o", color=color, linestyle=style, label=f"B{int(block_num)} / 作答 {response}")
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 30, 60], ["10", "20", "30", "60"])
    ax.set_xlabel("探针前窗口（s；对数轴）")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", ncol=2, fontsize=5.8)
    clean_axis(ax, grid_y=True)


def supplementary02_probe_objective_behavior(
    probe_behavior: pd.DataFrame,
    *,
    base: Path,
    formats: Iterable[str],
    raster_dpi: int,
) -> list[str]:
    df = probe_behavior.copy()
    if not df.empty:
        for num, den, name in (
            ("n_commission", "n_nogo", "commission_rate"),
            ("n_omission", "n_go", "omission_rate"),
            ("n_ambiguous_omission", "n_go", "ambiguous_omission_rate"),
            ("n_anticipatory_candidate", "n_go", "anticipatory_rate"),
        ):
            if num in df.columns and den in df.columns:
                numerator = _numeric(df[num])
                denominator = _numeric(df[den])
                df[name] = np.where(denominator > 0, numerator / denominator, np.nan)

    fig, axes = make_figure(width="full", height_cm=15.0, nrows=2, ncols=2)
    _probe_metric_panel(axes[0, 0], df, "go_rt_cv", "Go RT-CV")
    axes[0, 0].set_title("探针前 RT 变异性")
    panel_label(axes[0, 0], "A")
    _probe_metric_panel(axes[0, 1], df, "commission_rate", "误按率")
    axes[0, 1].set_title("探针前抑制错误")
    panel_label(axes[0, 1], "B")
    _probe_metric_panel(axes[1, 0], df, "omission_rate", "程序遗漏率")
    axes[1, 0].set_title("探针前遗漏")
    panel_label(axes[1, 0], "C")
    _probe_metric_panel(axes[1, 1], df, "anticipatory_rate", "预判候选率")
    axes[1, 1].set_title("探针前动作时序表型")
    panel_label(axes[1, 1], "D")
    finalize_layout(fig, wspace=0.34, hspace=0.44)
    return save_figure(fig, base, formats, raster_dpi=raster_dpi)


def _probe_rt_box(ax: plt.Axes, frame: pd.DataFrame, value_col: str, ylabel: str) -> None:
    if frame.empty or value_col not in frame.columns or "probe_response" not in frame.columns:
        _empty(ax, f"无 {value_col}")
        return
    subject = frame.dropna(subset=["probe_response", value_col]).groupby(["subject", "block_num", "probe_response"], as_index=False)[value_col].median()
    options = sorted(subject["probe_response"].dropna().unique())
    offsets = {1: -0.16, 2: 0.16}
    for block_num in (1, 2):
        color = PALETTE[f"block{block_num}"]
        for idx, option in enumerate(options):
            values = _numeric(subject.loc[subject["block_num"].eq(block_num) & subject["probe_response"].eq(option), value_col]).dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            pos = idx + 1 + offsets[block_num]
            ax.boxplot([values], positions=[pos], widths=0.26, showfliers=False, manage_ticks=False)
            rng = np.random.default_rng(1200 + block_num * 100 + idx)
            ax.scatter(pos + rng.normal(0, 0.025, len(values)), values, s=7, color=color, alpha=0.55, linewidths=0)
    ax.set_xticks(np.arange(1, len(options) + 1), [str(x) for x in options])
    ax.set_xlabel("probe_response 原始编码")
    ax.set_ylabel(ylabel)
    clean_axis(ax, grid_y=True)


def supplementary03_probe_response_times(
    probe_rt: pd.DataFrame,
    *,
    base: Path,
    formats: Iterable[str],
    raster_dpi: int,
) -> list[str]:
    fig, axes = make_figure(width="full", height_cm=13.5, nrows=1, ncols=2)
    _probe_rt_box(axes[0], probe_rt, "probe_rt", "probe_response RT（ms）")
    axes[0].set_title("探针作答决策时间 × 区块")
    panel_label(axes[0], "A")
    _probe_rt_box(axes[1], probe_rt, "probe_vigilance_rt", "probe_vigilance RT（ms）")
    axes[1].set_title("探针警觉决策时间 × 区块")
    panel_label(axes[1], "B")
    finalize_layout(fig, wspace=0.34, hspace=0.30)
    return save_figure(fig, base, formats, raster_dpi=raster_dpi)


def _stimulus_heatmap(ax: plt.Axes, visual_trial: pd.DataFrame, value_col: str, title: str) -> None:
    if visual_trial.empty or not {"stimulus_name", "stimulus_size", value_col}.issubset(visual_trial.columns):
        _empty(ax, f"无 {value_col}")
        return
    df = visual_trial.copy()
    df[value_col] = _numeric(df[value_col])
    matrix = df.pivot_table(index="stimulus_name", columns="stimulus_size", values=value_col, aggfunc="median")
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(matrix.columns)), [str(x) for x in matrix.columns])
    ax.set_yticks(np.arange(len(matrix.index)), [str(x).replace(".png", "") for x in matrix.index])
    ax.set_xlabel("刺激大小（%）")
    ax.set_title(title)
    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    cbar.ax.tick_params(labelsize=6)


def supplementary04_stimulus_identity_size(
    visual_trial: pd.DataFrame,
    *,
    base: Path,
    formats: Iterable[str],
    raster_dpi: int,
) -> list[str]:
    fig, axes = make_figure(width="full", height_cm=17.0, nrows=2, ncols=2)
    _stimulus_heatmap(axes[0, 0], visual_trial, "pupil_median", "按刺激身份/大小的试次前 PIR")
    panel_label(axes[0, 0], "A")
    _stimulus_heatmap(axes[0, 1], visual_trial, "current_central_rel_lum_mean", "中央相对亮度")
    panel_label(axes[0, 1], "B")
    _stimulus_heatmap(axes[1, 0], visual_trial, "current_central_rms_contrast", "中央 RMS 对比度")
    panel_label(axes[1, 0], "C")
    _stimulus_heatmap(axes[1, 1], visual_trial, "current_fruit_visible_area_fraction_central_roi", "可见面积占比")
    panel_label(axes[1, 1], "D")
    finalize_layout(fig, left=0.15, wspace=0.46, hspace=0.50)
    return save_figure(fig, base, formats, raster_dpi=raster_dpi)
