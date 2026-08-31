# -*- coding: utf-8 -*-
"""
文件名：regenerate_nir_figures_pupil_only_20260831.py
版本：v1.0
功能：从正式结果表与 session 级数据重画 NIR 瞳孔结果的四张正文图（图 7-10）与一张新增热力图，
      统一采用 pupil-only 直接瞳孔几何口径（瞳孔几何平均直径 + 瞳孔面积比例），
      去除已废止的 PIR 标签与中英混杂术语，纵轴标注单位。
用法：python regenerate_nir_figures_pupil_only_20260831.py
依赖：pandas, numpy, matplotlib, scipy
环境：使用 Python 3.14（anaconda 环境 numpy 会静默崩溃 0xc06d007f）
"""

from __future__ import annotations

import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import spearmanr

# =============================================================================
# 硬编码参数集中声明
# =============================================================================
# 数据根路径（本地，不进 Git）
NIR_TABLES = r"D:/Project/厚粲杯/11_数据/_FormalAnalysis/NIR/11_analysis_tables"
# 输出目录：重画后的 pupil-only 正式图
OUT_DIR = os.path.join(NIR_TABLES, "formal_figures", "final_pupil_only")

# 瞳孔主指标术语表（报告/图/附录统一口径，去 PIR）
METRIC_LABEL_ZH = {
    "pupil_geom_mean_diameter": "瞳孔几何平均直径",
    "hard_pupil_fraction": "瞳孔面积比例",
}
# 图 8/图 9 的 pupil_median 术语：底层即主指标几何平均直径，经场次×眼别基线中心化后含正负
PUPIL_MEDIAN_LABEL = "瞳孔几何平均直径（基线中心化，px）"

# Q1 注意内容（probe_response）四类中文标签
Q1_LABELS = {
    1.0: "完全专注",
    2.0: "在任务上没想目标",
    3.0: "走神",
    4.0: "大脑空白",
}
# Q2 警觉程度（probe_vigilance）四级中文标签
Q2_LABELS = {
    1.0: "非常困倦",
    2.0: "比较困倦",
    3.0: "比较清醒",
    4.0: "非常清醒",
}

# 试次级结局中文标签（图 10）
OUTCOME_LABEL_ZH = {
    "rt": "正确 Go 反应时",
    "omission": "Go 遗漏",
    "commission": "No-Go 误按",
}
# 参与者间/参与者内分量标签
PUPIL_TERM_LABEL_ZH = {
    "pupil_between": "参与者间",
    "pupil_within": "参与者内",
}

# 热力图选用的行为指标（对应多模态 M0 基准的 7 个行为特征）
BEHAVIOR_METRICS = [
    ("dprime", "辨别力 d′"),
    ("go_rt_median_ms", "正确 Go RT 中位数"),
    ("rt_cv", "RT 变异系数"),
    ("commission_rate", "No-Go 误按率"),
    ("clean_omission_rate", "真遗漏率"),
    ("program_omission_rate", "预判遗漏率"),
    ("ambiguous_omission_rate", "时序模糊遗漏率"),
]

# 出版级绘图常量
CM_TO_INCH = 1.0 / 2.54
RASTER_DPI = 600


# =============================================================================
# 字体与样式
# =============================================================================
def _detect_cjk_font() -> str:
    """检测本机可用中文字体，SimSun 优先，回退微软雅黑/黑体。"""
    for name in ("SimSun", "Microsoft YaHei", "SimHei"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "SimSun"


def _configure_style() -> str:
    """应用出版级样式，返回检测到的中文字体名。"""
    cjk = _detect_cjk_font()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [cjk, "Arial"],
            "axes.unicode_minus": False,
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.fontsize": 7.0,
            "legend.frameon": False,
            "lines.linewidth": 1.25,
            "lines.markersize": 4.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return cjk


def _clean_axis(ax: plt.Axes) -> None:
    """去除上/右边框并外置刻度。"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=2.0)


def _save(fig: plt.Figure, name: str) -> str:
    """保存图到输出目录并关闭 figure，返回输出路径。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=RASTER_DPI, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
    return path


# =============================================================================
# 数据加载
# =============================================================================
def _load_probe_windows() -> pd.DataFrame:
    """拼接全部 session 的 probe_pupil_windows.csv，仅保留 binocular_primary 主轨道的 pre_30s 窗口。

    返回 2180 行（109 场 × 20 探针）的 DataFrame，用于图 7 与图 9。
    """
    files = sorted(glob(os.path.join(NIR_TABLES, "sessions", "sub-*", "*_probe_pupil_windows.csv")))
    frames = []
    for f in files:
        df = pd.read_csv(f)
        # 只取主轨道（binocular_primary）与探针前 30s 窗口，与正式探针分析口径一致
        df = df[(df["track"] == "binocular_primary") & (df["window_name"] == "pre_30s")]
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    return full


def _load_visual_conditions() -> pd.DataFrame:
    """加载试次级视觉刺激条件汇总（stimulus_condition_summary.csv），用于图 8。"""
    path = os.path.join(NIR_TABLES, "tables", "publication_analysis", "stimulus_condition_summary.csv")
    return pd.read_csv(path)


def _load_trial_effects() -> pd.DataFrame:
    """加载试次级瞳孔效应表（trial_unadjusted_adjusted_effects.csv），用于图 10。"""
    path = os.path.join(NIR_TABLES, "reference_adjusted_models", "trial_unadjusted_adjusted_effects.csv")
    return pd.read_csv(path)


def _load_heatmap_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """构建热力图数据：两主指标与 7 个行为指标的参与者间 Spearman 相关矩阵。

    步骤：读候选指标（session×block）与行为指标（subject×block）→ 按 session_id↔subject 连接 →
    按 analysis_group_token 折叠重复场次（同一参与者多次到场）→ 计算 Spearman。
    返回 (left_matrix, right_matrix, behavior_labels)。
    """
    # 瞳孔主指标：session×block 层，取双眼原始中位数
    cand_path = os.path.join(NIR_TABLES, "candidate_validation", "nir_candidate_session_block_metrics.csv")
    cand = pd.read_csv(cand_path)
    cand = cand[cand["metric"].isin(["pupil_geom_mean_diameter", "hard_pupil_fraction"])]
    # 透视成 每个 (session, block) 一行、两个主指标两列
    pupil_wide = cand.pivot_table(
        index=["session_id", "block_num", "analysis_group_token"],
        columns="metric",
        values="binocular_raw_median",
    ).reset_index()

    # 行为指标：subject×block 层
    beh_path = os.path.join(NIR_TABLES, "tables", "publication_analysis", "advanced_behavior_subject_block.csv")
    beh = pd.read_csv(beh_path)

    # 连接：session_id（candidate）↔ subject（behavior）+ block_num
    merged = pupil_wide.merge(
        beh,
        left_on=["session_id", "block_num"],
        right_on=["subject", "block_num"],
        how="inner",
    )

    # 折叠重复场次：同一 analysis_group_token 的多次到场取均值（每个参与者一个代表值）
    group_cols = ["analysis_group_token", "block_num"]
    value_cols = [
        "pupil_geom_mean_diameter",
        "hard_pupil_fraction",
    ] + [m for m, _ in BEHAVIOR_METRICS]
    collapsed = merged.groupby(group_cols)[value_cols].mean().reset_index()

    # 计算 Spearman 相关：两主指标分别与 7 个行为指标
    behavior_labels = [label for _, label in BEHAVIOR_METRICS]
    left_row, right_row = [], []
    for m, _ in BEHAVIOR_METRICS:
        rl = spearmanr(collapsed["pupil_geom_mean_diameter"], collapsed[m]).correlation
        rr = spearmanr(collapsed["hard_pupil_fraction"], collapsed[m]).correlation
        left_row.append(rl)
        right_row.append(rr)
    return np.array(left_row), np.array(right_row), behavior_labels


# =============================================================================
# 图 7：瞳孔数据质量与覆盖
# =============================================================================
def fig07_data_quality(df: pd.DataFrame) -> str:
    """绘制图 7：探针前 30s 窗口的瞳孔数据质量与覆盖（三个面板）。"""
    fig, axes = plt.subplots(1, 3, figsize=(17 * CM_TO_INCH, 5.2 * CM_TO_INCH))

    panels = [
        ("pupil_valid_fraction", "有效瞳孔帧比例", (0, 1)),
        ("internal_coverage_fraction", "窗口内部时间覆盖", (0, 1)),
        ("source_mode_binocular_fraction", "双眼轨道可用比例", (0, 1)),
    ]
    for ax, (col, xlabel, xlim) in zip(axes, panels):
        vals = df[col].dropna()
        ax.hist(vals, bins=40, color="#4C78A8", edgecolor="white", linewidth=0.3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("观测数")
        ax.set_xlim(*xlim)
        _clean_axis(ax)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.90, wspace=0.32)
    return _save(fig, "fig07_pupil_data_quality.png")


# =============================================================================
# 图 8：视觉刺激属性与瞳孔变化
# =============================================================================
def fig08_visual_controls(vis: pd.DataFrame) -> str:
    """绘制图 8：刺激尺寸、亮度、对比度、可见面积与瞳孔大小的关系（四个面板）。"""
    fig, axes = plt.subplots(1, 4, figsize=(17 * CM_TO_INCH, 4.8 * CM_TO_INCH))

    # A：不同刺激尺寸下的瞳孔中位数分布
    ax = axes[0]
    sizes = sorted(vis["stimulus_size"].dropna().unique())
    data_by_size = [vis.loc[vis["stimulus_size"] == s, "pupil_median"].dropna() for s in sizes]
    ax.boxplot(data_by_size, tick_labels=[f"{s:.2f}" for s in sizes], widths=0.6)
    ax.set_xlabel("刺激尺寸")
    ax.set_ylabel(PUPIL_MEDIAN_LABEL)
    _clean_axis(ax)

    # B/C/D：亮度、对比度、可见面积与瞳孔的分箱散点
    scatter_panels = [
        ("current_central_rel_lum_mean", "数字相对亮度", axes[1]),
        ("current_central_rms_contrast", "RMS 对比度", axes[2]),
        ("current_fruit_visible_area_fraction_central_roi", "刺激可见面积比例", axes[3]),
    ]
    for col, xlabel, ax in scatter_panels:
        x = vis[col].dropna()
        y = vis.loc[x.index, "pupil_median"]
        ax.scatter(x, y, s=6, alpha=0.4, color="#2F5597", edgecolors="none")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(PUPIL_MEDIAN_LABEL)
        _clean_axis(ax)

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.90, wspace=0.42)
    return _save(fig, "fig08_visual_controls.png")


# =============================================================================
# 图 9：探针状态与瞳孔（探针前 30s 窗口，Q1 与 Q2 双面板）
# =============================================================================
def fig09_probe_states(df: pd.DataFrame) -> str:
    """绘制图 9：探针前 30s 窗口瞳孔大小按 Q1 注意内容与 Q2 警觉程度的分组分布。"""
    fig, axes = plt.subplots(1, 2, figsize=(14 * CM_TO_INCH, 5.0 * CM_TO_INCH))

    # 左面板：按 Q1 注意内容（probe_response）分组
    ax = axes[0]
    q1_data = []
    q1_ticks = []
    for code in sorted(df["probe_response"].dropna().unique()):
        q1_data.append(df.loc[df["probe_response"] == code, "pupil_median"].dropna())
        q1_ticks.append(Q1_LABELS.get(code, f"类别{int(code)}"))
    ax.boxplot(q1_data, tick_labels=q1_ticks, widths=0.55)
    ax.set_ylabel(PUPIL_MEDIAN_LABEL)
    ax.set_title("Q1 注意内容", fontsize=8)
    ax.tick_params(axis="x", rotation=15)
    _clean_axis(ax)

    # 右面板：按 Q2 警觉程度（probe_vigilance）分组
    ax = axes[1]
    q2_data = []
    q2_ticks = []
    for code in sorted(df["probe_vigilance"].dropna().unique()):
        q2_data.append(df.loc[df["probe_vigilance"] == code, "pupil_median"].dropna())
        q2_ticks.append(Q2_LABELS.get(code, f"级别{int(code)}"))
    ax.boxplot(q2_data, tick_labels=q2_ticks, widths=0.55)
    ax.set_ylabel(PUPIL_MEDIAN_LABEL)
    ax.set_title("Q2 警觉程度", fontsize=8)
    ax.tick_params(axis="x", rotation=15)
    _clean_axis(ax)

    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.24, top=0.88, wspace=0.35)
    return _save(fig, "fig09_probe_states.png")


# =============================================================================
# 图 10：试次级效应分解图（3 结局 × 参与者间/参与者内 × 调整态）
# =============================================================================
def fig10_trial_effect_forest(effects: pd.DataFrame) -> str:
    """绘制图 10：试次级瞳孔效应分解森林图（正确 Go RT / Go 遗漏 / No-Go 误按）。"""
    fig, axes = plt.subplots(1, 3, figsize=(17 * CM_TO_INCH, 5.5 * CM_TO_INCH))

    outcomes = ["rt", "omission", "commission"]
    # 每个面板内固定 4 行：参与者内/参与者间 × 未调整/调整后
    rows = [
        ("pupil_within", False),
        ("pupil_within", True),
        ("pupil_between", False),
        ("pupil_between", True),
    ]
    row_labels = [
        "参与者内·未调整",
        "参与者内·调整后",
        "参与者间·未调整",
        "参与者间·调整后",
    ]

    for ax, outcome in zip(axes, outcomes):
        sub = effects[effects["outcome"] == outcome]
        y_positions = list(range(len(rows)))[::-1]
        for y, (term, adj) in zip(y_positions, rows):
            r = sub[(sub["pupil_term"] == term) & (sub["adjusted"] == adj)]
            if r.empty:
                continue
            r = r.iloc[0]
            est, lo, hi = r["estimate"], r["ci_low"], r["ci_high"]
            # 参与者间用实心、参与者内用空心，区分分量
            is_between = term == "pupil_between"
            marker = "o" if is_between else "s"
            facecolor = "#2F5597" if is_between else "white"
            ax.errorbar(
                est, y, xerr=[[est - lo], [hi - est]],
                fmt=marker, color="#2F5597", markersize=4.5,
                markerfacecolor=facecolor, markeredgewidth=1.0,
                elinewidth=1.0, capsize=2.5,
            )
        ax.axvline(0, color="#888888", linewidth=0.7, linestyle="--")
        ax.set_yticks(y_positions)
        ax.set_yticklabels(row_labels)
        ax.set_title(OUTCOME_LABEL_ZH[outcome], fontsize=8)
        ax.set_xlabel("效应估计（95% CI）")
        _clean_axis(ax)

    # 图内脚注：说明瞳孔信号口径（底层即几何平均直径，中心化）
    fig.text(
        0.5, 0.015,
        "瞳孔信号为基线中心化的瞳孔几何平均直径（双眼融合参考轨道）；详见方法 4.5.2",
        ha="center", va="bottom", fontsize=6.5, color="#555555",
    )
    fig.subplots_adjust(left=0.22, right=0.99, bottom=0.20, top=0.90, wspace=0.32)
    return _save(fig, "fig10_trial_effect_forest.png")


# =============================================================================
# 新增热力图：两主指标 × 行为指标
# =============================================================================
def fig_heatmap_pupil_behavior(left_row, right_row, behavior_labels) -> str:
    """绘制热力图：左为瞳孔几何平均直径、右为瞳孔面积比例，分别与行为指标的 Spearman 相关。"""
    fig, axes = plt.subplots(1, 2, figsize=(14 * CM_TO_INCH, 5.0 * CM_TO_INCH))

    for ax, values, title in zip(
        axes,
        [left_row, right_row],
        [METRIC_LABEL_ZH["pupil_geom_mean_diameter"], METRIC_LABEL_ZH["hard_pupil_fraction"]],
    ):
        # 用 diverging 色标：负相关蓝、正相关红、零白
        vmax = max(0.2, np.nanmax(np.abs(values)))
        im = ax.imshow(
            values.reshape(1, -1), cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            aspect="auto",
        )
        ax.set_yticks([0])
        ax.set_yticklabels([title])
        ax.set_xticks(range(len(behavior_labels)))
        ax.set_xticklabels(behavior_labels, rotation=30, ha="right")
        # 每个格子标注数值
        for j, v in enumerate(values):
            ax.text(j, 0, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="black" if abs(v) < 0.5 * vmax else "white")
        # 每个面板附独立色标，标明相关系数范围
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        _clean_axis(ax)

    fig.subplots_adjust(left=0.20, right=0.90, bottom=0.30, top=0.85, wspace=0.35)
    return _save(fig, "fig_heatmap_pupil_behavior.png")


# =============================================================================
# 主流程
# =============================================================================
def main() -> None:
    """加载数据并生成全部 NIR 图。"""
    _configure_style()

    print("加载 probe 窗口数据（图 7 / 图 9）...")
    probe_df = _load_probe_windows()
    print(f"  probe 窗口行数: {len(probe_df)}")

    print("加载视觉刺激条件（图 8）...")
    vis_df = _load_visual_conditions()

    print("加载试次级效应（图 10）...")
    effects_df = _load_trial_effects()

    print("构建热力图数据...")
    left_row, right_row, beh_labels = _load_heatmap_data()

    outputs = [
        fig07_data_quality(probe_df),
        fig08_visual_controls(vis_df),
        fig09_probe_states(probe_df),
        fig10_trial_effect_forest(effects_df),
        fig_heatmap_pupil_behavior(left_row, right_row, beh_labels),
    ]
    print("已生成图：")
    for p in outputs:
        print(f"  {p}")


if __name__ == "__main__":
    main()
