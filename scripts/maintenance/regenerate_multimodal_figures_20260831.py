# -*- coding: utf-8 -*-
"""
文件名：regenerate_multimodal_figures_20260831.py
版本：v1.0
功能：从多模态融合结果重画两张补充图，增强结果直观性：
      （1）图 D：三模态平均边际贡献条形图（NIR / 毫米波 / RGB）；
      （2）图 E：八组合逐参与者 AUC 分布箱线图（M0 行为基准 → M7 完整三模态）。
      不重跑融合模型，仅读取已冻结的结果表。
用法：python regenerate_multimodal_figures_20260831.py
依赖：pandas, numpy, matplotlib
环境：Python 3.14
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# =============================================================================
# 硬编码参数集中声明
# =============================================================================
# 多模态融合结果根路径（本地，不进 Git）
MM_ROOT = r"D:/Project/厚粲杯/11_数据/_FormalAnalysis/MultiModal/full-20260831"
OUT_DIR = os.path.join(MM_ROOT, "figures")

# 组合定义（对应 configs/multimodal_fusion.yaml:93-100）
COMBINATION_LABELS = {
    "M0": "行为基准",
    "M1": "+NIR",
    "M2": "+毫米波",
    "M3": "+RGB",
    "M4": "+NIR+毫米波",
    "M5": "+NIR+RGB",
    "M6": "+毫米波+RGB",
    "M7": "完整三模态",
}

# 模态中文标签与配色
MODALITY_LABELS = {"nir": "NIR 瞳孔", "mmwave": "毫米波", "rgb": "RGB"}
MODALITY_COLORS = {"nir": "#2F5597", "mmwave": "#C55A11", "rgb": "#59A14F"}

# 出版级绘图常量
CM_TO_INCH = 1.0 / 2.54
RASTER_DPI = 600


def _detect_cjk_font() -> str:
    """检测本机可用中文字体，SimSun 优先，回退微软雅黑/黑体。"""
    for name in ("SimSun", "Microsoft YaHei", "SimHei"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "SimSun"


def _configure_style() -> None:
    """应用出版级样式。"""
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
            "legend.fontsize": 7.0,
            "legend.frameon": False,
            "lines.linewidth": 1.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _clean_axis(ax: plt.Axes) -> None:
    """去除上/右边框并外置刻度。"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=2.0)


def _save(fig: plt.Figure, name: str) -> str:
    """保存图到输出目录并关闭 figure。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=RASTER_DPI, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
    return path


# =============================================================================
# 图 D：三模态平均边际贡献条形图
# =============================================================================
def figD_marginal_contribution() -> str:
    """绘制图 D：各模态在已有信息基础上的平均边际贡献（AUC 增量），分结局与模型。"""
    mc = pd.read_csv(os.path.join(MM_ROOT, "marginal", "marginal_contribution.csv"))
    # 只取主指标 roc_auc_ovr_macro
    mc = mc[mc["metric"] == "roc_auc_ovr_macro"]

    fig, axes = plt.subplots(2, 2, figsize=(14 * CM_TO_INCH, 9.0 * CM_TO_INCH))
    panels = [
        ("q1", "logistic", axes[0, 0], "Q1 · 逻辑回归"),
        ("q1", "random_forest", axes[0, 1], "Q1 · 随机森林"),
        ("q2", "logistic", axes[1, 0], "Q2 · 逻辑回归"),
        ("q2", "random_forest", axes[1, 1], "Q2 · 随机森林"),
    ]
    for outcome, model, ax, title in panels:
        sub = mc[(mc["outcome"] == outcome) & (mc["model"] == model)]
        modalities = ["nir", "mmwave", "rgb"]
        values = [sub.loc[sub["modality"] == m, "mean_marginal"].iloc[0] for m in modalities]
        colors = [MODALITY_COLORS[m] for m in modalities]
        bars = ax.bar([MODALITY_LABELS[m] for m in modalities], values, color=colors, width=0.6)
        ax.axhline(0, color="#888888", linewidth=0.7)
        ax.set_ylabel("平均边际贡献（AUC 增量）")
        ax.set_title(title, fontsize=8)
        # 正值负值分别标注数值，便于读
        for bar, v in zip(bars, values):
            offset = 0.002 if v >= 0 else -0.002
            va = "bottom" if v >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, v + offset, f"{v:+.3f}",
                    ha="center", va=va, fontsize=7)
        _clean_axis(ax)

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.08, top=0.94, wspace=0.30, hspace=0.40)
    return _save(fig, "figD_marginal_contribution.png")


# =============================================================================
# 图 E：八组合逐参与者 AUC 分布箱线图
# =============================================================================
def figE_fold_auc_distribution() -> str:
    """绘制图 E：八种组合的逐参与者 AUC 分布，分 Q1 / Q2 两个面板。"""
    pf = pd.read_csv(os.path.join(MM_ROOT, "performance", "performance_by_fold.csv"))
    # 只取 logistic 主模型
    pf = pf[pf["model"] == "logistic"]

    fig, axes = plt.subplots(1, 2, figsize=(17 * CM_TO_INCH, 6.0 * CM_TO_INCH))
    for ax, outcome in zip(axes, ["q1", "q2"]):
        sub = pf[pf["outcome"] == outcome]
        combos = ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7"]
        data = [sub.loc[sub["combination"] == c, "roc_auc_ovr_macro"].dropna() for c in combos]
        positions = np.arange(len(combos))
        bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                        tick_labels=[COMBINATION_LABELS[c] for c in combos])
        # 箱体着色：M0 灰色、单模态浅色、双模态中色、M7 深色
        box_colors = ["#BBBBBB", "#D6E4F0", "#F5D9C7", "#DFF0D8",
                      "#9DC3E6", "#B9D8B0", "#E0B8A0", "#2F5597"]
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        # 标注 M0 基线均值虚线
        m0_median = np.nanmedian(data[0])
        ax.axhline(m0_median, color="#C55A11", linewidth=0.8, linestyle="--")
        ax.text(0.02, m0_median, "M0 基线", transform=ax.get_yaxis_transform(),
                fontsize=6.5, color="#C55A11", va="bottom")
        ax.set_xlabel("模态组合")
        ax.set_ylabel("逐参与者 AUC")
        ax.set_title(f"Q{outcome[1]}", fontsize=8)
        ax.tick_params(axis="x", rotation=30)
        _clean_axis(ax)

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.22, top=0.90, wspace=0.25)
    return _save(fig, "figE_fold_auc_distribution.png")


def main() -> None:
    """生成多模态补充图。"""
    _configure_style()
    outputs = [
        figD_marginal_contribution(),
        figE_fold_auc_distribution(),
    ]
    print("已生成图：")
    for p in outputs:
        print(f"  {p}")


if __name__ == "__main__":
    main()
