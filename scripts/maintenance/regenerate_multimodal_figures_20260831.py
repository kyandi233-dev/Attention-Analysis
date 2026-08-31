# -*- coding: utf-8 -*-
"""
文件名：regenerate_multimodal_figures_20260831.py
版本：v3.0
功能：复用 GitHub 正式样式重画多模态图：
      （1）图 D：三模态平均边际贡献条形图（q1/q2 × logistic/rf）；
      （2）图 E：性能提升阶梯图——M0 行为基准 → M7 完整三模态的 ROC-AUC 与宏 F1 提升轨迹，
           同时呈现逻辑回归与随机森林两种模型，Q1 与 Q2 各一张独立全宽图。
      样式：线细、图例无框且不挡、字号紧凑。
用法：python regenerate_multimodal_figures_20260831.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from attention_pipeline.nir_pipeline_validation.figure_style import (
    configure_publication_style,
    make_figure,
    clean_axis,
    panel_label,
    finalize_layout,
    save_figure,
)

# =============================================================================
# 硬编码参数
# =============================================================================
MM_ROOT = r"D:/Project/厚粲杯/11_数据/_FormalAnalysis/MultiModal/full-20260831"
OUT_DIR = r"D:/Project/厚粲杯/11_数据/_FormalAnalysis/new_figure"
FORMATS = ["png", "svg"]

COMBINATION_LABELS = {
    "M0": "行为基准", "M1": "+NIR", "M2": "+毫米波", "M3": "+RGB",
    "M4": "+NIR+毫米波", "M5": "+NIR+RGB", "M6": "+毫米波+RGB", "M7": "完整三模态",
}
COMBOS = ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7"]
MODALITY_LABELS = {"nir": "NIR 瞳孔", "mmwave": "毫米波", "rgb": "RGB"}
MODALITY_COLORS = {"nir": "#2F5597", "mmwave": "#C55A11", "rgb": "#59A14F"}
MODEL_COLORS = {"logistic": "#2F5597", "random_forest": "#C55A11"}
MODEL_LABELS = {"logistic": "逻辑回归", "random_forest": "随机森林"}


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    return save_figure(fig, Path(OUT_DIR) / name, FORMATS)


def figD_marginal_contribution() -> str:
    """图 D：三模态平均边际贡献条形图（分结局与模型）。"""
    mc = pd.read_csv(os.path.join(MM_ROOT, "marginal", "marginal_contribution.csv"))
    mc = mc[mc["metric"] == "roc_auc_ovr_macro"]
    fig, axes = make_figure(width="full", height_cm=5.2, nrows=2, ncols=2)
    panels = [("q1", "logistic", axes[0, 0], "A"), ("q1", "random_forest", axes[0, 1], "B"),
              ("q2", "logistic", axes[1, 0], "C"), ("q2", "random_forest", axes[1, 1], "D")]
    for outcome, model, ax, lbl in panels:
        sub = mc[(mc["outcome"] == outcome) & (mc["model"] == model)]
        values = [sub.loc[sub["modality"] == m, "mean_marginal"].iloc[0] for m in ["nir", "mmwave", "rgb"]]
        bars = ax.bar([MODALITY_LABELS[m] for m in ["nir", "mmwave", "rgb"]], values,
                      color=[MODALITY_COLORS[m] for m in ["nir", "mmwave", "rgb"]], width=0.55)
        ax.axhline(0, color="#888888", linewidth=0.6)
        ax.set_ylabel("平均边际贡献（AUC 增量）", fontsize=7)
        ax.tick_params(labelsize=6)
        for bar, v in zip(bars, values):
            offset = 0.002 if v >= 0 else -0.002
            ax.text(bar.get_x() + bar.get_width() / 2, v + offset, f"{v:+.3f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=6)
        panel_label(ax, lbl)
        clean_axis(ax)
    finalize_layout(fig, left=0.11, right=0.985, bottom=0.14, top=0.94, wspace=0.34, hspace=0.42)
    return _save(fig, "figD_marginal_contribution")[0]


def _progression_figure(ps, outcome, name):
    """单张全宽图：M0→M7 的 ROC-AUC 与宏 F1 提升轨迹（logistic + rf 两条）。"""
    fig, ax = make_figure(width="full", height_cm=4.6, nrows=1, ncols=1)
    for model in ["logistic", "random_forest"]:
        sub = ps[(ps["outcome"] == outcome) & (ps["model"] == model)]
        auc = [sub.loc[sub["combination"] == c, "roc_auc_ovr_macro_mean"].iloc[0] for c in COMBOS]
        f1 = [sub.loc[sub["combination"] == c, "macro_f1_mean"].iloc[0] for c in COMBOS]
        c = MODEL_COLORS[model]
        ax.plot(range(len(COMBOS)), auc, marker="o", color=c, linewidth=0.9,
                label=f"{MODEL_LABELS[model]} ROC-AUC")
        ax.plot(range(len(COMBOS)), f1, marker="s", color=c, linewidth=0.9, linestyle="--",
                label=f"{MODEL_LABELS[model]} 宏 F1")
    x = range(len(COMBOS))
    ax.set_xticks(x)
    ax.set_xticklabels([COMBINATION_LABELS[c] for c in COMBOS], rotation=30, ha="right", fontsize=6)
    ax.set_ylabel("性能（宏平均）", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.axhline(ps[(ps["outcome"] == outcome) & (ps["model"] == "logistic") & (ps["combination"] == "M0")]["roc_auc_ovr_macro_mean"].iloc[0],
               color="#888888", linewidth=0.6, linestyle=":")
    ax.legend(fontsize=6, frameon=False, loc="lower right", ncol=2)
    panel_label(ax, "A")
    clean_axis(ax)
    finalize_layout(fig, left=0.10, right=0.985, bottom=0.22, top=0.90)
    return _save(fig, name)[0]


def figE_q1_progression(ps) -> str:
    return _progression_figure(ps, "q1", "figE_q1_progression")


def figE_q2_progression(ps) -> str:
    return _progression_figure(ps, "q2", "figE_q2_progression")


def main():
    configure_publication_style()
    plt.rcParams.update({"lines.linewidth": 0.9, "lines.markersize": 2.5, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5})
    ps = pd.read_csv(os.path.join(MM_ROOT, "performance", "performance_summary.csv"))
    outputs = [
        figD_marginal_contribution(),
        figE_q1_progression(ps),
        figE_q2_progression(ps),
    ]
    print("已生成图（new_figure/）：")
    for p in outputs:
        print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
