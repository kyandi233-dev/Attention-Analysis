# -*- coding: utf-8 -*-
"""
文件名：regenerate_rgb_blink_figures_20260831.py
版本：v1.0
功能：补两张 RGB 眨眼率图：
      （1）眨眼率时间进程图（B1/B2 × 探针顺序）—— 呼应 5.5.1 的动程轨迹；
      （2）眨眼率与 Q1/Q2/行为/毫米波结局的效应森林图 —— 呼应 5.5.3 的关联。
      复用 GitHub 正式样式（figure_style.py），Paul Tol muted 配色。
用法：python regenerate_rgb_blink_figures_20260831.py
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

RGB_FEAT = r"D:/Project/厚粲杯/11_数据/_FormalAnalysis/RGB/11_analysis_tables_116cohort/rgb_probe_pre30s_features.csv"
RGB_MODELS = r"D:/Project/厚粲杯/11_数据/_FormalAnalysis/RGB/21_analysis_tables_5.5/models"
NIR_TABLES = r"D:/Project/厚粲杯/11_数据/_FormalAnalysis/NIR/11_analysis_tables"
OUT_DIR = r"D:/Project/厚粲杯/11_数据/_FormalAnalysis/new_figure"
FORMATS = ["png", "svg"]

PTOL = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"]
LW = {"line": 0.9, "err": 0.7, "frame": 0.6}


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    return save_figure(fig, Path(OUT_DIR) / name, FORMATS)


def fig_blink_timecourse():
    """眨眼率随 B1/B2 × 探针顺序的时间进程轨迹。"""
    f = pd.read_csv(RGB_FEAT)
    g = f.groupby(["block_id", "probe_order_in_block"])["blink_event_rate_per_min"].agg(["mean", "sem"]).reset_index()
    fig, ax = make_figure(width="full", height_cm=4.6, nrows=1, ncols=1)
    for i, (blk, c) in enumerate(zip(["B1", "B2"], PTOL[:2])):
        sub = g[g["block_id"] == blk].sort_values("probe_order_in_block")
        x = sub["probe_order_in_block"]
        ax.plot(x, sub["mean"], color=c, marker="o", markersize=3, linewidth=LW["line"],
                label=f"{blk} 区块")
        ax.fill_between(x, sub["mean"] - 1.96 * sub["sem"], sub["mean"] + 1.96 * sub["sem"],
                        color=c, alpha=0.14, linewidth=0)
    ax.set_xlabel("探针顺序（区内第 N 次）", fontsize=8)
    ax.set_ylabel("眨眼率（次/分）", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xticks(range(1, 11))
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), fontsize=7, frameon=False)
    panel_label(ax, "A")
    clean_axis(ax)
    finalize_layout(fig, left=0.11, right=0.985, bottom=0.18, top=0.88)
    return _save(fig, "fig_blink_timecourse")[0]


def _load_gee(path, predict_col):
    df = pd.read_csv(path)
    return df


def fig_blink_association():
    """眨眼率与 Q2 警觉 / 行为 / 毫米波结局的效应森林图。"""
    # 从各 GEE 表提取 blink 行的 estimate/CI
    rows = []
    q2 = pd.read_csv(os.path.join(RGB_MODELS, "rgb_q2_ordinal_gee.csv"))
    q2r = q2[q2["model_name"].str.contains("blink", na=False)]
    if len(q2r):
        r = q2r.iloc[0]
        rows.append(("Q2 警觉程度", r["estimate_per_predictor_sd"], r["ci_low"], r["ci_high"]))
    beh = pd.read_csv(os.path.join(RGB_MODELS, "rgb_behavior_window_gee.csv"))
    beh_map = {"omission_rate": "Go 遗漏率", "commission_rate": "No-Go 误按率",
               "dprime_loglinear": "辨别力 d′", "go_correct_rt_median_ms": "正确 Go RT 中位数"}
    for oc, zh in beh_map.items():
        r = beh[(beh["outcome"] == oc) & (beh["model_name"].str.contains("blink", na=False))]
        if len(r):
            r = r.iloc[0]
            rows.append((zh, r["estimate_per_predictor_sd"], r["ci_low"], r["ci_high"]))
    mm = pd.read_csv(os.path.join(RGB_MODELS, "rgb_mmwave_window_gee.csv"))
    mm_map = {"mmwave_motion_proxy_median": "毫米波运动代理"}
    for oc, zh in mm_map.items():
        r = mm[(mm["outcome"] == oc) & (mm["model_name"].str.contains("blink", na=False))]
        if len(r):
            r = r.iloc[0]
            rows.append((zh, r["estimate_per_predictor_sd"], r["ci_low"], r["ci_high"]))

    fig, ax = make_figure(width="full", height_cm=4.6, nrows=1, ncols=1)
    y_pos = list(range(len(rows)))[::-1]
    for y, (name, est, lo, hi) in zip(y_pos, rows):
        ax.errorbar(est, y, xerr=[[est - lo], [hi - est]], fmt="o", color=PTOL[0],
                    markersize=4, elinewidth=LW["err"], capsize=2.5)
    ax.axvline(0, color="#888888", linewidth=0.6, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([name for name, _, _, _ in rows], fontsize=7)
    ax.set_xlabel("效应估计（每 1 SD 眨眼率，95% CI）", fontsize=8)
    ax.tick_params(labelsize=7)
    panel_label(ax, "A")
    clean_axis(ax)
    finalize_layout(fig, left=0.22, right=0.985, bottom=0.16, top=0.90)
    return _save(fig, "fig_blink_association")[0]


def main():
    configure_publication_style()
    plt.rcParams.update({"lines.linewidth": LW["line"], "lines.markersize": 3, "axes.linewidth": LW["frame"],
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5})
    outputs = [fig_blink_timecourse(), fig_blink_association()]
    print("已生成图（new_figure/）：")
    for p in outputs:
        print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
