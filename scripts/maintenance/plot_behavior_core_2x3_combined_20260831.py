"""行为核心结果 2x3 合并总览图（报告 5.2 正文图，附录 C.5 方案 A）。

六面板：
A RT-CV B1/B2 配对；B RT-CV cycle 趋势；C 时序歧义遗漏率 B1/B2；
D 错误事件轨迹（参与者级聚类 mean±SEM）；E Q1 的 No-Go 误按率与 d′ 效应；
F Q2 四项标准化系数森林图。

数据输入：D:/Project/厚粲杯/11_数据/_FormalAnalysis/Behavior/formal_v3/ 的
b1_b2_pairs.csv、b1_b2_participant_cluster_bootstrap.csv、cycle_metrics.csv、
error_event_trajectories.csv、q1_nominal_models.csv、q2_ordinal_gee_models.csv
输出文件：figures_publication/behavior_core_2x3_combined.png（+svg）
项目：厚粲杯 FocusWave | 分析脚本 v1 | 创建日期：2026-08-31
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

DATA = Path(r"D:\Project\厚粲杯\11_数据\_FormalAnalysis\Behavior\formal_v3")
OUT = DATA / "figures_publication"

COLOR_B1 = "#4A7BA6"
COLOR_B2 = "#C2543D"
COLOR_OMISSION = "#C2543D"
COLOR_COMMISSION = "#4A7BA6"
COLOR_DIFF = "#5A6B7B"
GRAY_LINE = "#B9C4CC"

for name in ("SimSun", "Microsoft YaHei", "SimHei"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [name, "Arial"]
        break
plt.rcParams["axes.unicode_minus"] = False


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def panel_label(ax, label: str) -> None:
    ax.text(-0.14, 1.06, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left", color="#1F2D33")


def _clean(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#4A5B66")
    ax.spines["bottom"].set_color("#4A5B66")
    ax.tick_params(colors="#2A3B45", labelsize=8)


def _forest(ax, labels, estimates, lows, highs, colors, xlabel) -> None:
    y = np.arange(len(labels))
    ax.axvline(0, color="#9AA7B0", linewidth=0.9)
    for yi, (est, lo, hi, color) in enumerate(zip(estimates, lows, highs, colors)):
        ax.plot([lo, hi], [yi, yi], color=color, linewidth=1.6,
                solid_capstyle="round")
        ax.plot(est, yi, "o", color=color, markersize=4.6, markeredgecolor="white",
                markeredgewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8.5)
    _clean(ax)
    ax.tick_params(axis="y", length=0)


def main() -> None:
    pairs = pd.read_csv(DATA / "b1_b2_pairs.csv", encoding="utf-8-sig")
    boot = pd.read_csv(DATA / "b1_b2_participant_cluster_bootstrap.csv",
                       encoding="utf-8-sig")
    cycles = pd.read_csv(DATA / "cycle_metrics.csv", encoding="utf-8-sig")
    events = pd.read_csv(DATA / "error_event_trajectories.csv", encoding="utf-8-sig")
    q1 = pd.read_csv(DATA / "q1_nominal_models.csv", encoding="utf-8-sig")
    q2 = pd.read_csv(DATA / "q2_ordinal_gee_models.csv", encoding="utf-8-sig")

    fig = plt.figure(figsize=(7.2, 8.6), dpi=300)
    gs = fig.add_gridspec(3, 2, hspace=0.62, wspace=0.34,
                          left=0.085, right=0.975, top=0.965, bottom=0.055)

    # ---------- A: RT-CV B1/B2 ----------
    ax = fig.add_subplot(gs[0, 0])
    rt = pairs[pairs["metric"].astype(str).eq("go_correct_rt_cv")].copy()
    rt["b1"] = _num(rt["b1_value"])
    rt["b2"] = _num(rt["b2_value"])
    rt = rt.dropna(subset=["b1", "b2"])
    ax.plot([0, 1], rt[["b1", "b2"]].to_numpy().T, color=GRAY_LINE,
            linewidth=0.45, alpha=0.5, zorder=1)
    for x, col in ((0, "b1"), (1, "b2")):
        ax.plot(np.full(len(rt), x), rt[col], "o", markersize=2.6,
                color=GRAY_LINE, alpha=0.65, zorder=2)
        m = rt[col].mean()
        sd = rt[col].std(ddof=1)
        ax.errorbar(x, m, yerr=sd, fmt="o", markersize=5.5,
                    color=(COLOR_B1 if x == 0 else COLOR_B2),
                    ecolor=(COLOR_B1 if x == 0 else COLOR_B2),
                    elinewidth=1.4, capsize=3, zorder=3)
    ax.set_xticks([0, 1], ["B1", "B2"])
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylabel("RT-CV", fontsize=9)
    _clean(ax)
    panel_label(ax, "A")

    # ---------- B: RT-CV cycle ----------
    ax = fig.add_subplot(gs[0, 1])
    cyc = cycles[["block_id", "cycle_bin", "go_correct_rt_cv"]].copy()
    cyc["go_correct_rt_cv"] = _num(cyc["go_correct_rt_cv"])
    for block, color in (("B1", COLOR_B1), ("B2", COLOR_B2)):
        part = cyc[cyc["block_id"].astype(str).eq(block)]
        agg = part.groupby("cycle_bin")["go_correct_rt_cv"].agg(
            mean="mean", sem=lambda x: x.std(ddof=1) / np.sqrt(x.count()))
        x = agg.index.to_numpy(dtype=float)
        ax.plot(x, agg["mean"], color=color, linewidth=1.5, label=block,
                marker="o", markersize=3.4)
        ax.fill_between(x, agg["mean"] - agg["sem"], agg["mean"] + agg["sem"],
                        color=color, alpha=0.16, linewidth=0)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xlabel("周期", fontsize=8.5)
    ax.set_ylabel("RT-CV", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _clean(ax)
    panel_label(ax, "B")

    # ---------- C: 时序歧义遗漏率 B1/B2 ----------
    ax = fig.add_subplot(gs[1, 0])
    om = pairs[pairs["metric"].astype(str).eq("timing_ambiguous_go_omission_rate")].copy()
    om["b1"] = _num(om["b1_value"]) * 100.0
    om["b2"] = _num(om["b2_value"]) * 100.0
    om = om.dropna(subset=["b1", "b2"])
    ax.plot([0, 1], om[["b1", "b2"]].to_numpy().T, color=GRAY_LINE,
            linewidth=0.45, alpha=0.5, zorder=1)
    for x, col in ((0, "b1"), (1, "b2")):
        ax.plot(np.full(len(om), x), om[col], "o", markersize=2.6,
                color=GRAY_LINE, alpha=0.65, zorder=2)
        m = om[col].mean()
        sd = om[col].std(ddof=1)
        ax.errorbar(x, m, yerr=sd, fmt="o", markersize=5.5,
                    color=(COLOR_B1 if x == 0 else COLOR_B2),
                    ecolor=(COLOR_B1 if x == 0 else COLOR_B2),
                    elinewidth=1.4, capsize=3, zorder=3)
    ax.set_xticks([0, 1], ["B1", "B2"])
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylabel("时序歧义遗漏率（%）", fontsize=9)
    _clean(ax)
    panel_label(ax, "C")

    # ---------- D: 错误事件轨迹 ----------
    ax = fig.add_subplot(gs[1, 1])
    part = events.groupby(
        ["repeat_participant_id", "error_type", "relative_trial"]
    )["correct_go_rt_centered_ms"].mean().reset_index()
    for etype, color, label in (
        ("go_omission", COLOR_OMISSION, "Go 遗漏"),
        ("nogo_commission", COLOR_COMMISSION, "No-Go 误按"),
    ):
        sub = part[part["error_type"].astype(str).eq(etype)]
        agg = sub.groupby("relative_trial")["correct_go_rt_centered_ms"].agg(
            mean="mean", sem=lambda x: x.std(ddof=1) / np.sqrt(x.count()))
        x = agg.index.to_numpy(dtype=float)
        ax.errorbar(x, agg["mean"], yerr=agg["sem"], color=color, linewidth=1.5,
                    marker="o", markersize=3.4, elinewidth=1.1, capsize=2.5,
                    label=label)
    ax.axhline(0, color="#9AA7B0", linewidth=0.9)
    ax.axvline(0, color="#D8DFE4", linewidth=0.8, linestyle=(0, (2, 2)))
    ax.set_xticks([-3, -2, -1, 1, 2, 3])
    ax.set_xlabel("事件相对试次", fontsize=8.5)
    ax.set_ylabel("RT 偏移（ms）", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _clean(ax)
    panel_label(ax, "D")

    # ---------- E: Q1 核心效应 ----------
    ax = fig.add_subplot(gs[2, 0])
    rows = []
    for predictor, color, base in (
        ("commission_rate", COLOR_OMISSION, "误按率"),
        ("dprime_loglinear", COLOR_COMMISSION, "d′"),
    ):
        sub = q1[q1["predictor"].astype(str).eq(predictor)]
        for _, row in sub.iterrows():
            cat = str(row["contrast_category"])
            rows.append((f"{base} 类别{cat}比1", float(row["estimate_per_predictor_sd"]),
                         float(row["ci_low"]), float(row["ci_high"]), color))
    labels = [r[0] for r in rows]
    _forest(ax, labels, [r[1] for r in rows], [r[2] for r in rows],
            [r[3] for r in rows], [r[4] for r in rows],
            "标准化系数（95% CI）")
    panel_label(ax, "E")

    # ---------- F: Q2 四项森林图 ----------
    ax = fig.add_subplot(gs[2, 1])
    order = [
        ("commission_rate", "No-Go 误按率"),
        ("dprime_loglinear", "d′"),
        ("clean_go_omission_rate", "无时序歧义遗漏率"),
        ("go_correct_rt_cv", "RT-CV"),
    ]
    labels, ests, lows, highs, colors = [], [], [], [], []
    for predictor, label in order:
        row = q2[q2["predictor"].astype(str).eq(predictor)]
        if row.empty:
            continue
        row = row.iloc[0]
        labels.append(label)
        ests.append(float(row["estimate_per_predictor_sd"]))
        lows.append(float(row["ci_low"]))
        highs.append(float(row["ci_high"]))
        colors.append(COLOR_DIFF)
    _forest(ax, labels, ests, lows, highs, colors, "标准化系数（95% CI）")
    panel_label(ax, "F")

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        out = OUT / f"behavior_core_2x3_combined.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        print(out)
    plt.close(fig)


if __name__ == "__main__":
    main()
