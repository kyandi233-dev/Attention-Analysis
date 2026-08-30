"""生成 5.6 增量森林图：M1-M7 相对 M0 的 ROC-AUC 增量（95% CI）。

用法: python scripts/maintenance/plot_56_incremental_forest_20260831.py
数据: 11_数据/_FormalAnalysis/MultiModal/full-20260831/comparison/incremental_vs_M0.csv
输出: 11_数据/_FormalAnalysis/MultiModal/full-20260831/figures/incremental_forest_56.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATA = Path(r"D:/Project/厚粲杯/11_数据/_FormalAnalysis/MultiModal/full-20260831/comparison/incremental_vs_M0.csv")
OUT = Path(r"D:/Project/厚粲杯/11_数据/_FormalAnalysis/MultiModal/full-20260831/figures")
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["SimSun", "Arial"],
    "axes.unicode_minus": False,
    "savefig.dpi": 300,
})

df = pd.read_csv(DATA)
d = df[(df["model"] == "logistic") & (df["metric"] == "roc_auc_ovr_macro")].copy()
d["outcome_zh"] = d["outcome"].map({"q1": "Q1 注意内容", "q2": "Q2 警觉程度"})
d["comb_zh"] = d["comparison"].map({
    "M1_vs_M0": "+NIR", "M2_vs_M0": "+毫米波", "M3_vs_M0": "+RGB",
    "M4_vs_M0": "NIR+毫米波", "M5_vs_M0": "NIR+RGB",
    "M6_vs_M0": "毫米波+RGB", "M7_vs_M0": "完整三模态",
})

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharex=False)
for ax, outcome in zip(axes, ["q1", "q2"]):
    sub = d[d["outcome"] == outcome].sort_values("mean_diff")
    y = range(len(sub))
    for i, (_, row) in enumerate(sub.iterrows()):
        ci_ok = (row["ci_low"] > 0) or (row["ci_high"] < 0)
        color = "#C0392B" if ci_ok else "#2C3E50"
        ax.errorbar(row["mean_diff"], i, xerr=[[row["mean_diff"] - row["ci_low"]], [row["ci_high"] - row["mean_diff"]]],
                    fmt="o", ms=4, color=color, capsize=3, lw=1.2)
    ax.axvline(0, color="#888888", lw=0.8, ls="--")
    ax.set_yticks(list(y))
    ax.set_yticklabels([sub.iloc[i]["comb_zh"] for i in y], fontsize=8)
    ax.set_xlabel("ROC-AUC 相对基准的增量（95% CI）", fontsize=8)
    ax.set_title(sub.iloc[0]["outcome_zh"], fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_xlim(-0.075, 0.075)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout(w_pad=2.0)
fig.savefig(OUT / "incremental_forest_56.png", bbox_inches="tight")
fig.savefig(OUT / "incremental_forest_56.svg", bbox_inches="tight")
print("已保存:", OUT / "incremental_forest_56.png")
