"""用11人预实验行为数据生成正式实验修改建议报告与嵌入图表。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(r"D:\AAAWORK\07-竞赛\厚璨杯\021-analysisplan\attention-pipeline-v2")
DATA = Path(r"D:\_AttentionData\output-v2\040-pre-experiment\040-behavior")
OUT = ROOT / "docs" / "040-behavior"
PLOTS = OUT / "plots"
REPORT = OUT / "014-正式实验修改建议报告.md"
TRIALS_CSV = DATA / "041-trials.csv"
BLOCKS_CSV = DATA / "042-block_metrics.csv"
SEED = 20260813
BLOCK_ORDER = ["A", "B", "C", "C", "B", "A"]
COLORS = {"A": "#d95f59", "B": "#3478bf", "C": "#5b9b62"}


def setup_style() -> None:
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 130,
        "savefig.dpi": 180,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def ci95(values: pd.Series | np.ndarray) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    se = stats.sem(x)
    q = stats.t.ppf(0.975, len(x) - 1)
    return float(x.mean() - q * se), float(x.mean() + q * se)


def paired_summary(wide: pd.DataFrame, early: str, late: str) -> dict[str, float | int]:
    pair = wide[[early, late]].dropna()
    delta = pair[late] - pair[early]
    t = stats.ttest_rel(pair[late], pair[early]) if len(pair) >= 2 else None
    try:
        w = stats.wilcoxon(delta) if len(pair) >= 2 and np.any(delta != 0) else None
    except ValueError:
        w = None
    rng = np.random.default_rng(SEED)
    boots = np.array([rng.choice(delta.to_numpy(), len(delta), replace=True).mean() for _ in range(20000)])
    return {
        "n": int(len(pair)),
        "early_mean": float(pair[early].mean()),
        "late_mean": float(pair[late].mean()),
        "mean_delta": float(delta.mean()),
        "median_delta": float(delta.median()),
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
        "t": float(t.statistic) if t else np.nan,
        "t_p": float(t.pvalue) if t else np.nan,
        "w": float(w.statistic) if w else np.nan,
        "w_p": float(w.pvalue) if w else np.nan,
        "n_worse": int((delta > 0).sum()),
        "n_better": int((delta < 0).sum()),
        "n_same": int((delta == 0).sum()),
    }


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    trials = pd.read_csv(TRIALS_CSV, encoding="utf-8-sig", low_memory=False)
    blocks = pd.read_csv(BLOCKS_CSV, encoding="utf-8-sig")
    for col in ["rt", "is_no_go", "correct", "commission", "omission", "probe_response"]:
        trials[col] = pd.to_numeric(trials[col], errors="coerce")
    # 与v1修订口径一致：程序接受的0–1200 ms正确Go RT进入描述，快/慢反应另作QC，不静默删除。
    trials["go_rt_valid"] = trials["rt"].where(
        trials["is_no_go"].eq(0)
        & trials["correct"].eq(1)
        & trials["rt"].between(0, 1200, inclusive="both")
    )
    rt = (
        trials.groupby(["subject", "block_num"], as_index=False)["go_rt_valid"]
        .agg(rt_mean_ms="mean", rt_median_ms="median", rt_sd_ms="std", go_rt_n="count")
    )
    rt["rt_cv"] = rt["rt_sd_ms"] / rt["rt_mean_ms"]
    blocks = blocks.merge(rt, on=["subject", "block_num"], how="left", suffixes=("", "_recalc"))
    blocks["condition"] = pd.Categorical(blocks["condition"], ["A", "B", "C"], ordered=True)
    return trials, blocks


def plot_block_trajectory(blocks: pd.DataFrame) -> None:
    panels = [
        ("commission_rate", "No-Go误按率", "越高=抑制失败更多"),
        ("omission_rate", "Go漏按率", "越高=漏反应更多"),
        ("dprime_loglinear", "d′（Go命中与No-Go误按合成）", "越高=任务辨别表现更好"),
        ("rt_median_ms", "正确Go RT中位数（ms）", "速度，不等同于专注"),
        ("rt_cv", "Block级RT-CV", "越高=整段反应稳定性更差"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), constrained_layout=True)
    axes = axes.ravel()
    x = np.arange(1, 7)
    for ax, (metric, title, subtitle) in zip(axes, panels):
        for _, g in blocks.groupby("subject", observed=True):
            g = g.sort_values("block_num")
            ax.plot(g["block_num"], g[metric], color="#9aa4ad", alpha=.34, lw=1)
        means = blocks.groupby("block_num", observed=True)[metric].mean().reindex(x)
        lows, highs = [], []
        for b in x:
            lo, hi = ci95(blocks.loc[blocks["block_num"].eq(b), metric])
            lows.append(lo); highs.append(hi)
        ax.plot(x, means, color="#182b49", marker="o", lw=2.5, label="11人均值")
        ax.fill_between(x, lows, highs, color="#3478bf", alpha=.16, label="95% CI")
        for b, cond in enumerate(BLOCK_ORDER, 1):
            ax.scatter(b, means.loc[b], s=55, color=COLORS[cond], zorder=4)
        ax.set_title(title, fontsize=11.5)
        ax.text(.01, .98, subtitle, transform=ax.transAxes, va="top", fontsize=8.5, color="#555")
        ax.set_xticks(x, [f"B{i}\n{c}" for i, c in enumerate(BLOCK_ORDER, 1)])
        ax.grid(axis="y", alpha=.22)
    axes[-1].axis("off")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("11名被试在ABCCBA六个Block中的行为轨迹\n灰线=单个被试；深色线=组均值；阴影=被试间95%置信区间", fontsize=14)
    fig.savefig(PLOTS / "01-11人六Block行为轨迹.png", bbox_inches="tight")
    plt.close(fig)


def b2_b5_tables(blocks: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict]]:
    selected = blocks.loc[blocks["block_num"].isin([2, 5])].copy()
    metrics = {
        "commission_rate": "误按率",
        "dprime_loglinear": "d′",
        "rt_cv": "RT-CV",
        "rt_median_ms": "RT中位数",
        "omission_rate": "漏按率",
    }
    summaries: dict[str, dict] = {}
    pairs = []
    for metric, label in metrics.items():
        wide = selected.pivot(index="subject", columns="block_num", values=metric).rename(columns={2: "B2", 5: "B5"})
        summaries[metric] = paired_summary(wide, "B2", "B5")
        part = wide.reset_index()
        part["metric"] = metric
        part["metric_label"] = label
        part["delta_B5_minus_B2"] = part["B5"] - part["B2"]
        pairs.append(part)
    return pd.concat(pairs, ignore_index=True), summaries


def plot_b2_b5(pairs: pd.DataFrame, summaries: dict[str, dict]) -> None:
    specs = [
        ("commission_rate", "No-Go误按率", True),
        ("dprime_loglinear", "d′", False),
        ("rt_cv", "Block级RT-CV", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9), constrained_layout=True)
    for ax, (metric, label, higher_worse) in zip(axes, specs):
        d = pairs.loc[pairs["metric"].eq(metric)].copy()
        for _, r in d.iterrows():
            color = "#c74747" if ((r["B5"] > r["B2"]) == higher_worse) else "#4c8b59"
            if r["B5"] == r["B2"]:
                color = "#8b8b8b"
            ax.plot([0, 1], [r["B2"], r["B5"]], color=color, alpha=.65, lw=1.4)
            ax.scatter([0, 1], [r["B2"], r["B5"]], color=color, s=28)
        s = summaries[metric]
        ax.plot([0, 1], [s["early_mean"], s["late_mean"]], color="#111", lw=4, marker="D", ms=7)
        ax.set_xticks([0, 1], ["Block 2\n早期B", "Block 5\n后期B"])
        ax.set_title(label)
        ax.text(.5, .98, f"均值差(B5−B2)={s['mean_delta']:+.3f}\n配对t检验 p={s['t_p']:.3f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=9)
        ax.grid(axis="y", alpha=.22)
    fig.suptitle("相同B难度的被试内早—晚对比（n=11）\n每条细线=一名被试；黑色粗线=组均值；红=表现变差，绿=表现改善", fontsize=14)
    fig.savefig(PLOTS / "02-B2与B5被试内配对变化.png", bbox_inches="tight")
    plt.close(fig)


def pre_nogo_events(trials: pd.DataFrame, previous_go: int = 4) -> pd.DataFrame:
    records: list[dict] = []
    # 只分析B2/B5，避免把不同No-Go密度与位置混进“错误前预警”。
    b = trials.loc[trials["condition"].eq("B")].copy()
    for (subject, block_num), block in b.groupby(["subject", "block_num"], sort=True):
        block = block.sort_values("trial_num")
        block_baseline = float(block["go_rt_valid"].median())
        history: list[tuple[int, float]] = []
        for _, row in block.iterrows():
            if row["is_no_go"] == 0:
                if pd.notna(row["go_rt_valid"]):
                    history.append((int(row["trial_num"]), float(row["go_rt_valid"])))
                continue
            recent = history[-previous_go:]
            if len(recent) < previous_go:
                continue
            event_id = f"{subject}-B{int(block_num)}-T{int(row['trial_num'])}"
            for lag, (_, value) in zip(range(-previous_go, 0), recent):
                records.append({
                    "event_id": event_id,
                    "subject": subject,
                    "block_num": int(block_num),
                    "commission": int(row["commission"]),
                    "lag": lag,
                    "rt_ms": value,
                    "rt_offset_ms": value - block_baseline,
                })
    return pd.DataFrame(records)


def pre_nogo_stats(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 先在每名被试、结局、lag内取均值，保证图上的推断单位是被试而非重复事件。
    subject_lag = events.groupby(["subject", "commission", "lag"], as_index=False).agg(
        rt_offset_ms=("rt_offset_ms", "mean"), events=("event_id", "nunique")
    )
    rows = []
    for lag in sorted(subject_lag["lag"].unique()):
        wide = subject_lag.loc[subject_lag["lag"].eq(lag)].pivot(index="subject", columns="commission", values="rt_offset_ms")
        if 0 not in wide or 1 not in wide:
            continue
        wide = wide.rename(columns={0: "correct", 1: "error"})
        s = paired_summary(wide, "correct", "error")
        s["lag"] = int(lag)
        rows.append(s)
    result = pd.DataFrame(rows).sort_values("lag")
    if len(result):
        result["p_holm"] = np.minimum.accumulate((result["t_p"].sort_values(ascending=False) * np.arange(1, len(result) + 1)).clip(upper=1).to_numpy())[::-1]
        # 上式的顺序不适合回填；用明确Holm实现。
        order = np.argsort(result["t_p"].to_numpy())
        adjusted = np.empty(len(result))
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, result.iloc[idx]["t_p"] * (len(result) - rank))
            adjusted[idx] = min(running, 1.0)
        result["p_holm"] = adjusted
    return subject_lag, result


def plot_pre_nogo(events: pd.DataFrame, subject_lag: pd.DataFrame, stats_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.3, 5), constrained_layout=True)
    ax = axes[0]
    for outcome, label, color in [(0, "随后正确抑制", "#3478bf"), (1, "随后No-Go误按", "#d95f59")]:
        d = subject_lag.loc[subject_lag["commission"].eq(outcome)]
        g = d.groupby("lag")["rt_offset_ms"]
        mean = g.mean().sort_index()
        lo, hi = [], []
        for lag in mean.index:
            a, b = ci95(d.loc[d["lag"].eq(lag), "rt_offset_ms"]); lo.append(a); hi.append(b)
        ax.plot(mean.index, mean, marker="o", lw=2.4, color=color, label=label)
        ax.fill_between(mean.index, lo, hi, color=color, alpha=.14)
    ax.axhline(0, color="#777", ls="--", lw=1)
    ax.set_xticks([-4, -3, -2, -1], ["前4", "前3", "前2", "前1"])
    ax.set_xlabel("No-Go前的正确Go试次")
    ax.set_ylabel("相对该被试该Block RT中位数（ms）")
    ax.set_title("错误前是否已有RT轨迹差异？")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.22)

    ax = axes[1]
    counts = events.drop_duplicates("event_id").groupby(["subject", "commission"]).size().unstack(fill_value=0)
    x = np.arange(len(counts))
    ax.bar(x, counts.get(0, 0), label="正确抑制", color="#3478bf")
    ax.bar(x, counts.get(1, 0), bottom=counts.get(0, 0), label="误按", color="#d95f59")
    ax.set_xticks(x, [s.replace("sub-", "") for s in counts.index], rotation=0)
    ax.set_xlabel("被试")
    ax.set_ylabel("B2+B5 No-Go事件数")
    ax.set_title("每名被试的事件支撑量")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    n_error = int(events.loc[events["commission"].eq(1), "event_id"].nunique())
    n_correct = int(events.loc[events["commission"].eq(0), "event_id"].nunique())
    pmin = float(stats_df["p_holm"].min()) if len(stats_df) else np.nan
    fig.suptitle(f"B条件No-Go前4个正确Go的反应时轨迹（错误事件={n_error}，正确事件={n_correct}）\n先在被试内汇总，阴影=11人95% CI；4个位置中最小Holm校正p={pmin:.3f}", fontsize=13.5)
    fig.savefig(PLOTS / "03-B条件No-Go正确与错误前RT轨迹.png", bbox_inches="tight")
    plt.close(fig)


def plot_probe(trials: pd.DataFrame) -> pd.DataFrame:
    probes = trials.loc[trials["is_probe"].eq(1)].copy()
    labels = {1: "完全专注", 2: "关注实验\n未聚焦任务", 3: "任务无关思维", 4: "大脑空白"}
    colors = {1: "#3478bf", 2: "#e4a11b", 3: "#8064a2", 4: "#7f8c8d"}
    counts = probes["probe_response"].value_counts().reindex([1, 2, 3, 4], fill_value=0)
    by_subject = pd.crosstab(probes["subject"], probes["probe_response"]).reindex(columns=[1, 2, 3, 4], fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    axes[0].bar(range(1, 5), counts.values, color=[colors[i] for i in range(1, 5)])
    for i, v in enumerate(counts.values, 1):
        axes[0].text(i, v + 3, f"{v}\n({v/counts.sum():.1%})", ha="center", fontsize=9)
    axes[0].set_xticks(range(1, 5), [labels[i] for i in range(1, 5)])
    axes[0].set_ylabel("探针次数")
    axes[0].set_title("四类探针总体分布（264次）")
    bottom = np.zeros(len(by_subject))
    x = np.arange(len(by_subject))
    for state in range(1, 5):
        vals = by_subject[state].to_numpy()
        axes[1].bar(x, vals, bottom=bottom, color=colors[state], label=f"{state} {labels[state].replace(chr(10), '')}")
        bottom += vals
    axes[1].set_xticks(x, [s.replace("sub-", "") for s in by_subject.index])
    axes[1].set_ylabel("每人24次探针")
    axes[1].set_title("被试内类别变化")
    axes[1].legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    fig.suptitle("现有四分类探针高度偏向“完全专注”，且部分被试几乎不改变选项", fontsize=14)
    fig.savefig(PLOTS / "04-四类探针分布与被试内变化.png", bbox_inches="tight")
    plt.close(fig)
    return by_subject


def plot_design_evidence(blocks: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13.8, 7.2), gridspec_kw={"height_ratios": [1.15, 1]}, constrained_layout=True)
    ax = axes[0]
    means = blocks.groupby(["block_num", "condition"], observed=True)["commission_rate"].mean().reset_index()
    x = np.arange(1, 7)
    vals = means.set_index("block_num")["commission_rate"].reindex(x)
    ax.plot(x, vals, color="#182b49", marker="o", lw=2.4)
    for b, cond in enumerate(BLOCK_ORDER, 1):
        n_nogo = {"A": 48, "B": 24, "C": 12}[cond]
        ax.scatter(b, vals.loc[b], color=COLORS[cond], s=90, zorder=3)
        ax.text(b, vals.loc[b] + .018, f"{cond}\n{n_nogo}次No-Go", ha="center", fontsize=8.5)
    ax.set_xticks(x, [f"Block {i}" for i in x])
    ax.set_ylabel("11人平均No-Go误按率")
    ax.set_title("现有ABCCBA：条件、时间位置和No-Go机会数同时改变")
    ax.grid(axis="y", alpha=.22)
    ax.text(.01, .03, "因此A/B/C的表观差异不能单独归因于‘难度’；只有B2与B5是同难度、同机会数的早晚比较。", transform=ax.transAxes, fontsize=9.5)

    ax = axes[1]
    ax.set_xlim(0, 4); ax.set_ylim(0, 1); ax.axis("off")
    for i in range(4):
        rect = plt.Rectangle((i + .06, .33), .86, .34, facecolor=COLORS["B"], alpha=.88, edgecolor="white")
        ax.add_patch(rect)
        ax.text(i + .49, .55, f"B{i+1}\n216试次 / 24次No-Go", color="white", ha="center", va="center", fontsize=11, weight="bold")
    ax.annotate("时间在任务上增加 →", xy=(3.8, .18), xytext=(.2, .18), arrowprops={"arrowstyle": "->", "lw": 1.8}, ha="left", va="center")
    ax.text(.5, .82, "早期基础：抑制能力 + 规则适应", ha="center", fontsize=10.5)
    ax.text(2.5, .82, "同规则持续：个体内变化与末期维持", ha="center", fontsize=10.5)
    ax.set_title("建议BBBB：所有被试接受同一处理，保持规则与No-Go机会数不变", pad=8)
    fig.suptitle("为什么下一版主任务更适合使用单一B难度", fontsize=15)
    fig.savefig(PLOTS / "05-ABCCBA混杂与BBBB建议结构.png", bbox_inches="tight")
    plt.close(fig)


def fmt_p(p: float) -> str:
    return "<0.001" if p < .001 else f"={p:.3f}"


def generate_report(trials: pd.DataFrame, blocks: pd.DataFrame, pairs: pd.DataFrame,
                    bstats: dict[str, dict], events: pd.DataFrame,
                    event_stats: pd.DataFrame, probe_by_subject: pd.DataFrame) -> None:
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M（Asia/Shanghai）")
    block_summary = blocks.groupby(["block_num", "condition"], observed=True).agg(
        commission_rate=("commission_rate", "mean"), omission_rate=("omission_rate", "mean"),
        rt_cv=("rt_cv", "mean"), rt_median_ms=("rt_median_ms", "mean"),
        dprime=("dprime_loglinear", "mean")
    ).reset_index()
    block_rows = "\n".join(
        f"| {int(r.block_num)} | {r.condition} | {r.commission_rate:.3f} | {r.omission_rate:.3f} | {r.rt_cv:.3f} | {r.rt_median_ms:.0f} | {r.dprime:.2f} |"
        for r in block_summary.itertuples()
    )
    comm = bstats["commission_rate"]
    dp = bstats["dprime_loglinear"]
    cv = bstats["rt_cv"]
    bpair = pairs.loc[pairs["metric"].eq("commission_rate"), ["subject", "B2", "B5", "delta_B5_minus_B2"]]
    bpair_rows = "\n".join(
        f"| {r.subject} | {r.B2:.3f} | {r.B5:.3f} | {r.delta_B5_minus_B2:+.3f} |"
        for r in bpair.itertuples()
    )
    probe_counts = trials.loc[trials["is_probe"].eq(1), "probe_response"].value_counts().reindex([1,2,3,4], fill_value=0)
    fixed_state1 = int((probe_by_subject.get(1, 0) >= 22).sum())
    n_error = int(events.loc[events["commission"].eq(1), "event_id"].nunique())
    n_correct = int(events.loc[events["commission"].eq(0), "event_id"].nunique())
    lag_rows = "\n".join(
        f"| 前{abs(int(r.lag))} | {r.mean_delta:+.1f} ms | {r.t_p:.3f} | {r.p_holm:.3f} |"
        for r in event_stats.itertuples()
    )
    report = f"""# 014｜正式实验行为任务修改建议报告

> {now}｜基于11名被试预实验数据、v1行为分析及本轮讨论，建议将正式持续注意主任务由ABCCBA调整为BBBB，并把四分类时间段探针改为二分类即时试次探针；本报告只提出研究设计建议，不修改实验程序。

## 目录

1. [结论先行](#1-结论先行)
2. [我们讨论了什么](#2-我们讨论了什么)
3. [数据与指标口径](#3-数据与指标口径)
4. [11人数据证据](#4-11人数据证据)
5. [方案比较与正式实验建议](#5-方案比较与正式实验建议)
6. [建议的分析框架](#6-建议的分析框架)
7. [仍需验证与最终决策](#7-仍需验证与最终决策)

## 1. 结论先行

当前最合适的主方案是 **BBBB（4个相同B Block）**，而不是继续ABCCBA，也不建议为了“基础抑制能力”额外加入一个A Block。

- B1本身已经能提供个体早期抑制表现；它同时包含规则适应和初始投入，因此应称为“早期行为基线”，不能宣称是纯抑制能力。
- B2—B4在规则、No-Go频率和试次数相同时，变化更容易解释为时间在任务上的持续维持或衰减。
- A若只出现在第一个Block，会与新奇、学习和最高初始投入完全重合；它并不能更干净地分离抑制能力。
- 探针保留，但改成针对刚刚一个试次的二分问题；位置在不同Block中错开，同时所有被试使用完全相同的预生成方案。
- 持续专注不应由单个指标直接定义：Block级用commission、omission和RT-CV描述；错误前预警用短时RT轨迹，不能在四个试次上计算CV；d′只作任务总体表现摘要。

## 2. 我们讨论了什么

讨论从“现有ABCCBA是否能用难度变化制造专注差异”开始。预实验中B2与B5是同条件早晚重复，误按率的描述性差异最明显，因此先考虑保留B作为主难度。随后比较了两个方案：

1. **ABBB**：A作为个体抑制能力参考，后续B观察衰减。问题是A只出现一次且位于开头，A效应无法与初始适应和高投入分开。
2. **BBBB**：B1同时提供早期个体基线，后续相同B直接观察维持。它牺牲了难度对比，却显著减少了正式评分中的设计混杂。

最终讨论倾向BBBB。固定No-Go规律继续保留，以维持SART的规律性和自动化需求；也不做被试间顺序平衡，因为产品最终需要每位被试接受相同处理。探针不删除，但从“四类、回忆一段时间”简化为“二类、判断刚才一个试次”。

## 3. 数据与指标口径

- 数据：sub-000至sub-010，共11人、66个Block、14,256个正式试次、264次探针。
- 当前结构：每Block 216试次；A/B/C分别含48/24/12次No-Go；顺序固定为ABCCBA。
- RT：保留程序归属的反应；0–1200 ms正确Go RT进入本报告描述。`<100`、`<150`、`>1000`和`>1150 ms`只作质量与策略标签，不作为静默剔除规则。
- commission：No-Go误按数 / No-Go机会数。
- omission：Go漏按数 / Go机会数。
- d′：正确Go为Hit，No-Go误按为False Alarm，使用loglinear端点校正。它同时受Go命中和No-Go误按影响，并非“只算Go正确”。
- RT-CV：一个Block内正确Go RT标准差 / 均值，只用于具有足够试次的Block或较长窗口；不用于错误前四个试次。
- 推断单位：B2/B5比较按被试配对；错误前轨迹先在每名被试内汇总，再进行11人的配对比较，避免把重复事件冒充独立样本。

## 4. 11人数据证据

### 4.1 六个Block的整体变化

| Block | 条件 | commission | omission | RT-CV | RT中位数(ms) | d′ |
|---:|:---:|---:|---:|---:|---:|---:|
{block_rows}

![11人六个Block行为轨迹](plots/01-11人六Block行为轨迹.png)

图中灰线是一名被试，深色线是11人均值，阴影是被试间95%置信区间。它说明行为并不是简单地随Block单调恶化：起始Block也可能因为适应、规则切入或反应策略而不稳定。与此同时，现有顺序让条件与时间位置绑定，不能把A/B/C均值差直接解释成难度造成的专注差异。

RT-CV在这里有价值，因为每个Block约有168–204个Go机会，可以估计整段稳定性；但RT-CV高也可能由极短/极长反应、策略或数据质量造成，必须与RT分布和QC标签一起解释。

### 4.2 同一B难度：Block 2与Block 5

![B2与B5被试内配对变化](plots/02-B2与B5被试内配对变化.png)

11人中，B5误按率较B2上升者为 **{comm['n_worse']}/11**，下降者为{comm['n_better']}/11，不变者为{comm['n_same']}/11。组均值从{comm['early_mean']:.3f}变为{comm['late_mean']:.3f}，平均变化{comm['mean_delta']:+.3f}，bootstrap 95% CI [{comm['ci_low']:+.3f}, {comm['ci_high']:+.3f}]；配对t检验p{fmt_p(comm['t_p'])}，Wilcoxon p{fmt_p(comm['w_p'])}。

d′平均变化为{dp['mean_delta']:+.3f}（配对t检验p{fmt_p(dp['t_p'])}）；RT-CV平均变化为{cv['mean_delta']:+.3f}（p{fmt_p(cv['t_p'])}）。这些结果可以支持“B2—B5对部分个体有明显早晚变化、B适合做同条件追踪”，但如果95% CI跨0或p≥.05，就不能写成已经证明了统一的群体疲劳效应。

| 被试 | B2误按率 | B5误按率 | B5−B2 |
|---|---:|---:|---:|
{bpair_rows}

真正有产品价值的不是强迫所有人沿同一方向变化，而是同一任务下能否稳定识别“前期好、后期下降”“始终稳定”“前期适应后改善”等个体轨迹。BBBB比ABCCBA更适合验证这种个体内状态变化。

### 4.3 No-Go错误前是否已有预警

![B条件No-Go正确与错误前RT轨迹](plots/03-B条件No-Go正确与错误前RT轨迹.png)

本图只使用B2和B5，共{n_error}次误按事件、{n_correct}次正确抑制事件。纵轴不是原始RT，而是相对该被试该Block正确Go RT中位数的偏移；这样减少了被试之间固有快慢差异。每条轨迹先按被试汇总，统计检验的n是11人。

| 位置 | 随后误按−随后正确的RT差 | 未校正p | Holm校正p |
|---|---:|---:|---:|
{lag_rows}

如果某个位置为负，表示随后误按前的反应更快，可能对应自动化或预期性反应；为正则表示错误前反而更慢。当前结果用于判断“短时RT轨迹是否值得保留为候选预警特征”，不能把一次加速直接定义为不专注。即使出现显著差异，也仍需在新BBBB数据中做留一被试验证，确认它能预测尚未发生的错误，而不只是事后描述。

这里不计算错误前RT-CV：四个点估计标准差和均值比例极不稳定，且一次极端RT会完全控制结果。更合理的即时特征是相对个人局部基线的RT偏移、最近数次RT斜率、连续极短反应及其与固定周期位置的交互。

### 4.4 现有探针为什么需要简化

![四类探针分布与被试内变化](plots/04-四类探针分布与被试内变化.png)

四类选择分别为{int(probe_counts[1])}/{int(probe_counts[2])}/{int(probe_counts[3])}/{int(probe_counts[4])}次，其中“完全专注”占{probe_counts[1]/probe_counts.sum():.1%}；有{fixed_state1}名被试在24次探针中至少22次选择类别1。类别3和4只有{int(probe_counts[3] + probe_counts[4])}次，无法支撑稳定的四分类模型。

程序已核实没有默认选择，因此失衡不能解释成默认按钮造成；更可能涉及题目过细、短时间内难以区分心理内容、社会期许，以及样本确实较多处于任务投入状态。把1–4求均值也没有明确心理量尺含义。

当前四个探针固定在trial 30/82/137/191，对应18试次周期的位置12/10/11/11，全部跟在Go试次之后。探针类别因此还与固定周期位置及探针可预测性绑定，下一版应改变位置安排。

### 4.5 为什么A/B/C差异不能直接当作难度证据

![ABCCBA混杂与BBBB建议结构](plots/05-ABCCBA混杂与BBBB建议结构.png)

现有设计同时改变三件事：时间位置、No-Go密度和条件。A有48次No-Go，B有24次，C只有12次；一个错误在C中就改变8.3个百分点，在B中改变4.2个百分点，在A中改变2.1个百分点。因此条件间原始波动精度并不相同。ABCCBA适合探索“难度/投入变化是否带来生理差异”，却不适合把时间轨迹直接用于统一的专注评分。

## 5. 方案比较与正式实验建议

| 方案 | 优点 | 关键问题 | 结论 |
|---|---|---|---|
| ABCCBA | 难度和投入变化大，可能制造较明显生理对比 | 条件×时间×No-Go机会数混杂；个体评分难统一 | 不作为持续注意主方案 |
| ABBB | 有一个高No-Go密度Block，后面可看B衰减 | A只在开头，不能与适应和初始高投入分离；规则切换又引入新状态 | 不推荐作为主方案 |
| BBBB | 规则、机会数和试次数一致；可解释被试内时间变化；每人处理相同 | 失去难度对比；B1仍含适应 | **推荐** |

### 5.1 推荐任务结构

- 练习阶段继续存在，但正式主任务使用4个B Block，即 **BBBB**。
- 每Block 216试次、24次No-Go；全程共864试次、96次No-Go。固定18试次规律和B条件No-Go位置保持不变。
- B1标记为“早期基线/适应”，同时保留其前段与后段差异；B2—B4作为相同规则下的持续维持阶段。
- 不把B1误按率称为纯抑制控制能力。它可以作为个体基础表现参考，但仍混合了理解、适应、策略和初始投入。
- 如果未来确实需要独立测量抑制控制，应另设短校准任务并单独验证，而不是在持续注意主线开头塞入一个无法去混杂的A。

### 5.2 推荐探针

建议题干：**“刚才这个试次中，你的注意主要在任务上吗？”**

1. 是，主要在任务上；
2. 否，没有主要在任务上。

使用“试次”而不是“按键”，因为正确No-Go试次本来就不应按键。该题测的是即时主观任务聚焦，不再要求被试区分“关注实验但未聚焦任务、任务无关思维、大脑空白”。

探针建议每Block 4次、共16次。不同Block使用不同位置，但所有被试使用同一份固定调度表，并满足：

- 避开Block开头、结尾和紧邻休息的位置；
- 不再集中在18试次周期的10–12位；
- 在周期位置和前一试次类型上分层，包含Go后与No-Go后，但不根据被试当场正确/错误自适应触发；
- 位置间保持足够间隔，记录调度版本、前一试次类型、周期位置及探针反应时；
- 具体16个位置应在正式程序修改前单独生成并检查，本报告不擅自冻结。

二分探针不是专注“真值”。它只提供稀疏的即时主观标签，用于检查行为和生理特征在“是/否”之间是否有一致的被试内差异。

## 6. 建议的分析框架

应把“基础能力、持续状态、即时风险”分开，避免所有指标混成一个分数：

| 层次 | 时间尺度 | 主要指标 | 能回答的问题 |
|---|---|---|---|
| 早期行为基线 | B1及其稳定阶段 | commission、omission、d′、RT分布、Block级RT-CV | 这个人在相同任务上的起始表现如何？ |
| 持续维持 | B2—B4及Block内长窗 | commission/omission风险、RT-CV、RT漂移、极短RT比例 | 随时间是否变得更不稳定或更易犯错？ |
| 即时错误预警 | No-Go前数个Go或短时窗口 | 局部RT偏移、斜率、连续快反应、周期位置 | 错误发生前是否出现可预测变化？ |
| 主观状态锚点 | 探针前一个试次 | 二分探针及反应时 | 被试当时是否自报主要聚焦任务？ |

d′适合概括一个Block的整体任务辨别表现，但与commission和omission共享信息；未来复合评分不能再把三者当成完全独立证据重复加权。RT均值只表示速度，不能单独映射专注；RT-CV适合较长窗口的稳定性，不能用于少数错误前试次。

## 7. 仍需验证与最终决策

本报告的数据能支持“为什么要从ABCCBA转向同一B难度”和“为什么要简化探针”，但尚不能确定最终专注评分权重。正式采集前后仍需完成：

1. 审批BBBB、4个Block和二分探针题干；另行审查16个具体探针位置。
2. 用新BBBB小样本确认B1适应长度、B2—B4是否产生足够的个体内变化，以及任务总时长是否合适。
3. 对错误前RT候选特征做跨被试预测验证，并控制周期位置；只报告事后差异不等于可预警。
4. 检查二分探针是否仍几乎全选“是”。若仍严重失衡，应重新评估题干或探针价值，而不是强行用作监督标签。
5. 在行为定义稳定后，再检验NIR眼动/瞳孔指标是否能解释同一时间窗中的行为状态。

## 附：复现来源

- 11人逐试次数据：`D:/_AttentionData/output-v2/040-pre-experiment/040-behavior/041-trials.csv`
- 11人Block指标：`D:/_AttentionData/output-v2/040-pre-experiment/040-behavior/042-block_metrics.csv`
- v1既有报告：`D:/_AttentionData/output/040-pre-experiment/010-analysis/behavior/00-整体/00-行为整合分析-预实验.md`
- v1参考实现：`NIR-RGB-extraction/scripts/extract_beh.py`、`plotting/cores/behavior_group_summary.py`、`analyze_behavior_window_feasibility.py`
- 本报告生成脚本：`attention-pipeline-v2/scripts/build_formal_experiment_recommendation.py`
"""
    REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    setup_style()
    PLOTS.mkdir(parents=True, exist_ok=True)
    trials, blocks = load_data()
    if trials["subject"].nunique() != 11 or len(trials) != 14256:
        raise RuntimeError("输入数据不再是预期的11人/14,256试次，停止生成报告。")
    plot_block_trajectory(blocks)
    pairs, summaries = b2_b5_tables(blocks)
    plot_b2_b5(pairs, summaries)
    events = pre_nogo_events(trials)
    subject_lag, event_stats = pre_nogo_stats(events)
    plot_pre_nogo(events, subject_lag, event_stats)
    probe_by_subject = plot_probe(trials)
    plot_design_evidence(blocks)
    generate_report(trials, blocks, pairs, summaries, events, event_stats, probe_by_subject)
    audit = {
        "subjects": int(trials["subject"].nunique()),
        "trials": int(len(trials)),
        "blocks": int(len(blocks)),
        "probes": int(trials["is_probe"].sum()),
        "b2_b5": summaries,
        "pre_nogo_stats": event_stats.to_dict(orient="records"),
        "outputs": [str(REPORT), *[str(p) for p in sorted(PLOTS.glob("*.png"))]],
    }
    (PLOTS / "014-统计审计.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
