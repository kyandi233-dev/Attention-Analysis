"""正式 BBB SART 图表：20 张，中文 matplotlib，每个分析族 ≥1 图。

图存于 output_root/000-reports/，前缀 051-。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import Config
from ..behavior.reporting import _configure_chinese_font, _save_figure, _style_axis
from . import metrics as fmet
from . import stats as fstat

NEUTRAL = "#687386"
ACCENT = "#2878B5"
ACCENT2 = "#E69F00"
WARNING = "#D97706"
GOOD = "#4c8b59"
BAD = "#c74747"
GRID = "#D9DEE7"
Q1_COLORS = {1: "#2878B5", 2: "#E69F00", 3: "#7A5195", 4: "#6B7280"}
Q1_LABELS = {1: "完全专注", 2: "在任务上没想目标", 3: "走神", 4: "大脑空白"}
BLOCK_COLORS = {1: "#2878B5", 2: "#E69F00", 3: "#c74747"}

PRIMARY_EXCLUDE = {"sub-015"}


def _primary(trials: pd.DataFrame) -> pd.DataFrame:
    return trials.loc[~trials["subject"].isin(PRIMARY_EXCLUDE)]


def _ci95(x: pd.Series) -> tuple[float, float]:
    x = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < 2:
        return np.nan, np.nan
    from scipy import stats as sps
    se = sps.sem(x)
    q = sps.t.ppf(0.975, len(x) - 1)
    return float(x.mean() - q * se), float(x.mean() + q * se)


def _group_mean_ci(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    out = []
    for key, grp in df.groupby(x, sort=True):
        lo, hi = _ci95(grp[y])
        out.append({x: key, "mean": grp[y].mean(), "lo": lo, "hi": hi, "n": len(grp)})
    return pd.DataFrame(out)


def plot_01_completeness(config: Config, trials: pd.DataFrame, path) -> None:
    subj = [s for s in config.section("subjects")["include"]]
    counts = trials.pivot_table(index="subject", columns="block_num", values="trial_num", aggfunc="count")
    counts = counts.reindex(subj).reindex(columns=[1, 2, 3])
    fig, ax = plt.subplots(figsize=(8.5, 8))
    im = ax.imshow(counts.to_numpy(dtype=float), cmap="Blues", vmin=400, vmax=432)
    ax.set_xticks(range(3), ["Block1", "Block2", "Block3"])
    ax.set_yticks(range(len(subj)), [s.replace("sub-", "") for s in subj])
    for i in range(len(subj)):
        for j in range(3):
            v = counts.iloc[i, j]
            ax.text(j, i, f"{int(v)}", ha="center", va="center",
                    color="white" if v < 420 else "#1a1a1a", fontsize=8)
    ax.set_title("数据完整性：被试×Block 试次数（期望 432）\nsub-015 标红=完全无反应异常", fontsize=12)
    if "sub-015" in subj:
        idx = subj.index("sub-015")
        ax.get_yticklabels()[idx].set_color("red")
        ax.get_yticklabels()[idx].set_fontweight("bold")
    fig.colorbar(im, ax=ax, label="试次数")
    _save_figure(fig, path)


def plot_02_rt_distribution(config: Config, trials: pd.DataFrame, path) -> None:
    rt = _primary(trials)["go_rt_valid"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    ax = axes[0]
    ax.hist(rt, bins=80, color=ACCENT, alpha=0.85)
    for th in (100, 150, 1000, 1150):
        ax.axvline(th, color=WARNING, lw=1, ls="--")
    ax.set_xlim(0, 1300)
    ax.set_xlabel("正确 Go RT (ms)"); ax.set_ylabel("试次数")
    ax.set_title("正确 Go RT 直方图（QC 阈值仅标注，不删除）")
    _style_axis(ax)
    ax = axes[1]
    srt = np.sort(rt.to_numpy())
    ax.plot(srt, np.arange(1, len(srt) + 1) / len(srt), color=ACCENT2, lw=2)
    ax.set_xlabel("正确 Go RT (ms)"); ax.set_ylabel("累计比例")
    ax.set_title("ECDF")
    _style_axis(ax)
    fig.suptitle(f"RT 分布（n={len(rt)} 正确Go，主队列）", fontsize=13)
    _save_figure(fig, path)


def plot_03_rt_qc(config: Config, trials: pd.DataFrame, path) -> None:
    tr = _primary(trials)
    rt = tr["go_rt_valid"].dropna()
    cats = pd.cut(rt, bins=[0, 100, 150, 1000, 1150, np.inf],
                  labels=["<100", "100–<150", "150–1000", ">1000–1150", ">1150"], right=False)
    per_subj = tr.dropna(subset=["go_rt_valid"]).copy()
    per_subj["cat"] = pd.cut(per_subj["go_rt_valid"], bins=[0, 100, 150, 1000, 1150, np.inf],
                             labels=["<100", "100–<150", "150–1000", ">1000–1150", ">1150"], right=False)
    tab = per_subj.groupby(["subject", "cat"], observed=True).size().unstack(fill_value=0)
    tab = tab.div(tab.sum(axis=1), axis=0)
    order = [s for s in tab.index if s not in PRIMARY_EXCLUDE]
    tab = tab.reindex(order)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    cols = ["#c74747", "#E69F00", "#2878B5", "#7A5195", "#6B7280"]
    bottom = np.zeros(len(tab))
    for i, c in enumerate(tab.columns):
        ax.bar(range(len(tab)), tab[c], bottom=bottom, color=cols[i], label=str(c), width=0.72)
        bottom += tab[c]
    ax.set_xticks(range(len(tab)), [s.replace("sub-", "") for s in tab.index], rotation=90)
    ax.set_ylabel("占比")
    ax.set_title("每被试正确 Go RT 的 QC 区间组成")
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    _style_axis(ax)
    _save_figure(fig, path)


def plot_04_block_trajectory(config: Config, trials: pd.DataFrame, path) -> None:
    blocks = fmet.formal_block_metrics(config, _primary(trials))
    panels = [
        ("commission_rate", "No-Go 误按率", "越高=抑制失败更多"),
        ("omission_rate", "Go 漏按率", "越高=任务脱离更多"),
        ("dprime_loglinear", "d′", "越高=辨别表现更好"),
        ("go_rt_median_ms", "正确 Go RT 中位数(ms)", "速度"),
        ("rt_cv", "RT-CV", "越高=稳定性更差"),
        ("c", "反应标准 c", "c<0 宽松，c>0 保守"),
        ("exg_tau", "ex-Gaussian τ(ms)", "越高=慢尾更多"),
        ("beta", "似然比 β", "β>1 保守，β<1 宽松"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.6), constrained_layout=True)
    axes = axes.ravel()
    x = [1, 2, 3]
    for ax, (metric, title, sub) in zip(axes, panels):
        for _, g in blocks.groupby("subject"):
            g = g.sort_values("block_num")
            ax.plot(g["block_num"], g[metric], color="#9aa4ad", alpha=0.3, lw=1)
        for b in x:
            vals = blocks.loc[blocks["block_num"].eq(b), metric].dropna()
            if len(vals):
                ax.scatter([b], [vals.mean()], color=BLOCK_COLORS[b], s=42, zorder=4)
        means = _group_mean_ci(blocks, "block_num", metric)
        ax.plot(means["block_num"], means["mean"], color="#182b49", marker="o", lw=2.2)
        ax.fill_between(means["block_num"], means["lo"], means["hi"], color=ACCENT, alpha=0.16)
        ax.set_title(title, fontsize=10.5)
        ax.text(0.01, 0.97, sub, transform=ax.transAxes, va="top", fontsize=8, color="#555")
        ax.set_xticks(x)
        _style_axis(ax)
    fig.suptitle("block 主效应轨迹（主队列 n=19；灰线=单被试，深线=组均值，阴影=95%CI）", fontsize=14)
    _save_figure(fig, path)


def plot_05_b1_b3_paired(config: Config, trials: pd.DataFrame, path) -> None:
    blocks = fmet.formal_block_metrics(config, _primary(trials))
    specs = [("commission_rate", "No-Go 误按率", True), ("omission_rate", "Go 漏按率", True),
             ("dprime_loglinear", "d′", False), ("rt_cv", "RT-CV", True),
             ("c", "反应标准 c", None)]
    fig, axes = plt.subplots(1, len(specs), figsize=(17, 4.4), constrained_layout=True)
    for ax, (metric, label, higher_worse) in zip(axes, specs):
        wide = blocks.pivot_table(index="subject", columns="block_num", values=metric)[[1, 3]].dropna()
        for _, r in wide.iterrows():
            if higher_worse is None:
                color = "#8b8b8b"
            else:
                color = BAD if ((r[3] > r[1]) == higher_worse) else GOOD
                if r[3] == r[1]:
                    color = "#8b8b8b"
            ax.plot([0, 1], [r[1], r[3]], color=color, alpha=0.6, lw=1.3)
            ax.scatter([0, 1], [r[1], r[3]], color=color, s=26)
        ax.plot([0, 1], [wide[1].mean(), wide[3].mean()], color="#111", lw=3, marker="D", ms=6)
        d31 = wide[3] - wide[1]
        from scipy import stats as sps
        p = sps.wilcoxon(d31).pvalue if np.any(d31 != 0) else 1.0
        ax.set_xticks([0, 1], ["B1", "B3"])
        ax.set_title(f"{label}\nΔ(B3−B1)={d31.mean():+.3f}  p={p:.3f}", fontsize=10)
        _style_axis(ax)
    fig.suptitle("B1→B3 被试内配对变化（n=19；细线=被试，粗线=均值；红=恶化，绿=改善）", fontsize=14)
    _save_figure(fig, path)


def plot_06_block_bin_interaction(config: Config, trials: pd.DataFrame, path) -> None:
    bins = fmet.cycle_bin_metrics(config, _primary(trials))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for ax, metric, ylabel, title in [
        (axes[0], "go_rt_median_ms", "正确 Go RT 中位数(ms)", "RT 中位 × 周期bin"),
        (axes[1], "commission_jeffreys_mean", "No-Go 误按率(Jeffreys)", "误按率 × 周期bin"),
    ]:
        for block in (1, 2, 3):
            sub = bins.loc[bins["block_num"].eq(block)]
            m = _group_mean_ci(sub, "cycle_bin", metric)
            ax.plot(m["cycle_bin"], m["mean"], color=BLOCK_COLORS[block], marker="o", lw=2, label=f"Block{block}")
            ax.fill_between(m["cycle_bin"], m["lo"], m["hi"], color=BLOCK_COLORS[block], alpha=0.13)
        ax.set_xlabel("周期 bin（1=block 内前段 → 6=后段）")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
        _style_axis(ax)
    fig.suptitle("block × 周期bin 交互（组均值±95%CI，n=19）", fontsize=14)
    _save_figure(fig, path)


def plot_07_cycle_trends(config: Config, trials: pd.DataFrame, path) -> None:
    tr = _primary(trials)
    cycle = tr.groupby(["subject", "block_num", "cycle_num"], as_index=False).agg(
        rt_median=("go_rt_valid", "median"),
        comm=("commission", "mean"),
        omiss=("omission", "mean"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)
    for ax, (metric, title) in zip(axes, [("rt_median", "RT 中位(ms)"), ("comm", "误按率"), ("omiss", "漏按率")]):
        for block in (1, 2, 3):
            sub = cycle.loc[cycle["block_num"].eq(block)]
            m = _group_mean_ci(sub, "cycle_num", metric)
            ax.plot(m["cycle_num"], m["mean"], color=BLOCK_COLORS[block], lw=1.8, label=f"Block{block}")
            ax.fill_between(m["cycle_num"], m["lo"], m["hi"], color=BLOCK_COLORS[block], alpha=0.12)
        ax.set_xlabel("cycle_num（1–24）")
        ax.set_ylabel(title)
        ax.set_title(f"block 内 24 周期趋势：{title}")
        ax.legend(frameon=False, fontsize=8)
        _style_axis(ax)
    fig.suptitle("block 内周期趋势（组均值±95%CI，n=19）", fontsize=14)
    _save_figure(fig, path)


def plot_08_rt_drift(config: Config, trials: pd.DataFrame, path) -> None:
    from statsmodels.formula.api import mixedlm
    tr = _primary(trials).dropna(subset=["go_rt_valid"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True, sharey=True)
    for block, ax in zip((1, 2, 3), axes):
        sub = tr.loc[tr["block_num"].eq(block)]
        ax.scatter(sub["cycle_num"], sub["go_rt_valid"], s=2, alpha=0.06, color="#9aa4ad")
        model = mixedlm("go_rt_valid ~ cycle_num", sub, groups=sub["subject"]).fit()
        xs = np.linspace(1, 24, 50)
        ax.plot(xs, model.params["Intercept"] + model.params["cycle_num"] * xs,
                color=BLOCK_COLORS[block], lw=2.5)
        ax.set_title(f"Block{block}\n斜率={model.params['cycle_num']:+.2f} ms/cycle, p={model.pvalues['cycle_num']:.4f}")
        ax.set_xlabel("cycle_num")
        _style_axis(ax)
    axes[0].set_ylabel("正确 Go RT (ms)")
    fig.suptitle("block 内 RT 漂移 MixedLM（rt ~ cycle_num + (1|subject)，n=19）", fontsize=14)
    _save_figure(fig, path)


def plot_09_pre_nogo(config: Config, trials: pd.DataFrame, path) -> None:
    events = fstat.pre_nogo_events(_primary(trials))
    stats_df = fstat.pre_nogo_stats(events)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    ax = axes[0]
    for comm, label, color in [(0, "正确抑制", GOOD), (1, "误按(commission)", BAD)]:
        sub = events.loc[events["commission"].eq(comm)]
        g = sub.groupby("lag")["rt_offset_ms"].agg(["mean", "sem"])
        ax.errorbar(g.index, g["mean"], yerr=1.96 * g["sem"], color=color, marker="o", lw=2, label=label)
    ax.axhline(0, color="#555", lw=1, ls="--")
    ax.set_xlabel("No-Go 前的正确 Go 位置（lag −4..−1）")
    ax.set_ylabel("RT 偏移（相对 block 中位数，ms）")
    ax.set_title("错误前 RT 轨迹（组均值±SE）")
    ax.legend(frameon=False)
    _style_axis(ax)
    ax = axes[1]
    if len(stats_df):
        ax.bar(stats_df["lag"], stats_df["commission_minus_correct_ms"], color=[
            BAD if p < 0.05 else "#9aa4ad" for p in stats_df["wilcoxon_p_holm"]])
        ax.axhline(0, color="#555", lw=1)
        ax.set_xlabel("lag")
        ax.set_ylabel("误按 − 正确抑制 的 RT 偏移差(ms)")
        ax.set_title("误按 vs 正确抑制（Holm 校正 p<0.05 标红）")
        for i, r in stats_df.iterrows():
            ax.text(r["lag"], r["commission_minus_correct_ms"] + (2 if r["commission_minus_correct_ms"] >= 0 else -5),
                    f"p={r['wilcoxon_p_holm']:.3f}", ha="center", fontsize=8)
    _style_axis(ax)
    fig.suptitle("No-Go 前反应加速（前兆证据）", fontsize=14)
    _save_figure(fig, path)


def plot_10_prestimulus(config: Config, trials: pd.DataFrame, path) -> None:
    tr = _primary(trials)
    has_pre = tr["prestimulus_press_ms"].fillna("").astype(str).str.len().gt(0)
    by_block = tr.groupby("block_num").apply(
        lambda d: has_pre.loc[d.index].sum() / len(d), include_groups=False).rename("rate").reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    ax = axes[0]
    ax.bar(by_block["block_num"], by_block["rate"], color=ACCENT)
    ax.set_xticks([1, 2, 3], ["B1", "B2", "B3"])
    ax.set_ylabel("含预判按键的试次占比")
    ax.set_title("预判按键（prestimulus_press）按 block")
    _style_axis(ax)
    ax = axes[1]
    pre = tr.loc[has_pre]
    ax.hist(pre["go_rt_valid"].dropna(), bins=40, color=ACCENT2, alpha=0.8)
    ax.set_xlabel("含预判按键的试次正确 Go RT (ms)")
    ax.set_ylabel("计数")
    ax.set_title("预判按键试次的 RT 分布")
    _style_axis(ax)
    fig.suptitle(f"预判按键（n={int(has_pre.sum())} 试次）", fontsize=13)
    _save_figure(fig, path)


def plot_11_q1(config: Config, trials: pd.DataFrame, path) -> None:
    pr = _primary(trials).dropna(subset=["probe_response"])
    counts = pr["probe_response"].value_counts().reindex([1, 2, 3, 4], fill_value=0)
    per_subj = pr.groupby(["subject", "probe_response"]).size().unstack(fill_value=0).reindex(columns=[1, 2, 3, 4], fill_value=0)
    order = [s for s in per_subj.index if s not in PRIMARY_EXCLUDE]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    ax = axes[0]
    bars = ax.bar(range(1, 5), counts, color=[Q1_COLORS[i] for i in range(1, 5)])
    total = counts.sum()
    for b, i in zip(bars, range(1, 5)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 4, f"{int(counts[i])}\n({counts[i]/total*100:.1f}%)",
                ha="center", fontsize=9)
    ax.set_xticks(range(1, 5), [Q1_LABELS[i] for i in range(1, 5)], rotation=30)
    ax.set_ylabel("探针数")
    ax.set_title("Q1 注意状态分布（名义变量，不做均值）")
    _style_axis(ax)
    ax = axes[1]
    per_subj.div(per_subj.sum(axis=1), axis=0).reindex(order).plot(kind="bar", stacked=True, ax=ax,
        color=[Q1_COLORS[i] for i in range(1, 5)], legend=False)
    ax.set_xticks(range(len(order)), [s.replace("sub-", "") for s in order], rotation=90, fontsize=8)
    ax.set_ylabel("占比")
    ax.set_title("每被试 Q1 分布")
    _style_axis(ax)
    fig.suptitle(f"探针 Q1 注意状态（主队列，{int(total)} 探针）", fontsize=13)
    _save_figure(fig, path)


def plot_12_q2(config: Config, trials: pd.DataFrame, path) -> None:
    pr = _primary(trials).dropna(subset=["probe_vigilance"])
    counts = pr["probe_vigilance"].value_counts().reindex([1, 2, 3, 4], fill_value=0)
    cum = counts.cumsum() / counts.sum() * 100
    per_subj = pr.groupby(["subject", "probe_vigilance"]).size().unstack(fill_value=0).reindex(columns=[1, 2, 3, 4], fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    ax = axes[0]
    bars = ax.bar(range(1, 5), counts, color=[ACCENT, ACCENT2, "#7A5195", GOOD])
    total = counts.sum()
    for b, i in zip(bars, range(1, 5)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 4, f"{int(counts[i])}\n({counts[i]/total*100:.1f}%)",
                ha="center", fontsize=9)
    ax.set_xticks(range(1, 5), ["1 极困倦", "2", "3", "4 极清醒"])
    ax.set_ylabel("探针数")
    ax.set_title("Q2 警觉度分布（有序）")
    _style_axis(ax)
    ax = axes[1]
    ax2 = ax.twinx()
    ax.plot(range(1, 5), counts, marker="o", color=ACCENT, lw=2)
    ax2.plot(range(1, 5), cum, marker="s", color=WARNING, lw=2)
    ax2.set_ylabel("累计 %", color=WARNING)
    ax2.tick_params(axis="y", colors=WARNING)
    ax.set_xticks(range(1, 5), ["1 极困倦", "2", "3", "4 极清醒"])
    ax.set_title("Q2 分布与累计 %")
    _style_axis(ax)
    fig.suptitle(f"探针 Q2 警觉度（主队列，{int(total)} 探针）", fontsize=13)
    _save_figure(fig, path)


def plot_13_probe_behavior(config: Config, trials: pd.DataFrame, path) -> None:
    link = fmet.probe_behaviour_link(_primary(trials))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    ax = axes[0]
    d = link.dropna(subset=["probe_response", "post_go_rt_median_ms"])
    d.boxplot(column="post_go_rt_median_ms", by="probe_response", ax=ax, showfliers=False,
              patch_artist=True, boxprops=dict(facecolor="#E8F0F8"))
    ax.set_xticklabels([Q1_LABELS[i] for i in sorted(d["probe_response"].unique())], rotation=25)
    ax.set_ylabel("探针后正确 Go RT 中位(ms)")
    ax.set_title("探针后 RT × Q1 类别（名义，逐类展示）")
    _style_axis(ax)
    ax = axes[1]
    d = link.dropna(subset=["probe_vigilance", "post_go_rt_median_ms"])
    d.boxplot(column="post_go_rt_median_ms", by="probe_vigilance", ax=ax, showfliers=False,
              patch_artist=True, boxprops=dict(facecolor="#E8F0F8"))
    ax.set_xticklabels(["1 极困倦", "2", "3", "4 极清醒"])
    ax.set_ylabel("探针后正确 Go RT 中位(ms)")
    ax.set_title("探针后 RT × Q2 警觉度（有序）")
    _style_axis(ax)
    fig.suptitle("探针评分与探针后行为", fontsize=13)
    _save_figure(fig, path)


def plot_14_corr_heatmap(config: Config, trials: pd.DataFrame, path) -> None:
    blocks = fmet.formal_block_metrics(config, _primary(trials))
    corr = fstat.correlation_analysis(config, blocks)["corr_matrix"]
    labels = {
        "commission_rate": "误按率", "omission_rate": "漏按率", "dprime_loglinear": "d′",
        "c": "c", "go_rt_median_ms": "RT中位", "rt_cv": "RT-CV",
        "go_rt_lt_150_rate": "短RT率", "prestimulus_press_rate": "预判率",
    }
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    n = len(corr)
    for i in range(n):
        for j in range(n):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else "#1a1a1a")
    ax.set_xticks(range(n), [labels.get(c, c) for c in corr.columns], rotation=45, ha="right")
    ax.set_yticks(range(n), [labels.get(c, c) for c in corr.index])
    ax.set_title("指标相关矩阵（Spearman，subject×block 水平）")
    fig.colorbar(im, ax=ax, label="ρ")
    _save_figure(fig, path)


def plot_15_cross_block(config: Config, trials: pd.DataFrame, path) -> None:
    blocks = fmet.formal_block_metrics(config, _primary(trials))
    specs = [("commission_rate", "误按率", "%.2f"), ("omission_rate", "漏按率", "%.2f"),
             ("dprime_loglinear", "d′", "%.2f"), ("rt_cv", "RT-CV", "%.2f")]
    fig, axes = plt.subplots(1, len(specs), figsize=(16, 4.4), constrained_layout=True)
    for ax, (metric, label, fmt) in zip(axes, specs):
        wide = blocks.pivot_table(index="subject", columns="block_num", values=metric)[[1, 3]].dropna()
        from scipy import stats as sps
        rho, p = sps.spearmanr(wide[1], wide[3])
        ax.scatter(wide[1], wide[3], color=ACCENT, s=40, alpha=0.85)
        ax.plot([wide[1].min(), wide[1].max()], [wide[1].min(), wide[1].max()], color="#aaa", ls="--", lw=1)
        ax.set_xlabel(f"B1 {label}"); ax.set_ylabel(f"B3 {label}")
        ax.set_title(f"{label}\nρ={rho:.2f}, p={p:.3f}")
        _style_axis(ax)
    fig.suptitle("跨 block 一致性（B1 vs B3，n=19）", fontsize=14)
    _save_figure(fig, path)


def plot_16_sat(config: Config, trials: pd.DataFrame, path) -> None:
    blocks = fmet.formal_block_metrics(config, _primary(trials))
    d = blocks.dropna(subset=["dprime_loglinear", "rt_cv"])
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    sizes = (d["exg_tau"].fillna(d["exg_tau"].median()) / d["exg_tau"].median() * 120).clip(20, 300)
    sc = ax.scatter(d["dprime_loglinear"], d["rt_cv"], s=sizes, c=d["block_num"], cmap="viridis", alpha=0.85)
    from scipy import stats as sps
    rho, p = sps.spearmanr(d["dprime_loglinear"], d["rt_cv"])
    ax.set_xlabel("d′（loglinear）"); ax.set_ylabel("RT-CV")
    ax.set_title(f"速度-准确权衡：d′ × RT-CV\nρ={rho:.2f}, p<{p:.1e}；气泡大小=ex-Gaussian τ")
    cb = fig.colorbar(sc, ax=ax, label="block")
    cb.set_ticks([1, 2, 3])
    _style_axis(ax)
    _save_figure(fig, path)


def plot_17_window_evidence(config: Config, trials: pd.DataFrame, path) -> None:
    from .metrics import rolling_evidence_formal
    roll = rolling_evidence_formal(config, _primary(trials))
    roll = roll.loc[roll["time_window_sec"].eq(120)]
    status = roll.groupby(["subject", "block_num", "window_status"]).size().reset_index(name="n")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    ax = axes[0]
    tab = status.groupby(["block_num", "window_status"])["n"].sum().unstack(fill_value=0)
    tab = tab.reindex(columns=["insufficient_rt", "response_only", "insufficient_nogo", "full_evidence"], fill_value=0)
    labels_cn = {"insufficient_rt": "RT不足", "response_only": "仅反应", "insufficient_nogo": "No-Go不足", "full_evidence": "完整证据"}
    tab.columns = [labels_cn.get(c, c) for c in tab.columns]
    bottom = np.zeros(len(tab))
    for c in tab.columns:
        ax.bar(tab.index, tab[c], bottom=bottom, label=c)
        bottom += tab[c]
    ax.set_xticks([1, 2, 3], ["B1", "B2", "B3"])
    ax.set_ylabel("窗口数")
    ax.set_title("120s 滑窗证据状态组成")
    ax.legend(frameon=False, fontsize=8)
    _style_axis(ax)
    ax = axes[1]
    g = roll.groupby(["block_num", "window_end_ms"])["time_commission_jeffreys_mean"].agg(["mean", "sem"])
    for block in (1, 2, 3):
        sub = g.xs(block, level=0) if block in g.index.get_level_values(0) else None
        if sub is None or len(sub) == 0:
            continue
        x = (sub.index - sub.index.min()) / 60000
        ax.plot(x, sub["mean"], color=BLOCK_COLORS[block], lw=1.8, label=f"Block{block}")
        ax.fill_between(x, sub["mean"] - 1.96 * sub["sem"], sub["mean"] + 1.96 * sub["sem"],
                        color=BLOCK_COLORS[block], alpha=0.12)
    ax.set_xlabel("block 内时间（分钟）")
    ax.set_ylabel("误按率(Jeffreys)")
    ax.set_title("滑窗误按率随 block 时间轨迹")
    ax.legend(frameon=False, fontsize=8)
    _style_axis(ax)
    fig.suptitle("时间窗证据状态与误按风险", fontsize=14)
    _save_figure(fig, path)


def plot_18_window_trajectories(config: Config, trials: pd.DataFrame, path) -> None:
    bins = fmet.cycle_bin_metrics(config, _primary(trials))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for ax, (metric, title) in zip(axes, [
        ("rt_cv", "RT-CV"), ("commission_jeffreys_mean", "误按率(Jeffreys)"), ("omission_jeffreys_mean", "漏按率(Jeffreys)")]):
        for block in (1, 2, 3):
            sub = bins.loc[bins["block_num"].eq(block)]
            m = _group_mean_ci(sub, "cycle_bin", metric)
            ax.plot(m["cycle_bin"], m["mean"], color=BLOCK_COLORS[block], marker="o", lw=2, label=f"Block{block}")
            ax.fill_between(m["cycle_bin"], m["lo"], m["hi"], color=BLOCK_COLORS[block], alpha=0.13)
        ax.set_xlabel("周期 bin（1 前段 → 6 后段）")
        ax.set_ylabel(title)
        ax.set_title(f"block 内轨迹：{title}")
        ax.legend(frameon=False, fontsize=8)
        _style_axis(ax)
    fig.suptitle("block 内稳定性与错误轨迹（组均值±95%CI，n=19）", fontsize=14)
    _save_figure(fig, path)


def plot_19_probe_pre_post(config: Config, trials: pd.DataFrame, path) -> None:
    from .metrics import probe_evidence_formal
    pe = probe_evidence_formal(config, _primary(trials))
    pe = pe.loc[pe["time_window_sec"].eq(60) & pe["nogo_window_target"].eq(6)]
    g = pe.groupby(["window_kind"]).agg(
        rt=("go_rt_median_ms", "mean"), rt_sem=("go_rt_median_ms", "sem"),
        comm=("time_commission_jeffreys_mean", "mean"), comm_sem=("time_commission_jeffreys_mean", "sem"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    for ax, (metric, ylabel, title) in zip(axes, [
        ("rt", "正确 Go RT 中位(ms)", "探针前/后 60s 窗口 RT"),
        ("comm", "误按率(Jeffreys)", "探针前/后 60s 窗口误按率")]):
        x = [0, 1]
        vals = [g.loc["pre", metric], g.loc["post", metric]]
        errs = [1.96 * g.loc["pre", metric + "_sem"], 1.96 * g.loc["post", metric + "_sem"]]
        ax.errorbar(x, vals, yerr=errs, color=ACCENT, marker="o", lw=2, capsize=5)
        ax.set_xticks(x, ["探针前", "探针后"])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _style_axis(ax)
    fig.suptitle("探针前/后窗口行为对比（60s，组均值±95%CI）", fontsize=13)
    _save_figure(fig, path)


def plot_20_stimulus_size(config: Config, trials: pd.DataFrame, path) -> None:
    tr = _primary(trials).copy()
    tr["size_label"] = tr["stimulus_size"].astype(int).map({80: "80%", 100: "100%", 120: "120%"})
    go = tr.loc[tr["is_no_go"].eq(0)]
    nogo = tr.loc[tr["is_no_go"].eq(1)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    ax = axes[0]
    g = go.groupby(["block_num", "size_label"])["go_rt_valid"].agg(["median", "mean", "sem"])
    for block in (1, 2, 3):
        sub = g.xs(block, level=0) if block in g.index.get_level_values(0) else None
        if sub is None:
            continue
        ax.errorbar([1, 2, 3], sub["mean"], yerr=1.96 * sub["sem"], color=BLOCK_COLORS[block], marker="o", label=f"Block{block}")
    ax.set_xticks([1, 2, 3], ["80%", "100%", "120%"])
    ax.set_ylabel("正确 Go RT 均值(ms)")
    ax.set_title("刺激尺寸 × block → Go RT")
    ax.legend(frameon=False, fontsize=8)
    _style_axis(ax)
    ax = axes[1]
    g = nogo.groupby(["block_num", "size_label"])["commission"].agg(["mean", "sem"])
    for block in (1, 2, 3):
        sub = g.xs(block, level=0) if block in g.index.get_level_values(0) else None
        if sub is None:
            continue
        ax.errorbar([1, 2, 3], sub["mean"], yerr=1.96 * sub["sem"], color=BLOCK_COLORS[block], marker="o", label=f"Block{block}")
    ax.set_xticks([1, 2, 3], ["80%", "100%", "120%"])
    ax.set_ylabel("误按率")
    ax.set_title("刺激尺寸 × block → 误按率")
    ax.legend(frameon=False, fontsize=8)
    _style_axis(ax)
    fig.suptitle("刺激尺寸效应（经典 SART 尺寸变化检验）", fontsize=13)
    _save_figure(fig, path)


def generate_all(config: Config, trials: pd.DataFrame) -> dict:
    _configure_chinese_font()
    reports = config.path_value("output_root") / "000-reports"
    reports.mkdir(parents=True, exist_ok=True)
    jobs = [
        (plot_01_completeness, "051-01-数据完整性热图.png"),
        (plot_02_rt_distribution, "051-02-RT分布与ECDF.png"),
        (plot_03_rt_qc, "051-03-RT区间组成.png"),
        (plot_04_block_trajectory, "051-04-Block主效应轨迹.png"),
        (plot_05_b1_b3_paired, "051-05-B1与B3配对变化.png"),
        (plot_06_block_bin_interaction, "051-06-Block×bin交互.png"),
        (plot_07_cycle_trends, "051-07-周期内趋势.png"),
        (plot_08_rt_drift, "051-08-RT漂移混合模型.png"),
        (plot_09_pre_nogo, "051-09-错误前RT轨迹.png"),
        (plot_10_prestimulus, "051-10-预判按键.png"),
        (plot_11_q1, "051-11-探针Q1注意状态.png"),
        (plot_12_q2, "051-12-探针Q2警觉度.png"),
        (plot_13_probe_behavior, "051-13-探针与行为.png"),
        (plot_14_corr_heatmap, "051-14-相关热图.png"),
        (plot_15_cross_block, "051-15-跨block一致性.png"),
        (plot_16_sat, "051-16-速度准确权衡.png"),
        (plot_17_window_evidence, "051-17-窗口证据状态.png"),
        (plot_18_window_trajectories, "051-18-窗内轨迹.png"),
        (plot_19_probe_pre_post, "051-19-探针前后窗.png"),
        (plot_20_stimulus_size, "051-20-刺激尺寸效应.png"),
    ]
    made = []
    for fn, name in jobs:
        try:
            fn(config, trials, reports / name)
            made.append(name)
        except Exception as exc:  # 单图失败不阻塞整体
            print(f"[figures] FAIL {name}: {exc}")
    return {"figures": made, "count": len(made)}
