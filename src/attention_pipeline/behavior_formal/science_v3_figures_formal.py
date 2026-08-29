"""Chinese publication-facing figures for behavior science v3.

Formal inference is never reconstructed from pseudo-independent probe/trial
rows in this module.  Descriptive uncertainty is participant-first, while the
B1/B2 inferential panel consumes the participant-cluster bootstrap table.
"""
from __future__ import annotations

from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METRIC_LABELS = {
    "go_correct_rt_mean_ms": "正确 Go RT 均值（ms）",
    "go_correct_rt_median_ms": "正确 Go RT 中位数（ms）",
    "go_correct_rt_sd_ms": "正确 Go RT SD（ms）",
    "go_correct_rt_mad_ms": "正确 Go RT MAD（ms）",
    "go_correct_rt_iqr_ms": "正确 Go RT IQR（ms）",
    "go_correct_rt_cv": "正确 Go RT 变异系数",
    "go_correct_rt_theilsen_slope_ms_per_s": "正确 Go RT 斜率（ms/s）",
    "omission_rate": "Go 遗漏率",
    "commission_rate": "No-Go 误按率",
    "dprime_loglinear": "d′",
    "criterion_c": "判别标准 c",
    "beta": "β",
}

BEHAVIOR_FIGURE_CONTRACT = {
    "行为图01_B1B2配对轨迹.png": ("B1–B2 场次内配对轨迹", "区块", "正确 Go RT 中位数（ms）"),
    "行为图02_B1B2聚类效应.png": ("B1–B2 参与者聚类效应", "B2−B1 效应", "候选行为指标"),
    "行为图03_遗漏与误按.png": ("Go 遗漏与 No-Go 误按", "区块", "错误率（比例）"),
    "行为图04_Q1主探针.png": ("Q1 与探针前行为（30 秒主窗）", "Q1 类别（名义四分类）", "正确 Go RT 中位数（ms）"),
    "行为图05_Q2主探针.png": ("Q2 与探针前行为（30 秒主窗）", "Q2 警觉程度（有序 1–4）", "正确 Go RT 中位数（ms）"),
    "行为图06_错误事件轨迹.png": ("错误事件前后局部行为轨迹", "相对错误事件的试次位置", "被试内中心化正确 Go RT（ms）"),
    "行为图07_候选指标覆盖.png": ("候选行为指标的可计算覆盖", "分析尺度", "候选行为指标"),
    "行为图08_候选指标冗余.png": ("场次级候选行为指标冗余", "候选行为指标", "候选行为指标"),
    "行为图09_Q1Q2类别覆盖.png": ("主观探针类别覆盖", "类别", "参与者组数量"),
    "行为图10_任务时间进程.png": ("区块内任务时间进程", "cycle 序号", "正确 Go RT 中位数（ms）"),
    "行为图11_场次级核心指标分布.png": ("场次级核心行为指标原始分布", "指标取值", "场次数量"),
}


def formal_figure_contract_is_chinese() -> bool:
    """Machine-readable guard for publication-facing Chinese labels."""
    return all(
        re.search(r"[\u4e00-\u9fff]", text)
        for triple in BEHAVIOR_FIGURE_CONTRACT.values()
        for text in triple
    )


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _sem(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    return float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) >= 2 else math.nan


def _participant_first(
    frame: pd.DataFrame,
    *,
    group_cols: list[str],
    value_col: str,
    participant_col: str = "repeat_participant_id",
) -> pd.DataFrame:
    """Collapse repeated rows within participant before any descriptive SEM."""
    needed = {participant_col, value_col, *group_cols}
    if not needed.issubset(frame.columns):
        return pd.DataFrame()
    participant = (
        frame[[participant_col, *group_cols, value_col]]
        .assign(**{value_col: pd.to_numeric(frame[value_col], errors="coerce")})
        .dropna(subset=[participant_col, value_col])
        .groupby([*group_cols, participant_col], as_index=False, dropna=False)[value_col]
        .mean()
    )
    if participant.empty:
        return participant
    summary = (
        participant.groupby(group_cols, as_index=False, dropna=False)[value_col]
        .agg(mean="mean", participant_group_n="count")
    )
    sem = (
        participant.groupby(group_cols, dropna=False)[value_col]
        .apply(_sem)
        .rename("participant_sem")
        .reset_index()
    )
    return summary.merge(sem, on=group_cols, how="left", validate="one_to_one")


def _counts_text(frame: pd.DataFrame) -> str:
    participant_n = (
        int(frame["repeat_participant_id"].dropna().astype(str).nunique())
        if "repeat_participant_id" in frame else 0
    )
    session_n = (
        int(frame["session_id"].dropna().astype(str).nunique())
        if "session_id" in frame else 0
    )
    return f"观察单位按图中说明；参与者组 N={participant_n}，session N={session_n}"


def _metric_label(name: object) -> str:
    return METRIC_LABELS.get(str(name), str(name))


def generate_behavior_figures(
    block: pd.DataFrame,
    primary_probe: pd.DataFrame,
    output_dir: Path,
    *,
    session: pd.DataFrame | None = None,
    cycle: pd.DataFrame | None = None,
    error_summary: pd.DataFrame | None = None,
    b1b2_clustered: pd.DataFrame | None = None,
    candidate_validation: pd.DataFrame | None = None,
    metric_redundancy: pd.DataFrame | None = None,
) -> list[str]:
    """Generate Chinese formal/support figures with explicit observation units."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    # 01: session-internal paired trajectories; descriptive only.
    required = {"session_id", "repeat_participant_id", "block_id", "go_correct_rt_median_ms"}
    if not block.empty and required.issubset(block.columns):
        wide = block.pivot_table(
            index=["repeat_participant_id", "session_id"],
            columns="block_id",
            values="go_correct_rt_median_ms",
            aggfunc="first",
        ).dropna(subset=[c for c in ("B1", "B2") if c in block["block_id"].astype(str).unique()])
        if {"B1", "B2"}.issubset(wide.columns) and not wide.empty:
            fig, ax = plt.subplots(figsize=(7.0, 4.5))
            for _, row in wide.iterrows():
                ax.plot([1, 2], [row["B1"], row["B2"]], marker="o", alpha=0.25, linewidth=0.8)
            means = [float(wide["B1"].mean()), float(wide["B2"].mean())]
            ax.plot([1, 2], means, marker="o", linewidth=2.5, label="session 配对均值")
            title, xlabel, ylabel = BEHAVIOR_FIGURE_CONTRACT["行为图01_B1B2配对轨迹.png"]
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["B1", "B2"])
            ax.legend(title="描述性汇总")
            ax.text(
                0.01, 0.01,
                f"每条细线=一个 session；session N={len(wide)}；参与者组 N={wide.index.get_level_values(0).nunique()}。正式效应见图02。",
                transform=ax.transAxes, fontsize=8,
            )
            files.append(_save(fig, output_dir / "行为图01_B1B2配对轨迹.png"))

    # 02: participant-cluster bootstrap CI, not ordinary session SEM.
    if b1b2_clustered is not None and not b1b2_clustered.empty:
        needed = {"metric", "estimate_b2_minus_b1", "ci_low", "ci_high", "participant_group_n", "session_pair_n"}
        d = b1b2_clustered.dropna(subset=[c for c in needed if c in b1b2_clustered.columns]).copy()
        if needed.issubset(d.columns) and not d.empty:
            d = d.sort_values("metric", kind="stable").reset_index(drop=True)
            y = np.arange(len(d))
            est = pd.to_numeric(d["estimate_b2_minus_b1"], errors="coerce").to_numpy(dtype=float)
            lo = pd.to_numeric(d["ci_low"], errors="coerce").to_numpy(dtype=float)
            hi = pd.to_numeric(d["ci_high"], errors="coerce").to_numpy(dtype=float)
            fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.42 * len(d) + 1.5)))
            ax.errorbar(est, y, xerr=np.vstack([est - lo, hi - est]), fmt="o", capsize=3)
            ax.axvline(0, linestyle="--", linewidth=1)
            ax.set_yticks(y)
            ax.set_yticklabels([_metric_label(x) for x in d["metric"]])
            title, xlabel, ylabel = BEHAVIOR_FIGURE_CONTRACT["行为图02_B1B2聚类效应.png"]
            ax.set_title(title)
            ax.set_xlabel(f"{xlabel}（95% 参与者聚类 bootstrap CI）")
            ax.set_ylabel(ylabel)
            ax.text(
                0.01, 0.01,
                f"推断单位=参与者组；参与者组 N 范围 {int(d['participant_group_n'].min())}–{int(d['participant_group_n'].max())}；session 配对 N 范围 {int(d['session_pair_n'].min())}–{int(d['session_pair_n'].max())}。",
                transform=ax.transAxes, fontsize=8,
            )
            files.append(_save(fig, output_dir / "行为图02_B1B2聚类效应.png"))

    # 03: separate error mechanisms, participant-first descriptive uncertainty.
    if not block.empty and {"block_id", "omission_rate", "commission_rate", "repeat_participant_id"}.issubset(block.columns):
        long = block.melt(
            id_vars=["repeat_participant_id", "session_id", "block_id"],
            value_vars=["omission_rate", "commission_rate"],
            var_name="error_type", value_name="rate",
        )
        summary = _participant_first(long, group_cols=["block_id", "error_type"], value_col="rate")
        if not summary.empty:
            blocks = sorted(summary["block_id"].astype(str).unique())
            errors = [e for e in ("omission_rate", "commission_rate") if e in set(summary["error_type"])]
            x = np.arange(len(blocks), dtype=float)
            width = 0.36
            fig, ax = plt.subplots(figsize=(7.0, 4.5))
            for j, error in enumerate(errors):
                cur = summary[summary["error_type"].eq(error)].set_index("block_id").reindex(blocks)
                pos = x + (j - (len(errors) - 1) / 2.0) * width
                ax.bar(pos, cur["mean"], width=width, label=_metric_label(error))
                ax.errorbar(pos, cur["mean"], yerr=cur["participant_sem"], fmt="none", capsize=3)
            title, xlabel, ylabel = BEHAVIOR_FIGURE_CONTRACT["行为图03_遗漏与误按.png"]
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels(blocks)
            ax.legend(title="错误类型")
            ax.text(0.01, 0.01, _counts_text(block) + "；误差线=参与者先汇总后的描述性 SEM。", transform=ax.transAxes, fontsize=8)
            files.append(_save(fig, output_dir / "行为图03_遗漏与误按.png"))

    # 04/05: participant-first probe summaries. Q1 is never connected as ordered.
    probe_specs = [
        ("q1_nominal_4class", "行为图04_Q1主探针.png", False),
        ("q2_ordinal_4level", "行为图05_Q2主探针.png", True),
    ]
    for category, filename, ordered in probe_specs:
        needed = {category, "go_correct_rt_median_ms", "repeat_participant_id"}
        if primary_probe.empty or not needed.issubset(primary_probe.columns):
            continue
        summary = _participant_first(
            primary_probe.dropna(subset=[category, "go_correct_rt_median_ms"]),
            group_cols=[category], value_col="go_correct_rt_median_ms",
        )
        if summary.empty:
            continue
        summary[category] = pd.to_numeric(summary[category], errors="coerce")
        summary = summary.dropna(subset=[category]).sort_values(category)
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        x = summary[category].to_numpy(dtype=float)
        if ordered:
            ax.errorbar(x, summary["mean"], yerr=summary["participant_sem"], marker="o", capsize=4)
        else:
            ax.errorbar(x, summary["mean"], yerr=summary["participant_sem"], fmt="o", linestyle="none", capsize=4)
        title, xlabel, ylabel = BEHAVIOR_FIGURE_CONTRACT[filename]
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks([1, 2, 3, 4])
        ax.text(0.01, 0.01, _counts_text(primary_probe) + "；误差线=参与者先汇总后的描述性 SEM。", transform=ax.transAxes, fontsize=8)
        files.append(_save(fig, output_dir / filename))

    # 06: error trajectories are already participant-first in the upstream summary.
    if error_summary is not None and not error_summary.empty:
        needed = {"error_type", "relative_trial", "participant_centered_rt_mean_ms", "participant_centered_rt_sem_ms"}
        if needed.issubset(error_summary.columns):
            fig, ax = plt.subplots(figsize=(7.0, 4.5))
            names = {"go_omission": "Go 遗漏", "nogo_commission": "No-Go 误按"}
            for error_type, current in error_summary.groupby("error_type", sort=True):
                current = current.sort_values("relative_trial")
                ax.errorbar(
                    current["relative_trial"], current["participant_centered_rt_mean_ms"],
                    yerr=current["participant_centered_rt_sem_ms"], marker="o", capsize=3,
                    label=names.get(str(error_type), str(error_type)),
                )
            ax.axvline(0, linestyle="--", linewidth=1)
            ax.axhline(0, linestyle=":", linewidth=1)
            title, xlabel, ylabel = BEHAVIOR_FIGURE_CONTRACT["行为图06_错误事件轨迹.png"]
            ax.set_title(title)
            ax.set_xlabel(f"{xlabel}（0=错误事件）")
            ax.set_ylabel(ylabel)
            ax.legend(title="错误类型")
            if {"participant_group_n", "session_n", "error_event_n"}.issubset(error_summary.columns):
                ax.text(
                    0.01, 0.01,
                    f"汇总单位=参与者组；参与者组 N 最大={int(error_summary['participant_group_n'].max())}；session N 最大={int(error_summary['session_n'].max())}；错误事件 N 总计={int(error_summary['error_event_n'].max())}（各相对位置覆盖不同）。",
                    transform=ax.transAxes, fontsize=8,
                )
            files.append(_save(fig, output_dir / "行为图06_错误事件轨迹.png"))

    # 07: coverage heatmap across scales and candidates.
    if candidate_validation is not None and not candidate_validation.empty and {"scale", "metric", "coverage"}.issubset(candidate_validation.columns):
        pivot = candidate_validation.pivot_table(index="metric", columns="scale", values="coverage", aggfunc="first")
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(8.0, max(5.0, 0.42 * len(pivot) + 1.5)))
            image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1)
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels([str(x) for x in pivot.columns])
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels([_metric_label(x) for x in pivot.index])
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    value = pivot.iloc[i, j]
                    if pd.notna(value):
                        ax.text(j, i, f"{100*float(value):.0f}%", ha="center", va="center", fontsize=7)
            title, xlabel, ylabel = BEHAVIOR_FIGURE_CONTRACT["行为图07_候选指标覆盖.png"]
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            fig.colorbar(image, ax=ax, label="可计算比例")
            files.append(_save(fig, output_dir / "行为图07_候选指标覆盖.png"))

    # 08: session-scale redundancy, descriptive Spearman matrix.
    if metric_redundancy is not None and not metric_redundancy.empty and {"scale", "metric_a", "metric_b", "spearman_r"}.issubset(metric_redundancy.columns):
        d = metric_redundancy[metric_redundancy["scale"].astype(str).eq("session")].copy()
        if d.empty:
            first_scale = str(metric_redundancy["scale"].dropna().astype(str).iloc[0])
            d = metric_redundancy[metric_redundancy["scale"].astype(str).eq(first_scale)].copy()
        metrics = sorted(set(d["metric_a"].astype(str)) | set(d["metric_b"].astype(str)))
        if metrics:
            matrix = pd.DataFrame(np.eye(len(metrics)), index=metrics, columns=metrics)
            for row in d.itertuples(index=False):
                r = pd.to_numeric(pd.Series([row.spearman_r]), errors="coerce").iloc[0]
                matrix.loc[str(row.metric_a), str(row.metric_b)] = r
                matrix.loc[str(row.metric_b), str(row.metric_a)] = r
            fig, ax = plt.subplots(figsize=(max(7.0, 0.62 * len(metrics)), max(6.0, 0.58 * len(metrics))))
            image = ax.imshow(matrix.to_numpy(dtype=float), vmin=-1, vmax=1)
            ax.set_xticks(np.arange(len(metrics)))
            ax.set_xticklabels([_metric_label(x) for x in metrics], rotation=55, ha="right")
            ax.set_yticks(np.arange(len(metrics)))
            ax.set_yticklabels([_metric_label(x) for x in metrics])
            title, xlabel, ylabel = BEHAVIOR_FIGURE_CONTRACT["行为图08_候选指标冗余.png"]
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            fig.colorbar(image, ax=ax, label="Spearman 相关系数")
            files.append(_save(fig, output_dir / "行为图08_候选指标冗余.png"))

    # 09: category coverage is counts, not a psychometric score.
    if not primary_probe.empty and "repeat_participant_id" in primary_probe.columns:
        rows = []
        for label, column in (("Q1", "q1_nominal_4class"), ("Q2", "q2_ordinal_4level")):
            if column not in primary_probe:
                continue
            current = primary_probe.dropna(subset=[column]).copy()
            current[column] = pd.to_numeric(current[column], errors="coerce")
            for category, group in current.groupby(column):
                rows.append({
                    "question": label,
                    "category": int(category),
                    "participant_group_n": int(group["repeat_participant_id"].nunique()),
                    "session_n": int(group["session_id"].nunique()) if "session_id" in group else 0,
                    "probe_n": int(len(group)),
                })
        coverage = pd.DataFrame(rows)
        if not coverage.empty:
            fig, ax = plt.subplots(figsize=(7.2, 4.5))
            x = np.arange(4, dtype=float)
            width = 0.36
            for j, question in enumerate([q for q in ("Q1", "Q2") if q in set(coverage["question"])]):
                cur = coverage[coverage["question"].eq(question)].set_index("category").reindex([1, 2, 3, 4])
                pos = x + (j - 0.5) * width
                bars = ax.bar(pos, cur["participant_group_n"], width=width, label=question)
                for bar, category in zip(bars, [1, 2, 3, 4]):
                    if category in cur.index and pd.notna(cur.loc[category, "probe_n"]):
                        ax.text(
                            bar.get_x() + bar.get_width()/2, bar.get_height(),
                            f"探针{int(cur.loc[category, 'probe_n'])}\nsession{int(cur.loc[category, 'session_n'])}",
                            ha="center", va="bottom", fontsize=7,
                        )
            title, xlabel, ylabel = BEHAVIOR_FIGURE_CONTRACT["行为图09_Q1Q2类别覆盖.png"]
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels(["1", "2", "3", "4"])
            ax.legend(title="探针问题")
            files.append(_save(fig, output_dir / "行为图09_Q1Q2类别覆盖.png"))

    # 10: participant-first time-on-task trajectories.
    if cycle is not None and not cycle.empty and {"block_id", "cycle_bin", "repeat_participant_id", "go_correct_rt_median_ms"}.issubset(cycle.columns):
        summary = _participant_first(cycle, group_cols=["block_id", "cycle_bin"], value_col="go_correct_rt_median_ms")
        if not summary.empty:
            fig, ax = plt.subplots(figsize=(7.2, 4.5))
            for block_id, current in summary.groupby("block_id", sort=True):
                current = current.sort_values("cycle_bin")
                ax.errorbar(
                    current["cycle_bin"], current["mean"], yerr=current["participant_sem"],
                    marker="o", capsize=3, label=str(block_id),
                )
            title, xlabel, ylabel = BEHAVIOR_FIGURE_CONTRACT["行为图10_任务时间进程.png"]
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend(title="区块")
            ax.text(0.01, 0.01, _counts_text(cycle) + "；误差线=参与者先汇总后的描述性 SEM；正式趋势推断见 block×cycle GEE 表。", transform=ax.transAxes, fontsize=8)
            files.append(_save(fig, output_dir / "行为图10_任务时间进程.png"))

    # 11: raw session-level distributions for core candidates.
    if session is not None and not session.empty:
        metrics = [m for m in ("go_correct_rt_median_ms", "go_correct_rt_cv", "omission_rate", "commission_rate") if m in session]
        if metrics:
            fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.7))
            flat = list(np.ravel(axes))
            for ax, metric in zip(flat, metrics):
                values = pd.to_numeric(session[metric], errors="coerce").dropna()
                ax.hist(values, bins=min(12, max(4, int(np.sqrt(max(1, len(values)))))))
                ax.set_title(_metric_label(metric))
                ax.set_xlabel("指标取值")
                ax.set_ylabel("场次数量")
            for ax in flat[len(metrics):]:
                ax.axis("off")
            fig.suptitle(BEHAVIOR_FIGURE_CONTRACT["行为图11_场次级核心指标分布.png"][0])
            fig.text(0.02, 0.01, _counts_text(session) + "；本图仅描述原始分布，不作为显著性筛选依据。", fontsize=8)
            files.append(_save(fig, output_dir / "行为图11_场次级核心指标分布.png"))

    return files
