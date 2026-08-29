"""Chinese publication-facing figures for behavior science v3."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def generate_behavior_figures(
    block: pd.DataFrame,
    primary_probe: pd.DataFrame,
    output_dir: Path,
    *,
    error_summary: pd.DataFrame | None = None,
) -> list[str]:
    """Generate unit-safe Chinese charts without pseudo-independent sample claims."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    if not block.empty and {"block_id", "go_correct_rt_median_ms"}.issubset(block.columns):
        summary = block.groupby("block_id")["go_correct_rt_median_ms"].agg(["mean", "sem"]).reset_index()
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        x = np.arange(len(summary))
        ax.errorbar(x, summary["mean"], yerr=summary["sem"], marker="o", capsize=4)
        ax.set_xticks(x, summary["block_id"].astype(str))
        ax.set_xlabel("区块（场次内重复测量）")
        ax.set_ylabel("正确 Go RT 中位数（ms）")
        ax.set_title("B1–B2 反应时变化（描述性）")
        ax.text(
            0.01,
            0.01,
            "图中误差线仅为描述性；正式 B1–B2 不确定性按参与者组聚类 bootstrap",
            transform=ax.transAxes,
            fontsize=8,
        )
        files.append(_save(fig, output_dir / "行为图01_B1B2_RT.png"))

    if not block.empty and {"block_id", "omission_rate", "commission_rate"}.issubset(block.columns):
        long = block.melt(
            id_vars=["block_id"],
            value_vars=["omission_rate", "commission_rate"],
            var_name="error_type",
            value_name="rate",
        )
        summary = long.groupby(["block_id", "error_type"])["rate"].mean().unstack("error_type")
        summary = summary.rename(columns={"omission_rate": "Go 遗漏率", "commission_rate": "No-Go 误按率"})
        fig, ax = plt.subplots(figsize=(7, 4.3))
        summary.plot(kind="bar", ax=ax)
        ax.set_xlabel("区块")
        ax.set_ylabel("错误率（比例）")
        ax.set_title("Go 遗漏与 No-Go 误按分开报告")
        ax.legend(title="错误类型")
        files.append(_save(fig, output_dir / "行为图02_遗漏与误按.png"))

    if not primary_probe.empty and {"q1_nominal_4class", "go_correct_rt_median_ms"}.issubset(primary_probe.columns):
        d = primary_probe.dropna(subset=["q1_nominal_4class", "go_correct_rt_median_ms"]).copy()
        if not d.empty:
            summary = d.groupby("q1_nominal_4class")["go_correct_rt_median_ms"].agg(["mean", "sem"])
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            # Q1 is nominal: points/error bars are intentionally NOT connected.
            ax.errorbar(
                summary.index.astype(int),
                summary["mean"],
                yerr=summary["sem"],
                fmt="o",
                linestyle="none",
                capsize=4,
            )
            ax.set_xticks([1, 2, 3, 4])
            ax.set_xlabel("Q1 类别（名义四分类；类别之间无顺序假设）")
            ax.set_ylabel("探针前正确 Go RT 中位数（ms）")
            ax.set_title("Q1 与探针前行为（30 秒主窗）")
            files.append(_save(fig, output_dir / "行为图03_Q1主探针.png"))

    if not primary_probe.empty and {"q2_ordinal_4level", "go_correct_rt_median_ms"}.issubset(primary_probe.columns):
        d = primary_probe.dropna(subset=["q2_ordinal_4level", "go_correct_rt_median_ms"]).copy()
        if not d.empty:
            summary = d.groupby("q2_ordinal_4level")["go_correct_rt_median_ms"].agg(["mean", "sem"])
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            ax.errorbar(summary.index.astype(int), summary["mean"], yerr=summary["sem"], marker="o", capsize=4)
            ax.set_xticks([1, 2, 3, 4])
            ax.set_xlabel("Q2 警觉程度（有序 1–4）")
            ax.set_ylabel("探针前正确 Go RT 中位数（ms）")
            ax.set_title("Q2 与探针前行为（30 秒主窗）")
            files.append(_save(fig, output_dir / "行为图04_Q2主探针.png"))

    if error_summary is not None and not error_summary.empty:
        required = {"error_type", "relative_trial", "participant_centered_rt_mean_ms", "participant_centered_rt_sem_ms"}
        if required.issubset(error_summary.columns):
            fig, ax = plt.subplots(figsize=(7, 4.4))
            names = {"go_omission": "Go 遗漏", "nogo_commission": "No-Go 误按"}
            for error_type, current in error_summary.groupby("error_type", sort=True):
                current = current.sort_values("relative_trial")
                ax.errorbar(
                    current["relative_trial"],
                    current["participant_centered_rt_mean_ms"],
                    yerr=current["participant_centered_rt_sem_ms"],
                    marker="o",
                    capsize=3,
                    label=names.get(str(error_type), str(error_type)),
                )
            ax.axvline(0, linestyle="--", linewidth=1)
            ax.axhline(0, linestyle=":", linewidth=1)
            ax.set_xlabel("相对错误事件的试次位置（0=错误事件）")
            ax.set_ylabel("被试内中心化正确 Go RT（ms）")
            ax.set_title("错误事件前后局部行为轨迹")
            ax.legend(title="错误类型")
            files.append(_save(fig, output_dir / "行为图05_错误事件轨迹.png"))

    return files
