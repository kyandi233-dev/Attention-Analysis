"""Core figures for the current two-block formal behavior analysis."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import Config
from . import metrics as fmet


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def generate_all(config: Config, trials: pd.DataFrame, output_dir: Path | None = None) -> list[str]:
    output_dir = output_dir or (config.path_value("output_root") / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    counts = trials.pivot_table(index="subject", columns="block_num", values="trial_num", aggfunc="count").reindex(columns=[1, 2])
    fig, ax = plt.subplots(figsize=(6, max(4, 0.28 * len(counts))))
    im = ax.imshow(counts.to_numpy(dtype=float), aspect="auto")
    ax.set_xticks([0, 1], ["B1", "B2"])
    ax.set_yticks(range(len(counts)), counts.index)
    ax.set_title("Formal behavior completeness: subject x block")
    fig.colorbar(im, ax=ax, label="trials")
    generated.append(_save(fig, output_dir / "data-completeness.png"))

    rt = trials["go_rt_valid"].dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rt, bins=80)
    ax.set_xlabel("correct Go RT (ms)")
    ax.set_ylabel("trials")
    ax.set_title("Correct Go RT distribution")
    generated.append(_save(fig, output_dir / "go-rt-distribution.png"))

    blocks = fmet.formal_block_metrics(config, trials)
    metrics = [("commission_rate", "No-Go commission rate"), ("omission_rate", "Go omission rate"), ("dprime_loglinear", "d-prime"), ("go_rt_median_ms", "Go RT median (ms)"), ("rt_cv", "RT CV"), ("exg_tau", "ex-Gaussian tau (ms)")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for _, subject_df in blocks.groupby("subject"):
            subject_df = subject_df.sort_values("block_num")
            ax.plot(subject_df["block_num"], subject_df[metric], alpha=0.25)
        means = blocks.groupby("block_num")[metric].mean()
        ax.plot(means.index, means.values, marker="o", linewidth=2.5)
        ax.set_xticks([1, 2], ["B1", "B2"])
        ax.set_title(title)
    generated.append(_save(fig, output_dir / "block-paired-metrics.png"))

    bins = fmet.cycle_bin_metrics(config, trials)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for block_num in (1, 2):
        subset = bins.loc[bins["block_num"].eq(block_num)]
        rt_mean = subset.groupby("cycle_bin")["go_rt_median_ms"].mean()
        comm_mean = subset.groupby("cycle_bin")["commission_rate"].mean()
        axes[0].plot(rt_mean.index, rt_mean.values, marker="o", label=f"B{block_num}")
        axes[1].plot(comm_mean.index, comm_mean.values, marker="o", label=f"B{block_num}")
    axes[0].set_title("RT trajectory within block")
    axes[0].set_ylabel("Go RT median (ms)")
    axes[1].set_title("Commission trajectory within block")
    axes[1].set_ylabel("commission rate")
    for ax in axes:
        ax.set_xlabel("cycle bin")
        ax.legend()
    generated.append(_save(fig, output_dir / "block-by-cycle-bin.png"))

    probes = trials.loc[trials["is_probe"].eq(1)]
    if len(probes):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        probes["probe_response"].dropna().value_counts().sort_index().plot.bar(ax=axes[0])
        probes["probe_vigilance"].dropna().value_counts().sort_index().plot.bar(ax=axes[1])
        axes[0].set_title("Probe Q1 raw code")
        axes[1].set_title("Probe Q2 vigilance raw code")
        generated.append(_save(fig, output_dir / "probe-distributions.png"))
    return generated
