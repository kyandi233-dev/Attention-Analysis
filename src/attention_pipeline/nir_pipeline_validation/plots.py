from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from .analysis import VALIDATION_LABEL


def _banner(fig: plt.Figure) -> None:
    fig.text(0.5, 0.006, VALIDATION_LABEL, ha="center", va="bottom", fontsize=8)


def _save(fig: plt.Figure, base: Path, formats: list[str], dpi: int) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for fmt in formats:
        path = base.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def plot_pipeline_schematic(
    *,
    base: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axis("off")

    boxes = [
        (0.03, 0.58, 0.18, 0.22, "10_analysis_ready\ncontinuous subject-level PIR"),
        (0.29, 0.58, 0.18, 0.22, "11_analysis_tables\ntrial / probe / 1-s bins"),
        (0.55, 0.58, 0.18, 0.22, "12_pipeline_validation\nsmoke-test analyses"),
        (0.79, 0.58, 0.18, 0.22, "20_formal_statistics\nfuture only"),
    ]
    for x, y, w, h, label in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02",
            transform=ax.transAxes,
            fill=False,
            linewidth=1.5,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", transform=ax.transAxes)

    for x0, x1 in ((0.21, 0.29), (0.47, 0.55), (0.73, 0.79)):
        ax.annotate(
            "",
            xy=(x1, 0.69),
            xytext=(x0, 0.69),
            xycoords=ax.transAxes,
            arrowprops={"arrowstyle": "->", "linewidth": 1.4},
        )

    branches = [
        (0.58, 0.33, "Behavior / omission QC"),
        (0.68, 0.23, "Time-on-task / Block"),
        (0.54, 0.13, "Trial outcome / RT"),
        (0.42, 0.23, "Probe windows"),
        (0.34, 0.33, "Model smoke tests"),
    ]
    for x, y, label in branches:
        ax.text(x, y, label, ha="center", va="center", transform=ax.transAxes)
        ax.annotate(
            "",
            xy=(x, y + 0.045),
            xytext=(0.64, 0.58),
            xycoords=ax.transAxes,
            arrowprops={"arrowstyle": "->", "linewidth": 1.0},
        )

    ax.set_title("NIR downstream pipeline validation schematic", pad=14)
    _banner(fig)
    return _save(fig, base, formats, dpi)


def plot_time_on_task(
    coarse: pd.DataFrame,
    *,
    base: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for block_num in sorted(coarse["block_num"].dropna().unique()):
        block = coarse[coarse["block_num"].eq(block_num)].copy()
        for _, frame in block.groupby("subject"):
            ax.plot(
                frame["coarse_bin_start_sec"],
                frame["pir_median"],
                alpha=0.2,
                linewidth=0.8,
            )
        summary = block.groupby("coarse_bin_start_sec")["pir_median"].agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        )
        ax.plot(
            summary.index,
            summary["median"],
            linewidth=2.4,
            label=f"Block {int(block_num)} median",
        )
        ax.fill_between(
            summary.index,
            summary["q25"],
            summary["q75"],
            alpha=0.16,
        )
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Time in block (s)")
    ax.set_ylabel("Centered binocular PIR")
    ax.set_title("Time-on-task PIR trajectory")
    ax.legend()
    _banner(fig)
    return _save(fig, base, formats, dpi)


def plot_block_pairs(
    block_summary: pd.DataFrame,
    *,
    base: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    pivot = block_summary.pivot(index="subject", columns="block_num", values="pir_median")
    fig, ax = plt.subplots(figsize=(6.6, 5.3))
    for _, row in pivot.iterrows():
        xs: list[int] = []
        ys: list[float] = []
        for block in (1, 2):
            if block in row.index and pd.notna(row[block]):
                xs.append(block)
                ys.append(float(row[block]))
        if ys:
            ax.plot(xs, ys, marker="o", alpha=0.45, linewidth=1.0)
    medians = block_summary.groupby("block_num")["pir_median"].median()
    ax.plot(medians.index, medians.values, marker="o", linewidth=3.0, label="Cohort median")
    ax.axhline(0, linewidth=0.8)
    ax.set_xticks([1, 2], ["Block 1", "Block 2"])
    ax.set_ylabel("Median centered binocular PIR")
    ax.set_title("Within-subject Block comparison")
    ax.legend()
    _banner(fig)
    return _save(fig, base, formats, dpi)


def plot_trial_outcomes(
    trial_table: pd.DataFrame,
    *,
    base: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    order = ["go_correct", "go_omission_program", "nogo_correct", "nogo_commission"]
    subject = (
        trial_table.assign(
            pir_median=pd.to_numeric(trial_table["pir_median"], errors="coerce")
        )
        .groupby(["subject", "outcome"], as_index=False)["pir_median"]
        .median()
    )
    groups = [
        subject.loc[subject["outcome"].eq(name), "pir_median"].dropna().to_numpy()
        for name in order
    ]

    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    ax.boxplot(groups, labels=order, showfliers=False)
    rng = np.random.default_rng(0)
    for idx, values in enumerate(groups, start=1):
        if len(values) == 0:
            continue
        x = idx + rng.normal(0, 0.035, size=len(values))
        ax.scatter(x, values, s=26, alpha=0.55)
    ax.axhline(0, linewidth=0.8)
    ax.set_ylabel("Subject-median pre-trial PIR")
    ax.set_title("Program-scored trial outcome × pre-trial PIR")
    ax.tick_params(axis="x", rotation=18)
    _banner(fig)
    return _save(fig, base, formats, dpi)


def plot_omission_subtypes(
    trial_table: pd.DataFrame,
    *,
    base: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    order = [
        "clean_omission",
        "prestimulus_associated_omission",
        "carryover_associated_omission",
        "prestimulus_and_carryover_associated_omission",
        "go_omission_unclassified_qc_missing",
    ]
    labels = [
        "clean",
        "prestimulus",
        "carry-over",
        "pre + carry-over",
        "QC missing",
    ]
    omission = trial_table[
        trial_table["omission_qc_type"].astype(str).isin(order)
    ].copy()
    omission["pir_median"] = pd.to_numeric(omission["pir_median"], errors="coerce")
    subject = (
        omission.groupby(["subject", "omission_qc_type"], as_index=False)["pir_median"]
        .median()
    )
    groups = [
        subject.loc[
            subject["omission_qc_type"].astype(str).eq(name), "pir_median"
        ].dropna().to_numpy()
        for name in order
    ]

    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    nonempty = any(len(values) for values in groups)
    if nonempty:
        ax.boxplot(groups, labels=labels, showfliers=False)
        rng = np.random.default_rng(1)
        for idx, values in enumerate(groups, start=1):
            if len(values) == 0:
                continue
            x = idx + rng.normal(0, 0.035, size=len(values))
            ax.scatter(x, values, s=28, alpha=0.6)
    else:
        ax.text(
            0.5,
            0.5,
            "No Go omission rows available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.axhline(0, linewidth=0.8)
    ax.set_ylabel("Subject-median pre-trial PIR")
    ax.set_title("Go omission QC subtypes × pre-trial PIR")
    ax.tick_params(axis="x", rotation=18)
    _banner(fig)
    return _save(fig, base, formats, dpi)


def plot_probe_windows(
    probe_windows: pd.DataFrame,
    *,
    track: str,
    base: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    df = probe_windows[probe_windows["track"].astype(str).eq(track)].copy()
    df["pir_median"] = pd.to_numeric(df["pir_median"], errors="coerce")
    if "probe_vigilance" in df.columns:
        df["probe_vigilance_numeric"] = pd.to_numeric(
            df["probe_vigilance"], errors="coerce"
        )
    else:
        df["probe_vigilance_numeric"] = np.nan

    df = df.dropna(subset=["pir_median", "probe_vigilance_numeric"])
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    if df.empty:
        ax.text(
            0.5,
            0.5,
            "No numeric probe_vigilance available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        subject = (
            df.groupby(
                ["subject", "window_name", "probe_vigilance_numeric"],
                as_index=False,
            )["pir_median"]
            .median()
        )
        for window_name in list(dict.fromkeys(subject["window_name"].astype(str))):
            frame = subject[subject["window_name"].astype(str).eq(window_name)]
            summary = frame.groupby("probe_vigilance_numeric")["pir_median"].agg(
                mean="mean",
                se=lambda x: x.std(ddof=1) / np.sqrt(max(1, x.count())),
            )
            ax.errorbar(
                summary.index,
                summary["mean"],
                yerr=summary["se"],
                marker="o",
                capsize=3,
                label=window_name,
            )
        ax.legend(title="Pre-probe window")
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Probe vigilance")
    ax.set_ylabel("Pre-probe PIR median")
    ax.set_title("Probe vigilance × PIR across windows")
    _banner(fig)
    return _save(fig, base, formats, dpi)


def plot_coverage_heatmap(
    trial_coverage: pd.DataFrame,
    *,
    track: str,
    window_name: str,
    base: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    df = trial_coverage[
        trial_coverage["track"].astype(str).eq(track)
        & trial_coverage["window_name"].astype(str).eq(window_name)
    ].copy()
    if not df.empty:
        df["key"] = (
            df["subject"].astype(str)
            + " / B"
            + pd.to_numeric(df["block_num"], errors="coerce").astype("Int64").astype(str)
        )
        matrix = (
            df[["key", "pir_valid_fraction_median"]]
            .assign(
                pir_valid_fraction_median=lambda x: pd.to_numeric(
                    x["pir_valid_fraction_median"], errors="coerce"
                )
            )
            .set_index("key")
            .T
        )
    else:
        matrix = pd.DataFrame()

    fig_width = max(8.0, 0.44 * max(1, matrix.shape[1]))
    fig, ax = plt.subplots(figsize=(fig_width, 3.2))
    if matrix.empty:
        ax.text(0.5, 0.5, "No coverage rows", ha="center", va="center", transform=ax.transAxes)
    else:
        image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns, rotation=90)
        ax.set_yticks([0], ["PIR valid fraction"])
        fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="Fraction")
    ax.set_title(f"Coverage QC heatmap — {window_name}")
    _banner(fig)
    return _save(fig, base, formats, dpi)


def plot_model_forest(
    models: pd.DataFrame,
    *,
    base: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    df = models.copy()
    if not df.empty:
        df = df[
            ~df["term"].astype(str).str.contains(
                "Intercept|Group Var", regex=True, na=False
            )
        ].copy()
        df["estimate"] = pd.to_numeric(df["estimate"], errors="coerce")
        df["se"] = pd.to_numeric(df["se"], errors="coerce")
        df = df.dropna(subset=["estimate", "se"])

    fig, ax = plt.subplots(figsize=(9.2, max(4.2, 0.46 * max(1, len(df)))))
    if df.empty:
        ax.text(
            0.5,
            0.5,
            "No smoke-test model coefficients available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        labels = df["model"].astype(str) + " | " + df["term"].astype(str)
        y = np.arange(len(df))
        ax.errorbar(
            df["estimate"],
            y,
            xerr=1.96 * df["se"],
            fmt="o",
            capsize=3,
        )
        ax.set_yticks(y, labels)
        ax.axvline(0, linewidth=0.8)
        ax.invert_yaxis()
    ax.set_xlabel("Estimate (95% Wald interval)")
    ax.set_title("Model smoke-test coefficient forest")
    _banner(fig)
    return _save(fig, base, formats, dpi)
