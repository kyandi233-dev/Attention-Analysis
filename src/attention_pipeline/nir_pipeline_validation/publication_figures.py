from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .figure_style import (
    LINESTYLES,
    PALETTE,
    clean_axis,
    finalize_layout,
    make_figure,
    panel_label,
    save_figure,
)


def _empty(ax: plt.Axes, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _add_zero(ax: plt.Axes, *, vertical: bool = False) -> None:
    if vertical:
        ax.axvline(0, color="#777777", linewidth=0.65, linestyle=":", zorder=0)
    else:
        ax.axhline(0, color="#777777", linewidth=0.65, linestyle=":", zorder=0)


def _heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "coolwarm",
    cbar_label: str = "",
    annotate: bool = False,
) -> None:
    if matrix.empty:
        _empty(ax, "No data")
        return
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    x_labels = [str(x) for x in matrix.columns]
    y_labels = [textwrap.fill(str(x).replace("_", " "), width=18, break_long_words=False) for x in matrix.index]
    ax.set_xticks(np.arange(len(matrix.columns)), x_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), y_labels)
    if annotate and matrix.shape[0] <= 8 and matrix.shape[1] <= 8:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.8)
    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=7)
    cbar.ax.tick_params(labelsize=6)


def _subject_block_box(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    category_col: str,
    value_col: str,
    order: list[str],
    ylabel: str,
    tick_labels: list[str] | None = None,
    tick_rotation: float = 22,
) -> None:
    if frame.empty or value_col not in frame.columns:
        _empty(ax, "No condition rows")
        return
    subject = (
        frame.assign(**{value_col: _numeric(frame[value_col])})
        .groupby(["subject", "block_num", category_col], as_index=False)[value_col]
        .median()
    )
    offsets = {1: -0.17, 2: 0.17}
    width = 0.27
    legend_handles = []
    legend_labels = []
    for block_num in (1, 2):
        color = PALETTE[f"block{block_num}"]
        for idx, category in enumerate(order):
            values = subject.loc[
                subject["block_num"].eq(block_num)
                & subject[category_col].astype(str).eq(str(category)),
                value_col,
            ].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            pos = idx + 1 + offsets[block_num]
            bp = ax.boxplot(
                [values],
                positions=[pos],
                widths=width,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                medianprops={"color": "black", "linewidth": 0.8},
                whiskerprops={"linewidth": 0.65},
                capprops={"linewidth": 0.65},
                boxprops={"linewidth": 0.65, "edgecolor": color},
            )
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.30)
            rng = np.random.default_rng(1000 + block_num * 100 + idx)
            x = pos + rng.normal(0, 0.025, size=len(values))
            ax.scatter(x, values, s=7, alpha=0.48, color=color, linewidths=0, zorder=3)
        handle = plt.Line2D([0], [0], color=color, linewidth=2)
        legend_handles.append(handle)
        legend_labels.append(f"Block {block_num}")
    labels = tick_labels if tick_labels is not None else order
    ax.set_xticks(
        np.arange(1, len(order) + 1),
        labels,
        rotation=tick_rotation,
        ha="right" if tick_rotation else "center",
    )
    ax.set_ylabel(ylabel)
    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc="best", ncol=2)
    _add_zero(ax)
    clean_axis(ax, grid_y=True)


def _trajectory_by_condition(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    condition_col: str = "event_condition",
    value_col: str = "pir_median",
    block_num: int | None = None,
    title: str = "",
    ylabel: str = "Centered PIR",
) -> None:
    if frame.empty or value_col not in frame.columns:
        _empty(ax, "No continuous trajectory")
        return
    df = frame.copy()
    if block_num is not None:
        df = df[_numeric(df["block_num"]).eq(block_num)]
    if df.empty:
        _empty(ax, f"No Block {block_num} trajectory")
        return
    for idx, (condition, current) in enumerate(df.groupby(condition_col, sort=True)):
        subject = current.groupby(["subject", "time_bin_mid_sec"], as_index=False)[value_col].median()
        summary = subject.groupby("time_bin_mid_sec")[value_col].agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        ).sort_index()
        color = PALETTE.get(str(condition), plt.get_cmap("tab10")(idx % 10))
        style = LINESTYLES.get(str(condition), "-")
        ax.plot(summary.index, summary["median"], color=color, linestyle=style, label=str(condition))
        ax.fill_between(summary.index, summary["q25"], summary["q75"], color=color, alpha=0.10, linewidth=0)
    _add_zero(ax, vertical=True)
    _add_zero(ax)
    ax.set_xlabel("Time relative to event (s)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    clean_axis(ax, grid_y=True)


def figure01_global_landscape(
    global_detail: pd.DataFrame,
    global_summary: pd.DataFrame,
    distribution: pd.DataFrame,
    transition: pd.DataFrame,
    *,
    base: Path,
    formats: Iterable[str],
    raster_dpi: int,
) -> list[str]:
    fig, axes = make_figure(width="full", height_cm=15.5, nrows=2, ncols=2)
    ax = axes[0, 0]
    if global_detail.empty:
        _empty(ax, "No global trajectory")
    else:
        for _, subject in global_detail.groupby("subject", sort=True):
            ax.plot(subject["global_time_sec"] / 60.0, subject["pir_median"], color="#BDBDBD", linewidth=0.35, alpha=0.22)
        for block_num in (1, 2):
            current = global_summary[_numeric(global_summary["block_num"]).eq(block_num)]
            if current.empty:
                continue
            color = PALETTE[f"block{block_num}"]
            ax.plot(current["global_bin_sec"] / 60.0, current["median"], color=color, linewidth=1.7, label=f"Block {block_num}")
            ax.fill_between(current["global_bin_sec"] / 60.0, current["q25"], current["q75"], color=color, alpha=0.14, linewidth=0)
        b1_end = float(global_detail["block1_display_end_sec"].median()) / 60.0
        b2_start = float(global_detail["block2_display_start_sec"].median()) / 60.0
        ax.axvspan(b1_end, b2_start, color="#E6E6E6", alpha=0.75, linewidth=0)
        ax.text((b1_end + b2_start) / 2.0, ax.get_ylim()[1], "between-block interval", ha="center", va="top", fontsize=6)
        ax.set_xlabel("Global experimental time (min)")
        ax.set_ylabel("Centered PIR")
        ax.legend(loc="best", ncol=2)
        _add_zero(ax)
        clean_axis(ax, grid_y=True)
    ax.set_title("Whole-experiment PIR landscape")
    panel_label(ax, "A")

    ax = axes[0, 1]
    if global_detail.empty:
        _empty(ax, "No aligned Block trajectory")
    else:
        work = global_detail.copy()
        work["aligned_bin"] = np.floor(_numeric(work["time_in_block_sec"]) / 30.0) * 30.0 + 15.0
        for block_num in (1, 2):
            current = work[_numeric(work["block_num"]).eq(block_num)]
            subject = current.groupby(["subject", "aligned_bin"], as_index=False)["pir_median"].median()
            summary = subject.groupby("aligned_bin")["pir_median"].agg(
                median="median",
                q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75),
            ).sort_index()
            color = PALETTE[f"block{block_num}"]
            ax.plot(summary.index / 60.0, summary["median"], color=color, linestyle=LINESTYLES[f"block{block_num}"], label=f"Block {block_num}")
            ax.fill_between(summary.index / 60.0, summary["q25"], summary["q75"], color=color, alpha=0.12, linewidth=0)
        ax.set_xlabel("Time within Block (min)")
        ax.set_ylabel("Centered PIR")
        ax.legend(loc="best")
        _add_zero(ax)
        clean_axis(ax, grid_y=True)
    ax.set_title("Block-aligned time-on-task")
    panel_label(ax, "B")

    ax = axes[1, 0]
    if distribution.empty:
        _empty(ax, "No Block distribution")
    else:
        groups = [distribution.loc[_numeric(distribution["block_num"]).eq(block), "pir_median"].dropna().to_numpy(dtype=float) for block in (1, 2)]
        vp = ax.violinplot(groups, positions=[1, 2], showmedians=True, showextrema=False, widths=0.72)
        for idx, body in enumerate(vp["bodies"], start=1):
            body.set_facecolor(PALETTE[f"block{idx}"])
            body.set_edgecolor(PALETTE[f"block{idx}"])
            body.set_alpha(0.28)
        for _, row in distribution.pivot(index="subject", columns="block_num", values="pir_median").iterrows():
            if 1 in row.index and 2 in row.index and pd.notna(row[1]) and pd.notna(row[2]):
                ax.plot([1, 2], [row[1], row[2]], color="#9B9B9B", linewidth=0.45, alpha=0.45)
        ax.set_xticks([1, 2], ["Block 1", "Block 2"])
        ax.set_ylabel("Subject-median centered PIR")
        _add_zero(ax)
        clean_axis(ax, grid_y=True)
    ax.set_title("Global Block distributions")
    panel_label(ax, "C")

    ax = axes[1, 1]
    if transition.empty:
        _empty(ax, "No Block transition trajectory")
    else:
        for block_num, label in ((1, "Block 1 end"), (2, "Block 2 start")):
            current = transition[_numeric(transition["block_num"]).eq(block_num)].copy()
            current["bin"] = np.floor(_numeric(current["transition_time_sec"]) / 10.0) * 10.0 + 5.0
            subject = current.groupby(["subject", "bin"], as_index=False)["pir_median"].median()
            summary = subject.groupby("bin")["pir_median"].agg(
                median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)
            ).sort_index()
            color = PALETTE[f"block{block_num}"]
            ax.plot(summary.index, summary["median"], color=color, label=label)
            ax.fill_between(summary.index, summary["q25"], summary["q75"], color=color, alpha=0.12, linewidth=0)
        _add_zero(ax, vertical=True)
        _add_zero(ax)
        ax.set_xlabel("Time relative to Block boundary (s)")
        ax.set_ylabel("Centered PIR")
        ax.legend(loc="best")
        clean_axis(ax, grid_y=True)
    ax.set_title("End-of-Block / start-of-Block recovery")
    panel_label(ax, "D")

    finalize_layout(fig, wspace=0.28, hspace=0.42)
    return save_figure(fig, base, formats, raster_dpi=raster_dpi)


def figure02_block_time_on_task(
    block_summary: pd.DataFrame,
    heterogeneity: pd.DataFrame,
    recovery_summary: pd.DataFrame,
    global_detail: pd.DataFrame,
    *,
    base: Path,
    formats: Iterable[str],
    raster_dpi: int,
) -> list[str]:
    fig, axes = make_figure(width="full", height_cm=14.5, nrows=2, ncols=2)
    ax = axes[0, 0]
    if block_summary.empty:
        _empty(ax, "No paired Block summary")
    else:
        pivot = block_summary.pivot(index="subject", columns="block_num", values="pir_median")
        for _, row in pivot.iterrows():
            if 1 in row.index and 2 in row.index and pd.notna(row[1]) and pd.notna(row[2]):
                ax.plot([1, 2], [row[1], row[2]], marker="o", markersize=2.6, color="#8E8E8E", linewidth=0.55, alpha=0.55)
        med = block_summary.groupby("block_num")["pir_median"].median()
        ax.plot(med.index, med.values, marker="o", markersize=4.5, color="black", linewidth=1.8, label="Cohort median")
        ax.set_xticks([1, 2], ["Block 1", "Block 2"])
        ax.set_ylabel("Median centered PIR")
        ax.legend(loc="best")
        _add_zero(ax)
        clean_axis(ax, grid_y=True)
    ax.set_title("Within-subject Block change")
    panel_label(ax, "A")

    ax = axes[0, 1]
    slope_cols = [col for col in ("block1_time_slope_per_sec", "block2_time_slope_per_sec") if col in heterogeneity.columns]
    if heterogeneity.empty or len(slope_cols) < 2:
        _empty(ax, "No subject time-on-task slopes")
    else:
        x = _numeric(heterogeneity[slope_cols[0]])
        y = _numeric(heterogeneity[slope_cols[1]])
        ok = x.notna() & y.notna()
        ax.scatter(x[ok], y[ok], s=14, facecolors="none", edgecolors="#555555", linewidths=0.75)
        lim = np.nanmax(np.abs(np.concatenate([x[ok].to_numpy(), y[ok].to_numpy()]))) if int(ok.sum()) else 1.0
        lim = max(float(lim), 1e-6)
        ax.plot([-lim, lim], [-lim, lim], color="#999999", linestyle=":", linewidth=0.7)
        ax.axhline(0, color="#BBBBBB", linewidth=0.6)
        ax.axvline(0, color="#BBBBBB", linewidth=0.6)
        ax.set_xlabel("Block 1 PIR slope / s")
        ax.set_ylabel("Block 2 PIR slope / s")
        clean_axis(ax, grid_y=False)
    ax.set_title("Individual time-on-task slopes")
    panel_label(ax, "B")

    ax = axes[1, 0]
    if recovery_summary.empty or "recovery_delta_block2_minus_block1" not in recovery_summary.columns:
        _empty(ax, "No recovery delta")
    else:
        values = _numeric(recovery_summary["recovery_delta_block2_minus_block1"]).dropna().sort_values().to_numpy(dtype=float)
        ax.axhline(0, color="#777777", linestyle=":", linewidth=0.7)
        ax.scatter(np.arange(1, len(values) + 1), values, s=12, color=PALETTE["block2"], alpha=0.75, linewidths=0)
        ax.set_xlabel("Subjects sorted by recovery delta")
        ax.set_ylabel("B2 first 60 s − B1 last 60 s")
        clean_axis(ax, grid_y=True)
    ax.set_title("Block transition recovery heterogeneity")
    panel_label(ax, "C")

    ax = axes[1, 1]
    if global_detail.empty:
        _empty(ax, "No early/late comparison")
    else:
        work = global_detail.copy()
        work["half"] = work.groupby(["subject", "block_num"])["time_in_block_sec"].transform(
            lambda x: np.where(x <= x.median(), "first half", "second half")
        )
        subject = work.groupby(["subject", "block_num", "half"], as_index=False)["pir_median"].median()
        for block_num in (1, 2):
            pivot = subject[_numeric(subject["block_num"]).eq(block_num)].pivot(index="subject", columns="half", values="pir_median")
            if {"first half", "second half"}.issubset(pivot.columns):
                delta = (pivot["second half"] - pivot["first half"]).dropna()
                pos = block_num
                bp = ax.boxplot([delta], positions=[pos], widths=0.5, patch_artist=True, showfliers=False, manage_ticks=False)
                bp["boxes"][0].set_facecolor(PALETTE[f"block{block_num}"])
                bp["boxes"][0].set_alpha(0.28)
                bp["boxes"][0].set_edgecolor(PALETTE[f"block{block_num}"])
                rng = np.random.default_rng(230 + block_num)
                ax.scatter(pos + rng.normal(0, 0.035, len(delta)), delta, s=8, color=PALETTE[f"block{block_num}"], alpha=0.55, linewidths=0)
        ax.set_xticks([1, 2], ["Block 1", "Block 2"])
        ax.set_ylabel("Second-half − first-half PIR")
        _add_zero(ax)
        clean_axis(ax, grid_y=True)
    ax.set_title("Within-Block deterioration contrast")
    panel_label(ax, "D")

    finalize_layout(fig, wspace=0.30, hspace=0.42)
    return save_figure(fig, base, formats, raster_dpi=raster_dpi)


def figure03_trial_states(
    trial_conditions: pd.DataFrame,
    advanced_behavior: pd.DataFrame,
    *,
    base: Path,
    formats: Iterable[str],
    raster_dpi: int,
) -> list[str]:
    fig, axes = make_figure(width="full", height_cm=15.0, nrows=2, ncols=2)
    order = ["go_correct", "go_omission_program", "nogo_correct", "nogo_commission"]
    _subject_block_box(
        axes[0, 0], trial_conditions, category_col="outcome", value_col="pir_median", order=order, ylabel="Pre-trial centered PIR"
    )
    axes[0, 0].set_title("Program-scored outcome × Block")
    panel_label(axes[0, 0], "A")

    omission_order = [
        "clean_omission",
        "prestimulus_associated_omission",
        "carryover_associated_omission",
        "prestimulus_and_carryover_associated_omission",
    ]
    omission = trial_conditions[trial_conditions.get("omission_qc_type", pd.Series(index=trial_conditions.index, dtype=str)).astype(str).isin(omission_order)].copy()
    _subject_block_box(
        axes[0, 1], omission, category_col="omission_qc_type", value_col="pir_median", order=omission_order,
        ylabel="Pre-trial centered PIR",
        tick_labels=[
            "clean",
            "pre",
            "carry",
            "pre+carry",
        ],
        tick_rotation=0,
    )
    axes[0, 1].set_xlim(0.5, 4.7)
    axes[0, 1].set_title("Omission motor-timing subtypes × Block")
    panel_label(axes[0, 1], "B")

    ax = axes[1, 0]
    if trial_conditions.empty or "rt" not in trial_conditions.columns:
        _empty(ax, "No Go RT/PIR rows")
    else:
        df = trial_conditions[trial_conditions["outcome"].astype(str).eq("go_correct")].copy()
        for block_num in (1, 2):
            current = df[_numeric(df["block_num"]).eq(block_num)].copy()
            current["rt"] = _numeric(current["rt"])
            current["pir_median"] = _numeric(current["pir_median"])
            ok = current["rt"].notna() & current["pir_median"].notna()
            current = current.loc[ok]
            if current.empty:
                continue
            bins = pd.qcut(current["pir_median"], q=min(8, max(2, current["pir_median"].nunique())), duplicates="drop")
            current["pir_bin"] = bins
            summary = current.groupby("pir_bin", observed=True).agg(pir=("pir_median", "median"), rt=("rt", "median")).sort_values("pir")
            ax.plot(summary["pir"], summary["rt"], marker="o", color=PALETTE[f"block{block_num}"], linestyle=LINESTYLES[f"block{block_num}"], label=f"Block {block_num}")
        ax.set_xlabel("Binned pre-trial PIR")
        ax.set_ylabel("Correct-Go RT (ms)")
        ax.legend(loc="best")
        clean_axis(ax, grid_y=True)
    ax.set_title("Correct-Go behavior across PIR state")
    panel_label(ax, "C")

    ax = axes[1, 1]
    metrics = [col for col in ("commission_rate", "program_omission_rate", "clean_omission_rate", "ambiguous_omission_rate") if col in advanced_behavior.columns]
    if advanced_behavior.empty or not metrics:
        _empty(ax, "No error-rate summary")
    else:
        x = np.arange(len(metrics))
        width = 0.34
        for block_num, offset in ((1, -width / 2), (2, width / 2)):
            frame = advanced_behavior[_numeric(advanced_behavior["block_num"]).eq(block_num)]
            means = [float(_numeric(frame[m]).median()) for m in metrics]
            ax.bar(x + offset, means, width=width, color=PALETTE[f"block{block_num}"], alpha=0.65, label=f"Block {block_num}")
        ax.set_xticks(x, [m.replace("_rate", "") for m in metrics], rotation=25, ha="right")
        ax.set_ylabel("Subject-median rate")
        ax.set_ylim(bottom=0)
        ax.legend(loc="best")
        clean_axis(ax, grid_y=True)
    ax.set_title("Behavioral error/QC profiles")
    panel_label(ax, "D")

    finalize_layout(fig, wspace=0.34, hspace=0.46)
    return save_figure(fig, base, formats, raster_dpi=raster_dpi)


def figure04_error_precursors(
    nogo_continuous: pd.DataFrame,
    omission_continuous: pd.DataFrame,
    nogo_trial_lag: pd.DataFrame,
    *,
    base: Path,
    formats: Iterable[str],
    raster_dpi: int,
) -> list[str]:
    fig, axes = make_figure(width="full", height_cm=15.0, nrows=2, ncols=2)
    _trajectory_by_condition(axes[0, 0], nogo_continuous, block_num=1, title="No-Go precursor — Block 1")
    panel_label(axes[0, 0], "A")
    _trajectory_by_condition(axes[0, 1], nogo_continuous, block_num=2, title="No-Go precursor — Block 2")
    panel_label(axes[0, 1], "B")
    _trajectory_by_condition(axes[1, 0], omission_continuous, title="Go omission precursor — both Blocks")
    panel_label(axes[1, 0], "C")

    ax = axes[1, 1]
    if nogo_trial_lag.empty:
        _empty(ax, "No trial-lag precursor")
    else:
        for condition, current in nogo_trial_lag.groupby("event_outcome", sort=True):
            subject = current.groupby(["subject", "lag"], as_index=False).agg(rt=("go_rt_ms", "median"), pir=("pir_median", "median"))
            rt = subject.groupby("lag")["rt"].median().sort_index()
            color = PALETTE.get(str(condition), "#555555")
            style = LINESTYLES.get(str(condition), "-")
            ax.plot(rt.index, rt.values, marker="o", color=color, linestyle=style, label=str(condition))
        ax.axvline(0, color="#777777", linestyle=":", linewidth=0.65)
        ax.set_xlabel("Correct-Go trial lag before No-Go")
        ax.set_ylabel("Correct-Go RT (ms)")
        ax.legend(loc="best")
        clean_axis(ax, grid_y=True)
    ax.set_title("Discrete trial-lag behavioral precursor")
    panel_label(ax, "D")

    finalize_layout(fig, wspace=0.30, hspace=0.42)
    return save_figure(fig, base, formats, raster_dpi=raster_dpi)


def figure05_probe_states(
    probe_events: pd.DataFrame,
    probe_continuous: pd.DataFrame,
    probe_transitions: pd.DataFrame,
    *,
    base: Path,
    formats: Iterable[str],
    raster_dpi: int,
) -> list[str]:
    fig, axes = make_figure(width="full", height_cm=15.0, nrows=2, ncols=2)
    ax = axes[0, 0]
    if probe_events.empty or "probe_response" not in probe_events.columns:
        _empty(ax, "No probe_response")
    else:
        counts = probe_events.groupby(["block_num", "probe_response"]).size().rename("n").reset_index()
        totals = counts.groupby("block_num")["n"].transform("sum")
        counts["fraction"] = counts["n"] / totals
        options = sorted(counts["probe_response"].dropna().unique())
        x = np.arange(len(options))
        width = 0.34
        for block_num, offset in ((1, -width / 2), (2, width / 2)):
            current = counts[_numeric(counts["block_num"]).eq(block_num)].set_index("probe_response")
            vals = [current["fraction"].get(opt, 0.0) for opt in options]
            ax.bar(x + offset, vals, width=width, color=PALETTE[f"block{block_num}"], alpha=0.68, label=f"Block {block_num}")
        ax.set_xticks(x, [str(int(v)) if float(v).is_integer() else str(v) for v in options])
        ax.set_xlabel("probe_response raw code")
        ax.set_ylabel("Fraction of probes")
        ax.legend(loc="best")
        clean_axis(ax, grid_y=True)
    ax.set_title("Probe response distribution × Block")
    panel_label(ax, "A")

    ax = axes[0, 1]
    if probe_events.empty or not {"probe_response", "probe_vigilance"}.issubset(probe_events.columns):
        _empty(ax, "No response × vigilance")
    else:
        df = probe_events.copy()
        df["probe_vigilance"] = _numeric(df["probe_vigilance"])
        for block_num in (1, 2):
            current = df[_numeric(df["block_num"]).eq(block_num)]
            summary = current.groupby("probe_response")["probe_vigilance"].median().sort_index()
            ax.plot(summary.index, summary.values, marker="o", color=PALETTE[f"block{block_num}"], linestyle=LINESTYLES[f"block{block_num}"], label=f"Block {block_num}")
        ax.set_xlabel("probe_response raw code")
        ax.set_ylabel("Median probe_vigilance")
        ax.legend(loc="best")
        clean_axis(ax, grid_y=True)
    ax.set_title("Probe response × vigilance structure")
    panel_label(ax, "B")

    _trajectory_by_condition(axes[1, 0], probe_continuous, title="Continuous PIR trajectory before Probe")
    panel_label(axes[1, 0], "C")

    ax = axes[1, 1]
    if probe_transitions.empty or "response_transition" not in probe_transitions.columns:
        _empty(ax, "No sequential Probe transitions")
    else:
        counts = probe_transitions.groupby("response_transition").size().rename("n").sort_values(ascending=False)
        top = counts.head(12).sort_values()
        ax.barh(np.arange(len(top)), top.values, color="#777777", alpha=0.72)
        ax.set_yticks(np.arange(len(top)), top.index)
        ax.set_xlabel("Transition count")
        clean_axis(ax, grid_y=False)
    ax.set_title("Sequential probe-state transitions")
    panel_label(ax, "D")

    finalize_layout(fig, wspace=0.32, hspace=0.44)
    return save_figure(fig, base, formats, raster_dpi=raster_dpi)


def _binned_scatter(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x_col: str,
    *,
    title: str,
    legend_loc: str = "best",
) -> None:
    if frame.empty or x_col not in frame.columns or "pir_median" not in frame.columns:
        _empty(ax, "Visual covariate unavailable")
        return
    for block_num in (1, 2):
        current = frame[_numeric(frame["block_num"]).eq(block_num)].copy()
        current[x_col] = _numeric(current[x_col])
        current["pir_median"] = _numeric(current["pir_median"])
        current = current.dropna(subset=[x_col, "pir_median"])
        if current.empty:
            continue
        q = min(8, max(2, current[x_col].nunique()))
        current["bin"] = pd.qcut(current[x_col], q=q, duplicates="drop")
        summary = current.groupby("bin", observed=True).agg(x=(x_col, "median"), y=("pir_median", "median")).sort_values("x")
        ax.plot(summary["x"], summary["y"], marker="o", color=PALETTE[f"block{block_num}"], linestyle=LINESTYLES[f"block{block_num}"], label=f"Block {block_num}")
    ax.set_xlabel(x_col.replace("current_", ""))
    ax.set_ylabel("Pre-trial centered PIR")
    ax.set_title(title)
    ax.legend(loc=legend_loc, fontsize=6.5)
    _add_zero(ax)
    clean_axis(ax, grid_y=True)


def figure06_visual_controls(
    visual_trial: pd.DataFrame,
    stimulus_summary: pd.DataFrame,
    visual_correlations: pd.DataFrame,
    *,
    base: Path,
    formats: Iterable[str],
    raster_dpi: int,
) -> list[str]:
    fig, axes = make_figure(width="full", height_cm=15.0, nrows=2, ncols=2)
    ax = axes[0, 0]
    if stimulus_summary.empty or not {"stimulus_size", "pir_median"}.issubset(stimulus_summary.columns):
        _empty(ax, "No stimulus-size summary")
    else:
        for block_num in (1, 2):
            for is_nogo, marker in ((0, "o"), (1, "s")):
                current = stimulus_summary[
                    _numeric(stimulus_summary["block_num"]).eq(block_num)
                    & _numeric(stimulus_summary["is_no_go"]).eq(is_nogo)
                ]
                summary = current.groupby("stimulus_size")["pir_median"].median().sort_index()
                if summary.empty:
                    continue
                ax.plot(
                    summary.index,
                    summary.values,
                    marker=marker,
                    color=PALETTE[f"block{block_num}"],
                    linestyle="-" if is_nogo == 0 else ":",
                    label=f"B{block_num} {'No-Go' if is_nogo else 'Go'}",
                )
        ax.set_xlabel("Stimulus size (%)")
        ax.set_ylabel("Pre-trial centered PIR")
        ax.legend(loc="best", ncol=2)
        _add_zero(ax)
        clean_axis(ax, grid_y=True)
    ax.set_title("Stimulus size / task type control")
    panel_label(ax, "A")

    _binned_scatter(axes[0, 1], visual_trial, "current_central_rel_lum_mean", title="Current-stimulus luminance")
    panel_label(axes[0, 1], "B")
    _binned_scatter(
        axes[1, 0],
        visual_trial,
        "current_central_rms_contrast",
        title="Current-stimulus contrast",
        legend_loc="upper left",
    )
    panel_label(axes[1, 0], "C")

    ax = axes[1, 1]
    if visual_correlations.empty:
        _empty(ax, "No visual-covariate correlations")
    else:
        df = visual_correlations.sort_values("spearman_rho_with_pir")
        ax.barh(np.arange(len(df)), df["spearman_rho_with_pir"], color="#777777", alpha=0.72)
        ax.set_yticks(
            np.arange(len(df)),
            [
                (
                    ("cur" if str(x).startswith("current_") else "prev")
                    + ": "
                    + ("fruit" if "fruit_" in str(x) else "central")
                    + "\n"
                    + (
                        "visible area"
                        if "visible_area" in str(x)
                        else "contrast"
                        if "rms_contrast" in str(x)
                        else "delta luminance"
                        if "delta_" in str(x) and "rel_lum" in str(x)
                        else "luminance"
                    )
                )
                for x in df["covariate"]
            ],
        )
        ax.tick_params(axis="y", labelsize=5.5)
        ax.axvline(0, color="#777777", linestyle=":", linewidth=0.65)
        ax.set_xlabel("Spearman ρ with PIR (validation only)")
        clean_axis(ax, grid_y=False)
    ax.set_title("Current vs previous visual covariates")
    panel_label(ax, "D")

    finalize_layout(fig, left=0.16, wspace=0.42, hspace=0.44)
    return save_figure(fig, base, formats, raster_dpi=raster_dpi)


def figure07_individual_differences(
    heterogeneity: pd.DataFrame,
    raw_pir: pd.DataFrame,
    *,
    base: Path,
    formats: Iterable[str],
    raster_dpi: int,
) -> list[str]:
    fig, axes = make_figure(width="full", height_cm=14.5, nrows=2, ncols=2)
    ax = axes[0, 0]
    if heterogeneity.empty or not {"block1_time_slope_per_sec", "block2_time_slope_per_sec"}.issubset(heterogeneity.columns):
        _empty(ax, "No subject slope data")
    else:
        x = _numeric(heterogeneity["block1_time_slope_per_sec"])
        y = _numeric(heterogeneity["block2_time_slope_per_sec"])
        ok = x.notna() & y.notna()
        ax.scatter(x[ok], y[ok], s=18, color="#555555", alpha=0.72, linewidths=0)
        for subject, xv, yv in zip(heterogeneity.loc[ok, "subject"], x[ok], y[ok]):
            ax.annotate(str(subject).replace("sub-", ""), (xv, yv), xytext=(2, 2), textcoords="offset points", fontsize=5.2, color="#666666")
        ax.axhline(0, color="#BBBBBB", linewidth=0.6)
        ax.axvline(0, color="#BBBBBB", linewidth=0.6)
        ax.set_xlabel("Block 1 slope / s")
        ax.set_ylabel("Block 2 slope / s")
        clean_axis(ax)
    ax.set_title("Time-on-task slope heterogeneity")
    panel_label(ax, "A")

    ax = axes[0, 1]
    if raw_pir.empty or "raw_PIR_subject_median" not in raw_pir.columns:
        _empty(ax, "No raw between-person PIR")
    else:
        df = raw_pir.sort_values("raw_PIR_subject_median")
        ax.scatter(np.arange(1, len(df) + 1), df["raw_PIR_subject_median"], s=15, color="#555555", alpha=0.78, linewidths=0)
        ax.set_xlabel("Subjects sorted by raw PIR baseline")
        ax.set_ylabel("Raw binocular PIR median")
        clean_axis(ax, grid_y=True)
    ax.set_title("Between-person raw PIR characteristics")
    panel_label(ax, "B")

    ax = axes[1, 0]
    if heterogeneity.empty or "recovery_delta_block2_minus_block1" not in heterogeneity.columns:
        _empty(ax, "No recovery heterogeneity")
    else:
        df = heterogeneity[["subject", "recovery_delta_block2_minus_block1"]].dropna().sort_values("recovery_delta_block2_minus_block1")
        ax.bar(np.arange(len(df)), df["recovery_delta_block2_minus_block1"], color=np.where(df["recovery_delta_block2_minus_block1"].ge(0), PALETTE["block2"], PALETTE["block1"]), alpha=0.70)
        ax.axhline(0, color="#777777", linewidth=0.65)
        ax.set_xlabel("Subjects sorted by Block-transition change")
        ax.set_ylabel("Recovery delta")
        clean_axis(ax, grid_y=True)
    ax.set_title("Block-transition heterogeneity")
    panel_label(ax, "C")

    ax = axes[1, 1]
    effect_cols = [col for col in heterogeneity.columns if col.startswith("median_effect__")]
    if heterogeneity.empty or not effect_cols:
        _empty(ax, "No subject event-effect heterogeneity")
    else:
        long = heterogeneity[["subject", *effect_cols]].melt(id_vars="subject", var_name="effect", value_name="value")
        long["effect"] = long["effect"].str.replace("median_effect__", "", regex=False)
        effects = list(dict.fromkeys(long["effect"]))
        for idx, effect in enumerate(effects):
            values = _numeric(long.loc[long["effect"].eq(effect), "value"]).dropna().to_numpy(dtype=float)
            if len(values):
                pos = idx + 1
                ax.boxplot([values], positions=[pos], widths=0.52, showfliers=False, manage_ticks=False)
                rng = np.random.default_rng(500 + idx)
                ax.scatter(pos + rng.normal(0, 0.035, len(values)), values, s=8, color="#666666", alpha=0.55, linewidths=0)
        short_effects = [
            textwrap.fill(
                effect.replace("_minus_", " − ").replace("_", " "),
                width=20,
                break_long_words=False,
            )
            for effect in effects
        ]
        ax.set_xticks(np.arange(1, len(effects) + 1), short_effects, rotation=0, ha="center")
        ax.set_ylabel("Subject-level median contrast")
        _add_zero(ax)
        clean_axis(ax, grid_y=True)
    ax.set_title("Event-association heterogeneity")
    panel_label(ax, "D")

    finalize_layout(fig, wspace=0.32, hspace=0.44)
    return save_figure(fig, base, formats, raster_dpi=raster_dpi)


def _corr_matrix(long: pd.DataFrame, row: str, col: str, value: str) -> pd.DataFrame:
    return long.pivot(index=row, columns=col, values=value) if not long.empty else pd.DataFrame()


def figure08_feature_structure(
    feature_within: pd.DataFrame,
    within_metrics: pd.DataFrame,
    between_metrics: pd.DataFrame,
    window_stability: pd.DataFrame,
    *,
    base: Path,
    formats: Iterable[str],
    raster_dpi: int,
) -> list[str]:
    fig, axes = make_figure(width="full", height_cm=17.0, nrows=2, ncols=2)
    matrix = _corr_matrix(feature_within, "feature_a", "feature_b", "r")
    _heatmap(axes[0, 0], matrix, vmin=-1, vmax=1, cmap="coolwarm", cbar_label="within-person r")
    axes[0, 0].set_title("PIR feature redundancy")
    panel_label(axes[0, 0], "A")

    matrix = _corr_matrix(within_metrics, "metric_a", "metric_b", "r")
    _heatmap(axes[0, 1], matrix, vmin=-1, vmax=1, cmap="coolwarm", cbar_label="within-person r", annotate=True)
    axes[0, 1].set_title("Within-person NIR–behavior structure")
    panel_label(axes[0, 1], "B")

    matrix = _corr_matrix(between_metrics, "metric_a", "metric_b", "r")
    _heatmap(axes[1, 0], matrix, vmin=-1, vmax=1, cmap="coolwarm", cbar_label="between-person r")
    axes[1, 0].set_title("Between-person raw-PIR structure")
    panel_label(axes[1, 0], "C")

    ax = axes[1, 1]
    if window_stability.empty:
        _empty(ax, "No multiscale stability summary")
    else:
        for idx, (contrast, current) in enumerate(window_stability.groupby("contrast", sort=True)):
            current = current.sort_values("window_sec")
            color = plt.get_cmap("tab10")(idx % 10)
            ax.plot(current["window_sec"], current["median"], marker="o", color=color, label=str(contrast))
            ax.fill_between(current["window_sec"], current["q25"], current["q75"], color=color, alpha=0.12, linewidth=0)
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 5, 10, 20, 30, 60], ["1", "3", "5", "10", "20", "30", "60"])
        ax.set_xlabel("Pre-event window (s; log scale)")
        ax.set_ylabel("Subject-level effect/contrast")
        ax.legend(loc="best")
        _add_zero(ax)
        clean_axis(ax, grid_y=True)
    ax.set_title("Effect stability across prespecified windows")
    panel_label(ax, "D")

    finalize_layout(fig, left=0.20, bottom=0.15, wspace=0.50, hspace=0.50)
    return save_figure(fig, base, formats, raster_dpi=raster_dpi)


def _coverage_fraction_matrix(
    coverage: pd.DataFrame,
    *,
    level: str,
    track: str,
) -> pd.DataFrame:
    if coverage.empty:
        return pd.DataFrame()
    fraction_metrics = [
        "pir_valid_fraction_median",
        "available_duration_fraction_median",
        "internal_coverage_fraction_median",
        "boundary_truncated_fraction",
    ]
    df = coverage[
        coverage["analysis_level"].astype(str).eq(level)
        & coverage["track"].astype(str).eq(track)
        & coverage["coverage_metric"].astype(str).isin(fraction_metrics)
    ].copy()
    if df.empty:
        return pd.DataFrame()
    df["subject_block"] = df["subject"].astype(str).str.replace("sub-", "", regex=False) + "/B" + _numeric(df["block_num"]).astype("Int64").astype(str)
    return df.pivot_table(index="coverage_metric", columns="subject_block", values="coverage_value", aggfunc="median")


def figure09_quality_control(
    coverage: pd.DataFrame,
    source_mode: pd.DataFrame,
    *,
    track: str,
    base: Path,
    formats: Iterable[str],
    raster_dpi: int,
) -> list[str]:
    fig, axes = make_figure(width="full", height_cm=16.5, nrows=2, ncols=2)
    trial_matrix = _coverage_fraction_matrix(coverage, level="trial", track=track)
    _heatmap(axes[0, 0], trial_matrix, vmin=0, vmax=1, cmap="viridis", cbar_label="fraction")
    axes[0, 0].set_title("Trial-window coverage dimensions")
    panel_label(axes[0, 0], "A")

    probe_matrix = _coverage_fraction_matrix(coverage, level="probe", track=track)
    _heatmap(axes[0, 1], probe_matrix, vmin=0, vmax=1, cmap="viridis", cbar_label="fraction")
    axes[0, 1].set_title("Probe-window coverage dimensions")
    panel_label(axes[0, 1], "B")

    ax = axes[1, 0]
    if source_mode.empty:
        _empty(ax, "No binocular source-mode QC")
    else:
        df = source_mode[source_mode["track"].astype(str).eq(track)].copy() if "track" in source_mode.columns else source_mode.copy()
        cols = [col for col in ("source_mode_binocular_fraction", "source_mode_left_only_fraction", "source_mode_right_only_fraction", "source_mode_missing_fraction") if col in df.columns]
        summary = df.groupby("block_num")[cols].mean().sort_index()
        x = np.arange(len(summary.index))
        bottom = np.zeros(len(x))
        colors = ["#4C78A8", "#59A14F", "#F28E2B", "#B8B8B8"]
        for col, color in zip(cols, colors):
            values = summary[col].to_numpy(dtype=float)
            ax.bar(x, values, bottom=bottom, color=color, alpha=0.78, label=col.replace("source_mode_", "").replace("_fraction", ""))
            bottom += np.nan_to_num(values)
        ax.set_xticks(x, [f"Block {int(v)}" for v in summary.index])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Mean window fraction")
        ax.legend(loc="best", ncol=2)
        clean_axis(ax, grid_y=True)
    ax.set_title("Binocular / single-eye / missing composition")
    panel_label(ax, "C")

    ax = axes[1, 1]
    if coverage.empty:
        _empty(ax, "No temporal-gap QC")
    else:
        df = coverage[
            coverage["track"].astype(str).eq(track)
            & coverage["coverage_metric"].astype(str).eq("max_temporal_gap_sec_p95")
        ].copy()
        if df.empty:
            _empty(ax, "No max-gap metric")
        else:
            df["coverage_value"] = _numeric(df["coverage_value"])
            for idx, level in enumerate(("trial", "probe"), start=1):
                values = df.loc[df["analysis_level"].astype(str).eq(level), "coverage_value"].dropna().to_numpy(dtype=float)
                if len(values):
                    ax.boxplot([values], positions=[idx], widths=0.48, showfliers=False, manage_ticks=False)
                    rng = np.random.default_rng(900 + idx)
                    ax.scatter(idx + rng.normal(0, 0.035, len(values)), values, s=7, color="#666666", alpha=0.45, linewidths=0)
            ax.set_xticks([1, 2], ["Trial", "Probe"])
            ax.set_ylabel("95th percentile max temporal gap (s)")
            clean_axis(ax, grid_y=True)
    ax.set_title("Temporal discontinuity QC")
    panel_label(ax, "D")

    finalize_layout(fig, left=0.20, wspace=0.44, hspace=0.48)
    return save_figure(fig, base, formats, raster_dpi=raster_dpi)


def figure10_robustness(
    track_correlations: pd.DataFrame,
    track_agreement: pd.DataFrame,
    model_results: pd.DataFrame,
    *,
    base: Path,
    formats: Iterable[str],
    raster_dpi: int,
) -> list[str]:
    fig, axes = make_figure(width="full", height_cm=15.5, nrows=2, ncols=2)
    matrix = _corr_matrix(track_correlations, "track_a", "track_b", "correlation")
    _heatmap(axes[0, 0], matrix, vmin=-1, vmax=1, cmap="coolwarm", cbar_label="Pearson r", annotate=True)
    axes[0, 0].set_title("Six-track agreement")
    panel_label(axes[0, 0], "A")

    ax = axes[0, 1]
    if track_agreement.empty:
        _empty(ax, "No track agreement summary")
    else:
        df = track_agreement.sort_values("pearson_r")
        ax.barh(np.arange(len(df)), df["pearson_r"], color="#777777", alpha=0.72)
        ax.set_yticks(np.arange(len(df)), df["comparison_track"])
        ax.set_xlim(-1, 1)
        ax.axvline(0, color="#777777", linewidth=0.65)
        ax.set_xlabel("Correlation with main track")
        clean_axis(ax)
    ax.set_title("Agreement with binocular-primary")
    panel_label(ax, "B")

    ax = axes[1, 0]
    if track_agreement.empty:
        _empty(ax, "No absolute-difference summary")
    else:
        df = track_agreement.sort_values("median_absolute_difference")
        ax.barh(np.arange(len(df)), df["median_absolute_difference"], color="#999999", alpha=0.72)
        ax.set_yticks(np.arange(len(df)), df["comparison_track"])
        ax.set_xlabel("Median absolute PIR difference")
        clean_axis(ax)
    ax.set_title("Magnitude of track disagreement")
    panel_label(ax, "C")

    ax = axes[1, 1]
    if model_results.empty:
        _empty(ax, "No smoke-model coefficients")
    else:
        df = model_results.copy()
        df["estimate"] = _numeric(df["estimate"])
        df["se"] = _numeric(df["se"])
        df = df.dropna(subset=["estimate", "se"])
        df = df[~df["term"].astype(str).str.contains("Intercept|Group Var", regex=True, na=False)].head(12)
        if df.empty:
            _empty(ax, "No non-intercept coefficients")
        else:
            model_labels = {
                "lmm_time_on_task_pir": "time-on-task LMM",
                "lmm_go_rt_pir_within_between": "Go-RT LMM",
                "gee_nogo_commission_pir": "commission GEE",
                "gee_go_program_omission_pir": "program omission GEE",
                "gee_go_clean_omission_sensitivity": "clean omission GEE",
            }
            term_labels = {
                "C(block_num)[T.2]": "Block 2",
                "pir_median_within": "PIR within",
                "pir_median_between": "PIR between",
                "time_z": "time",
                "pir_median": "PIR",
            }
            labels = [
                f"{model_labels.get(str(model), str(model).replace('_', ' '))}: "
                f"{term_labels.get(str(term), str(term).replace('_', ' '))}"
                for model, term in zip(df["model"], df["term"])
            ]
            labels = [
                textwrap.fill(
                    label,
                    width=24,
                    break_long_words=False,
                )
                for label in labels
            ]
            y = np.arange(len(df))
            ax.errorbar(df["estimate"], y, xerr=1.96 * df["se"], fmt="o", markersize=2.8, color="#555555", ecolor="#888888", elinewidth=0.7, capsize=1.5)
            ax.set_yticks(y, labels)
            ax.tick_params(axis="y", labelsize=5.5)
            ax.invert_yaxis()
            ax.axvline(0, color="#777777", linestyle=":", linewidth=0.65)
            ax.set_xscale("symlog", linthresh=1.0, linscale=1.0)
            # Singular smoke-test fits can produce very wide, non-scientific
            # confidence intervals. Keep them visible while using a sparse,
            # deterministic set of readable symlog ticks.
            smoke_ticks = np.array([-1e10, -1e6, -1e2, 0.0, 1e2, 1e6, 1e10])
            smoke_tick_labels = ["−10¹⁰", "−10⁶", "−10²", "0", "10²", "10⁶", "10¹⁰"]
            ax.set_xticks(smoke_ticks)
            ax.set_xticklabels(smoke_tick_labels, fontsize=5.5)
            ax.xaxis.get_offset_text().set_visible(False)
            ax.set_xlabel("Validation-only coefficient ± 95% CI (symlog)", fontsize=7, labelpad=5)
            clean_axis(ax)
    ax.set_title("Model-interface smoke test")
    panel_label(ax, "D")

    finalize_layout(fig, left=0.28, bottom=0.15, wspace=0.50, hspace=0.46)
    return save_figure(fig, base, formats, raster_dpi=raster_dpi)


def write_publication_suite(
    *,
    output_dir: Path,
    formats: Iterable[str],
    raster_dpi: int,
    main_track: str,
    global_detail: pd.DataFrame,
    global_summary: pd.DataFrame,
    global_distribution: pd.DataFrame,
    block_transition: pd.DataFrame,
    block_summary: pd.DataFrame,
    heterogeneity: pd.DataFrame,
    recovery_summary: pd.DataFrame,
    trial_conditions: pd.DataFrame,
    advanced_behavior: pd.DataFrame,
    nogo_continuous: pd.DataFrame,
    omission_continuous: pd.DataFrame,
    nogo_trial_lag: pd.DataFrame,
    probe_events: pd.DataFrame,
    probe_continuous: pd.DataFrame,
    probe_transitions: pd.DataFrame,
    visual_trial: pd.DataFrame,
    stimulus_summary: pd.DataFrame,
    visual_correlations: pd.DataFrame,
    raw_pir: pd.DataFrame,
    feature_within: pd.DataFrame,
    within_metrics: pd.DataFrame,
    between_metrics: pd.DataFrame,
    window_stability: pd.DataFrame,
    coverage: pd.DataFrame,
    source_mode: pd.DataFrame,
    track_correlations: pd.DataFrame,
    track_agreement: pd.DataFrame,
    model_results: pd.DataFrame,
) -> dict[str, list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = {
        "Figure01_global_PIR_landscape": lambda: figure01_global_landscape(
            global_detail, global_summary, global_distribution, block_transition,
            base=output_dir / "Figure01_global_PIR_landscape", formats=formats, raster_dpi=raster_dpi,
        ),
        "Figure02_Block_time_on_task": lambda: figure02_block_time_on_task(
            block_summary, heterogeneity, recovery_summary, global_detail,
            base=output_dir / "Figure02_Block_time_on_task", formats=formats, raster_dpi=raster_dpi,
        ),
        "Figure03_trial_behavior_states": lambda: figure03_trial_states(
            trial_conditions, advanced_behavior,
            base=output_dir / "Figure03_trial_behavior_states", formats=formats, raster_dpi=raster_dpi,
        ),
        "Figure04_error_precursor_trajectories": lambda: figure04_error_precursors(
            nogo_continuous, omission_continuous, nogo_trial_lag,
            base=output_dir / "Figure04_error_precursor_trajectories", formats=formats, raster_dpi=raster_dpi,
        ),
        "Figure05_probe_states_trajectories": lambda: figure05_probe_states(
            probe_events, probe_continuous, probe_transitions,
            base=output_dir / "Figure05_probe_states_trajectories", formats=formats, raster_dpi=raster_dpi,
        ),
        "Figure06_visual_PLR_controls": lambda: figure06_visual_controls(
            visual_trial, stimulus_summary, visual_correlations,
            base=output_dir / "Figure06_visual_PLR_controls", formats=formats, raster_dpi=raster_dpi,
        ),
        "Figure07_individual_differences": lambda: figure07_individual_differences(
            heterogeneity, raw_pir,
            base=output_dir / "Figure07_individual_differences", formats=formats, raster_dpi=raster_dpi,
        ),
        "Figure08_feature_structure_multiscale": lambda: figure08_feature_structure(
            feature_within, within_metrics, between_metrics, window_stability,
            base=output_dir / "Figure08_feature_structure_multiscale", formats=formats, raster_dpi=raster_dpi,
        ),
        "Figure09_data_quality_coverage": lambda: figure09_quality_control(
            coverage, source_mode, track=main_track,
            base=output_dir / "Figure09_data_quality_coverage", formats=formats, raster_dpi=raster_dpi,
        ),
        "Figure10_robustness_models": lambda: figure10_robustness(
            track_correlations, track_agreement, model_results,
            base=output_dir / "Figure10_robustness_models", formats=formats, raster_dpi=raster_dpi,
        ),
    }
    outputs: dict[str, list[str]] = {}
    for name, job in jobs.items():
        outputs[name] = job()
    return outputs
