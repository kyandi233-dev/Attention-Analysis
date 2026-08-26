from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def plot_dynamic_feature_matrix(
    dynamic_long: pd.DataFrame,
    *,
    track: str,
    base: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    df = dynamic_long[dynamic_long["track"].astype(str).eq(track)].copy() if not dynamic_long.empty else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(11.2, 7.2))
    if df.empty:
        ax.text(0.5, 0.5, "No dynamic-feature rows", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        summary = (
            df.groupby(["feature", "outcome", "window_name"], as_index=False)["value"]
            .median()
        )
        summary["row"] = summary["feature"].astype(str) + " | " + summary["outcome"].astype(str)
        windows = sorted(
            summary["window_name"].astype(str).unique(),
            key=lambda value: (
                float(value.replace("pre_", "").replace("s", ""))
                if value.startswith("pre_") and value.endswith("s") and value[4:-1].replace(".", "", 1).isdigit()
                else 9999.0
            ),
        )
        matrix = summary.pivot(index="row", columns="window_name", values="value").reindex(columns=windows)
        standardized = matrix.copy()
        for row in standardized.index:
            values = standardized.loc[row].to_numpy(dtype=float)
            finite = np.isfinite(values)
            if finite.sum() >= 2:
                mean = np.nanmean(values)
                sd = np.nanstd(values)
                standardized.loc[row] = (values - mean) / sd if sd > 0 else np.zeros_like(values)
        image = ax.imshow(standardized.to_numpy(dtype=float), aspect="auto")
        ax.set_xticks(np.arange(len(standardized.columns)), standardized.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(standardized.index)), standardized.index)
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Within-row standardized median")
        ax.set_title("Trial dynamic PIR feature matrix across pre-event windows")
    _banner(fig)
    return _save(fig, base, formats, dpi)


def plot_trial_multiscale_trajectory(
    trajectory: pd.DataFrame,
    *,
    feature: str,
    base: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    if trajectory.empty or feature not in trajectory.columns:
        ax.text(0.5, 0.5, "No multiscale trajectory rows", ha="center", va="center", transform=ax.transAxes)
    else:
        for outcome, frame in trajectory.groupby("outcome", sort=True):
            summary = frame.groupby("seconds_before_event")[feature].agg(
                median="median",
                q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75),
            ).sort_index(ascending=False)
            x = -summary.index.to_numpy(dtype=float)
            ax.plot(x, summary["median"], marker="o", linewidth=1.8, label=str(outcome))
            ax.fill_between(x, summary["q25"], summary["q75"], alpha=0.12)
        ax.axhline(0, linewidth=0.8)
        ax.set_xlabel("Seconds relative to trial onset")
        ax.set_ylabel(feature)
        ax.legend(title="Program-scored outcome", fontsize=8)
    ax.set_title(f"Multiscale pre-trial trajectory — {feature}")
    _banner(fig)
    return _save(fig, base, formats, dpi)


def _plot_nogo_precursor(
    precursor: pd.DataFrame,
    *,
    value_col: str,
    ylabel: str,
    title: str,
    base: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    if precursor.empty or value_col not in precursor.columns:
        ax.text(0.5, 0.5, "No No-Go precursor rows", ha="center", va="center", transform=ax.transAxes)
    else:
        for outcome, frame in precursor.groupby("event_outcome", sort=True):
            summary = frame.groupby("lag")[value_col].agg(
                median="median",
                q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75),
            ).sort_index()
            ax.plot(summary.index, summary["median"], marker="o", linewidth=2.0, label=str(outcome))
            ax.fill_between(summary.index, summary["q25"], summary["q75"], alpha=0.12)
        ax.axvline(0, linewidth=0.8)
        ax.set_xticks(sorted(precursor["lag"].dropna().astype(int).unique()))
        ax.set_xlabel("Trial lag relative to No-Go event")
        ax.set_ylabel(ylabel)
        ax.legend(title="No-Go outcome")
    ax.set_title(title)
    _banner(fig)
    return _save(fig, base, formats, dpi)


def plot_nogo_precursor_rt(
    precursor: pd.DataFrame,
    *,
    base: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    return _plot_nogo_precursor(
        precursor,
        value_col="go_rt_ms",
        ylabel="Preceding correct-Go RT (ms)",
        title="Behavioral precursor trajectory before No-Go events",
        base=base,
        formats=formats,
        dpi=dpi,
    )


def plot_nogo_precursor_pir(
    precursor: pd.DataFrame,
    *,
    base: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    return _plot_nogo_precursor(
        precursor,
        value_col="pir_median",
        ylabel="Preceding-trial PIR median",
        title="PIR precursor trajectory before No-Go events",
        base=base,
        formats=formats,
        dpi=dpi,
    )


def plot_probe_rt_by_response(
    probe_rt: pd.DataFrame,
    *,
    value_col: str,
    base: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(8.3, 5.2))
    if probe_rt.empty or "probe_response" not in probe_rt.columns or value_col not in probe_rt.columns:
        ax.text(0.5, 0.5, "Probe RT field unavailable", ha="center", va="center", transform=ax.transAxes)
    else:
        subject = (
            probe_rt.dropna(subset=["probe_response", value_col])
            .groupby(["subject", "probe_response"], as_index=False)[value_col]
            .median()
        )
        options = sorted(subject["probe_response"].dropna().unique())
        groups = [subject.loc[subject["probe_response"].eq(option), value_col].to_numpy(dtype=float) for option in options]
        if groups:
            ax.boxplot(groups, labels=[str(x) for x in options], showfliers=False)
            rng = np.random.default_rng(1)
            for idx, values in enumerate(groups, start=1):
                ax.scatter(idx + rng.normal(0, 0.035, size=len(values)), values, s=24, alpha=0.55)
        ax.set_xlabel("probe_response raw code")
        ax.set_ylabel(value_col)
    ax.set_title(f"Probe response option × {value_col}")
    _banner(fig)
    return _save(fig, base, formats, dpi)


def plot_probe_behavior_multiscale(
    probe_behavior: pd.DataFrame,
    *,
    metric: str,
    base: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    if probe_behavior.empty or metric not in probe_behavior.columns or "probe_response" not in probe_behavior.columns:
        ax.text(0.5, 0.5, "No probe-behavior multiscale rows", ha="center", va="center", transform=ax.transAxes)
    else:
        data = probe_behavior.copy()
        data["window_sec"] = data["window_name"].astype(str).str.extract(r"pre_(\d+(?:\.\d+)?)s", expand=False)
        data["window_sec"] = pd.to_numeric(data["window_sec"], errors="coerce")
        data[metric] = pd.to_numeric(data[metric], errors="coerce")
        data = data.dropna(subset=["window_sec", metric, "probe_response"])
        for option, frame in data.groupby("probe_response", sort=True):
            subject = frame.groupby(["subject", "window_sec"], as_index=False)[metric].median()
            summary = subject.groupby("window_sec")[metric].median().sort_index()
            ax.plot(summary.index, summary.values, marker="o", linewidth=1.8, label=f"response {option}")
        ax.set_xlabel("Pre-probe window (s)")
        ax.set_ylabel(metric)
        ax.legend(title="Raw probe_response")
    ax.set_title(f"Objective behavior before probe across windows — {metric}")
    _banner(fig)
    return _save(fig, base, formats, dpi)


def plot_advanced_behavior_block(
    advanced: pd.DataFrame,
    *,
    base: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    metrics = [
        col
        for col in ("go_rt_median_ms", "rt_cv", "exg_tau", "dprime", "commission_rate", "program_omission_rate")
        if col in advanced.columns
    ]
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    if advanced.empty or not metrics:
        ax.text(0.5, 0.5, "No advanced behavior summary", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        subject = advanced[["subject", "block_num", *metrics]].copy()
        long = subject.melt(id_vars=["subject", "block_num"], value_vars=metrics, var_name="metric", value_name="value")
        long["z"] = long.groupby("metric")["value"].transform(
            lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) and np.isfinite(x.std(ddof=0)) else 0.0
        )
        summary = long.groupby(["metric", "block_num"])["z"].median().unstack("block_num")
        image = ax.imshow(summary.to_numpy(dtype=float), aspect="auto")
        ax.set_yticks(np.arange(len(summary.index)), summary.index)
        ax.set_xticks(np.arange(len(summary.columns)), [f"Block {int(x)}" for x in summary.columns])
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="Standardized cohort median")
        ax.set_title("Advanced SART behavior profile by Block")
    _banner(fig)
    return _save(fig, base, formats, dpi)


def plot_track_robustness(
    correlations: pd.DataFrame,
    *,
    base: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    if correlations.empty:
        ax.text(0.5, 0.5, "No track robustness rows", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        matrix = correlations.pivot(index="track_a", columns="track_b", values="correlation")
        image = ax.imshow(matrix.to_numpy(dtype=float), aspect="equal", vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="Pearson r")
        ax.set_title("Primary / strict / eye-preserved track agreement")
    _banner(fig)
    return _save(fig, base, formats, dpi)


def plot_source_mode_qc(
    source_mode: pd.DataFrame,
    *,
    window_name: str,
    base: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    if source_mode.empty:
        ax.text(0.5, 0.5, "No source-mode QC rows", ha="center", va="center", transform=ax.transAxes)
    else:
        df = source_mode[source_mode["window_name"].astype(str).eq(window_name)].copy()
        cols = [
            col
            for col in (
                "source_mode_binocular_fraction",
                "source_mode_left_only_fraction",
                "source_mode_right_only_fraction",
                "source_mode_missing_fraction",
            )
            if col in df.columns
        ]
        summary = df.groupby("block_num")[cols].mean().sort_index()
        bottom = np.zeros(len(summary))
        for col in cols:
            values = summary[col].to_numpy(dtype=float)
            ax.bar([f"Block {int(x)}" for x in summary.index], values, bottom=bottom, label=col.replace("source_mode_", "").replace("_fraction", ""))
            bottom = bottom + np.nan_to_num(values)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Mean fraction")
        ax.legend(title="Binocular source mode", fontsize=8)
    ax.set_title(f"Binocular source composition QC — {window_name}")
    _banner(fig)
    return _save(fig, base, formats, dpi)


def plot_coverage_multidimensional(
    coverage_long: pd.DataFrame,
    *,
    analysis_level: str,
    track: str,
    window_name: str,
    base: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    if coverage_long.empty:
        ax.text(0.5, 0.5, "No multidimensional coverage rows", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        df = coverage_long[
            coverage_long["analysis_level"].astype(str).eq(analysis_level)
            & coverage_long["track"].astype(str).eq(track)
            & coverage_long["window_name"].astype(str).eq(window_name)
        ].copy()
        if df.empty:
            ax.text(0.5, 0.5, "Requested coverage slice unavailable", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
        else:
            df["subject_block"] = df["subject"].astype(str) + " / B" + df["block_num"].astype(str)
            matrix = df.pivot(index="coverage_metric", columns="subject_block", values="coverage_value")
            row_scaled = matrix.copy()
            for metric in row_scaled.index:
                values = row_scaled.loc[metric].to_numpy(dtype=float)
                finite = np.isfinite(values)
                if finite.sum() >= 2:
                    lo = np.nanmin(values)
                    hi = np.nanmax(values)
                    row_scaled.loc[metric] = (values - lo) / (hi - lo) if hi > lo else np.zeros_like(values)
            image = ax.imshow(row_scaled.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1)
            ax.set_yticks(np.arange(len(row_scaled.index)), row_scaled.index)
            ax.set_xticks(np.arange(len(row_scaled.columns)), row_scaled.columns, rotation=90)
            fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Within-metric scaled value")
            ax.set_title(f"Multidimensional coverage QC — {analysis_level} / {window_name}")
    _banner(fig)
    return _save(fig, base, formats, dpi)


def plot_visual_covariate_association(
    visual_trial: pd.DataFrame,
    *,
    x_col: str,
    base: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    if visual_trial.empty or x_col not in visual_trial.columns or "pir_median" not in visual_trial.columns:
        ax.text(0.5, 0.5, "Visual covariate table unavailable", ha="center", va="center", transform=ax.transAxes)
    else:
        x = pd.to_numeric(visual_trial[x_col], errors="coerce")
        y = pd.to_numeric(visual_trial["pir_median"], errors="coerce")
        ok = x.notna() & y.notna()
        ax.scatter(x[ok], y[ok], s=13, alpha=0.22)
        if int(ok.sum()) >= 3:
            coefficients = np.polyfit(x[ok], y[ok], deg=1)
            grid = np.linspace(float(x[ok].min()), float(x[ok].max()), 100)
            ax.plot(grid, coefficients[0] * grid + coefficients[1], linewidth=1.8)
        ax.set_xlabel(x_col)
        ax.set_ylabel("Pre-trial PIR median")
    ax.set_title("Stimulus visual covariate × PIR validation view")
    _banner(fig)
    return _save(fig, base, formats, dpi)


def plot_raw_between_person(
    raw_pir: pd.DataFrame,
    *,
    base: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    if raw_pir.empty or "raw_PIR_subject_median" not in raw_pir.columns:
        ax.text(0.5, 0.5, "Raw between-person PIR table unavailable", ha="center", va="center", transform=ax.transAxes)
    else:
        ordered = raw_pir.sort_values("raw_PIR_subject_median").reset_index(drop=True)
        ax.scatter(np.arange(len(ordered)), ordered["raw_PIR_subject_median"], s=30)
        ax.set_xticks(np.arange(len(ordered)), ordered["subject"], rotation=90)
        ax.set_ylabel("Subject raw PIR median")
        ax.set_xlabel("Subject")
    ax.set_title("Between-person raw PIR baseline characteristics")
    _banner(fig)
    return _save(fig, base, formats, dpi)
