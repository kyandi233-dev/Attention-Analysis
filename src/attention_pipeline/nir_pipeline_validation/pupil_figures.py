from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIGURE_SUITE_VERSION = "nir-pupil-publication-figure-suite-v2"


def _save(fig: plt.Figure, base: Path, formats: Iterable[str], dpi: int) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for fmt in formats:
        suffix = str(fmt).lower().lstrip(".")
        path = base.with_suffix(f".{suffix}")
        kwargs = {"dpi": dpi} if suffix in {"png", "jpg", "jpeg", "tif", "tiff"} else {}
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def _empty(ax: plt.Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _title(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.text(0.5, 0.985, subtitle, ha="center", va="top", fontsize=8)


def figure01_global_landscape(time_on_task: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    work = time_on_task[time_on_task["track"].astype(str).eq("binocular_primary")].copy()
    if work.empty:
        _empty(axes[0], "No binocular-primary time-on-task rows")
        _empty(axes[1], "No block distribution")
    else:
        for block_num, current in work.groupby("block_num", sort=True):
            subject = current.groupby(["session_id", "time_in_block_mid_sec"], as_index=False)["pupil_median"].median()
            summary = subject.groupby("time_in_block_mid_sec")["pupil_median"].agg(["median", "quantile"])
            med = subject.groupby("time_in_block_mid_sec")["pupil_median"].median()
            q25 = subject.groupby("time_in_block_mid_sec")["pupil_median"].quantile(0.25)
            q75 = subject.groupby("time_in_block_mid_sec")["pupil_median"].quantile(0.75)
            axes[0].plot(med.index / 60.0, med.values, label=f"Block {int(block_num)}")
            axes[0].fill_between(med.index / 60.0, q25.values, q75.values, alpha=0.15)
        axes[0].axhline(0, linewidth=0.7, linestyle=":")
        axes[0].set_xlabel("Time within block (min)")
        axes[0].set_ylabel("Median-centered pupil diameter")
        axes[0].legend()
        session_block = work.groupby(["session_id", "block_num"], as_index=False)["pupil_median"].median()
        groups = [
            _numeric(session_block[session_block["block_num"].eq(block)], "pupil_median").dropna().to_numpy()
            for block in (1, 2)
        ]
        if all(len(group) for group in groups):
            axes[1].boxplot(groups, labels=["Block 1", "Block 2"], showfliers=False)
        else:
            _empty(axes[1], "Block 1/2 coverage incomplete")
        axes[1].set_ylabel("Session-median centered pupil")
    _title(fig, "Figure01 — Whole-experiment pupil landscape", "Descriptive only; session and analysis-group counts are distinct axes")
    return fig


def figure02_eye_block_hierarchy(time_on_task: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    work = time_on_task.copy()
    tracks = ["left_primary", "right_primary"]
    eye = work[work["track"].astype(str).isin(tracks)].copy()
    if eye.empty:
        _empty(axes[0], "No left/right tracks")
    else:
        session = eye.groupby(["session_id", "block_num", "track"], as_index=False)["pupil_median"].median()
        labels, groups = [], []
        for block in (1, 2):
            for track in tracks:
                values = _numeric(session[(session["block_num"].eq(block)) & (session["track"].eq(track))], "pupil_median").dropna().to_numpy()
                labels.append(f"B{block}\n{track.split('_')[0]}")
                groups.append(values)
        axes[0].boxplot(groups, labels=labels, showfliers=False)
        axes[0].set_ylabel("Centered pupil diameter")
    if {"session_id", "analysis_group_token"}.issubset(work.columns):
        hierarchy = work[["session_id", "analysis_group_token"]].drop_duplicates()
        counts = hierarchy.groupby("analysis_group_token")["session_id"].nunique()
        axes[1].bar(["single-session groups", "double-session groups"], [(counts == 1).sum(), (counts == 2).sum()])
        axes[1].set_ylabel("Analysis groups")
    else:
        _empty(axes[1], "Hierarchy columns unavailable")
    _title(fig, "Figure02 — Eye, block, session and analysis-group hierarchy", "Left/right sensitivity is preserved; repeat sessions are not independent participants")
    return fig


def figure03_sart_errors(trials: pd.DataFrame, cv_metrics: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    go = pd.to_numeric(trials.get("go_omission_target"), errors="coerce").dropna()
    nogo = pd.to_numeric(trials.get("nogo_commission_target"), errors="coerce").dropna()
    axes[0].bar(["Go omission", "NoGo commission"], [float(go.mean()) if len(go) else np.nan, float(nogo.mean()) if len(nogo) else np.nan])
    axes[0].set_ylabel("Observed rate")
    axes[0].set_ylim(bottom=0)
    if cv_metrics.empty:
        _empty(axes[1], "No participant-exclusive CV metrics")
    else:
        summary = cv_metrics.groupby(["target", "model_stage"], as_index=False)["balanced_accuracy"].mean()
        labels = [f"{row.target}\n{row.model_stage}" for row in summary.itertuples(index=False)]
        axes[1].bar(labels, summary["balanced_accuracy"].to_numpy(dtype=float))
        axes[1].axhline(0.5, linewidth=0.7, linestyle=":")
        axes[1].set_ylabel("Balanced accuracy")
        axes[1].tick_params(axis="x", rotation=30)
    _title(fig, "Figure03 — SART failure modes and prediction baselines", "Go omissions and NoGo commissions have separate denominators and imbalance metrics")
    return fig


def figure04_probe_state(probe_windows: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    primary = probe_windows[probe_windows["track"].astype(str).eq("binocular_primary")].copy()
    if primary.empty:
        _empty(axes[0], "No probe pupil windows")
        _empty(axes[1], "No probe response rows")
    else:
        windows = primary.groupby("window_name", as_index=False)["pupil_median"].median()
        axes[0].plot(np.arange(len(windows)), windows["pupil_median"], marker="o")
        axes[0].set_xticks(np.arange(len(windows)), windows["window_name"], rotation=30, ha="right")
        axes[0].set_ylabel("Median centered pupil")
        if "probe_response" in primary:
            response = primary.dropna(subset=["probe_response"]).groupby("probe_response")["pupil_median"].median()
            if len(response):
                axes[1].bar(response.index.astype(str), response.values)
                axes[1].set_ylabel("Median centered pupil")
            else:
                _empty(axes[1], "Probe responses unavailable")
        else:
            _empty(axes[1], "Probe responses unavailable")
    _title(fig, "Figure04 — Probe-state multiscale pupil summary", "Q2 stream is reported descriptively across prespecified pre-probe windows")
    return fig


def figure05_visual(visual_trials: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if visual_trials.empty:
        _empty(axes[0], "Visual covariates unavailable")
        _empty(axes[1], "Visual direction/coverage unavailable")
    else:
        current_cols = [c for c in visual_trials if c.startswith("current_visual__") and c.endswith("__mean")]
        previous_cols = [c for c in visual_trials if c.startswith("previous_visual__") and c.endswith("__mean")]
        pupil = _numeric(visual_trials, "pupil_median")
        labels, values = [], []
        for column in current_cols[:4] + previous_cols[:4]:
            x = _numeric(visual_trials, column)
            valid = x.notna() & pupil.notna()
            labels.append(column.replace("current_visual__", "cur:").replace("previous_visual__", "prev:"))
            values.append(float(x[valid].corr(pupil[valid])) if valid.sum() >= 3 else np.nan)
        if labels:
            axes[0].barh(labels, values)
            axes[0].axvline(0, linewidth=0.7, linestyle=":")
            axes[0].set_xlabel("Pearson r with centered pupil")
        else:
            _empty(axes[0], "No numeric visual metrics")
        matched = [
            float(visual_trials.get("current_visual_matched", pd.Series(False, index=visual_trials.index)).mean()),
            float(visual_trials.get("previous_visual_matched", pd.Series(False, index=visual_trials.index)).mean()),
        ]
        axes[1].bar(["current", "strict previous"], matched)
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("Visual-property match fraction")
    _title(fig, "Figure05 — Stimulus brightness/visual covariate audit", "Current and strictly previous stimulus directions are separated; multi-component coverage is retained")
    return fig


def figure06_feature_families(feature_audit: pd.DataFrame, phasic: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if feature_audit.empty:
        _empty(axes[0], "No feature-family audit")
    else:
        axes[0].bar(feature_audit["family"].astype(str), feature_audit["n_present"].to_numpy(dtype=float))
        axes[0].set_ylabel("Expected features present")
        axes[0].tick_params(axis="x", rotation=25)
    if phasic.empty:
        _empty(axes[1], "No explicit phasic pupil deltas")
    else:
        values = _numeric(phasic, "phasic_pupil_delta").dropna()
        axes[1].hist(values, bins=min(30, max(5, int(np.sqrt(max(1, len(values)))))))
        axes[1].axvline(0, linewidth=0.7, linestyle=":")
        axes[1].set_xlabel("Response minus pre-event baseline")
        axes[1].set_ylabel("Trial count")
    _title(fig, "Figure06 — Tonic, phasic, derivative and variability separation", "Phasic features require explicit baseline metadata; families are not pooled into one pupil construct")
    return fig


def figure07_multiscale(trial_windows: pd.DataFrame, dependency: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    primary = trial_windows[trial_windows["track"].astype(str).eq("binocular_primary")].copy()
    if primary.empty:
        _empty(axes[0], "No trial windows")
    else:
        summary = primary.groupby("window_name", as_index=False).agg(
            pupil_median=("pupil_median", "median"),
            valid_fraction=("pupil_valid_fraction", "median"),
        )
        axes[0].plot(np.arange(len(summary)), summary["pupil_median"], marker="o")
        axes[0].set_xticks(np.arange(len(summary)), summary["window_name"], rotation=40, ha="right")
        axes[0].set_ylabel("Median centered pupil")
    if dependency.empty:
        _empty(axes[1], "No overlap audit")
    else:
        dep = dependency[dependency["track"].astype(str).eq("binocular_primary")]
        summary = dep.groupby("window_name", as_index=False)["overlap_with_previous_fraction"].median()
        axes[1].bar(summary["window_name"].astype(str), summary["overlap_with_previous_fraction"])
        axes[1].set_ylabel("Window overlap fraction")
        axes[1].tick_params(axis="x", rotation=40)
    _title(fig, "Figure07 — Prespecified multiscale windows and dependence audit", "Overlapping state windows remain descriptive and are not treated as independent trial observations")
    return fig


def figure08_heterogeneity(time_on_task: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    primary = time_on_task[time_on_task["track"].astype(str).eq("binocular_primary")].copy()
    if primary.empty:
        _empty(axes[0], "No session summaries")
        _empty(axes[1], "No group summaries")
    else:
        session = primary.groupby("session_id", as_index=False)["pupil_median"].agg(["median", "std"]).reset_index()
        axes[0].scatter(session["median"], session["std"], s=16)
        axes[0].set_xlabel("Session median centered pupil")
        axes[0].set_ylabel("Session SD")
        group = primary.groupby("analysis_group_token", as_index=False)["pupil_median"].median()
        axes[1].hist(group["pupil_median"].dropna(), bins=min(20, max(5, int(np.sqrt(max(1, len(group)))))))
        axes[1].set_xlabel("Analysis-group median centered pupil")
        axes[1].set_ylabel("Group count")
    _title(fig, "Figure08 — Session and analysis-group heterogeneity", "Between-session/group summaries are kept separate from within-session centered effects")
    return fig


def figure09_repeat_qc(repeat_summary: pd.DataFrame, qc_axes: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if repeat_summary.empty:
        _empty(axes[0], "No double-session repeat summaries")
    else:
        axes[0].scatter(repeat_summary["session_a_pupil_median"], repeat_summary["session_b_pupil_median"], s=20)
        finite = pd.concat([repeat_summary["session_a_pupil_median"], repeat_summary["session_b_pupil_median"]]).dropna()
        if len(finite):
            lo, hi = float(finite.min()), float(finite.max())
            axes[0].plot([lo, hi], [lo, hi], linestyle=":")
        axes[0].set_xlabel("Repeat session A")
        axes[0].set_ylabel("Repeat session B")
    if qc_axes.empty:
        _empty(axes[1], "No QC count axes")
    else:
        axes[1].bar(qc_axes["axis"].astype(str), qc_axes["count"].to_numpy(dtype=float))
        axes[1].tick_params(axis="x", rotation=40)
        axes[1].set_ylabel("Count (axis-specific)")
    _title(fig, "Figure09 — Repeat-session descriptive check and QC axes", "B1/B2 repeat summaries are descriptive only; QC counts are not collapsed across hierarchy levels")
    return fig


def figure10_model_admission(
    cv_metrics: pd.DataFrame,
    model_failures: pd.DataFrame,
    admission: Mapping[str, object],
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if cv_metrics.empty:
        _empty(axes[0], "No leakage-safe prediction metrics")
    else:
        summary = cv_metrics.groupby("model_stage", as_index=False).agg(
            balanced_accuracy=("balanced_accuracy", "mean"),
            majority_accuracy=("majority_accuracy", "mean"),
        )
        x = np.arange(len(summary))
        axes[0].bar(x - 0.18, summary["majority_accuracy"], width=0.36, label="majority")
        axes[0].bar(x + 0.18, summary["balanced_accuracy"], width=0.36, label="model")
        axes[0].set_xticks(x, summary["model_stage"].astype(str), rotation=25)
        axes[0].set_ylabel("Accuracy metric")
        axes[0].legend()
    gates = admission.get("gates", {}) if isinstance(admission, Mapping) else {}
    labels = list(gates.keys())
    values = [1 if bool(gates[name]) else 0 for name in labels]
    if labels:
        axes[1].barh(labels, values)
        axes[1].set_xlim(0, 1.05)
        axes[1].set_xlabel("Gate passed")
        axes[1].set_title(f"Model failures recorded: {len(model_failures)}")
    else:
        _empty(axes[1], "No report-admission gates")
    _title(fig, "Figure10 — Baseline → NIR increment and report admission", "Participant-exclusive outer folds; failed/singular models are failure-table entries, never formal statistics")
    return fig


def write_pupil_figure_suite(
    *,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
    time_on_task: pd.DataFrame,
    trials: pd.DataFrame,
    trial_windows: pd.DataFrame,
    probe_windows: pd.DataFrame,
    dependency: pd.DataFrame,
    visual_trials: pd.DataFrame,
    feature_audit: pd.DataFrame,
    phasic: pd.DataFrame,
    repeat_summary: pd.DataFrame,
    qc_axes: pd.DataFrame,
    cv_metrics: pd.DataFrame,
    model_failures: pd.DataFrame,
    admission: Mapping[str, object],
) -> dict[str, list[str]]:
    figures = {
        "Figure01": figure01_global_landscape(time_on_task),
        "Figure02": figure02_eye_block_hierarchy(time_on_task),
        "Figure03": figure03_sart_errors(trials, cv_metrics),
        "Figure04": figure04_probe_state(probe_windows),
        "Figure05": figure05_visual(visual_trials),
        "Figure06": figure06_feature_families(feature_audit, phasic),
        "Figure07": figure07_multiscale(trial_windows, dependency),
        "Figure08": figure08_heterogeneity(time_on_task),
        "Figure09": figure09_repeat_qc(repeat_summary, qc_axes),
        "Figure10": figure10_model_admission(cv_metrics, model_failures, admission),
    }
    outputs: dict[str, list[str]] = {}
    for name, fig in figures.items():
        outputs[name] = _save(fig, output_dir / name, formats, dpi)
    return outputs
