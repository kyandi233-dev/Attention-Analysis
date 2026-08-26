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


def _sorted_codes(values: pd.Series) -> list[str]:
    unique = [str(value) for value in values.dropna().astype(str).unique()]
    def key(value: str) -> tuple[int, float | str]:
        try:
            return (0, float(value))
        except ValueError:
            return (1, value)
    return sorted(unique, key=key)


def plot_probe_response_distribution(
    events: pd.DataFrame,
    *,
    base: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    if events.empty or "probe_response_code" not in events.columns:
        ax.text(0.5, 0.5, "No probe_response data", ha="center", va="center", transform=ax.transAxes)
    else:
        df = events.dropna(subset=["probe_response_code"]).copy()
        codes = _sorted_codes(df["probe_response_code"])
        blocks = sorted(pd.to_numeric(df["block_num"], errors="coerce").dropna().astype(int).unique())
        x = np.arange(len(blocks), dtype=float)
        bottom = np.zeros(len(blocks), dtype=float)
        for code in codes:
            fractions: list[float] = []
            for block in blocks:
                frame = df[pd.to_numeric(df["block_num"], errors="coerce").eq(block)]
                denom = len(frame)
                fractions.append(float(frame["probe_response_code"].astype(str).eq(code).mean()) if denom else 0.0)
            values = np.asarray(fractions, dtype=float)
            ax.bar(x, values, bottom=bottom, label=f"raw option {code}")
            bottom += values
        ax.set_xticks(x, [f"Block {block}" for block in blocks])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction of probe responses")
        ax.legend(title="probe_response", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_title("Probe response option distribution")
    _banner(fig)
    return _save(fig, base, formats, dpi)


def plot_probe_response_pir_windows(
    summary: pd.DataFrame,
    *,
    base: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    if summary.empty or "pir_median" not in summary.columns:
        ax.text(0.5, 0.5, "No probe-response PIR rows", ha="center", va="center", transform=ax.transAxes)
    else:
        df = summary.copy()
        df["pir_median"] = pd.to_numeric(df["pir_median"], errors="coerce")
        codes = _sorted_codes(df["probe_response_code"])
        x_map = {code: idx for idx, code in enumerate(codes)}
        for window_name in list(dict.fromkeys(df["window_name"].astype(str))):
            frame = df[df["window_name"].astype(str).eq(window_name)].copy()
            grouped = frame.groupby("probe_response_code")["pir_median"].agg(
                median="median",
                q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75),
            )
            xs = [x_map[str(code)] for code in grouped.index]
            y = grouped["median"].to_numpy(dtype=float)
            lower = y - grouped["q25"].to_numpy(dtype=float)
            upper = grouped["q75"].to_numpy(dtype=float) - y
            ax.errorbar(xs, y, yerr=np.vstack([lower, upper]), marker="o", capsize=3, label=window_name)
        ax.set_xticks(range(len(codes)), [f"raw {code}" for code in codes])
        ax.axhline(0, linewidth=0.8)
        ax.set_ylabel("Subject-level pre-probe PIR median")
        ax.legend(title="Pre-probe window")
    ax.set_xlabel("probe_response raw option")
    ax.set_title("Probe response option × pre-probe PIR")
    _banner(fig)
    return _save(fig, base, formats, dpi)


def plot_probe_response_behavior(
    summary: pd.DataFrame,
    *,
    window_name: str,
    base: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    df = summary[summary["window_name"].astype(str).eq(window_name)].copy() if not summary.empty else pd.DataFrame()
    if df.empty or "go_rt_median_ms" not in df.columns:
        ax.text(0.5, 0.5, "No pre-probe Go RT rows", ha="center", va="center", transform=ax.transAxes)
    else:
        df["go_rt_median_ms"] = pd.to_numeric(df["go_rt_median_ms"], errors="coerce")
        codes = _sorted_codes(df["probe_response_code"])
        groups = [
            df.loc[df["probe_response_code"].astype(str).eq(code), "go_rt_median_ms"].dropna().to_numpy()
            for code in codes
        ]
        labels = [f"raw {code}" for code in codes]
        ax.boxplot(groups, showfliers=False)
        ax.set_xticks(range(1, len(labels) + 1), labels=labels)
        rng = np.random.default_rng(0)
        for idx, values in enumerate(groups, start=1):
            if len(values):
                ax.scatter(idx + rng.normal(0, 0.035, len(values)), values, s=24, alpha=0.55)
        ax.set_ylabel("Subject-level pre-probe Go RT median (ms)")
    ax.set_xlabel("probe_response raw option")
    ax.set_title(f"Probe response option × pre-probe behavior ({window_name})")
    _banner(fig)
    return _save(fig, base, formats, dpi)


def plot_probe_response_vigilance(
    events: pd.DataFrame,
    *,
    base: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    if events.empty or "probe_vigilance" not in events.columns:
        ax.text(0.5, 0.5, "No probe vigilance rows", ha="center", va="center", transform=ax.transAxes)
    else:
        df = events.dropna(subset=["probe_response_code", "probe_vigilance"]).copy()
        df["probe_vigilance"] = pd.to_numeric(df["probe_vigilance"], errors="coerce")
        df = df.dropna(subset=["probe_vigilance"])
        codes = _sorted_codes(df["probe_response_code"])
        groups = [
            df.loc[df["probe_response_code"].astype(str).eq(code), "probe_vigilance"].dropna().to_numpy()
            for code in codes
        ]
        if codes:
            labels = [f"raw {code}" for code in codes]
            ax.boxplot(groups, showfliers=False)
            ax.set_xticks(range(1, len(labels) + 1), labels=labels)
            rng = np.random.default_rng(1)
            for idx, values in enumerate(groups, start=1):
                if len(values):
                    ax.scatter(idx + rng.normal(0, 0.035, len(values)), values, s=20, alpha=0.45)
        else:
            ax.text(0.5, 0.5, "No paired probe_response / vigilance rows", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("probe_vigilance raw numeric value")
    ax.set_xlabel("probe_response raw option")
    ax.set_title("Joint structure of the two probe responses")
    _banner(fig)
    return _save(fig, base, formats, dpi)
