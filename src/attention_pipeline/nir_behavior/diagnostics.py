from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .contract import EYES, OAR_COLUMN, PIR_COLUMN, PIR_VALID_COLUMN


def _one_second_median(
    frame: pd.DataFrame, value_col: str, *, valid_col: str | None = None
) -> pd.DataFrame:
    work = frame[["unix_ms", "eye", value_col] + ([valid_col] if valid_col else [])].copy()
    if valid_col:
        valid = work[valid_col].fillna(False).astype(bool)
        work.loc[~valid, value_col] = np.nan
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work["second"] = np.floor(pd.to_numeric(work["unix_ms"], errors="coerce") / 1000.0)
    result = (
        work.dropna(subset=["second"])
        .groupby(["eye", "second"], as_index=False)[value_col]
        .median()
    )
    result["unix_ms"] = result["second"] * 1000.0
    return result


def _break_large_gaps(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_gap_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaNs so matplotlib never draws across long periods with no data."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size <= 1:
        return x, y
    if max_gap_sec <= 0:
        raise ValueError("max_gap_sec must be positive")

    out_x: list[float] = [float(x[0])]
    out_y: list[float] = [float(y[0])]
    for index in range(1, len(x)):
        if np.isfinite(x[index - 1]) and np.isfinite(x[index]):
            if float(x[index] - x[index - 1]) > float(max_gap_sec):
                out_x.append(np.nan)
                out_y.append(np.nan)
        out_x.append(float(x[index]))
        out_y.append(float(y[index]))
    return np.asarray(out_x), np.asarray(out_y)


def _block_starts(trials: pd.DataFrame) -> list[tuple[int, float]]:
    result: list[tuple[int, float]] = []
    for block_num, frame in trials.groupby("block_num", sort=True):
        onset = pd.to_numeric(frame["absolute_onset_time"], errors="coerce").dropna()
        if not onset.empty:
            result.append((int(block_num), float(onset.min())))
    return result


def _timeline_plot(
    subject: str,
    nir: pd.DataFrame,
    trials: pd.DataFrame,
    *,
    value_col: str,
    valid_col: str | None,
    ylabel: str,
    path: Path,
    max_line_gap_sec: float,
) -> str:
    binned = _one_second_median(nir, value_col, valid_col=valid_col)
    origin_candidates = [
        pd.to_numeric(nir["unix_ms"], errors="coerce").min(),
        pd.to_numeric(trials["absolute_onset_time"], errors="coerce").min(),
    ]
    origin = float(np.nanmin(origin_candidates))

    fig, ax = plt.subplots(figsize=(13, 5))
    for eye in EYES:
        eye_frame = binned[binned["eye"] == eye].sort_values("unix_ms")
        if eye_frame.empty:
            continue
        x = (eye_frame["unix_ms"].to_numpy(dtype=float) - origin) / 1000.0
        y = eye_frame[value_col].to_numpy(dtype=float)
        x, y = _break_large_gaps(x, y, max_gap_sec=max_line_gap_sec)
        ax.plot(x, y, linewidth=1.0, label=eye)

    probe_times = pd.to_numeric(
        trials.loc[
            pd.to_numeric(trials["is_probe"], errors="coerce").eq(1),
            "probe_onset_time",
        ],
        errors="coerce",
    ).dropna()
    for timestamp in probe_times:
        ax.axvline((float(timestamp) - origin) / 1000.0, linewidth=0.5, alpha=0.25)

    for block_num, timestamp in _block_starts(trials):
        x_block = (timestamp - origin) / 1000.0
        ax.axvline(x_block, linewidth=1.0, linestyle="--", alpha=0.65)
        ax.text(
            x_block,
            0.985,
            f"B{block_num} start",
            rotation=90,
            va="top",
            ha="right",
            transform=ax.get_xaxis_transform(),
            fontsize=8,
        )

    commission_times = pd.to_numeric(
        trials.loc[
            pd.to_numeric(trials["commission"], errors="coerce").eq(1),
            "absolute_onset_time",
        ],
        errors="coerce",
    ).dropna()
    if len(commission_times):
        y_min, y_max = ax.get_ylim()
        ax.scatter(
            (commission_times.to_numpy(dtype=float) - origin) / 1000.0,
            np.full(len(commission_times), y_min),
            marker="x",
            s=18,
            label="commission onset",
        )
        ax.set_ylim(y_min, y_max)

    ax.set_title(f"{subject} alignment QC: {ylabel}")
    ax.set_xlabel("Seconds from first formal block data")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def _probe_centered_pir(
    subject: str,
    nir: pd.DataFrame,
    trials: pd.DataFrame,
    path: Path,
    *,
    lookback_sec: int = 60,
    bin_sec: int = 5,
) -> str:
    probes = trials[
        pd.to_numeric(trials["is_probe"], errors="coerce").eq(1)
        & pd.to_numeric(trials["probe_onset_time"], errors="coerce").notna()
    ][["block_num", "probe_onset_time"]].copy()

    records: list[dict[str, Any]] = []
    edges = np.arange(-lookback_sec, 0 + bin_sec, bin_sec, dtype=float)
    for probe in probes.itertuples(index=False):
        onset = float(probe.probe_onset_time)
        block_num = int(probe.block_num)
        block = nir[nir["block_num"] == block_num]
        for eye in EYES:
            eye_frame = block[block["eye"] == eye]
            times = pd.to_numeric(eye_frame["unix_ms"], errors="coerce").to_numpy(
                dtype=float
            )
            pir = pd.to_numeric(eye_frame[PIR_COLUMN], errors="coerce").to_numpy(
                dtype=float
            )
            gate = eye_frame[PIR_VALID_COLUMN].fillna(False).astype(bool).to_numpy()
            rel_sec = (times - onset) / 1000.0
            for left, right in zip(edges[:-1], edges[1:], strict=False):
                mask = gate & np.isfinite(pir) & (rel_sec >= left) & (rel_sec < right)
                if mask.any():
                    records.append(
                        {
                            "eye": eye,
                            "bin_mid": (left + right) / 2.0,
                            "value": float(np.median(pir[mask])),
                        }
                    )
    frame = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(9, 5))
    if not frame.empty:
        grouped = frame.groupby(["eye", "bin_mid"], as_index=False)["value"].median()
        for eye in EYES:
            eye_frame = grouped[grouped["eye"] == eye]
            if eye_frame.empty:
                continue
            ax.plot(
                eye_frame["bin_mid"], eye_frame["value"], marker="o", label=eye
            )
    ax.axvline(0, linewidth=0.8)
    ax.set_title(
        f"{subject} alignment QC: probe-centered PIR "
        f"({bin_sec}-s bins; median within probe, median across {len(probes)} probes)"
    )
    ax.set_xlabel("Seconds before probe onset")
    ax.set_ylabel("Pupil / outer-iris diameter ratio")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def generate_diagnostics(
    subject: str,
    nir: pd.DataFrame,
    trials: pd.DataFrame,
    probe_windows: pd.DataFrame,
    paths: dict[str, Path],
    *,
    max_line_gap_sec: float = 2.5,
) -> dict[str, str]:
    del probe_windows  # Coverage is summarized by the dedicated coverage report.
    return {
        "timeline_pir": _timeline_plot(
            subject,
            nir,
            trials,
            value_col=PIR_COLUMN,
            valid_col=PIR_VALID_COLUMN,
            ylabel="Pupil / outer-iris diameter ratio",
            path=paths["qc_timeline_pir"],
            max_line_gap_sec=max_line_gap_sec,
        ),
        "timeline_oar": _timeline_plot(
            subject,
            nir,
            trials,
            value_col=OAR_COLUMN,
            valid_col=None,
            ylabel="RITnet ocular aperture ratio (median)",
            path=paths["qc_timeline_oar"],
            max_line_gap_sec=max_line_gap_sec,
        ),
        "probe_centered_pir": _probe_centered_pir(
            subject,
            nir,
            trials,
            paths["qc_probe_pir"],
        ),
    }
