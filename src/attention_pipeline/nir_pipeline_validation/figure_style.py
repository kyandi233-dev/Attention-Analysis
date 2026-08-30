from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .analysis import VALIDATION_LABEL
from attention_pipeline.formal_analysis.publication_style import (
    FONT_FALLBACKS,
    finalize_publication_figure,
)

CM_TO_INCH = 1.0 / 2.54

# Journal-oriented widths following the project Figure specification:
# single column ~8 cm, medium ~14 cm, full width ~17 cm.
FIGURE_WIDTH_CM = {
    "single": 8.0,
    "medium": 14.0,
    "full": 17.0,
}

# Compact, color-blind-friendly palette. Grayscale/linestyle redundancy is used
# in publication figures so interpretation never depends on color alone.
PALETTE = {
    "block1": "#2F5597",
    "block2": "#C55A11",
    "go_correct": "#4C78A8",
    "go_omission_program": "#E45756",
    "nogo_correct": "#59A14F",
    "nogo_commission": "#B279A2",
    "correct_inhibition": "#59A14F",
    "commission": "#E45756",
    "clean_omission": "#4C78A8",
    "ambiguous_omission": "#E45756",
    "neutral": "#666666",
    "light": "#B8B8B8",
}

LINESTYLES = {
    "block1": "-",
    "block2": "--",
    "correct_inhibition": "-",
    "commission": "--",
    "go_correct": "-",
    "go_omission_program": "--",
    "nogo_correct": "-",
    "nogo_commission": "--",
}


def configure_publication_style() -> None:
    """Apply one centralized style to every manuscript-oriented NIR figure."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": FONT_FALLBACKS,
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": 7.0,
            "legend.title_fontsize": 7.0,
            "legend.frameon": False,
            "lines.linewidth": 1.25,
            "lines.markersize": 4.0,
            "patch.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def figure_size(width: str = "full", height_cm: float = 10.0) -> tuple[float, float]:
    width_cm = FIGURE_WIDTH_CM.get(width, FIGURE_WIDTH_CM["full"])
    return width_cm * CM_TO_INCH, float(height_cm) * CM_TO_INCH


def make_figure(
    *,
    width: str = "full",
    height_cm: float = 10.0,
    nrows: int = 1,
    ncols: int = 1,
    sharex: bool = False,
    sharey: bool = False,
    width_ratios: list[float] | None = None,
    height_ratios: list[float] | None = None,
):
    configure_publication_style()
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figure_size(width, height_cm),
        sharex=sharex,
        sharey=sharey,
        gridspec_kw={
            **({"width_ratios": width_ratios} if width_ratios else {}),
            **({"height_ratios": height_ratios} if height_ratios else {}),
        },
        constrained_layout=False,
    )
    return fig, axes


def clean_axis(ax: plt.Axes, *, grid_y: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=2.0)
    if grid_y:
        ax.grid(axis="y", linewidth=0.45, alpha=0.22, zorder=0)
    else:
        ax.grid(False)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        clip_on=False,
    )


def validation_banner(fig: plt.Figure) -> None:
    fig.text(
        0.5,
        0.006,
        VALIDATION_LABEL,
        ha="center",
        va="bottom",
        fontsize=5.8,
        color="#555555",
    )


def finalize_layout(
    fig: plt.Figure,
    *,
    left: float = 0.09,
    right: float = 0.985,
    bottom: float = 0.11,
    top: float = 0.94,
    wspace: float = 0.30,
    hspace: float = 0.36,
    banner: bool = True,
) -> None:
    if banner:
        validation_banner(fig)
        bottom = max(bottom, 0.12)
    fig.subplots_adjust(
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        wspace=wspace,
        hspace=hspace,
    )


def save_figure(
    fig: plt.Figure,
    base: Path,
    formats: Iterable[str],
    *,
    raster_dpi: int = 600,
) -> list[str]:
    """Save manuscript figures without changing the physical canvas dimensions.

    Vector outputs (PDF/SVG/EPS) preserve line/text geometry. Raster outputs use
    the configured high DPI; TIFF uses LZW compression when supported.
    """
    finalize_publication_figure(fig)
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for raw_fmt in formats:
        fmt = str(raw_fmt).lower().lstrip(".")
        path = base.with_suffix(f".{fmt}")
        kwargs: dict = {
            "format": fmt,
            "bbox_inches": None,
            "pad_inches": 0.0,
        }
        if fmt in {"png", "tif", "tiff", "jpg", "jpeg"}:
            kwargs["dpi"] = int(raster_dpi)
        if fmt in {"tif", "tiff"}:
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        outputs.append(str(path))
    plt.close(fig)
    return outputs
