"""Shared manuscript figure style for formal Behavior, NIR, and RGB outputs."""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.figure import Figure


FONT_FAMILY = "Times New Roman"
FONT_FALLBACKS = ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"]


def configure_publication_style() -> None:
    """Use English-compatible serif typography and frameless legends."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": FONT_FALLBACKS,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "legend.frameon": False,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def finalize_publication_figure(fig: Figure, *, remove_titles: bool = True) -> None:
    """Apply non-negotiable image-level rules before saving."""
    for ax in fig.axes:
        if remove_titles:
            ax.set_title("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out")
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_visible(False)
    if remove_titles and getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text("")


configure_publication_style()
