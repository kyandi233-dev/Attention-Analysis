"""Backward-compatible Behavior figure entrypoint routed to the science layer.

Historical metric-by-scale figure generators remain importable for reproduction,
but current formal runs and redraws share one organized scientific-output path.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .science_output import build_behavior_science_output
from .science_v3_figures_formal import BEHAVIOR_FIGURE_CONTRACT, formal_figure_contract_is_english


def publication_figure_contract() -> dict[str, object]:
    return {
        "internal_title_allowed": False,
        "caption_is_external": True,
        "in_image_language": "English",
        "font_family": "Times New Roman",
        "legend_frame": False,
        "question_driven_allowlist_required": True,
        "cartesian_metric_by_scale_pack_allowed": False,
        "metric_scale_coverage_audit_required": False,
        "current_output_layer": "FormalScience/Behavior",
    }


def generate_behavior_figures(
    block: pd.DataFrame,
    primary_probe: pd.DataFrame,
    output_dir: Path,
    *,
    error_summary: pd.DataFrame | None = None,
) -> list[str]:
    """Build the current Behavior science package after producer tables exist.

    ``block``, ``primary_probe`` and ``error_summary`` remain accepted only for
    API compatibility. The authoritative source is the already-written producer
    directory so main figures, qualification figures, QC and handoff metadata
    are built from exactly the same saved tables.
    """
    del block, primary_probe, error_summary
    producer_root = Path(output_dir).parent.resolve()
    authoritative = producer_root.name == "formal_v3"
    science_root = (
        producer_root.parent.parent / "FormalScience"
        if authoritative
        else producer_root / "science_output"
    )
    result = build_behavior_science_output(
        producer_root,
        science_root,
        authoritative=authoritative,
        replace=True,
    )
    root = Path(result["science_output_root"])
    return [str(root / rel) for rel in result["figure_files"]]


__all__ = [
    "BEHAVIOR_FIGURE_CONTRACT",
    "formal_figure_contract_is_english",
    "publication_figure_contract",
    "generate_behavior_figures",
]
