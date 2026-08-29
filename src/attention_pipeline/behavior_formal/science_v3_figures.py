"""Backward-compatible entrypoint for the expanded behavior formal figure pack."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .science_v3_figures_formal import (
    BEHAVIOR_FIGURE_CONTRACT,
    formal_figure_contract_is_chinese,
    generate_behavior_figures as _generate_behavior_figures,
)


def _read_optional(root: Path, name: str) -> pd.DataFrame | None:
    path = root / name
    if not path.is_file():
        return None
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def generate_behavior_figures(
    block: pd.DataFrame,
    primary_probe: pd.DataFrame,
    output_dir: Path,
    *,
    error_summary: pd.DataFrame | None = None,
) -> list[str]:
    """Preserve the existing runner API while loading newly written support tables."""
    output_dir = Path(output_dir)
    root = output_dir.parent
    return _generate_behavior_figures(
        block,
        primary_probe,
        output_dir,
        session=_read_optional(root, "session_metrics.csv"),
        cycle=_read_optional(root, "cycle_metrics.csv"),
        error_summary=error_summary,
        b1b2_clustered=_read_optional(root, "b1_b2_participant_cluster_bootstrap.csv"),
        candidate_validation=_read_optional(root, "behavior_candidate_metric_validation.csv"),
        metric_redundancy=_read_optional(root, "behavior_metric_redundancy.csv"),
    )


__all__ = [
    "BEHAVIOR_FIGURE_CONTRACT",
    "formal_figure_contract_is_chinese",
    "generate_behavior_figures",
]
