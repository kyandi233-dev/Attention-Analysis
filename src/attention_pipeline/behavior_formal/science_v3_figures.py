"""Backward-compatible concise Behavior figure entrypoint.

The old metric-by-scale Cartesian figure pack remains available in historical
modules for audit/reproduction, but the formal runner no longer calls it.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .science_output import (
    _coverage_figure,
    _feature_ecdf,
    _rt_level_figures,
    build_behavior_feature_handoff,
)
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
    }


def generate_behavior_figures(
    block: pd.DataFrame,
    primary_probe: pd.DataFrame,
    output_dir: Path,
    *,
    error_summary: pd.DataFrame | None = None,
) -> list[str]:
    """Generate only the compact qualification/QC figures available at this stage.

    Full main scientific figures (Q1/Q2 coefficients and participant-clustered
    B1/B2 estimates) are assembled by ``build_behavior_science_output.py`` after
    producer tables exist. ``block`` and ``error_summary`` remain accepted for
    API compatibility but do not trigger automatic plot proliferation.
    """
    del block, error_summary
    root = Path(output_dir).parent
    for rel in ("figures/qualification", "figures/qc"):
        (root / rel).mkdir(parents=True, exist_ok=True)

    handoff = build_behavior_feature_handoff(primary_probe)
    rows = _rt_level_figures(primary_probe, root)
    for item in (
        _feature_ecdf(
            primary_probe, root, "go_correct_rt_cv", "behavior_rt_cv_ecdf",
            "Correct-Go RT CV", "ratio", "探针前30秒反应时变异系数的经验累积分布。",
        ),
        _feature_ecdf(
            primary_probe, root, "go_correct_rt_theilsen_slope_ms_per_s", "behavior_rt_slope_ecdf",
            "Theil-Sen RT slope (ms/s)", "ms/s", "探针前30秒Theil–Sen反应时斜率的经验累积分布。",
        ),
        _coverage_figure(handoff, root),
    ):
        if item is not None:
            rows.append(item)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(root / "behavior_figure_manifest.csv", index=False, encoding="utf-8-sig")
    handoff[["predictor_column", "display_name", "report_role", "coverage_summary"]].to_csv(
        root / "behavior_figure_coverage_audit.csv", index=False, encoding="utf-8-sig"
    )
    files: list[str] = []
    if not manifest.empty:
        for column in ("png_path", "svg_path"):
            files.extend(str(root / path) for path in manifest[column].dropna().astype(str))
    return files


__all__ = [
    "BEHAVIOR_FIGURE_CONTRACT",
    "formal_figure_contract_is_english",
    "publication_figure_contract",
    "generate_behavior_figures",
]
