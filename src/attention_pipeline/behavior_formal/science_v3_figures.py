"""Backward-compatible entrypoint for the expanded behavior formal figure pack."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from matplotlib.axes import Axes

from .omission_candidate_validation import validate_omission_candidates
from .science_v3_figures_formal import (
    BEHAVIOR_FIGURE_CONTRACT,
    formal_figure_contract_is_chinese,
    generate_behavior_figures as _generate_behavior_figures,
)
from .science_v3_metric_figures import generate_complete_metric_figure_pack


def _read_optional(root: Path, name: str) -> pd.DataFrame | None:
    path = root / name
    if not path.is_file():
        return None
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


@contextmanager
def _suppress_internal_titles():
    """Force publication images to keep titles outside the image as captions."""
    original = Axes.set_title

    def _title_guard(self, label, *args, **kwargs):  # type: ignore[no-untyped-def]
        return original(self, "", *args, **kwargs)

    Axes.set_title = _title_guard  # type: ignore[method-assign]
    try:
        yield
    finally:
        Axes.set_title = original  # type: ignore[method-assign]


def publication_figure_contract() -> dict[str, object]:
    return {
        "internal_title_allowed": False,
        "caption_is_external": True,
        "chinese_axes_and_legends_required": True,
        "metric_scale_coverage_audit_required": True,
    }


def generate_behavior_figures(
    block: pd.DataFrame,
    primary_probe: pd.DataFrame,
    output_dir: Path,
    *,
    error_summary: pd.DataFrame | None = None,
) -> list[str]:
    """Generate overview plus systematic metric figures with external captions.

    The historical overview pack is retained for continuity, but its internal
    titles are forcibly suppressed. A second systematic pack enumerates every
    canonical and omission-taxonomy metric across each scientifically relevant
    scale and writes an explicit generated/not-estimable coverage audit. The
    omission taxonomy also receives a non-p-value candidate science audit for
    coverage, floor/ceiling, within/between structure and redundancy.
    """
    output_dir = Path(output_dir)
    root = output_dir.parent
    session = _read_optional(root, "session_metrics.csv")
    cycle = _read_optional(root, "cycle_metrics.csv")

    omission_validation, omission_redundancy = validate_omission_candidates(
        {
            "session": session if session is not None else pd.DataFrame(),
            "block": block,
            "cycle": cycle if cycle is not None else pd.DataFrame(),
        },
        primary_probe,
    )
    omission_validation.to_csv(
        root / "behavior_omission_candidate_validation.csv", index=False, encoding="utf-8-sig"
    )
    omission_redundancy.to_csv(
        root / "behavior_omission_candidate_redundancy.csv", index=False, encoding="utf-8-sig"
    )

    with _suppress_internal_titles():
        overview = _generate_behavior_figures(
            block,
            primary_probe,
            output_dir,
            session=session,
            cycle=cycle,
            error_summary=error_summary,
            b1b2_clustered=_read_optional(root, "b1_b2_participant_cluster_bootstrap.csv"),
            candidate_validation=_read_optional(root, "behavior_candidate_metric_validation.csv"),
            metric_redundancy=_read_optional(root, "behavior_metric_redundancy.csv"),
        )

    systematic, metric_manifest, coverage = generate_complete_metric_figure_pack(
        session=session,
        block=block,
        cycle=cycle,
        probe=primary_probe,
        output_dir=output_dir,
    )

    overview_rows: list[dict[str, object]] = []
    for path in overview:
        name = Path(path).name
        contract = BEHAVIOR_FIGURE_CONTRACT.get(name)
        caption = str(contract[0]) if contract else name
        overview_rows.append({
            "metric": "multi_metric_or_contract_overview",
            "metric_label_zh": "多指标/契约概览",
            "figure_family": "overview",
            "analysis_scale": "mixed",
            "status": "generated",
            "reason": "generated",
            "filename": name,
            "caption_zh": caption,
            "internal_title_allowed": False,
            "caption_is_external": True,
            "participant_n": pd.NA,
            "session_n": pd.NA,
            "report_layer": "core_or_support_by_figure_contract",
        })
    overview_manifest = pd.DataFrame(overview_rows)
    manifest = pd.concat([overview_manifest, metric_manifest], ignore_index=True, sort=False)
    manifest.to_csv(root / "behavior_figure_manifest.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(root / "behavior_figure_coverage_audit.csv", index=False, encoding="utf-8-sig")

    return [*overview, *systematic]


__all__ = [
    "BEHAVIOR_FIGURE_CONTRACT",
    "formal_figure_contract_is_chinese",
    "publication_figure_contract",
    "generate_behavior_figures",
]
