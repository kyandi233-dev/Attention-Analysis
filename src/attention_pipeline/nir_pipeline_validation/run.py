from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from attention_pipeline.config import load_config

from .analysis import (
    PIPELINE_VERSION,
    SCHEMA_VERSION,
    VALIDATION_LABEL,
    behavior_subject_summary,
    block_pir_summary,
    coarse_time_on_task,
    fit_smoke_models,
    load_cohort_tables,
    output_root,
    probe_analysis_table,
    selected_subjects,
    trial_analysis_table,
)
from .plots import (
    plot_block_pairs,
    plot_coverage_heatmap,
    plot_model_forest,
    plot_pipeline_schematic,
    plot_probe_windows,
    plot_time_on_task,
    plot_trial_outcomes,
)


def run_validation(
    config_path: str | Path,
    *,
    subjects: Iterable[str] | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    selected = selected_subjects(config, subjects)
    if not selected:
        raise ValueError("No completed 11_analysis_tables subjects selected")

    tables = load_cohort_tables(config, selected)
    analysis = config.section("analysis")
    figures_config = config.section("figures")

    track = str(analysis.get("track", "binocular_primary"))
    trial_window = str(analysis.get("trial_display_window", "pre_5s"))
    probe_model_window = str(analysis.get("probe_model_window", "pre_20s"))
    coarse_bin_sec = int(analysis.get("time_on_task_display_bin_sec", 30))
    min_subjects = int(analysis.get("minimum_subjects_for_models", 3))
    formats = [
        str(value).lower()
        for value in figures_config.get("formats", ["png", "pdf"])
    ]
    dpi = int(figures_config.get("dpi", 180))

    out = output_root(config)
    table_dir = out / "tables"
    figure_dir = out / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    behavior_summary = behavior_subject_summary(tables["trial_level"])
    block_summary = block_pir_summary(tables["time_on_task"], track)
    coarse = coarse_time_on_task(
        tables["time_on_task"],
        track=track,
        bin_sec=coarse_bin_sec,
    )
    trial_table = trial_analysis_table(
        tables["trial_level"],
        tables["trial_windows"],
        track=track,
        window_name=trial_window,
    )
    probe_table = probe_analysis_table(
        tables["probe_windows"],
        track=track,
        window_name=probe_model_window,
    )
    model_results, model_status = fit_smoke_models(
        trial_table,
        probe_table,
        tables["time_on_task"],
        track=track,
        min_subjects=min_subjects,
    )

    table_outputs = {
        "behavior_subject_summary": table_dir / "behavior_subject_summary.csv",
        "block_pir_summary": table_dir / "block_pir_summary.csv",
        "time_on_task_coarse": table_dir / f"time_on_task_{coarse_bin_sec}s.csv",
        "trial_analysis": table_dir / f"trial_analysis_{trial_window}.csv",
        "probe_analysis": table_dir / f"probe_analysis_{probe_model_window}.csv",
        "model_smoke_results": table_dir / "model_smoke_results.csv",
    }
    behavior_summary.to_csv(
        table_outputs["behavior_subject_summary"], index=False, encoding="utf-8-sig"
    )
    block_summary.to_csv(
        table_outputs["block_pir_summary"], index=False, encoding="utf-8-sig"
    )
    coarse.to_csv(
        table_outputs["time_on_task_coarse"], index=False, encoding="utf-8-sig"
    )
    trial_table.to_csv(
        table_outputs["trial_analysis"], index=False, encoding="utf-8-sig"
    )
    probe_table.to_csv(
        table_outputs["probe_analysis"], index=False, encoding="utf-8-sig"
    )
    model_results.to_csv(
        table_outputs["model_smoke_results"], index=False, encoding="utf-8-sig"
    )

    figure_outputs: dict[str, list[str]] = {}
    figure_outputs["pipeline_schematic"] = plot_pipeline_schematic(
        base=figure_dir / "fig00_pipeline_validation_schematic",
        formats=formats,
        dpi=dpi,
    )
    figure_outputs["time_on_task"] = plot_time_on_task(
        coarse,
        base=figure_dir / "fig01_time_on_task_trajectory",
        formats=formats,
        dpi=dpi,
    )
    figure_outputs["block_pairs"] = plot_block_pairs(
        block_summary,
        base=figure_dir / "fig02_block_paired_pir",
        formats=formats,
        dpi=dpi,
    )
    figure_outputs["trial_outcomes"] = plot_trial_outcomes(
        trial_table,
        base=figure_dir / f"fig03_trial_outcome_pir_{trial_window}",
        formats=formats,
        dpi=dpi,
    )
    figure_outputs["probe_windows"] = plot_probe_windows(
        tables["probe_windows"],
        track=track,
        base=figure_dir / "fig04_probe_vigilance_windows",
        formats=formats,
        dpi=dpi,
    )
    figure_outputs["coverage_heatmap"] = plot_coverage_heatmap(
        tables["trial_coverage"],
        track=track,
        window_name=trial_window,
        base=figure_dir / f"fig05_coverage_heatmap_{trial_window}",
        formats=formats,
        dpi=dpi,
    )
    figure_outputs["model_forest"] = plot_model_forest(
        model_results,
        base=figure_dir / "fig06_model_smoke_forest",
        formats=formats,
        dpi=dpi,
    )

    summary = {
        "status": "complete",
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_only": True,
        "nir_values_known_invalid": True,
        "validation_label": VALIDATION_LABEL,
        "subjects": selected,
        "n_subjects": len(selected),
        "track": track,
        "trial_display_window": trial_window,
        "probe_model_window": probe_model_window,
        "time_on_task_display_bin_sec": coarse_bin_sec,
        "rows": {
            "behavior_subject_summary": len(behavior_summary),
            "block_pir_summary": len(block_summary),
            "time_on_task_coarse": len(coarse),
            "trial_analysis": len(trial_table),
            "probe_analysis": len(probe_table),
            "model_smoke_results": len(model_results),
        },
        "model_status": model_status,
        "tables": {key: str(path) for key, path in table_outputs.items()},
        "figures": figure_outputs,
        "interpretation_prohibited": [
            "PIR effect direction",
            "PIR p-values",
            "Block PIR scientific differences",
            "PIR-behavior associations",
            "PIR-probe associations",
            "window selection based on apparent effects",
        ],
    }
    (out / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return summary
