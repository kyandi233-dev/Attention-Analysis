from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from attention_pipeline.config import load_config

from .analysis import load_cohort_tables, output_root, selected_subjects
from .extended import (
    attach_visual_covariates,
    load_visual_covariates,
    nogo_precursor_trajectory,
    probe_behavior_multiscale,
    probe_rt_summary,
)
from .landscape import (
    build_event_catalogs,
    continuous_event_trajectory,
    load_continuous_analysis_ready,
)
from .supplementary_figures import (
    supplementary01_error_dynamics,
    supplementary02_probe_objective_behavior,
    supplementary03_probe_response_times,
    supplementary04_stimulus_identity_size,
)


SUPPLEMENTARY_SUITE_VERSION = "nir-publication-supplementary-v1"


def run_supplementary_validation(
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
    figures = config.section("figures")

    track = str(analysis.get("main_track", "binocular_primary"))
    trial_window = str(analysis.get("trial_display_window", "pre_5s"))
    precursor_window = str(analysis.get("precursor_trial_window", trial_window))
    n_preceding_go = int(analysis.get("no_go_preceding_correct_go_trials", 5))
    start_sec = float(analysis.get("event_trajectory_start_sec", -60.0))
    end_sec = float(analysis.get("event_trajectory_end_sec", 2.0))
    bin_sec = float(analysis.get("event_trajectory_bin_sec", 1.0))
    max_go_reference = int(analysis.get("max_go_reference_events_per_subject_block", 60))
    formats = [str(v).lower() for v in figures.get("publication_formats", ["pdf", "svg", "png", "tiff"])]
    raster_dpi = int(figures.get("raster_dpi", 600))

    continuous, continuous_status = load_continuous_analysis_ready(config, selected, track=track)
    catalogs = build_event_catalogs(
        tables["trial_level"],
        tables["probe_windows"],
        track=track,
        max_go_reference_per_subject_block=max_go_reference,
    )
    nogo_continuous = continuous_event_trajectory(
        continuous,
        catalogs["nogo"],
        start_sec=start_sec,
        end_sec=end_sec,
        bin_sec=bin_sec,
    )
    omission_continuous = continuous_event_trajectory(
        continuous,
        catalogs["omission"],
        start_sec=start_sec,
        end_sec=end_sec,
        bin_sec=bin_sec,
    )
    nogo_trial_lag = nogo_precursor_trajectory(
        tables["trial_level"],
        tables["trial_windows"],
        track=track,
        window_name=precursor_window,
        n_preceding_go=n_preceding_go,
    )
    probe_behavior = probe_behavior_multiscale(tables["probe_windows"], track=track)
    probe_rt = probe_rt_summary(tables["probe_windows"], track=track)

    visual_table, visual_status = load_visual_covariates(config)
    visual_trial = attach_visual_covariates(
        tables["trial_level"],
        tables["trial_windows"],
        visual_table,
        track=track,
        window_name=trial_window,
    )

    out = output_root(config)
    figure_dir = out / "figures" / "publication" / "supplementary"
    outputs = {
        "FigureS01_error_dynamics": supplementary01_error_dynamics(
            nogo_continuous,
            omission_continuous,
            nogo_trial_lag,
            base=figure_dir / "FigureS01_error_dynamics",
            formats=formats,
            raster_dpi=raster_dpi,
        ),
        "FigureS02_probe_objective_behavior": supplementary02_probe_objective_behavior(
            probe_behavior,
            base=figure_dir / "FigureS02_probe_objective_behavior",
            formats=formats,
            raster_dpi=raster_dpi,
        ),
        "FigureS03_probe_response_times": supplementary03_probe_response_times(
            probe_rt,
            base=figure_dir / "FigureS03_probe_response_times",
            formats=formats,
            raster_dpi=raster_dpi,
        ),
        "FigureS04_stimulus_identity_size": supplementary04_stimulus_identity_size(
            visual_trial,
            base=figure_dir / "FigureS04_stimulus_identity_size",
            formats=formats,
            raster_dpi=raster_dpi,
        ),
    }

    summary = {
        "status": "complete",
        "suite_version": SUPPLEMENTARY_SUITE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_only": True,
        "nir_values_known_invalid": False,
        "endpoint_freeze": "pending_real_data_scientific_review",
        "scientific_inference_authorized": False,
        "subjects": selected,
        "continuous_status": continuous_status,
        "visual_status": visual_status,
        "figures": outputs,
        "purpose": [
            "retain PIR-variability and discrete behavioral precursors without overcrowding Figure 4",
            "retain pre-Probe objective behavior across 10/20/30/60-s windows",
            "retain both Probe response-time measures",
            "retain stimulus identity×size visual-control detail",
        ],
    }
    path = out / "publication_supplementary_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    summary["summary_path"] = str(path)
    return summary
