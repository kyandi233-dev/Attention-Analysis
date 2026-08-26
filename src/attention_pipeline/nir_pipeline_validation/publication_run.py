from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from attention_pipeline.config import load_config

from .analysis import (
    block_pir_summary,
    load_cohort_tables,
    output_root,
    selected_subjects,
)
from .extended import (
    advanced_behavior_summary,
    attach_visual_covariates,
    load_visual_covariates,
    multidimensional_coverage,
    nogo_precursor_trajectory,
    raw_between_person_summary,
    source_mode_qc,
    subject_level_summary,
    track_robustness,
    visual_covariate_correlation_table,
)
from .landscape import (
    block_transition_recovery,
    build_event_catalogs,
    continuous_event_trajectory,
    feature_redundancy,
    global_pir_distribution,
    global_pir_trajectory,
    individual_heterogeneity,
    load_continuous_analysis_ready,
    probe_transition_table,
    stimulus_condition_summary,
    trial_condition_summary,
    window_effect_stability,
    within_between_correlation_tables,
)
from .probe_analysis import probe_event_table
from .publication_figures import write_publication_suite


PUBLICATION_SUITE_VERSION = "nir-publication-figure-suite-v1"
PUBLICATION_SUITE_SCHEMA = 1


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def run_publication_validation(
    config_path: str | Path,
    *,
    subjects: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build the complete manuscript-oriented NIR analysis/figure validation layer.

    This function never writes 10_analysis_ready or 11_analysis_tables. It may
    read 10_analysis_ready continuously for true event-aligned trajectories and
    raw between-person baselines, and otherwise consumes completed 11 tables.
    """
    config = load_config(config_path)
    selected = selected_subjects(config, subjects)
    if not selected:
        raise ValueError("No completed 11_analysis_tables subjects selected")
    tables = load_cohort_tables(config, selected)
    analysis = config.section("analysis")
    figures = config.section("figures")

    track = str(analysis.get("main_track", analysis.get("track", "binocular_primary")))
    trial_window = str(analysis.get("trial_display_window", "pre_5s"))
    probe_window = str(analysis.get("probe_model_window", "pre_20s"))
    precursor_window = str(analysis.get("precursor_trial_window", trial_window))
    n_preceding_go = int(analysis.get("no_go_preceding_correct_go_trials", 5))
    global_gap_sec = float(analysis.get("global_display_block_gap_sec", 60.0))
    global_bin_sec = float(analysis.get("global_summary_bin_sec", 10.0))
    transition_window_sec = float(analysis.get("block_transition_window_sec", 120.0))
    recovery_summary_sec = float(analysis.get("block_recovery_summary_sec", 60.0))
    event_start_sec = float(analysis.get("event_trajectory_start_sec", -60.0))
    event_bin_sec = float(analysis.get("event_trajectory_bin_sec", 1.0))
    event_end_sec = float(analysis.get("event_trajectory_end_sec", 2.0))
    probe_end_sec = float(analysis.get("probe_trajectory_end_sec", 0.0))
    max_go_reference = int(analysis.get("max_go_reference_events_per_subject_block", 60))

    formats = [str(value).lower() for value in figures.get("publication_formats", figures.get("formats", ["pdf", "svg", "png", "tiff"]))]
    raster_dpi = int(figures.get("raster_dpi", figures.get("dpi", 600)))

    out = output_root(config)
    table_dir = out / "tables" / "publication_analysis"
    figure_dir = out / "figures" / "publication"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Whole-experiment / Block structure
    # ------------------------------------------------------------------
    global_detail, global_summary = global_pir_trajectory(
        tables["time_on_task"],
        track=track,
        display_gap_sec=global_gap_sec,
        summary_bin_sec=global_bin_sec,
    )
    global_distribution = global_pir_distribution(tables["time_on_task"], track=track)
    block_summary = block_pir_summary(tables["time_on_task"], track)
    block_transition, recovery_summary = block_transition_recovery(
        tables["time_on_task"],
        track=track,
        transition_window_sec=transition_window_sec,
        recovery_summary_sec=recovery_summary_sec,
    )

    # ------------------------------------------------------------------
    # Trial conditions and higher-order behavior
    # ------------------------------------------------------------------
    trial_conditions = trial_condition_summary(
        tables["trial_level"],
        tables["trial_windows"],
        track=track,
        window_name=trial_window,
    )
    advanced_behavior = advanced_behavior_summary(tables["trial_level"])
    nogo_trial_lag = nogo_precursor_trajectory(
        tables["trial_level"],
        tables["trial_windows"],
        track=track,
        window_name=precursor_window,
        n_preceding_go=n_preceding_go,
    )

    # ------------------------------------------------------------------
    # True continuous event/probe trajectories from read-only 10 layer
    # ------------------------------------------------------------------
    continuous, continuous_status = load_continuous_analysis_ready(
        config, selected, track=track
    )
    catalogs = build_event_catalogs(
        tables["trial_level"],
        tables["probe_windows"],
        track=track,
        max_go_reference_per_subject_block=max_go_reference,
    )
    nogo_continuous = continuous_event_trajectory(
        continuous,
        catalogs["nogo"],
        start_sec=event_start_sec,
        end_sec=event_end_sec,
        bin_sec=event_bin_sec,
        extra_event_columns=("outcome", "commission"),
    )
    omission_continuous = continuous_event_trajectory(
        continuous,
        catalogs["omission"],
        start_sec=event_start_sec,
        end_sec=event_end_sec,
        bin_sec=event_bin_sec,
        extra_event_columns=("outcome", "omission_qc_type"),
    )
    probe_continuous = continuous_event_trajectory(
        continuous,
        catalogs["probe"],
        start_sec=event_start_sec,
        end_sec=probe_end_sec,
        bin_sec=event_bin_sec,
        extra_event_columns=("probe_response", "probe_vigilance"),
    )

    # ------------------------------------------------------------------
    # Probe state transitions
    # ------------------------------------------------------------------
    probe_events = probe_event_table(tables["probe_windows"], track=track)
    probe_transitions = probe_transition_table(
        tables["probe_windows"], track=track, window_name=probe_window
    )

    # ------------------------------------------------------------------
    # Visual/PLR controls
    # ------------------------------------------------------------------
    visual_table, visual_status = load_visual_covariates(config)
    visual_trial = attach_visual_covariates(
        tables["trial_level"],
        tables["trial_windows"],
        visual_table,
        track=track,
        window_name=trial_window,
    )
    visual_correlations = visual_covariate_correlation_table(visual_trial)
    stimulus_summary = stimulus_condition_summary(visual_trial)

    # ------------------------------------------------------------------
    # Feature structure, within/between decomposition, multiscale stability
    # ------------------------------------------------------------------
    feature_within, feature_between = feature_redundancy(
        tables["trial_windows"], track=track, window_name=trial_window
    )
    raw_pir, raw_status = raw_between_person_summary(config, selected)
    subject_summary = subject_level_summary(advanced_behavior, raw_pir)
    within_metrics, between_metrics = within_between_correlation_tables(
        tables["trial_level"],
        tables["trial_windows"],
        subject_summary,
        track=track,
        window_name=trial_window,
    )
    window_effects, window_stability = window_effect_stability(
        tables["trial_level"],
        tables["trial_windows"],
        tables["probe_windows"],
        track=track,
    )
    heterogeneity = individual_heterogeneity(
        tables["time_on_task"], recovery_summary, window_effects, track=track
    )

    # ------------------------------------------------------------------
    # QC / robustness
    # ------------------------------------------------------------------
    coverage = multidimensional_coverage(
        tables["trial_coverage"], tables["probe_coverage"]
    )
    source_mode = source_mode_qc(tables["trial_windows"])
    robustness_tracks = [
        str(value)
        for value in analysis.get(
            "robustness_tracks",
            [
                "binocular_primary",
                "left_primary",
                "right_primary",
                "binocular_strict",
                "left_strict",
                "right_strict",
            ],
        )
    ]
    track_correlations, track_agreement = track_robustness(
        tables["trial_windows"],
        window_name=trial_window,
        main_track=track,
        tracks=robustness_tracks,
    )

    model_path = out / "tables" / "model_smoke_results.csv"
    model_results = (
        pd.read_csv(model_path, encoding="utf-8-sig", low_memory=False)
        if model_path.is_file()
        else pd.DataFrame()
    )

    # ------------------------------------------------------------------
    # Persist complete publication-analysis tables
    # ------------------------------------------------------------------
    outputs = {
        "global_pir_trajectory_detail": global_detail,
        "global_pir_trajectory_summary": global_summary,
        "global_pir_distribution_subject_block": global_distribution,
        "block_transition_trajectory": block_transition,
        "block_recovery_subject_summary": recovery_summary,
        "trial_conditions": trial_conditions,
        "advanced_behavior_subject_block": advanced_behavior,
        "nogo_trial_lag_precursor": nogo_trial_lag,
        "nogo_continuous_event_trajectory": nogo_continuous,
        "omission_continuous_event_trajectory": omission_continuous,
        "probe_continuous_event_trajectory": probe_continuous,
        "probe_transitions": probe_transitions,
        "visual_trial_covariates": visual_trial,
        "stimulus_condition_summary": stimulus_summary,
        "visual_covariate_correlations": visual_correlations,
        "pir_feature_redundancy_within": feature_within,
        "pir_feature_redundancy_between": feature_between,
        "within_person_metric_correlations": within_metrics,
        "between_person_metric_correlations": between_metrics,
        "window_effects_subject": window_effects,
        "window_effect_stability": window_stability,
        "individual_heterogeneity": heterogeneity,
        "coverage_multidimensional": coverage,
        "source_mode_qc": source_mode,
        "track_correlations": track_correlations,
        "track_agreement": track_agreement,
        "raw_between_person_pir": raw_pir,
        "subject_level_summary": subject_summary,
    }
    table_paths: dict[str, str] = {}
    for name, frame in outputs.items():
        path = table_dir / f"{name}.csv"
        _write_csv(frame, path)
        table_paths[name] = str(path)

    publication_figures = write_publication_suite(
        output_dir=figure_dir,
        formats=formats,
        raster_dpi=raster_dpi,
        main_track=track,
        global_detail=global_detail,
        global_summary=global_summary,
        global_distribution=global_distribution,
        block_transition=block_transition,
        block_summary=block_summary,
        heterogeneity=heterogeneity,
        recovery_summary=recovery_summary,
        trial_conditions=trial_conditions,
        advanced_behavior=advanced_behavior,
        nogo_continuous=nogo_continuous,
        omission_continuous=omission_continuous,
        nogo_trial_lag=nogo_trial_lag,
        probe_events=probe_events,
        probe_continuous=probe_continuous,
        probe_transitions=probe_transitions,
        visual_trial=visual_trial,
        stimulus_summary=stimulus_summary,
        visual_correlations=visual_correlations,
        raw_pir=raw_pir,
        feature_within=feature_within,
        within_metrics=within_metrics,
        between_metrics=between_metrics,
        window_stability=window_stability,
        coverage=coverage,
        source_mode=source_mode,
        track_correlations=track_correlations,
        track_agreement=track_agreement,
        model_results=model_results,
    )

    summary = {
        "status": "complete",
        "suite_version": PUBLICATION_SUITE_VERSION,
        "schema_version": PUBLICATION_SUITE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_only": True,
        "nir_values_known_invalid": True,
        "subjects": selected,
        "n_subjects": len(selected),
        "main_track": track,
        "figure_system": {
            "publication_width_cm": 17,
            "panel_labels": "A/B/C/D",
            "font_family": "Arial with DejaVu Sans fallback",
            "vector_formats": [fmt for fmt in formats if fmt in {"pdf", "svg", "eps"}],
            "raster_formats": [fmt for fmt in formats if fmt in {"png", "tif", "tiff", "jpg", "jpeg"}],
            "raster_dpi": raster_dpi,
            "fixed_canvas": True,
            "legacy_diagnostic_figures_are_not_manuscript_figures": True,
        },
        "continuous_trajectory_policy": {
            "source": "read-only 10_analysis_ready",
            "track": track,
            "start_sec": event_start_sec,
            "end_sec": event_end_sec,
            "probe_end_sec": probe_end_sec,
            "bin_sec": event_bin_sec,
            "status": continuous_status,
        },
        "visual_covariates": visual_status,
        "raw_between_person": raw_status,
        "rows": {name: int(len(frame)) for name, frame in outputs.items()},
        "tables": table_paths,
        "figures": publication_figures,
        "scientific_boundaries": [
            "formal FocusWave condition is B; manuscript figures stratify Block, Go/No-Go, correctness/error, omission subtype, stimulus properties and Probe states rather than inventing A/B conditions",
            "current NIR/PIR values are known invalid and all displayed directions/effects are validation artifacts",
            "multiscale windows remain prespecified; apparent current effects cannot select a preferred window",
            "continuous trajectories read only 10_analysis_ready and never production",
            "primary/strict and eye tracks remain sensitivity analyses rather than significance-selected alternatives",
        ],
    }
    summary_path = out / "publication_suite_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path)
    return summary
