from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

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
    omission_subject_summary,
    output_root,
    probe_analysis_table,
    selected_subjects,
    trial_analysis_table,
)
from .plots import (
    plot_block_pairs,
    plot_coverage_heatmap,
    plot_model_forest,
    plot_omission_subtypes,
    plot_pipeline_schematic,
    plot_probe_windows,
    plot_time_on_task,
    plot_trial_outcomes,
)
from .probe_analysis import (
    fit_probe_option_smoke_models,
    probe_event_table,
    probe_response_subject_summary,
    probe_response_vigilance_table,
    probe_response_window_table,
)
from .probe_plots import (
    plot_probe_response_behavior,
    plot_probe_response_distribution,
    plot_probe_response_pir_windows,
    plot_probe_response_vigilance,
)
from .extended import (
    advanced_behavior_summary,
    attach_visual_covariates,
    extension_readiness,
    load_visual_covariates,
    multidimensional_coverage,
    nogo_precursor_trajectory,
    probe_behavior_multiscale,
    probe_dynamic_feature_long,
    probe_rt_summary,
    questionnaire_correlations,
    raw_between_person_summary,
    source_mode_qc,
    subject_level_summary,
    track_robustness,
    trial_dynamic_feature_long,
    trial_multiscale_trajectory,
    visual_covariate_correlation_table,
)
from .extended_models import fit_extended_smoke_models
from .extended_plots import (
    plot_advanced_behavior_block,
    plot_coverage_multidimensional,
    plot_dynamic_feature_matrix,
    plot_nogo_precursor_pir,
    plot_nogo_precursor_rt,
    plot_probe_behavior_multiscale,
    plot_probe_rt_by_response,
    plot_raw_between_person,
    plot_source_mode_qc,
    plot_track_robustness,
    plot_trial_multiscale_trajectory,
    plot_visual_covariate_association,
)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


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

    track = str(analysis.get("main_track", analysis.get("track", "binocular_primary")))
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
    trial_window = str(analysis.get("trial_display_window", "pre_5s"))
    probe_model_window = str(analysis.get("probe_model_window", "pre_20s"))
    precursor_window = str(analysis.get("precursor_trial_window", trial_window))
    n_preceding_go = int(analysis.get("no_go_preceding_correct_go_trials", 5))
    trial_state_windows = [str(v) for v in analysis.get("trial_state_windows", [])]
    trial_event_windows = [str(v) for v in analysis.get("trial_event_windows", [])]
    all_trial_windows = list(dict.fromkeys([*trial_state_windows, *trial_event_windows]))
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

    # ------------------------------------------------------------------
    # Core validation layer retained from v1.0/v1.1
    # ------------------------------------------------------------------
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
    omission_summary = omission_subject_summary(trial_table)
    probe_table = probe_analysis_table(
        tables["probe_windows"],
        track=track,
        window_name=probe_model_window,
    )

    probe_events = probe_event_table(tables["probe_windows"], track=track)
    probe_response_summary = probe_response_subject_summary(probe_events)
    probe_response_windows = probe_response_window_table(
        tables["probe_windows"], track=track
    )
    probe_response_vigilance = probe_response_vigilance_table(probe_events)

    # ------------------------------------------------------------------
    # Extended comprehensive validation layer
    # ------------------------------------------------------------------
    advanced_behavior = advanced_behavior_summary(tables["trial_level"])
    trial_dynamic = trial_dynamic_feature_long(
        tables["trial_level"],
        tables["trial_windows"],
        tracks=robustness_tracks,
        window_names=all_trial_windows or None,
    )
    probe_dynamic = probe_dynamic_feature_long(
        tables["probe_windows"],
        tracks=robustness_tracks,
    )
    trial_trajectory = trial_multiscale_trajectory(
        tables["trial_level"],
        tables["trial_windows"],
        track=track,
        feature="pupil_median",
    )
    nogo_precursor = nogo_precursor_trajectory(
        tables["trial_level"],
        tables["trial_windows"],
        track=track,
        window_name=precursor_window,
        n_preceding_go=n_preceding_go,
    )
    probe_rt = probe_rt_summary(tables["probe_windows"], track=track)
    probe_behavior_windows = probe_behavior_multiscale(
        tables["probe_windows"], track=track
    )
    track_correlations, track_agreement = track_robustness(
        tables["trial_windows"],
        window_name=trial_window,
        main_track=track,
        tracks=robustness_tracks,
    )
    source_mode = source_mode_qc(tables["trial_windows"])
    coverage_multi = multidimensional_coverage(
        tables["trial_coverage"], tables["probe_coverage"]
    )

    visual_table, visual_status = load_visual_covariates(config)
    visual_trial = attach_visual_covariates(
        tables["trial_level"],
        tables["trial_windows"],
        visual_table,
        track=track,
        window_name=trial_window,
    )
    visual_correlations = visual_covariate_correlation_table(visual_trial)

    raw_pir, raw_pir_status = raw_between_person_summary(config, selected)
    subject_summary = subject_level_summary(advanced_behavior, raw_pir)
    questionnaire_results, questionnaire_status = questionnaire_correlations(
        config, subject_summary
    )
    readiness = extension_readiness(config, selected)
    readiness["visual_covariate_loader"] = visual_status
    readiness["raw_between_person"] = raw_pir_status
    readiness["questionnaire_loader"] = questionnaire_status

    # ------------------------------------------------------------------
    # Model smoke tests: existing + extended
    # ------------------------------------------------------------------
    model_results, model_status = fit_smoke_models(
        trial_table,
        probe_table,
        tables["time_on_task"],
        track=track,
        min_subjects=min_subjects,
    )
    probe_model_results, probe_model_status = fit_probe_option_smoke_models(
        tables["probe_windows"],
        track=track,
        window_name=probe_model_window,
        min_subjects=min_subjects,
    )
    extended_model_results, extended_model_status = fit_extended_smoke_models(
        tables["trial_level"],
        tables["trial_windows"],
        probe_rt,
        nogo_precursor,
        visual_trial,
        track=track,
        window_name=trial_window,
        min_subjects=min_subjects,
    )
    model_parts = [frame for frame in (model_results, probe_model_results, extended_model_results) if not frame.empty]
    model_results = (
        pd.concat(model_parts, ignore_index=True)
        if model_parts
        else pd.DataFrame(columns=["model", "term", "estimate", "se", "p_value"])
    )
    model_status = [*model_status, *probe_model_status, *extended_model_status]

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    table_outputs = {
        "behavior_subject_summary": table_dir / "behavior_subject_summary.csv",
        "behavior_advanced_subject_block": table_dir / "behavior_advanced_subject_block.csv",
        "omission_qc_subject_summary": table_dir / "omission_qc_subject_summary.csv",
        "block_pir_summary": table_dir / "block_pir_summary.csv",
        "time_on_task_coarse": table_dir / f"time_on_task_{coarse_bin_sec}s.csv",
        "trial_analysis": table_dir / f"trial_analysis_{trial_window}.csv",
        "trial_dynamic_features": table_dir / "trial_dynamic_features_long.csv",
        "trial_multiscale_trajectory": table_dir / "trial_multiscale_trajectory.csv",
        "nogo_precursor_trajectory": table_dir / "nogo_precursor_trajectory.csv",
        "probe_analysis": table_dir / f"probe_analysis_{probe_model_window}.csv",
        "probe_event_table": table_dir / "probe_event_table.csv",
        "probe_response_subject_summary": table_dir / "probe_response_subject_summary.csv",
        "probe_response_window_summary": table_dir / "probe_response_window_summary.csv",
        "probe_response_vigilance_joint": table_dir / "probe_response_vigilance_joint.csv",
        "probe_dynamic_features": table_dir / "probe_dynamic_features_long.csv",
        "probe_rt": table_dir / "probe_rt.csv",
        "probe_behavior_multiscale": table_dir / "probe_behavior_multiscale.csv",
        "track_correlations": table_dir / "track_correlations.csv",
        "track_agreement": table_dir / "track_agreement.csv",
        "source_mode_qc": table_dir / "source_mode_qc.csv",
        "coverage_multidimensional": table_dir / "coverage_multidimensional.csv",
        "trial_visual_covariates": table_dir / "trial_visual_covariates.csv",
        "visual_covariate_correlations": table_dir / "visual_covariate_correlations.csv",
        "raw_between_person_pir": table_dir / "raw_between_person_pir.csv",
        "subject_level_summary": table_dir / "subject_level_summary.csv",
        "questionnaire_correlations": table_dir / "questionnaire_correlations.csv",
        "model_smoke_results": table_dir / "model_smoke_results.csv",
    }
    frames_to_write = {
        "behavior_subject_summary": behavior_summary,
        "behavior_advanced_subject_block": advanced_behavior,
        "omission_qc_subject_summary": omission_summary,
        "block_pir_summary": block_summary,
        "time_on_task_coarse": coarse,
        "trial_analysis": trial_table,
        "trial_dynamic_features": trial_dynamic,
        "trial_multiscale_trajectory": trial_trajectory,
        "nogo_precursor_trajectory": nogo_precursor,
        "probe_analysis": probe_table,
        "probe_event_table": probe_events,
        "probe_response_subject_summary": probe_response_summary,
        "probe_response_window_summary": probe_response_windows,
        "probe_response_vigilance_joint": probe_response_vigilance,
        "probe_dynamic_features": probe_dynamic,
        "probe_rt": probe_rt,
        "probe_behavior_multiscale": probe_behavior_windows,
        "track_correlations": track_correlations,
        "track_agreement": track_agreement,
        "source_mode_qc": source_mode,
        "coverage_multidimensional": coverage_multi,
        "trial_visual_covariates": visual_trial,
        "visual_covariate_correlations": visual_correlations,
        "raw_between_person_pir": raw_pir,
        "subject_level_summary": subject_summary,
        "questionnaire_correlations": questionnaire_results,
        "model_smoke_results": model_results,
    }
    for key, frame in frames_to_write.items():
        _write_csv(frame, table_outputs[key])

    (out / "extension_readiness.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Code-generated professional figures
    # ------------------------------------------------------------------
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
    figure_outputs["omission_subtypes"] = plot_omission_subtypes(
        trial_table,
        base=figure_dir / f"fig03b_omission_qc_subtypes_{trial_window}",
        formats=formats,
        dpi=dpi,
    )
    figure_outputs["probe_vigilance_windows"] = plot_probe_windows(
        tables["probe_windows"],
        track=track,
        base=figure_dir / "fig04_probe_vigilance_windows",
        formats=formats,
        dpi=dpi,
    )
    figure_outputs["probe_response_distribution"] = plot_probe_response_distribution(
        probe_events,
        base=figure_dir / "fig04a_probe_response_distribution",
        formats=formats,
        dpi=dpi,
    )
    figure_outputs["probe_response_pir_windows"] = plot_probe_response_pir_windows(
        probe_response_windows,
        base=figure_dir / "fig04b_probe_response_pir_windows",
        formats=formats,
        dpi=dpi,
    )
    figure_outputs["probe_response_behavior"] = plot_probe_response_behavior(
        probe_response_windows,
        window_name=probe_model_window,
        base=figure_dir / f"fig04c_probe_response_preprobe_behavior_{probe_model_window}",
        formats=formats,
        dpi=dpi,
    )
    figure_outputs["probe_response_vigilance_joint"] = plot_probe_response_vigilance(
        probe_events,
        base=figure_dir / "fig04d_probe_response_vigilance_joint",
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

    figure_outputs["dynamic_feature_matrix"] = plot_dynamic_feature_matrix(
        trial_dynamic,
        track=track,
        base=figure_dir / "fig07_dynamic_pir_feature_matrix",
        formats=formats,
        dpi=dpi,
    )
    figure_outputs["trial_multiscale_trajectory"] = plot_trial_multiscale_trajectory(
        trial_trajectory,
        feature="pupil_median",
        base=figure_dir / "fig08_trial_multiscale_pir_trajectory",
        formats=formats,
        dpi=dpi,
    )
    figure_outputs["nogo_precursor_rt"] = plot_nogo_precursor_rt(
        nogo_precursor,
        base=figure_dir / "fig09a_nogo_precursor_rt",
        formats=formats,
        dpi=dpi,
    )
    figure_outputs["nogo_precursor_pir"] = plot_nogo_precursor_pir(
        nogo_precursor,
        base=figure_dir / "fig09b_nogo_precursor_pir",
        formats=formats,
        dpi=dpi,
    )
    figure_outputs["probe_response_rt"] = plot_probe_rt_by_response(
        probe_rt,
        value_col="probe_rt",
        base=figure_dir / "fig10a_probe_response_rt",
        formats=formats,
        dpi=dpi,
    )
    figure_outputs["probe_vigilance_rt"] = plot_probe_rt_by_response(
        probe_rt,
        value_col="probe_vigilance_rt",
        base=figure_dir / "fig10b_probe_vigilance_rt_by_response",
        formats=formats,
        dpi=dpi,
    )
    figure_outputs["probe_behavior_rt_cv"] = plot_probe_behavior_multiscale(
        probe_behavior_windows,
        metric="go_rt_cv",
        base=figure_dir / "fig10c_probe_prebehavior_rt_cv_multiscale",
        formats=formats,
        dpi=dpi,
    )
    figure_outputs["probe_behavior_ambiguous_omission"] = plot_probe_behavior_multiscale(
        probe_behavior_windows,
        metric="n_ambiguous_omission",
        base=figure_dir / "fig10d_probe_prebehavior_ambiguous_omission_multiscale",
        formats=formats,
        dpi=dpi,
    )
    figure_outputs["advanced_behavior_block"] = plot_advanced_behavior_block(
        advanced_behavior,
        base=figure_dir / "fig11_advanced_behavior_block_profile",
        formats=formats,
        dpi=dpi,
    )
    figure_outputs["track_robustness"] = plot_track_robustness(
        track_correlations,
        base=figure_dir / f"fig12_track_robustness_{trial_window}",
        formats=formats,
        dpi=dpi,
    )
    figure_outputs["source_mode_qc"] = plot_source_mode_qc(
        source_mode[source_mode["track"].astype(str).eq(track)].copy() if not source_mode.empty and "track" in source_mode.columns else source_mode,
        window_name=trial_window,
        base=figure_dir / f"fig13_source_mode_qc_{trial_window}",
        formats=formats,
        dpi=dpi,
    )
    figure_outputs["coverage_multidimensional_trial"] = plot_coverage_multidimensional(
        coverage_multi,
        analysis_level="trial",
        track=track,
        window_name=trial_window,
        base=figure_dir / f"fig14a_coverage_multidimensional_trial_{trial_window}",
        formats=formats,
        dpi=dpi,
    )
    figure_outputs["coverage_multidimensional_probe"] = plot_coverage_multidimensional(
        coverage_multi,
        analysis_level="probe",
        track=track,
        window_name=probe_model_window,
        base=figure_dir / f"fig14b_coverage_multidimensional_probe_{probe_model_window}",
        formats=formats,
        dpi=dpi,
    )
    figure_outputs["visual_covariate_pir"] = plot_visual_covariate_association(
        visual_trial,
        x_col="current_central_rel_lum_mean",
        base=figure_dir / "fig15_visual_luminance_pir",
        formats=formats,
        dpi=dpi,
    )
    figure_outputs["raw_between_person_pir"] = plot_raw_between_person(
        raw_pir,
        base=figure_dir / "fig16_raw_between_person_pir",
        formats=formats,
        dpi=dpi,
    )

    summary = {
        "status": "complete",
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "validation_suite_version": 2,
        "validation_suite_schema": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_only": True,
        "nir_values_known_invalid": True,
        "validation_label": VALIDATION_LABEL,
        "subjects": selected,
        "n_subjects": len(selected),
        "main_track": track,
        "robustness_tracks": robustness_tracks,
        "trial_display_window": trial_window,
        "probe_model_window": probe_model_window,
        "precursor_trial_window": precursor_window,
        "time_on_task_display_bin_sec": coarse_bin_sec,
        "analysis_logic": {
            "within_person_state": "centered PIR / dynamic PIR features across time, trial and probe windows",
            "between_person_characteristics": "raw PIR subject baselines from read-only 10_analysis_ready; do not use mean-centered PIR as the primary between-person measure",
            "behavior": "program scoring + QC-aware omission sensitivity + motor-timing phenotype + higher-order SART metrics",
            "probe": "raw probe_response category + probe_vigilance + both RTs + pre-probe objective behavior + pre-probe PIR",
            "robustness": "primary/strict and binocular/left/right tracks are prespecified sensitivity tracks, never significance-selected alternatives",
            "confounds": "current/previous stimulus visual properties enter as optional covariates when the versioned visual table is available",
            "qc": "source-mode composition and multidimensional coverage are descriptive measurement-quality layers, not outcome-driven hard gates",
        },
        "behavior_policy": {
            "program_scoring_preserved": True,
            "omission_qc_subtypes_preserved": True,
            "ambiguous_omission_not_silently_recoded": True,
            "anticipatory_candidate_kept_as_separate_motor_timing_phenotype": True,
            "rt_cv_exgaussian_and_sdt_interfaces_present": True,
            "nogo_precursor_trajectory_present": True,
        },
        "probe_policy": {
            "probe_response_preserved_as_raw_categorical_code": True,
            "probe_response_not_semantically_relabelled": True,
            "probe_vigilance_preserved_as_separate_probe_dimension": True,
            "probe_response_and_vigilance_joint_structure_tested": True,
            "probe_response_rt_and_vigilance_rt_preserved": True,
            "objective_behavior_multiscale_windows_preserved": True,
            "all_configured_preprobe_windows_compared_descriptively": True,
        },
        "nir_feature_policy": {
            "level": True,
            "variability": True,
            "trend": True,
            "short_term_instability": True,
            "multiscale_trial_trajectory": True,
            "primary_strict_eye_robustness": True,
            "raw_between_person_baseline": True,
        },
        "extension_readiness": readiness,
        "rows": {key: int(len(frame)) for key, frame in frames_to_write.items()},
        "model_status": model_status,
        "tables": {key: str(path) for key, path in table_outputs.items()},
        "figures": figure_outputs,
        "interpretation_prohibited": [
            "PIR effect direction",
            "PIR p-values",
            "Block PIR scientific differences",
            "PIR-behavior associations",
            "PIR-probe associations",
            "dynamic PIR feature scientific effects",
            "visual-covariate effect interpretation",
            "semantic interpretation of raw probe_response codes before task-source verification",
            "questionnaire-NIR scientific associations before questionnaire data contract is frozen",
            "window selection based on apparent effects",
            "QC threshold selection based on significance",
        ],
    }
    (out / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return summary
