"""FocusWave formal SART behavior science-v3 runner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from attention_pipeline.config import load_config
from attention_pipeline.behavior_formal import extract as fex
from attention_pipeline.behavior_formal.candidate_validation import (
    build_candidate_validation,
    build_sensitivity_status,
    decompose_within_between,
)
from attention_pipeline.behavior_formal.science_v3 import (
    BehaviorScienceConfig,
    CANONICAL_METRICS,
    MODEL_FAILURE_COLUMNS,
    build_b1_b2_pairs,
    build_multiscale_tables,
    build_probe_windows,
    fit_q1_nominal,
    fit_q2_ordinal,
    participant_disjoint_folds,
    qc_denominators,
    validate_topology,
    write_chinese_result_summary,
)
from attention_pipeline.behavior_formal.science_v3_extensions import (
    build_error_trajectories,
    cluster_bootstrap_b1_b2,
    fit_block_cycle_gee,
    repeat_stability_boundary,
    summarize_error_trajectories,
)
from attention_pipeline.behavior_formal.science_v3_figures import generate_behavior_figures
from attention_pipeline.formal_analysis.behavior_adapter import (
    attach_behavior_groups,
    prepare_behavior_runtime_config,
)
from attention_pipeline.formal_analysis.provenance import resolve_git_checkout


def _cfg(config) -> BehaviorScienceConfig:
    behavior = config.section("behavior")
    stats = config.section("stats")
    return BehaviorScienceConfig(
        primary_probe_window_seconds=int(behavior.get("primary_probe_window_seconds", 30)),
        sensitivity_probe_windows_seconds=tuple(
            int(x) for x in behavior.get("probe_windows_seconds", [10, 20, 30])
        ),
        q1_reference_category=int(stats.get("q1_reference_category", 1)),
        min_model_rows=int(stats.get("minimum_model_rows", 24)),
        min_participant_groups=int(stats.get("minimum_participant_groups", 6)),
        rt_min_ms=float(behavior.get("rt_valid_min_ms", 100)),
        rt_max_ms=(
            None
            if behavior.get("rt_valid_max_ms") in (None, "")
            else float(behavior["rt_valid_max_ms"])
        ),
        sdt_min_go=int(stats.get("sdt_min_go_opportunities", 4)),
        sdt_min_nogo=int(stats.get("sdt_min_nogo_opportunities", 2)),
    )


def _expected(config) -> dict[str, int] | None:
    cohort = config.section("cohort")
    keys = {
        "sessions": cohort.get("expected_session_count"),
        "analysis_groups": cohort.get("expected_group_count"),
        "double_session_repeat_groups": cohort.get("expected_double_session_repeat_groups"),
    }
    return None if any(v is None for v in keys.values()) else {k: int(v) for k, v in keys.items()}


def _normalize_failures(*frames: pd.DataFrame) -> pd.DataFrame:
    nonempty = [f.copy() for f in frames if f is not None and not f.empty]
    if not nonempty:
        return pd.DataFrame(columns=MODEL_FAILURE_COLUMNS)
    failures = pd.concat(nonempty, ignore_index=True, sort=False)
    if "reason" not in failures:
        failures["reason"] = failures.get("failure_detail", pd.NA)
    for column in MODEL_FAILURE_COLUMNS:
        if column not in failures:
            failures[column] = pd.NA
    if "status" in failures:
        failures["status"] = failures["status"].fillna("not_estimable")
    return failures


def _annotate_scale(
    frame: pd.DataFrame,
    *,
    unit: str,
    rt_cv_min_n: int,
    rt_slope_min_n: int,
    is_probe: bool = False,
) -> pd.DataFrame:
    """Make observation units/sample denominators and RT estimability explicit."""
    out = frame.copy()
    if out.empty:
        return out
    out["unit"] = unit
    out["session_n"] = 1
    out["participant_group_n"] = 1
    out["probe_n"] = 1 if is_probe else 0
    n = pd.to_numeric(out.get("correct_go_rt_opportunities"), errors="coerce")
    out["rt_variability_valid_n"] = n
    out["rt_cv_min_n"] = int(rt_cv_min_n)
    out["rt_slope_min_n"] = int(rt_slope_min_n)
    out["rt_cv_status"] = np.where(n.ge(rt_cv_min_n), "estimable", "not_estimable_low_rt_n")
    out["rt_slope_status"] = np.where(n.ge(rt_slope_min_n), "estimable", "not_estimable_low_rt_n")
    if "go_correct_rt_cv" in out:
        out.loc[n.lt(rt_cv_min_n) | n.isna(), "go_correct_rt_cv"] = np.nan
    if "go_correct_rt_theilsen_slope_ms_per_s" in out:
        out.loc[n.lt(rt_slope_min_n) | n.isna(), "go_correct_rt_theilsen_slope_ms_per_s"] = np.nan
    out["rt_slope_fit_scope"] = "within current observation unit using valid correct-Go RT time positions"
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/behavior_formal_v2.yaml")
    parser.add_argument("--paths-config", default=None)
    parser.add_argument(
        "--force", action="store_true", help="replace the derived formal_v3 output directory"
    )
    args = parser.parse_args()

    config = load_config(args.config, paths_config=args.paths_config)
    runtime, cohort = prepare_behavior_runtime_config(config)
    output_root = config.path_value("output_root") / "formal_v3"
    if output_root.exists():
        if not args.force:
            raise FileExistsError(
                f"formal behavior output already exists: {output_root}; use --force to rebuild derived outputs"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)

    repo_root = Path(__file__).resolve().parents[1]
    code_provenance = resolve_git_checkout(repo_root, role="code", require_clean=True)

    subjects = fex.discover_subjects(runtime)
    trials = fex.load_cohort(runtime, subjects)
    trials = attach_behavior_groups(trials, cohort, require_all=True)
    validation = fex.validate_formal(runtime, trials)
    cfg = _cfg(config)

    tables = build_multiscale_tables(trials, cfg)
    primary_probe, probe_sensitivity = build_probe_windows(trials, cfg)
    behavior_cfg = config.section("behavior")
    rt_cv_min_n = int(behavior_cfg.get("rt_cv_min_n", 20))
    rt_slope_min_n = int(behavior_cfg.get("rt_slope_min_n", 5))
    tables["session"] = _annotate_scale(
        tables["session"], unit="session", rt_cv_min_n=rt_cv_min_n, rt_slope_min_n=rt_slope_min_n
    )
    tables["block"] = _annotate_scale(
        tables["block"], unit="block", rt_cv_min_n=rt_cv_min_n, rt_slope_min_n=rt_slope_min_n
    )
    tables["cycle"] = _annotate_scale(
        tables["cycle"], unit="cycle", rt_cv_min_n=rt_cv_min_n, rt_slope_min_n=rt_slope_min_n
    )
    primary_probe = _annotate_scale(
        primary_probe,
        unit="probe",
        rt_cv_min_n=rt_cv_min_n,
        rt_slope_min_n=rt_slope_min_n,
        is_probe=True,
    )
    probe_sensitivity = _annotate_scale(
        probe_sensitivity,
        unit="probe_window_sensitivity",
        rt_cv_min_n=rt_cv_min_n,
        rt_slope_min_n=rt_slope_min_n,
        is_probe=True,
    )

    pairs, pair_failures = build_b1_b2_pairs(tables["block"])
    topology = validate_topology(tables["session"], expected=_expected(config))

    stats_cfg = config.section("stats")
    b1b2_clustered, b1b2_failures = cluster_bootstrap_b1_b2(
        pairs,
        iterations=int(stats_cfg.get("bootstrap_iterations", 20000)),
        seed=int(stats_cfg.get("seed", 20260829)),
    )
    cycle_gee, cycle_failures = fit_block_cycle_gee(tables["cycle"])
    error_events = build_error_trajectories(trials)
    error_summary = summarize_error_trajectories(error_events)
    stability_boundary = repeat_stability_boundary(tables["session"])

    q1_results, q1_failures = fit_q1_nominal(primary_probe, cfg)
    q2_results, q2_failures = fit_q2_ordinal(primary_probe, cfg)
    failures = _normalize_failures(
        pair_failures,
        b1b2_failures,
        cycle_failures,
        q1_failures,
        q2_failures,
    )

    folds = (
        participant_disjoint_folds(primary_probe, int(stats_cfg.get("prediction_folds", 5)))
        if not primary_probe.empty
        else pd.DataFrame()
    )
    qc = qc_denominators(trials, primary_probe, tables)

    candidate_validation, metric_redundancy, endpoint_decisions = build_candidate_validation(
        tables, primary_probe
    )
    sensitivity_status = build_sensitivity_status(tables["session"])
    decomposition_metrics = [m for m in CANONICAL_METRICS if m in primary_probe.columns]
    probe_within_between = (
        decompose_within_between(primary_probe, decomposition_metrics)
        if not primary_probe.empty else primary_probe.copy()
    )
    session_decomposition_metrics = [m for m in CANONICAL_METRICS if m in tables["session"].columns]
    session_within_between = (
        decompose_within_between(tables["session"], session_decomposition_metrics)
        if not tables["session"].empty else tables["session"].copy()
    )

    trials.to_csv(output_root / "trial_metrics.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(output_root / "validation.csv", index=False, encoding="utf-8-sig")
    tables["session"].to_csv(output_root / "session_metrics.csv", index=False, encoding="utf-8-sig")
    tables["block"].to_csv(output_root / "block_metrics.csv", index=False, encoding="utf-8-sig")
    tables["cycle"].to_csv(output_root / "cycle_metrics.csv", index=False, encoding="utf-8-sig")
    primary_probe.to_csv(output_root / "probe_primary_30s.csv", index=False, encoding="utf-8-sig")
    probe_sensitivity.to_csv(
        output_root / "probe_window_sensitivity.csv", index=False, encoding="utf-8-sig"
    )
    probe_within_between.to_csv(
        output_root / "probe_primary_30s_within_between.csv", index=False, encoding="utf-8-sig"
    )
    session_within_between.to_csv(
        output_root / "session_metrics_within_between.csv", index=False, encoding="utf-8-sig"
    )
    candidate_validation.to_csv(
        output_root / "behavior_candidate_metric_validation.csv", index=False, encoding="utf-8-sig"
    )
    metric_redundancy.to_csv(
        output_root / "behavior_metric_redundancy.csv", index=False, encoding="utf-8-sig"
    )
    endpoint_decisions.to_csv(
        output_root / "behavior_endpoint_decisions.csv", index=False, encoding="utf-8-sig"
    )
    sensitivity_status.to_csv(
        output_root / "behavior_sensitivity_status.csv", index=False, encoding="utf-8-sig"
    )
    pairs.to_csv(output_root / "b1_b2_pairs.csv", index=False, encoding="utf-8-sig")
    b1b2_clustered.to_csv(
        output_root / "b1_b2_participant_cluster_bootstrap.csv", index=False, encoding="utf-8-sig"
    )
    cycle_gee.to_csv(output_root / "block_cycle_gee.csv", index=False, encoding="utf-8-sig")
    error_events.to_csv(output_root / "error_event_trajectories.csv", index=False, encoding="utf-8-sig")
    error_summary.to_csv(output_root / "error_trajectory_summary.csv", index=False, encoding="utf-8-sig")
    stability_boundary.to_csv(
        output_root / "repeat_session_stability_boundary.csv", index=False, encoding="utf-8-sig"
    )
    q1_results.to_csv(output_root / "q1_nominal_models.csv", index=False, encoding="utf-8-sig")
    q2_results.to_csv(output_root / "q2_ordinal_gee_models.csv", index=False, encoding="utf-8-sig")
    failures.to_csv(output_root / "model_failures.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(output_root / "participant_disjoint_folds.csv", index=False, encoding="utf-8-sig")
    qc.to_csv(output_root / "qc_denominators.csv", index=False, encoding="utf-8-sig")

    figure_files = generate_behavior_figures(
        tables["block"], primary_probe, output_root / "figures", error_summary=error_summary
    )
    write_chinese_result_summary(
        output_root / "结果说明.md", topology, q1_results, q2_results, failures
    )

    manifest = {
        "pipeline": "behavior-science-v3",
        "schema_version": 4,
        "config": str(config.path),
        "config_digest": config.digest,
        "runtime_provenance": {"code": code_provenance},
        "sessions": topology["sessions"],
        "analysis_groups": topology["analysis_groups"],
        "double_session_repeat_groups": topology["double_session_repeat_groups"],
        "primary_probe_window_seconds": cfg.primary_probe_window_seconds,
        "sensitivity_probe_windows_seconds": list(cfg.sensitivity_probe_windows_seconds),
        "probe_primary_rows": int(len(primary_probe)),
        "error_event_rows": int(len(error_events)),
        "model_failure_rows": int(len(failures)),
        "candidate_validation_rows": int(len(candidate_validation)),
        "endpoint_decision_rows": int(len(endpoint_decisions)),
        "figure_files": figure_files,
        "formal_inference_contract": "session pairing then repeat_participant_id clustering",
        "cycle_contract": "cycle nested block/session/repeat_participant_id; block_by_cycle GEE",
        "prediction_split_contract": "participant_disjoint",
        "go_nogo_error_contract": "separate_denominators",
        "q1_contract": "nominal_four_class_reference_1",
        "q2_contract": "ordinal_gee_repeat_participant_cluster",
        "repeat_reliability_contract": "descriptive_only_when_two_sessions",
        "candidate_selection_contract": "coverage/distribution/within-between/redundancy; no p-value screening; final freeze pending real-data review",
        "sensitivity_contract": "visit-order analyses fail closed until a verified order field exists; never infer order from session_id",
        "rt_cv_min_n": rt_cv_min_n,
        "rt_slope_min_n": rt_slope_min_n,
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
