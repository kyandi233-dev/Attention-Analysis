"""FocusWave formal SART behavior science-v3 runner.

The governed cohort remains authoritative. Optional scientific layers can be
selected with ``--only-steps`` or omitted with ``--skip-steps``; required
structural/QC steps always run. Every step is timed and recorded. ``--subjects``
creates an isolated smoke/subset output and never overwrites the full-cohort
formal output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

from attention_pipeline.config import load_config
from attention_pipeline.behavior_formal import extract as fex
from attention_pipeline.behavior_formal.behavior_error_taxonomy import (
    FORMAL_OMISSION_ENDPOINT_METRICS,
    add_omission_taxonomy,
    build_taxonomy_validation,
    enrich_multiscale_taxonomy,
    enrich_probe_taxonomy,
)
from attention_pipeline.behavior_formal.candidate_validation import (
    FORMAL_BEHAVIOR_ENDPOINT_METRICS,
    build_candidate_validation,
    decompose_within_between,
)
from attention_pipeline.behavior_formal.omission_candidate_validation import (
    validate_omission_candidates,
)
from attention_pipeline.behavior_formal.omission_endpoints import (
    build_omission_b1_b2_pairs,
    formal_omission_partition_audit,
)
from attention_pipeline.behavior_formal.science_v3 import (
    BehaviorScienceConfig,
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
from attention_pipeline.behavior_formal.visit_sensitivity import (
    build_visit_sensitivity_status,
    fit_b1_b2_visit_order_sensitivity,
    subset_verified_first_visit,
)
from attention_pipeline.formal_analysis.behavior_adapter import (
    attach_behavior_groups,
    prepare_behavior_runtime_config,
)
from attention_pipeline.formal_analysis.cohort import canonical_session_id
from attention_pipeline.formal_analysis.execution_control import (
    ExecutionLedger,
    resolve_optional_steps,
)
from attention_pipeline.formal_analysis.identity_questionnaire import (
    attach_identity_metadata,
    build_identity_audit,
    build_questionnaire_variable_audit,
    left_join_questionnaire,
    load_questionnaire_data,
    load_repeat_registry,
    validate_questionnaire_registry_consistency,
)
from attention_pipeline.formal_analysis.provenance import resolve_git_checkout
from attention_pipeline.nir_behavior.behavior_qc import add_behavior_qc


OPTIONAL_STEPS = (
    "core_inference",
    "probe_models",
    "visit_sensitivity",
    "candidate_validation",
    "prediction_folds",
    "first_visit_sensitivity",
    "figures",
)

STRUCTURAL_STEPS = (
    "setup",
    "load_trials",
    "build_tables",
    "identity_questionnaire",
    "topology_qc",
    "write_outputs",
    "write_manifest",
)

CYCLE_INFERENCE_METRICS = (
    "go_correct_rt_median_ms",
    "go_correct_rt_cv",
    "omission_rate",
    "raw_go_omission_rate",
    "clean_go_omission_rate",
    "timing_ambiguous_go_omission_rate",
    "commission_rate",
    "dprime_loglinear",
)


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
    }
    return None if any(v is None for v in keys.values()) else {k: int(v) for k, v in keys.items()}


def _observed_topology(session_metrics: pd.DataFrame) -> dict[str, Any]:
    pairs = session_metrics[["repeat_participant_id", "session_id"]].drop_duplicates()
    sizes = pairs.groupby("repeat_participant_id")["session_id"].nunique().astype(int)
    distribution = {str(size): int(sizes.eq(size).sum()) for size in sorted(sizes.unique())}
    return {
        "sessions": int(session_metrics["session_id"].astype(str).nunique()),
        "analysis_groups": int(pairs["repeat_participant_id"].astype(str).nunique()),
        "repeated_participant_groups": int(sizes.gt(1).sum()),
        "max_sessions_per_participant": int(sizes.max()) if len(sizes) else 0,
        "group_size_distribution": distribution,
    }


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attach_identity_to_tables(tables: dict[str, pd.DataFrame], cohort: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        scale: attach_identity_metadata(table, cohort) if table is not None and not table.empty else table
        for scale, table in tables.items()
    }


def _build_all_b1_b2_pairs(block_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical_pairs, canonical_fail = build_b1_b2_pairs(block_metrics)
    omission_pairs, omission_fail = build_omission_b1_b2_pairs(block_metrics)
    pair_frames = [x for x in (canonical_pairs, omission_pairs) if x is not None and not x.empty]
    pairs = pd.concat(pair_frames, ignore_index=True, sort=False) if pair_frames else pd.DataFrame()
    failures = _normalize_failures(canonical_fail, omission_fail)
    return pairs, failures


def _partition_audits(tables: dict[str, pd.DataFrame], probe: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for scale, frame in {**tables, "probe": probe}.items():
        if frame is None or frame.empty:
            continue
        audit = formal_omission_partition_audit(frame)
        audit.insert(0, "scale", scale)
        rows.append(audit)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def _parse_subjects(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    requested = [canonical_session_id(x) for x in raw.split(",") if x.strip()]
    return list(dict.fromkeys(requested))


def _run_first_visit_bundle(
    *,
    prefix: str,
    tables: dict[str, pd.DataFrame],
    probe: pd.DataFrame,
    cfg: BehaviorScienceConfig,
    stats_cfg: dict,
    output_root: Path,
    same_stage: bool,
) -> tuple[dict[str, int], pd.DataFrame]:
    selected = {
        scale: subset_verified_first_visit(table, same_stage=same_stage)
        for scale, table in tables.items()
    }
    selected_probe = subset_verified_first_visit(probe, same_stage=same_stage)
    for scale, table in selected.items():
        table.to_csv(output_root / f"{prefix}_{scale}_metrics.csv", index=False, encoding="utf-8-sig")
    selected_probe.to_csv(output_root / f"{prefix}_probe_primary_30s.csv", index=False, encoding="utf-8-sig")

    pairs, pair_fail = _build_all_b1_b2_pairs(selected.get("block", pd.DataFrame())) if not selected.get("block", pd.DataFrame()).empty else (pd.DataFrame(), pd.DataFrame())
    clustered, cluster_fail = cluster_bootstrap_b1_b2(
        pairs,
        iterations=int(stats_cfg.get("bootstrap_iterations", 20000)),
        seed=int(stats_cfg.get("seed", 20260829)),
    ) if not pairs.empty else (pd.DataFrame(), pd.DataFrame())
    cycle_result, cycle_fail = fit_block_cycle_gee(
        selected.get("cycle", pd.DataFrame()), metrics=CYCLE_INFERENCE_METRICS
    )
    q1, q1_fail = fit_q1_nominal(selected_probe, cfg) if not selected_probe.empty else (pd.DataFrame(), pd.DataFrame())
    q2, q2_fail = fit_q2_ordinal(selected_probe, cfg) if not selected_probe.empty else (pd.DataFrame(), pd.DataFrame())

    pairs.to_csv(output_root / f"{prefix}_b1_b2_pairs.csv", index=False, encoding="utf-8-sig")
    clustered.to_csv(output_root / f"{prefix}_b1_b2_participant_cluster_bootstrap.csv", index=False, encoding="utf-8-sig")
    cycle_result.to_csv(output_root / f"{prefix}_block_cycle_gee.csv", index=False, encoding="utf-8-sig")
    q1.to_csv(output_root / f"{prefix}_q1_nominal_models.csv", index=False, encoding="utf-8-sig")
    q2.to_csv(output_root / f"{prefix}_q2_ordinal_gee_models.csv", index=False, encoding="utf-8-sig")
    failures = _normalize_failures(pair_fail, cluster_fail, cycle_fail, q1_fail, q2_fail)
    failures.to_csv(output_root / f"{prefix}_model_failures.csv", index=False, encoding="utf-8-sig")
    return {
        "session_rows": int(len(selected.get("session", pd.DataFrame()))),
        "probe_rows": int(len(selected_probe)),
        "model_failure_rows": int(len(failures)),
    }, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/behavior_formal_v2.yaml")
    parser.add_argument("--paths-config", default=None)
    parser.add_argument("--subjects", help="Optional comma-separated governed sessions for isolated smoke/subset run.")
    parser.add_argument("--only-steps", help="Comma-separated optional analyses to run; structural steps always run.")
    parser.add_argument("--skip-steps", help="Comma-separated optional analyses to omit; structural steps always run.")
    parser.add_argument("--list-steps", action="store_true", help="Print structural/optional step names and exit.")
    parser.add_argument(
        "--force", action="store_true", help="replace the exact derived output directory for this run scope"
    )
    args = parser.parse_args()

    if args.list_steps:
        print(json.dumps({
            "structural_steps": STRUCTURAL_STEPS,
            "optional_steps": OPTIONAL_STEPS,
            "contract": "skipped optional analyses are explicit partial_run; structural/QC steps cannot be skipped",
        }, ensure_ascii=False, indent=2))
        return 0

    selected_optional = resolve_optional_steps(
        OPTIONAL_STEPS, only=args.only_steps, skip=args.skip_steps
    )
    requested_subjects = _parse_subjects(args.subjects)
    ledger = ExecutionLedger(pipeline="behavior-science-v3")
    output_root: Path | None = None

    try:
        def _setup():
            config = load_config(args.config, paths_config=args.paths_config)
            runtime, cohort = prepare_behavior_runtime_config(config)
            cohort_cfg = config.section("cohort")
            registry_key = str(cohort_cfg.get("identity_registry_path_key", "repeat_registry"))
            questionnaire_key = str(cohort_cfg.get("questionnaire_path_key", "questionnaire_derived_data"))
            registry = load_repeat_registry(config, path_key=registry_key)
            questionnaire = load_questionnaire_data(config, path_key=questionnaire_key)
            identity_consistency = validate_questionnaire_registry_consistency(questionnaire, registry)
            return config, runtime, cohort, cohort_cfg, registry_key, questionnaire_key, questionnaire, identity_consistency

        setup = ledger.run("setup", _setup, required=True)
        assert setup is not None
        config, runtime, cohort, cohort_cfg, registry_key, questionnaire_key, questionnaire, identity_consistency = setup

        base_output = config.path_value("output_root")
        if requested_subjects:
            label = "__".join(x.replace("sub-", "s") for x in requested_subjects)
            output_root = base_output / "formal_v3_smoke" / label
            run_scope = "governed_subset_smoke"
        else:
            output_root = base_output / "formal_v3"
            run_scope = "current_analysis_cohort"
        if output_root.exists():
            if not args.force:
                raise FileExistsError(
                    f"formal behavior output already exists: {output_root}; use --force to rebuild this derived scope"
                )
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=False)

        repo_root = Path(__file__).resolve().parents[1]
        code_provenance = resolve_git_checkout(repo_root, role="code", require_clean=True)

        def _load_trials():
            discovered = fex.discover_subjects(runtime)
            if requested_subjects:
                missing = sorted(set(requested_subjects) - set(discovered))
                if missing:
                    raise ValueError(f"requested smoke sessions are not complete governed Behavior sessions: {missing}")
                subjects = requested_subjects
            else:
                subjects = discovered
            trials = fex.load_cohort(runtime, subjects)
            trials = attach_behavior_groups(trials, cohort, require_all=True)
            behavior_cfg = config.section("behavior")
            trials = add_behavior_qc(
                trials,
                carryover_ms=float(behavior_cfg.get("carryover_candidate_ms", 200.0)),
            )
            trials = add_omission_taxonomy(trials)
            validation = fex.validate_formal(runtime, trials)
            return subjects, trials, validation, behavior_cfg

        loaded = ledger.run("load_trials", _load_trials, required=True)
        assert loaded is not None
        subjects, trials, validation, behavior_cfg = loaded
        cfg = _cfg(config)

        def _build_tables():
            tables = build_multiscale_tables(trials, cfg)
            primary_probe, probe_sensitivity = build_probe_windows(trials, cfg)
            tables = enrich_multiscale_taxonomy(trials, tables)
            primary_probe = enrich_probe_taxonomy(trials, primary_probe)
            probe_sensitivity = enrich_probe_taxonomy(trials, probe_sensitivity)
            tables = _attach_identity_to_tables(tables, cohort)
            primary_probe = attach_identity_metadata(primary_probe, cohort) if not primary_probe.empty else primary_probe
            probe_sensitivity = attach_identity_metadata(probe_sensitivity, cohort) if not probe_sensitivity.empty else probe_sensitivity

            rt_cv_min_n = int(behavior_cfg.get("rt_cv_min_n", 20))
            rt_slope_min_n = int(behavior_cfg.get("rt_slope_min_n", 5))
            tables["session"] = _annotate_scale(tables["session"], unit="session", rt_cv_min_n=rt_cv_min_n, rt_slope_min_n=rt_slope_min_n)
            tables["block"] = _annotate_scale(tables["block"], unit="block", rt_cv_min_n=rt_cv_min_n, rt_slope_min_n=rt_slope_min_n)
            tables["cycle"] = _annotate_scale(tables["cycle"], unit="cycle", rt_cv_min_n=rt_cv_min_n, rt_slope_min_n=rt_slope_min_n)
            primary_probe = _annotate_scale(primary_probe, unit="probe", rt_cv_min_n=rt_cv_min_n, rt_slope_min_n=rt_slope_min_n, is_probe=True)
            probe_sensitivity = _annotate_scale(probe_sensitivity, unit="probe_window_sensitivity", rt_cv_min_n=rt_cv_min_n, rt_slope_min_n=rt_slope_min_n, is_probe=True)
            return tables, primary_probe, probe_sensitivity, rt_cv_min_n, rt_slope_min_n

        built = ledger.run("build_tables", _build_tables, required=True)
        assert built is not None
        tables, primary_probe, probe_sensitivity, rt_cv_min_n, rt_slope_min_n = built

        def _identity_questionnaire():
            identity_audit = build_identity_audit(cohort, questionnaire)
            variable_audit = build_questionnaire_variable_audit(questionnaire)
            session_questionnaire = left_join_questionnaire(tables["session"], questionnaire)
            return identity_audit, variable_audit, session_questionnaire

        identity = ledger.run("identity_questionnaire", _identity_questionnaire, required=True)
        assert identity is not None
        identity_audit, variable_audit, session_questionnaire = identity

        def _topology_qc():
            if requested_subjects:
                topology = _observed_topology(tables["session"])
            else:
                topology = validate_topology(tables["session"], expected=_expected(config))
            qc = qc_denominators(trials, primary_probe, tables)
            partition = _partition_audits(tables, primary_probe)
            if not partition.empty and (
                partition["status"].eq("complete").any()
                and not partition.loc[partition["status"].eq("complete"), "partition_exact_within_1e_12"].all()
            ):
                raise RuntimeError("formal omission partition identity failed: raw != clean + timing_ambiguous")
            return topology, qc, partition

        topology_qc = ledger.run("topology_qc", _topology_qc, required=True)
        assert topology_qc is not None
        topology, qc, omission_partition_audit = topology_qc

        pairs, pair_failures = _build_all_b1_b2_pairs(tables["block"])
        stats_cfg = config.section("stats")

        b1b2_clustered = pd.DataFrame()
        cycle_gee = pd.DataFrame()
        error_events = pd.DataFrame()
        error_summary = pd.DataFrame()
        stability_boundary = pd.DataFrame()
        core_failures = pair_failures.copy()

        def _core_inference():
            clustered, cluster_failures = cluster_bootstrap_b1_b2(
                pairs,
                iterations=int(stats_cfg.get("bootstrap_iterations", 20000)),
                seed=int(stats_cfg.get("seed", 20260829)),
            ) if not pairs.empty else (pd.DataFrame(), pd.DataFrame())
            cycle_result, cycle_failures = fit_block_cycle_gee(
                tables["cycle"], metrics=CYCLE_INFERENCE_METRICS
            )
            events = build_error_trajectories(trials)
            if not events.empty and "omission_subtype" in trials.columns:
                omission_labels = trials[["session_id", "block_num", "trial_num", "omission_subtype"]].drop_duplicates()
                events = events.merge(
                    omission_labels,
                    left_on=["session_id", "block_num", "error_trial_num"],
                    right_on=["session_id", "block_num", "trial_num"],
                    how="left",
                    validate="many_to_one",
                ).drop(columns=["trial_num"], errors="ignore")
            summary = summarize_error_trajectories(events)
            stability = repeat_stability_boundary(tables["session"])
            return clustered, cycle_result, events, summary, stability, _normalize_failures(cluster_failures, cycle_failures)

        core = ledger.run(
            "core_inference", _core_inference, required=False,
            requested="core_inference" in selected_optional,
            skip_reason="user omitted core_inference",
        )
        if core is not None:
            b1b2_clustered, cycle_gee, error_events, error_summary, stability_boundary, extra_core_fail = core
            core_failures = _normalize_failures(pair_failures, extra_core_fail)

        q1_results = pd.DataFrame()
        q2_results = pd.DataFrame()
        probe_failures = pd.DataFrame(columns=MODEL_FAILURE_COLUMNS)

        def _probe_models():
            q1, q1_fail = fit_q1_nominal(primary_probe, cfg)
            q2, q2_fail = fit_q2_ordinal(primary_probe, cfg)
            return q1, q2, _normalize_failures(q1_fail, q2_fail)

        probe_models = ledger.run(
            "probe_models", _probe_models, required=False,
            requested="probe_models" in selected_optional,
            skip_reason="user omitted probe_models",
        )
        if probe_models is not None:
            q1_results, q2_results, probe_failures = probe_models

        visit_adjusted = pd.DataFrame()
        visit_adjusted_failures = pd.DataFrame(columns=MODEL_FAILURE_COLUMNS)

        def _visit_sensitivity():
            return fit_b1_b2_visit_order_sensitivity(pairs, tables["session"])

        visit_result = ledger.run(
            "visit_sensitivity", _visit_sensitivity, required=False,
            requested="visit_sensitivity" in selected_optional,
            skip_reason="user omitted visit_sensitivity",
        )
        if visit_result is not None:
            visit_adjusted, visit_adjusted_failures = visit_result

        candidate_validation = pd.DataFrame()
        metric_redundancy = pd.DataFrame()
        endpoint_decisions = pd.DataFrame()
        taxonomy_validation = pd.DataFrame()
        omission_validation = pd.DataFrame()
        omission_redundancy = pd.DataFrame()
        sensitivity_status = build_visit_sensitivity_status(tables["session"])
        probe_within_between = primary_probe.copy()
        session_within_between = tables["session"].copy()

        def _candidate_validation():
            candidate, redundancy, decisions = build_candidate_validation(tables, primary_probe)
            taxonomy = build_taxonomy_validation(tables, primary_probe)
            omission_candidate, omission_corr = validate_omission_candidates(tables, primary_probe)
            decomposition_metrics = [m for m in FORMAL_BEHAVIOR_ENDPOINT_METRICS if m in primary_probe.columns]
            probe_wb = decompose_within_between(primary_probe, decomposition_metrics) if not primary_probe.empty else primary_probe.copy()
            session_metrics = [m for m in FORMAL_BEHAVIOR_ENDPOINT_METRICS if m in tables["session"].columns]
            session_wb = decompose_within_between(tables["session"], session_metrics) if not tables["session"].empty else tables["session"].copy()
            return candidate, redundancy, decisions, taxonomy, omission_candidate, omission_corr, probe_wb, session_wb

        candidate_result = ledger.run(
            "candidate_validation", _candidate_validation, required=False,
            requested="candidate_validation" in selected_optional,
            skip_reason="user omitted candidate_validation",
        )
        if candidate_result is not None:
            (
                candidate_validation, metric_redundancy, endpoint_decisions,
                taxonomy_validation, omission_validation, omission_redundancy,
                probe_within_between, session_within_between,
            ) = candidate_result

        folds = pd.DataFrame()
        folds_result = ledger.run(
            "prediction_folds",
            lambda: participant_disjoint_folds(primary_probe, int(stats_cfg.get("prediction_folds", 5))) if not primary_probe.empty else pd.DataFrame(),
            required=False,
            requested="prediction_folds" in selected_optional,
            skip_reason="user omitted prediction_folds",
        )
        if folds_result is not None:
            folds = folds_result

        first_any_summary: dict[str, int] = {"status": "skipped"}  # type: ignore[assignment]
        first_stage_summary: dict[str, int] = {"status": "skipped"}  # type: ignore[assignment]
        sensitivity_failures = pd.DataFrame(columns=MODEL_FAILURE_COLUMNS)

        def _first_visit_sensitivity():
            any_summary, any_fail = _run_first_visit_bundle(
                prefix="first_any_visit_only", tables=tables, probe=primary_probe,
                cfg=cfg, stats_cfg=stats_cfg, output_root=output_root, same_stage=False,
            )
            stage_summary, stage_fail = _run_first_visit_bundle(
                prefix="first_same_stage_visit_only", tables=tables, probe=primary_probe,
                cfg=cfg, stats_cfg=stats_cfg, output_root=output_root, same_stage=True,
            )
            failures = _normalize_failures(visit_adjusted_failures, any_fail, stage_fail)
            return any_summary, stage_summary, failures

        first_visit = ledger.run(
            "first_visit_sensitivity", _first_visit_sensitivity, required=False,
            requested="first_visit_sensitivity" in selected_optional,
            skip_reason="user omitted first_visit_sensitivity",
        )
        if first_visit is not None:
            first_any_summary, first_stage_summary, sensitivity_failures = first_visit

        failures = _normalize_failures(core_failures, probe_failures)

        def _write_outputs():
            identity_consistency.to_csv(output_root / "questionnaire_identity_consistency.csv", index=False, encoding="utf-8-sig")
            identity_audit.to_csv(output_root / "behavior_identity_audit.csv", index=False, encoding="utf-8-sig")
            variable_audit.to_csv(output_root / "questionnaire_variable_audit.csv", index=False, encoding="utf-8-sig")
            session_questionnaire.to_csv(output_root / "session_questionnaire.csv", index=False, encoding="utf-8-sig")
            trials.to_csv(output_root / "trial_metrics.csv", index=False, encoding="utf-8-sig")
            validation.to_csv(output_root / "validation.csv", index=False, encoding="utf-8-sig")
            tables["session"].to_csv(output_root / "session_metrics.csv", index=False, encoding="utf-8-sig")
            tables["block"].to_csv(output_root / "block_metrics.csv", index=False, encoding="utf-8-sig")
            tables["cycle"].to_csv(output_root / "cycle_metrics.csv", index=False, encoding="utf-8-sig")
            primary_probe.to_csv(output_root / "probe_primary_30s.csv", index=False, encoding="utf-8-sig")
            probe_sensitivity.to_csv(output_root / "probe_window_sensitivity.csv", index=False, encoding="utf-8-sig")
            probe_within_between.to_csv(output_root / "probe_primary_30s_within_between.csv", index=False, encoding="utf-8-sig")
            session_within_between.to_csv(output_root / "session_metrics_within_between.csv", index=False, encoding="utf-8-sig")
            candidate_validation.to_csv(output_root / "behavior_candidate_metric_validation.csv", index=False, encoding="utf-8-sig")
            taxonomy_validation.to_csv(output_root / "behavior_error_taxonomy_validation.csv", index=False, encoding="utf-8-sig")
            omission_validation.to_csv(output_root / "behavior_omission_candidate_validation.csv", index=False, encoding="utf-8-sig")
            omission_redundancy.to_csv(output_root / "behavior_omission_candidate_redundancy.csv", index=False, encoding="utf-8-sig")
            omission_partition_audit.to_csv(output_root / "behavior_omission_partition_audit.csv", index=False, encoding="utf-8-sig")
            metric_redundancy.to_csv(output_root / "behavior_metric_redundancy.csv", index=False, encoding="utf-8-sig")
            endpoint_decisions.to_csv(output_root / "behavior_endpoint_decisions.csv", index=False, encoding="utf-8-sig")
            sensitivity_status.to_csv(output_root / "behavior_sensitivity_status.csv", index=False, encoding="utf-8-sig")
            pairs.to_csv(output_root / "b1_b2_pairs.csv", index=False, encoding="utf-8-sig")
            b1b2_clustered.to_csv(output_root / "b1_b2_participant_cluster_bootstrap.csv", index=False, encoding="utf-8-sig")
            visit_adjusted.to_csv(output_root / "b1_b2_visit_order_adjusted_sensitivity.csv", index=False, encoding="utf-8-sig")
            visit_adjusted_failures.to_csv(output_root / "b1_b2_visit_order_adjusted_failures.csv", index=False, encoding="utf-8-sig")
            cycle_gee.to_csv(output_root / "block_cycle_gee.csv", index=False, encoding="utf-8-sig")
            error_events.to_csv(output_root / "error_event_trajectories.csv", index=False, encoding="utf-8-sig")
            error_summary.to_csv(output_root / "error_trajectory_summary.csv", index=False, encoding="utf-8-sig")
            stability_boundary.to_csv(output_root / "repeat_session_stability_boundary.csv", index=False, encoding="utf-8-sig")
            q1_results.to_csv(output_root / "q1_nominal_models.csv", index=False, encoding="utf-8-sig")
            q2_results.to_csv(output_root / "q2_ordinal_gee_models.csv", index=False, encoding="utf-8-sig")
            failures.to_csv(output_root / "model_failures.csv", index=False, encoding="utf-8-sig")
            folds.to_csv(output_root / "participant_disjoint_folds.csv", index=False, encoding="utf-8-sig")
            qc.to_csv(output_root / "qc_denominators.csv", index=False, encoding="utf-8-sig")
            sensitivity_failures.to_csv(output_root / "sensitivity_model_failures.csv", index=False, encoding="utf-8-sig")
            return True

        ledger.run("write_outputs", _write_outputs, required=True)

        figure_files: list[str] = []
        figure_result = ledger.run(
            "figures",
            lambda: generate_behavior_figures(
                tables["block"], primary_probe, output_root / "figures", error_summary=error_summary
            ),
            required=False,
            requested="figures" in selected_optional,
            skip_reason="user omitted figures",
        )
        if figure_result is not None:
            figure_files = figure_result

        write_chinese_result_summary(
            output_root / "结果说明.md", topology, q1_results, q2_results, failures
        )
        with (output_root / "结果说明.md").open("a", encoding="utf-8") as handle:
            handle.write(
                "\n## Go omission 正式结局\n\n"
                "正式保留 raw_go_omission、clean_go_omission 与 timing_ambiguous_go_omission。"
                "三者共享 Go 机会数分母，且 raw = clean + timing-ambiguous。"
                "clean 仅表示未检测到预定义的 motor-timing ambiguity，不等同于已证明的注意失败；"
                "更细的 prestimulus/carry-over 子型只作 QC/机制描述。\n"
            )

        cohort_path = config.registry_path(str(cohort_cfg.get("manifest_path_key", "cohort_manifest")))
        registry_path = config.registry_path(registry_key)
        questionnaire_path = config.registry_path(questionnaire_key)
        manifest = {
            "pipeline": "behavior-science-v3",
            "schema_version": 6,
            "run_scope": run_scope,
            "selected_sessions": subjects,
            "config": str(config.path),
            "config_digest": config.digest,
            "path_registry_digest": config.path_registry.digest if config.path_registry is not None else None,
            "runtime_provenance": {"code": code_provenance},
            "input_provenance": {
                "cohort_manifest": {"logical_key": str(cohort_cfg.get("manifest_path_key", "cohort_manifest")), "sha256": _file_sha256(cohort_path)},
                "repeat_registry": {"logical_key": registry_key, "sha256": _file_sha256(registry_path)},
                "questionnaire_derived_data": {"logical_key": questionnaire_key, "sha256": _file_sha256(questionnaire_path)},
            },
            "current_analysis_cohort_topology": topology,
            "study_total_scope_contract": "registered session universe = 149; governed formal cohort = 116 sessions / 61 participant groups; modality availability does not redefine cohort membership",
            "questionnaire_present_session_n": int(session_questionnaire["questionnaire_present"].sum()),
            "questionnaire_missing_session_n": int(session_questionnaire["questionnaire_present"].eq(0).sum()),
            "participant_key_present_session_n": int(tables["session"].get("participant_key", pd.Series(dtype=object)).notna().sum()),
            "identity_conflict_flag_session_n": int(pd.to_numeric(tables["session"].get("identity_conflict_flag"), errors="coerce").eq(1).sum()),
            "primary_probe_window_seconds": cfg.primary_probe_window_seconds,
            "sensitivity_probe_windows_seconds": list(cfg.sensitivity_probe_windows_seconds),
            "probe_primary_rows": int(len(primary_probe)),
            "error_event_rows": int(len(error_events)),
            "model_failure_rows": int(len(failures)),
            "candidate_validation_rows": int(len(candidate_validation)),
            "error_taxonomy_validation_rows": int(len(taxonomy_validation)),
            "omission_validation_rows": int(len(omission_validation)),
            "endpoint_decision_rows": int(len(endpoint_decisions)),
            "first_any_visit_sensitivity": first_any_summary,
            "first_same_stage_visit_sensitivity": first_stage_summary,
            "figure_files": figure_files,
            "formal_inference_contract": "governed cohort sessions; participant grouping overlaid from participant_key with audited legacy compatibility",
            "prediction_contract": "prediction is separate from inference; folds are participant-group disjoint and must not be used for endpoint selection",
            "cycle_contract": "cycle nested block/session/participant group; block_by_cycle GEE",
            "go_nogo_error_contract": "raw/clean/timing-ambiguous Go omission share the Go denominator; raw=clean+timing-ambiguous; No-Go commission uses a separate No-Go denominator",
            "omission_endpoint_contract": {
                "formal_endpoints": list(FORMAL_OMISSION_ENDPOINT_METRICS),
                "partition": "raw_go_omission_rate = clean_go_omission_rate + timing_ambiguous_go_omission_rate",
                "clean_interpretation": "no detected prestimulus/carry-over ambiguity; not proven attentional lapse",
                "finer_timing_subtypes": "QC/descriptive only",
            },
            "q1_contract": "nominal_four_class_reference_1",
            "q2_contract": "ordinal_gee_participant_cluster",
            "repeat_reliability_contract": "visit_count_1_to_n_is_descriptive; reliability_requires_metric_specific_design",
            "candidate_selection_contract": "prespecified roles + coverage/distribution/within-between/redundancy; no p-value screening",
            "sensitivity_contract": "verified questionnaire-registry visit_order only; missing order is never inferred from session_id; first-ever and first-same-stage visits remain distinct",
            "questionnaire_contract": "session tables are left-joined; missing questionnaire never deletes a governed behavior session; questionnaire variables are not automatically selected into models",
            "figure_contract": "English in-image text; Times New Roman; no internal titles; frameless legends; external Chinese captions; machine-readable metric-by-scale coverage audit",
            "rt_cv_min_n": rt_cv_min_n,
            "rt_slope_min_n": rt_slope_min_n,
            "selected_optional_steps": sorted(selected_optional),
        }

        def _write_manifest():
            (output_root / "run_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return True

        ledger.run("write_manifest", _write_manifest, required=True)
        ledger.write(output_root)
        manifest["execution"] = ledger.as_dict()
        manifest["run_status"] = ledger.run_status
        (output_root / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    except Exception as exc:
        if output_root is not None:
            ledger.write(output_root)
        print(json.dumps({
            "pipeline": "behavior-science-v3",
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "execution": ledger.as_dict(),
        }, ensure_ascii=False, indent=2, default=str))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
