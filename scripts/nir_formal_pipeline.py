"""Run the authoritative downstream pupil-only NIR pipeline.

Stages:
  materialize  topology pupil-only source manifest -> 10_analysis_ready
  tables       10_analysis_ready + existing formal Behavior -> 11_analysis_tables
  all          materialize then tables

This runner never invokes YOLO or RITnet, never reconstructs PIR/OAR, and never
writes production extraction roots.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from attention_pipeline.nir_analysis_ready import (
    run_candidate_materialization,
    run_materialization,
)
from attention_pipeline.nir_formal_analysis.adjustment_audit import run_adjustment_audit
from attention_pipeline.nir_formal_analysis.baseline_contract import run_baseline_contract
from attention_pipeline.nir_formal_analysis.candidate_validation import run_candidate_validation
from attention_pipeline.nir_formal_analysis.event_response import run_event_response_candidates
from attention_pipeline.nir_formal_analysis.figures import generate_nir_figure_pack
from attention_pipeline.nir_formal_analysis.identity_audit import run_nir_identity_audit
from attention_pipeline.nir_formal_analysis.probe_contract import run_probe_contract_repair
from attention_pipeline.nir_formal_analysis.pupil_tables import run_cohort
from attention_pipeline.nir_formal_analysis.scientific_models import run_reference_adjusted_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("materialize", "tables", "all"),
        default="tables",
        help="Default is tables; candidate validation requires matching candidate sidecars in 10_analysis_ready.",
    )
    parser.add_argument("--subjects", help="Optional comma-separated session-key override.")
    parser.add_argument("--paths-config", default=None, help="Machine-local path registry used by identity audit.")
    parser.add_argument("--materialize-config", default="configs/nir_analysis_ready.yaml")
    parser.add_argument("--tables-config", default="configs/nir_formal_analysis.yaml")
    parser.add_argument("--materialize-output-root", help="Optional derived 10_analysis_ready output override.")
    parser.add_argument("--overwrite-derived", action="store_true", help="Allow rebuilding derived 10_analysis_ready outputs only.")
    parser.add_argument("--force-tables", action="store_true", help="Allow rebuilding derived 11_analysis_tables outputs.")
    return parser.parse_args()


def _sessions(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def _emit_and_fail(result: dict[str, object]) -> int:
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 2


def main() -> int:
    args = parse_args()
    sessions = _sessions(args.subjects)
    result: dict[str, object] = {"stage": args.stage, "signal_semantics": "pupil_geometry_only"}

    if args.stage in {"materialize", "all"}:
        materialized = run_materialization(
            Path(args.materialize_config),
            subjects=sessions,
            output_root_override=args.materialize_output_root,
            overwrite_derived=bool(args.overwrite_derived),
        )
        result["materialize"] = materialized
        if int(materialized["summary"].get("n_sessions_failed_this_run", 0)) > 0:
            return _emit_and_fail(result)
        candidate_materialized = run_candidate_materialization(
            Path(args.materialize_config),
            subjects=sessions,
            output_root_override=args.materialize_output_root,
            overwrite_derived=bool(args.overwrite_derived),
        )
        result["candidate_materialize"] = candidate_materialized
        if int(candidate_materialized.get("n_sessions_failed", 0)) > 0:
            return _emit_and_fail(result)

    if args.stage in {"tables", "all"}:
        table_result = run_cohort(
            Path(args.tables_config), subjects=sessions, force=bool(args.force_tables)
        )
        result["tables"] = table_result
        if int(table_result.get("n_sessions_failed", 0)) > 0:
            return _emit_and_fail(result)

        identity_audit = run_nir_identity_audit(
            Path(args.tables_config), subjects=sessions, paths_config=args.paths_config
        )
        result["identity_audit"] = identity_audit
        if identity_audit.get("status") != "complete":
            return _emit_and_fail(result)

        baseline_contract = run_baseline_contract(Path(args.tables_config))
        result["baseline_contract"] = baseline_contract
        if baseline_contract.get("status") != "complete":
            return _emit_and_fail(result)

        probe_contract = run_probe_contract_repair(Path(args.tables_config), subjects=sessions)
        result["probe_contract"] = probe_contract
        if probe_contract.get("status") != "complete":
            return _emit_and_fail(result)

        candidate_validation = run_candidate_validation(Path(args.tables_config), subjects=sessions)
        result["candidate_validation"] = candidate_validation
        if int(candidate_validation.get("n_sessions_failed", 0)) > 0 or candidate_validation.get("status") != "complete":
            return _emit_and_fail(result)

        event_response = run_event_response_candidates(Path(args.tables_config), subjects=sessions)
        result["event_response_candidates"] = event_response
        if int(event_response.get("n_sessions_failed", 0)) > 0:
            return _emit_and_fail(result)

        reference_models = run_reference_adjusted_models(Path(args.tables_config), subjects=sessions)
        result["reference_adjusted_models"] = reference_models
        if reference_models.get("status") == "blocked":
            return _emit_and_fail(result)

        adjustment_audit = run_adjustment_audit(Path(args.tables_config))
        result["adjustment_audit"] = adjustment_audit
        if adjustment_audit.get("status") == "not_estimable":
            return _emit_and_fail(result)

        figures = generate_nir_figure_pack(Path(args.tables_config), subjects=sessions)
        result["figures"] = figures
        if figures.get("status") != "complete":
            return _emit_and_fail(result)

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
