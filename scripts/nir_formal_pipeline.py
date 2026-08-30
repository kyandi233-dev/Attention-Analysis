"""Run the authoritative downstream pupil-only NIR pipeline.

Stages:
  materialize  topology pupil-only source manifest -> 10_analysis_ready
  tables       10_analysis_ready + existing formal Behavior -> 11_analysis_tables
  all          materialize then tables

This runner never invokes YOLO or RITnet, never reconstructs PIR/OAR, and never
writes production extraction roots.

Structural gates always run for the requested stage. Optional scientific layers
can be selected with ``--only-steps`` or omitted with ``--skip-steps``. Every
step is timed and written to ``execution_steps.csv`` / ``execution_manifest.json``.
A run with deliberately skipped optional analyses is explicitly ``partial_run``.

The resting-period layer is an observability audit, not a mandatory baseline.
It uses ``baseline_start``/``baseline_stop`` from each formal
``beh/master_timeline.csv``. Until observability thresholds are pre-frozen, the
step reports ``audit_only_thresholds_not_frozen`` and never authorizes a resting
pupil reference merely because pupil samples are present.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable

from attention_pipeline.config import load_config
from attention_pipeline.formal_analysis.execution_control import (
    ExecutionLedger,
    resolve_optional_steps,
)
from attention_pipeline.nir_analysis_ready import (
    run_candidate_materialization,
    run_materialization,
)
from attention_pipeline.nir_formal_analysis.adjustment_audit import run_adjustment_audit
from attention_pipeline.nir_formal_analysis.adjustment_figures import run_adjustment_figures
from attention_pipeline.nir_formal_analysis.baseline_contract import run_baseline_contract
from attention_pipeline.nir_formal_analysis.block_session_models import run_block_session_models
from attention_pipeline.nir_formal_analysis.probe_pupil_models import run_probe_pupil_models
from attention_pipeline.nir_formal_analysis.candidate_validation import run_candidate_validation
from attention_pipeline.nir_formal_analysis.event_response import run_event_response_candidates
from attention_pipeline.nir_formal_analysis.figures import generate_nir_figure_pack
from attention_pipeline.nir_formal_analysis.identity_audit import run_nir_identity_audit
from attention_pipeline.nir_formal_analysis.probe_contract import run_probe_contract_repair
from attention_pipeline.nir_formal_analysis.pupil_tables_hardened import run_cohort
from attention_pipeline.nir_formal_analysis.repeat_visit_sensitivity import run_candidate_visit_sensitivity
from attention_pipeline.nir_formal_analysis.resting_observability import run_resting_observability
from attention_pipeline.nir_formal_analysis.scientific_models import run_reference_adjusted_models


OPTIONAL_TABLE_STEPS = (
    "resting_observability",
    "candidate_validation",
    "visit_sensitivity",
    "event_response",
    "reference_models",
    "probe_pupil_models",
    "block_session_models",
    "adjustment_audit",
    "figures",
    "adjustment_figures",
)

STRUCTURAL_STEPS = (
    "materialize",
    "candidate_materialize",
    "tables",
    "identity_audit",
    "baseline_contract",
    "probe_contract",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("materialize", "tables", "all"),
        default="tables",
        help="Default is tables; candidate validation requires matching candidate sidecars in 10_analysis_ready.",
    )
    parser.add_argument("--subjects", help="Optional comma-separated session-key override for smoke/subset runs.")
    parser.add_argument("--paths-config", default=None, help="Machine-local path registry for all @path references.")
    parser.add_argument("--materialize-config", default="configs/nir_analysis_ready.yaml")
    parser.add_argument("--tables-config", default="configs/nir_formal_analysis.yaml")
    parser.add_argument("--materialize-output-root", help="Optional derived 10_analysis_ready output override.")
    parser.add_argument("--overwrite-derived", action="store_true", help="Allow rebuilding derived 10_analysis_ready outputs only.")
    parser.add_argument("--force-tables", action="store_true", help="Allow rebuilding derived 11_analysis_tables outputs.")
    parser.add_argument(
        "--only-steps",
        help="Comma-separated optional table analyses to run. Structural gates always run.",
    )
    parser.add_argument(
        "--skip-steps",
        help="Comma-separated optional table analyses to skip. Structural gates always run.",
    )
    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="Print structural/optional step names and exit without touching data.",
    )
    return parser.parse_args()


def _sessions(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def _require_result(
    name: str,
    func: Callable[[], dict[str, object]],
    *,
    invalid: Callable[[dict[str, object]], bool],
) -> dict[str, object]:
    value = func()
    if invalid(value):
        raise RuntimeError(
            f"NIR formal step {name} did not satisfy its release gate: "
            + json.dumps(value, ensure_ascii=False, default=str)
        )
    return value


def _resolve_ledger_root(args: argparse.Namespace) -> Path | None:
    try:
        if args.stage in {"tables", "all"}:
            cfg = load_config(args.tables_config, paths_config=args.paths_config)
            return cfg.path_value("output_root")
        cfg = load_config(args.materialize_config, paths_config=args.paths_config)
        if args.materialize_output_root:
            return Path(args.materialize_output_root)
        return cfg.path_value("output_root")
    except Exception:
        return None


def main() -> int:
    args = parse_args()
    if args.list_steps:
        print(json.dumps({
            "structural_steps": STRUCTURAL_STEPS,
            "optional_table_steps": OPTIONAL_TABLE_STEPS,
            "contract": (
                "structural steps cannot be skipped; omitted optional steps make "
                "run_status=partial_run; resting_observability is audit-only until "
                "its thresholds are pre-frozen"
            ),
        }, ensure_ascii=False, indent=2))
        return 0

    # Make the CLI path registry authoritative for all nested functions that
    # independently load configs. This avoids hidden dependence on the current
    # machine's environment state.
    if args.paths_config:
        os.environ["ATTENTION_ANALYSIS_PATHS_CONFIG"] = str(Path(args.paths_config).resolve())

    selected_optional = resolve_optional_steps(
        OPTIONAL_TABLE_STEPS,
        only=args.only_steps,
        skip=args.skip_steps,
    )
    sessions = _sessions(args.subjects)
    ledger = ExecutionLedger(pipeline="nir-formal-pupil-only")
    ledger_root = _resolve_ledger_root(args)
    result: dict[str, object] = {
        "stage": args.stage,
        "signal_semantics": "pupil_geometry_only",
        "session_override": sessions,
        "selected_optional_steps": sorted(selected_optional),
    }

    try:
        if args.stage in {"materialize", "all"}:
            materialized = ledger.run(
                "materialize",
                lambda: _require_result(
                    "materialize",
                    lambda: run_materialization(
                        Path(args.materialize_config),
                        subjects=sessions,
                        output_root_override=args.materialize_output_root,
                        overwrite_derived=bool(args.overwrite_derived),
                    ),
                    invalid=lambda x: int(x["summary"].get("n_sessions_failed_this_run", 0)) > 0,
                ),
                required=True,
            )
            result["materialize"] = materialized

            candidate_materialized = ledger.run(
                "candidate_materialize",
                lambda: _require_result(
                    "candidate_materialize",
                    lambda: run_candidate_materialization(
                        Path(args.materialize_config),
                        subjects=sessions,
                        output_root_override=args.materialize_output_root,
                        overwrite_derived=bool(args.overwrite_derived),
                    ),
                    invalid=lambda x: int(x.get("n_sessions_failed", 0)) > 0,
                ),
                required=True,
            )
            result["candidate_materialize"] = candidate_materialized

        if args.stage in {"tables", "all"}:
            table_result = ledger.run(
                "tables",
                lambda: _require_result(
                    "tables",
                    lambda: run_cohort(
                        Path(args.tables_config), subjects=sessions, force=bool(args.force_tables)
                    ),
                    invalid=lambda x: int(x.get("n_sessions_failed", 0)) > 0,
                ),
                required=True,
            )
            result["tables"] = table_result

            identity_audit = ledger.run(
                "identity_audit",
                lambda: _require_result(
                    "identity_audit",
                    lambda: run_nir_identity_audit(
                        Path(args.tables_config), subjects=sessions, paths_config=args.paths_config
                    ),
                    invalid=lambda x: x.get("status") != "complete",
                ),
                required=True,
            )
            result["identity_audit"] = identity_audit

            baseline_contract = ledger.run(
                "baseline_contract",
                lambda: _require_result(
                    "baseline_contract",
                    lambda: run_baseline_contract(Path(args.tables_config)),
                    invalid=lambda x: x.get("status") != "complete",
                ),
                required=True,
            )
            result["baseline_contract"] = baseline_contract

            # Resting-period observability is deliberately optional.  Audit-only
            # status is not a release failure: it means the timeline/source facts
            # were summarized but observability thresholds have not yet been
            # authorized.  Session-specific problems are preserved in its failure
            # table rather than being reinterpreted as closed eyes.
            resting_observability = ledger.run(
                "resting_observability",
                lambda: run_resting_observability(
                    Path(args.materialize_config),
                    Path(args.tables_config),
                    subjects=sessions,
                    paths_config=args.paths_config,
                ),
                required=False,
                requested="resting_observability" in selected_optional,
                skip_reason="user omitted resting_observability",
            )
            result["resting_observability"] = resting_observability

            probe_contract = ledger.run(
                "probe_contract",
                lambda: _require_result(
                    "probe_contract",
                    lambda: run_probe_contract_repair(Path(args.tables_config), subjects=sessions),
                    invalid=lambda x: x.get("status") != "complete",
                ),
                required=True,
            )
            result["probe_contract"] = probe_contract

            candidate_validation = ledger.run(
                "candidate_validation",
                lambda: _require_result(
                    "candidate_validation",
                    lambda: run_candidate_validation(Path(args.tables_config), subjects=sessions),
                    invalid=lambda x: int(x.get("n_sessions_failed", 0)) > 0 or x.get("status") != "complete",
                ),
                required=False,
                requested="candidate_validation" in selected_optional,
                skip_reason="user omitted candidate_validation",
            )
            result["candidate_validation"] = candidate_validation

            visit_sensitivity = ledger.run(
                "visit_sensitivity",
                lambda: _require_result(
                    "visit_sensitivity",
                    lambda: run_candidate_visit_sensitivity(
                        Path(args.tables_config), subjects=sessions, paths_config=args.paths_config
                    ),
                    invalid=lambda x: x.get("status") != "complete",
                ),
                required=False,
                requested="visit_sensitivity" in selected_optional,
                skip_reason="user omitted visit_sensitivity",
            )
            result["candidate_visit_sensitivity"] = visit_sensitivity

            event_response = ledger.run(
                "event_response",
                lambda: _require_result(
                    "event_response",
                    lambda: run_event_response_candidates(Path(args.tables_config), subjects=sessions),
                    invalid=lambda x: int(x.get("n_sessions_failed", 0)) > 0,
                ),
                required=False,
                requested="event_response" in selected_optional,
                skip_reason="user omitted event_response",
            )
            result["event_response_candidates"] = event_response

            reference_models = ledger.run(
                "reference_models",
                lambda: _require_result(
                    "reference_models",
                    lambda: run_reference_adjusted_models(Path(args.tables_config), subjects=sessions),
                    invalid=lambda x: x.get("status") == "blocked",
                ),
                required=False,
                requested="reference_models" in selected_optional,
                skip_reason="user omitted reference_models",
            )
            result["reference_adjusted_models"] = reference_models

            # Frozen-endpoint formal association layers (previously deferred).
            # These run after probe_contract so the strict pre-probe behavior
            # repair is already applied to the probe windows they read.
            probe_pupil_models = ledger.run(
                "probe_pupil_models",
                lambda: _require_result(
                    "probe_pupil_models",
                    lambda: run_probe_pupil_models(Path(args.tables_config), subjects=sessions),
                    invalid=lambda x: x.get("status") == "blocked",
                ),
                required=False,
                requested="probe_pupil_models" in selected_optional,
                skip_reason="user omitted probe_pupil_models",
            )
            result["probe_pupil_models"] = probe_pupil_models

            block_session_models = ledger.run(
                "block_session_models",
                lambda: _require_result(
                    "block_session_models",
                    lambda: run_block_session_models(Path(args.tables_config), subjects=sessions),
                    invalid=lambda x: x.get("status") == "blocked",
                ),
                required=False,
                requested="block_session_models" in selected_optional,
                skip_reason="user omitted block_session_models",
            )
            result["block_session_models"] = block_session_models

            adjustment_audit = ledger.run(
                "adjustment_audit",
                lambda: _require_result(
                    "adjustment_audit",
                    lambda: run_adjustment_audit(Path(args.tables_config)),
                    invalid=lambda x: x.get("status") == "not_estimable",
                ),
                required=False,
                requested="adjustment_audit" in selected_optional,
                skip_reason="user omitted adjustment_audit",
            )
            result["adjustment_audit"] = adjustment_audit

            figures = ledger.run(
                "figures",
                lambda: _require_result(
                    "figures",
                    lambda: generate_nir_figure_pack(Path(args.tables_config), subjects=sessions),
                    invalid=lambda x: x.get("status") != "complete",
                ),
                required=False,
                requested="figures" in selected_optional,
                skip_reason="user omitted figures",
            )
            result["figures"] = figures

            adjustment_figures = ledger.run(
                "adjustment_figures",
                lambda: _require_result(
                    "adjustment_figures",
                    lambda: run_adjustment_figures(Path(args.tables_config)),
                    invalid=lambda x: x.get("status") != "complete",
                ),
                required=False,
                requested="adjustment_figures" in selected_optional,
                skip_reason="user omitted adjustment_figures",
            )
            result["adjustment_figures"] = adjustment_figures

    except Exception as exc:
        result["status"] = "failed"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        if ledger_root is not None:
            ledger.write(ledger_root)
        result["execution"] = ledger.as_dict()
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 2

    if ledger_root is not None:
        ledger.write(ledger_root)
    result["status"] = ledger.run_status
    result["execution"] = ledger.as_dict()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
