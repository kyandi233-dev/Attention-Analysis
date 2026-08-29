"""Build formal pupil-only NIR downstream tables from 10_analysis_ready.

This command never reads production NIR directly. It consumes the authoritative
pupil-only analysis-ready layer plus the already-produced formal Behavior tables
and writes 11_analysis_tables. Long overlapping windows are preserved for
multiscale description but receive an explicit dependence audit. Participant
identity parity, baseline semantics, strict pre-probe semantics, candidate
validation, event-response candidates, adjustment auditing and title-free figure
coverage are required before success is returned.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    parser.add_argument("--config", default="configs/nir_formal_analysis.yaml")
    parser.add_argument("--paths-config", default=None, help="Machine-local path registry used by identity audit.")
    parser.add_argument("--subjects", help="Optional comma-separated session-key override.")
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild derived 11_analysis_tables outputs even when completion identity matches.",
    )
    return parser.parse_args()


def _fail(payload: dict[str, object]) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 2


def main() -> int:
    args = parse_args()
    sessions = None
    if args.subjects:
        sessions = [item.strip() for item in args.subjects.split(",") if item.strip()]
    result = run_cohort(Path(args.config), subjects=sessions, force=bool(args.force))
    payload: dict[str, object] = {"tables": result}
    if int(result.get("n_sessions_failed", 0)):
        return _fail(payload)

    identity_audit = run_nir_identity_audit(
        Path(args.config), subjects=sessions, paths_config=args.paths_config
    )
    payload["identity_audit"] = identity_audit
    if identity_audit.get("status") != "complete":
        return _fail(payload)

    baseline_contract = run_baseline_contract(Path(args.config))
    payload["baseline_contract"] = baseline_contract
    if baseline_contract.get("status") != "complete":
        return _fail(payload)

    probe_contract = run_probe_contract_repair(Path(args.config), subjects=sessions)
    payload["probe_contract"] = probe_contract
    if probe_contract.get("status") != "complete":
        return _fail(payload)

    candidate_validation = run_candidate_validation(Path(args.config), subjects=sessions)
    payload["candidate_validation"] = candidate_validation
    if candidate_validation.get("status") != "complete":
        return _fail(payload)

    event_response = run_event_response_candidates(Path(args.config), subjects=sessions)
    payload["event_response_candidates"] = event_response
    if int(event_response.get("n_sessions_failed", 0)):
        return _fail(payload)

    reference_models = run_reference_adjusted_models(Path(args.config), subjects=sessions)
    payload["reference_adjusted_models"] = reference_models
    if reference_models.get("status") == "blocked":
        return _fail(payload)

    adjustment_audit = run_adjustment_audit(Path(args.config))
    payload["adjustment_audit"] = adjustment_audit
    if adjustment_audit.get("status") == "not_estimable":
        return _fail(payload)

    figures = generate_nir_figure_pack(Path(args.config), subjects=sessions)
    payload["figures"] = figures
    if figures.get("status") != "complete":
        return _fail(payload)

    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
