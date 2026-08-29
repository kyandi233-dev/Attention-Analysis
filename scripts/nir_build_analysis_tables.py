"""Build formal pupil-only NIR downstream tables from 10_analysis_ready.

This command never reads production NIR directly. It consumes the authoritative
pupil-only analysis-ready layer plus the already-produced formal Behavior tables
and writes 11_analysis_tables. Long overlapping windows are preserved for
multiscale description but receive an explicit dependence audit. After table
construction, strict pre-probe semantics, candidate validation, and reference
unadjusted/adjusted model interfaces are audited before success is returned.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from attention_pipeline.nir_formal_analysis.candidate_validation import run_candidate_validation
from attention_pipeline.nir_formal_analysis.probe_contract import run_probe_contract_repair
from attention_pipeline.nir_formal_analysis.pupil_tables import run_cohort
from attention_pipeline.nir_formal_analysis.scientific_models import run_reference_adjusted_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/nir_formal_analysis.yaml")
    parser.add_argument(
        "--subjects",
        help="Optional comma-separated session-key override.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild derived 11_analysis_tables outputs even when completion identity matches.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sessions = None
    if args.subjects:
        sessions = [item.strip() for item in args.subjects.split(",") if item.strip()]
    result = run_cohort(
        Path(args.config),
        subjects=sessions,
        force=bool(args.force),
    )
    payload: dict[str, object] = {"tables": result}
    if int(result.get("n_sessions_failed", 0)):
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 2

    probe_contract = run_probe_contract_repair(Path(args.config), subjects=sessions)
    payload["probe_contract"] = probe_contract
    if probe_contract.get("status") != "complete":
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 2

    candidate_validation = run_candidate_validation(Path(args.config), subjects=sessions)
    payload["candidate_validation"] = candidate_validation
    if candidate_validation.get("status") != "complete":
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 2

    reference_models = run_reference_adjusted_models(Path(args.config), subjects=sessions)
    payload["reference_adjusted_models"] = reference_models
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0 if reference_models.get("status") != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
