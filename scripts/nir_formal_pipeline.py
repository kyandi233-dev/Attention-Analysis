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
from attention_pipeline.nir_formal_analysis.pupil_tables import run_cohort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("materialize", "tables", "all"),
        default="tables",
        help="Default is tables because 10_analysis_ready may already be frozen.",
    )
    parser.add_argument(
        "--subjects",
        help="Optional comma-separated session-key override.",
    )
    parser.add_argument("--materialize-config", default="configs/nir_analysis_ready.yaml")
    parser.add_argument("--tables-config", default="configs/nir_formal_analysis.yaml")
    parser.add_argument(
        "--materialize-output-root",
        help="Optional derived 10_analysis_ready output override.",
    )
    parser.add_argument(
        "--overwrite-derived",
        action="store_true",
        help="Allow rebuilding derived 10_analysis_ready outputs only.",
    )
    parser.add_argument(
        "--force-tables",
        action="store_true",
        help="Allow rebuilding derived 11_analysis_tables outputs.",
    )
    return parser.parse_args()


def _sessions(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


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
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            return 2
        candidate_materialized = run_candidate_materialization(
            Path(args.materialize_config),
            subjects=sessions,
            output_root_override=args.materialize_output_root,
            overwrite_derived=bool(args.overwrite_derived),
        )
        result["candidate_materialize"] = candidate_materialized
        if int(candidate_materialized.get("n_sessions_failed", 0)) > 0:
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            return 2

    if args.stage in {"tables", "all"}:
        table_result = run_cohort(
            Path(args.tables_config),
            subjects=sessions,
            force=bool(args.force_tables),
        )
        result["tables"] = table_result
        if int(table_result.get("n_sessions_failed", 0)) > 0:
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            return 2

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
