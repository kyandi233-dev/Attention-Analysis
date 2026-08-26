"""Run the formal downstream NIR pipeline after frozen production extraction.

Stages:
  materialize  frozen production -> 10_analysis_ready
  tables       10_analysis_ready + formal Behavior -> 11_analysis_tables
  all          materialize then tables

This runner never invokes YOLO or RITnet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from attention_pipeline.nir_analysis_ready.materialize import run_materialization
from attention_pipeline.nir_formal_analysis.tables import run_cohort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("materialize", "tables", "all"),
        default="tables",
        help="Default is tables because 10_analysis_ready is normally already frozen.",
    )
    parser.add_argument(
        "--subjects",
        help="Optional comma-separated subject override.",
    )
    parser.add_argument(
        "--materialize-config",
        default="configs/nir_analysis_ready.yaml",
    )
    parser.add_argument(
        "--tables-config",
        default="configs/nir_formal_analysis.yaml",
    )
    parser.add_argument(
        "--overwrite-derived",
        action="store_true",
        help="Allow rebuilding 10_analysis_ready derived outputs only.",
    )
    parser.add_argument(
        "--force-tables",
        action="store_true",
        help="Allow rebuilding 11_analysis_tables when completion identity already matches.",
    )
    return parser.parse_args()


def _subjects(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> int:
    args = parse_args()
    subjects = _subjects(args.subjects)
    result: dict[str, object] = {"stage": args.stage}

    if args.stage in {"materialize", "all"}:
        result["materialize"] = run_materialization(
            Path(args.materialize_config),
            subjects=subjects,
            overwrite_derived=bool(args.overwrite_derived),
        )

    if args.stage in {"tables", "all"}:
        table_result = run_cohort(
            Path(args.tables_config),
            subjects=subjects,
            force=bool(args.force_tables),
        )
        result["tables"] = table_result
        if int(table_result.get("n_subjects_failed", 0)) > 0:
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            return 2

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
