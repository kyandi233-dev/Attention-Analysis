"""Build formal pupil-only NIR downstream tables from 10_analysis_ready.

This command never reads production NIR directly. It consumes the authoritative
pupil-only analysis-ready layer plus the already-produced formal Behavior tables
and writes 11_analysis_tables. Long overlapping windows are preserved for
multiscale description but receive an explicit dependence audit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from attention_pipeline.nir_formal_analysis.pupil_tables import run_cohort


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
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if int(result.get("n_sessions_failed", 0)) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
