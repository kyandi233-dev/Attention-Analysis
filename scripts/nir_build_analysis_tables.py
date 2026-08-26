"""Build formal NIR downstream analysis tables from 10_analysis_ready.

This command never reads frozen production NIR directly. It consumes the
analysis-ready frame tables plus formal FocusWave behavior files and writes
analysis-specific derived tables under 11_analysis_tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from attention_pipeline.nir_formal_analysis.tables import run_cohort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/nir_formal_analysis.yaml")
    parser.add_argument(
        "--subjects",
        help="Optional comma-separated subject override, e.g. sub-031,sub-032",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild derived 11_analysis_tables outputs even when completion identity matches.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subjects = None
    if args.subjects:
        subjects = [item.strip() for item in args.subjects.split(",") if item.strip()]
    result = run_cohort(
        Path(args.config),
        subjects=subjects,
        force=bool(args.force),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if int(result.get("n_subjects_failed", 0)) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
