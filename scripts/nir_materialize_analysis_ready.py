"""Materialize the read-only NIR analysis-ready derived dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from attention_pipeline.nir_analysis_ready import (
    run_candidate_materialization,
    run_materialization,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/nir_analysis_ready.yaml",
        help="Analysis-ready materialization config",
    )
    parser.add_argument(
        "--subjects",
        help="Optional comma-separated session override, e.g. sub-031,sub-032",
    )
    parser.add_argument(
        "--output-root",
        help="Optional derived output-root override; production roots are rejected",
    )
    parser.add_argument(
        "--overwrite-derived",
        action="store_true",
        help="Allow rebuilding files inside the derived analysis-ready output only",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subjects = None
    if args.subjects:
        subjects = [value.strip() for value in args.subjects.split(",") if value.strip()]

    try:
        canonical = run_materialization(
            Path(args.config),
            subjects=subjects,
            output_root_override=args.output_root,
            overwrite_derived=bool(args.overwrite_derived),
        )
        candidates = run_candidate_materialization(
            Path(args.config),
            subjects=subjects,
            output_root_override=args.output_root,
            overwrite_derived=bool(args.overwrite_derived),
        )
        result = {"canonical": canonical, "candidate_metrics": candidates}
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, ensure_ascii=False, indent=2))
    canonical_failed = int(canonical["summary"].get("n_sessions_failed_this_run", 0))
    candidate_failed = int(candidates.get("n_sessions_failed", 0))
    return 0 if canonical_failed == 0 and candidate_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
