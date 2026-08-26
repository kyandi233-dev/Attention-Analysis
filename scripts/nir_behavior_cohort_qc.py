"""Run Phase 1 cohort preflight（运行前检查）and QC（质量控制）.

Examples:
    PYTHONPATH=src python scripts/nir_behavior_cohort_qc.py --subjects sub-031 --output-root "D:/_AttentionData/Beijing-NIR/analysis/nir-behavior-v2/smoke/sub-031"
    PYTHONPATH=src python scripts/nir_behavior_cohort_qc.py

This command reads frozen production full-class NIR and formal Behavior inputs and
writes only to the external nir-behavior-v2 analysis area.
"""
from __future__ import annotations

import argparse
import json

from attention_pipeline.config import load_config
from attention_pipeline.nir_behavior_cohort import run_phase1_cohort_qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 NIR cohort preflight/QC")
    parser.add_argument("--config", default="configs/nir_behavior_cohort.yaml")
    parser.add_argument(
        "--subjects",
        help="Optional comma-separated smoke-test subject override, e.g. sub-031,sub-032",
    )
    parser.add_argument(
        "--output-root",
        help="Optional output-root override; use this for isolated smoke tests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    subjects = (
        [item.strip() for item in args.subjects.split(",") if item.strip()]
        if args.subjects
        else None
    )
    result = run_phase1_cohort_qc(
        config,
        subjects=subjects,
        output_root_override=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 1 if int(result.get("subjects_preflight_failed", 0)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
