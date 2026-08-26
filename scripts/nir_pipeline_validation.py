"""Run validation-only downstream NIR analyses and publication-grade figures.

This script writes only to 12_pipeline_validation. Current NIR values are known
invalid, so all PIR effects and model outputs are smoke-test artifacts only.
No image-generation model is used: every figure is produced by Python/Matplotlib.
"""
from __future__ import annotations

import argparse
import json

from attention_pipeline.nir_pipeline_validation.publication_run import (
    run_publication_validation,
)
from attention_pipeline.nir_pipeline_validation.run import run_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/nir_pipeline_validation.yaml",
    )
    parser.add_argument(
        "--subjects",
        help="Optional comma-separated override, e.g. sub-031,sub-032",
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Run legacy/core diagnostic validation only; skip publication Figure 1-10 suite.",
    )
    parser.add_argument(
        "--publication-only",
        action="store_true",
        help="Run publication analysis/Figure 1-10 only; assumes core validation tables/models already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.core_only and args.publication_only:
        raise SystemExit("--core-only and --publication-only are mutually exclusive")
    subjects = (
        [value.strip() for value in args.subjects.split(",") if value.strip()]
        if args.subjects
        else None
    )

    result: dict = {}
    if not args.publication_only:
        result["core_validation"] = run_validation(args.config, subjects=subjects)
    if not args.core_only:
        result["publication_validation"] = run_publication_validation(
            args.config, subjects=subjects
        )

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
