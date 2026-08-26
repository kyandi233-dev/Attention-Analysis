"""Run validation-only downstream NIR analyses and code-generated figures.

This script writes only to 12_pipeline_validation. Current NIR values are known
invalid, so all PIR effects and model outputs are smoke-test artifacts only.
"""
from __future__ import annotations

import argparse
import json

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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subjects = (
        [value.strip() for value in args.subjects.split(",") if value.strip()]
        if args.subjects
        else None
    )
    result = run_validation(args.config, subjects=subjects)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
