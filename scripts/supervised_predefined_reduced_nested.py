"""Run the prespecified reduced-model nested LOSO comparison for binary Q1."""
from __future__ import annotations

import argparse
import json

from attention_pipeline.supervised_learning.predefined_reduced import (
    run_predefined_reduced_from_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design-config",
        default="configs/supervised_predefined_reduced_v1.yaml",
    )
    parser.add_argument(
        "--base-config",
        default=None,
        help="Optional override for the frozen binary supervised-learning config.",
    )
    parser.add_argument("--input-table", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_predefined_reduced_from_config(
        args.design_config,
        base_config_path=args.base_config,
        input_table=args.input_table,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

