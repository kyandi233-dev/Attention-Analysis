"""Run FocusWave Task A Q1 binary supervised learning on an upstream probe table."""
from __future__ import annotations

import argparse
import json

from attention_pipeline.supervised_learning.entrypoint import run_supervised_from_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/supervised_learning_v1.yaml")
    parser.add_argument("--paths-config", default=None)
    parser.add_argument("--input-table", default=None, help="Optional runtime override for the upstream admitted probe table.")
    parser.add_argument("--output-root", default=None, help="Optional runtime override for the external output root.")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    manifest = run_supervised_from_config(
        args.config,
        paths_config=args.paths_config,
        input_table=args.input_table,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
