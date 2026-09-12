"""Run FocusWave formal Q1 binary supervised learning on one comparison-specific probe table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from attention_pipeline.config import load_config
from attention_pipeline.supervised_learning.entrypoint import run_supervised_from_config
from attention_pipeline.supervised_learning.outcome_scope import validate_task_a_required_outcomes


def _read_probe_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"unsupported supervised input format {suffix!r}; use CSV or Parquet")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/supervised_learning_v1.yaml")
    parser.add_argument("--paths-config", default=None)
    parser.add_argument("--input-table", default=None, help="Optional runtime override for the comparison-specific admitted probe table.")
    parser.add_argument("--output-root", default=None, help="Optional runtime override for the external output root.")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    if args.input_table is not None:
        input_path = Path(args.input_table).expanduser().resolve()
    else:
        config = load_config(args.config, paths_config=args.paths_config)
        input_path = config.path_value("input_table")
    if not input_path.is_file():
        raise FileNotFoundError(f"supervised input probe table not found: {input_path}")

    # The comparison-specific sample may use outcome validity to define membership.
    # Verify that it was based only on the frozen source of the current Q1 target
    # before any model selection or participant-disjoint fitting starts.
    validate_task_a_required_outcomes(_read_probe_table(input_path))

    manifest = run_supervised_from_config(
        args.config,
        paths_config=args.paths_config,
        input_table=input_path,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
