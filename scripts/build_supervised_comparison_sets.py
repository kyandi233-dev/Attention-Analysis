"""Build every comparison-specific Task-B analysis set for the frozen registry.

Usage
-----
::

    $env:PYTHONPATH = "<worktree>\\src"   # REQUIRED: see the note below
    python scripts/build_supervised_comparison_sets.py `
      --config configs/supervised_learning_v1.yaml `
      --behavior-probes <FormalScience/Behavior/formal_v3/probe_primary_30s.csv> `
      --ocular-probes  <FormalScience/Ocular/tables/ocular_probe_features_wide.csv> `
      --movement-probes <FormalScience/Movement/tables/movement_probe_descriptive_source.csv> `
      --output-root <external comparison-set root>

Why ``PYTHONPATH`` is mandatory
-------------------------------
The analysis virtual environment contains an *editable install* of
``attention-analysis`` that points at a different worktree. Pytest is unaffected because
``pyproject.toml`` prepends ``src`` to ``sys.path``, but a plain script run resolves
``attention_pipeline`` to that other checkout and would silently execute another tree's
code. Always export ``PYTHONPATH`` to this worktree's ``src``.

This command only normalises keys, audits availability and writes analysis sets. It never
imputes, never zero-fills, never deletes rows, never trains a model and never mutates the
frozen feature registry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from attention_pipeline.config import load_config
from attention_pipeline.supervised_learning.feature_registry import (
    build_feature_comparison_plan,
    load_registered_features,
)
from attention_pipeline.multimodal_formal.supervised_comparison_sets import (
    DEFAULT_REQUIRED_OUTCOMES,
    build_supervised_comparison_sets,
    write_supervised_comparison_sets,
)


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"unsupported table format for {path}; expected csv/txt/parquet")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/supervised_learning_v1.yaml")
    parser.add_argument("--paths-config", default=None)
    parser.add_argument("--behavior-probes", required=True, type=Path)
    parser.add_argument("--ocular-probes", required=True, type=Path)
    parser.add_argument("--movement-probes", required=True, type=Path)
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="External directory for the comparison sets. Refuses to overwrite.",
    )
    parser.add_argument(
        "--required-outcome",
        action="append",
        default=None,
        help=f"Outcome column that defines membership (default: {list(DEFAULT_REQUIRED_OUTCOMES)})",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    for label, path in (
        ("behavior_probes", args.behavior_probes),
        ("ocular_probes", args.ocular_probes),
        ("movement_probes", args.movement_probes),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    config = load_config(args.config, paths_config=args.paths_config)
    registry_features = config.data["feature_registry"]["features"]
    plan = build_feature_comparison_plan(load_registered_features(config.data["feature_registry"]))

    result = build_supervised_comparison_sets(
        plan=plan,
        registry_features=registry_features,
        behavior_probes=_read_table(args.behavior_probes),
        ocular_probes=_read_table(args.ocular_probes),
        movement_probes=_read_table(args.movement_probes),
        required_outcomes=tuple(args.required_outcome or DEFAULT_REQUIRED_OUTCOMES),
    )
    paths = write_supervised_comparison_sets(args.output_root, result)

    print(
        json.dumps(
            {
                "config_digest": config.digest,
                "config_path": str(Path(args.config).resolve()),
                "outputs": paths,
                "manifest": result.manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
