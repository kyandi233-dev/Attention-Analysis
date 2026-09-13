#!/usr/bin/env python
"""Run the canonical mmWave -> Cardiopulmonary ingest audit.

This command never re-runs the mmWave estimator and never trains a supervised model.
It validates the versioned integration snapshot against the Behavior-authoritative
probe denominator, writes focused ingest/coverage/provenance outputs, and exercises
the current comparison-specific analysis-set interface without consuming Q1/Q2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from attention_pipeline.multimodal_formal.mmwave_cardiopulmonary_ingest import (
    audit_mmwave_cardiopulmonary_snapshot,
    write_mmwave_cardiopulmonary_ingest_audit,
)


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"unsupported table format for {path}; expected csv/txt/parquet")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit canonical mmWave snapshot ingestion as Cardiopulmonary science data."
    )
    parser.add_argument(
        "--snapshot",
        required=True,
        type=Path,
        help="MMWAVE_INTEGRATION_SNAPSHOT_V1 probe-level CSV/Parquet.",
    )
    parser.add_argument(
        "--behavior-probes",
        required=True,
        type=Path,
        help="Behavior-authoritative governed probe table containing canonical keys and participant_group_id.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Directory for ingest audit, coverage, manifest, handoff, and interface-smoke outputs.",
    )
    parser.add_argument(
        "--allow-subset-smoke",
        action="store_true",
        help=(
            "Allow a deliberately small schema smoke instead of enforcing the canonical "
            "116-session / 61-group / 2320-probe and 2180/40/100 denominators. "
            "Never use this flag for the final governed-cohort audit."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    snapshot = _read_table(args.snapshot)
    behavior = _read_table(args.behavior_probes)
    result = audit_mmwave_cardiopulmonary_snapshot(
        snapshot,
        behavior,
        strict_canonical_counts=not args.allow_subset_smoke,
    )
    paths = write_mmwave_cardiopulmonary_ingest_audit(args.output_root, result)
    print(
        json.dumps(
            {
                "manifest": result.manifest,
                "outputs": paths,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
