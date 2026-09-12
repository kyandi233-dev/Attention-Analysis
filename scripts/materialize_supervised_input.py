"""Materialize one Task-B analysis-set membership into a Task-A probe table.

This is a file-level handoff only. It does not construct analysis sets, impute,
scale, select features, fit models, or invoke the historical all-modality common
subset path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from attention_pipeline.multimodal_formal.supervised_input import (
    materialize_supervised_input,
)


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"unsupported table format {suffix!r}; use CSV or Parquet")


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return
    if suffix in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
        return
    raise ValueError(f"unsupported output format {suffix!r}; use CSV or Parquet")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-sets", required=True, help="Task-B analysis_sets CSV/Parquet")
    parser.add_argument(
        "--probe-feature-status",
        required=True,
        help="Task-B probe_feature_status CSV/Parquet",
    )
    parser.add_argument(
        "--probe-metadata",
        default=None,
        help="Optional Behavior-authoritative probe metadata CSV/Parquet",
    )
    parser.add_argument("--analysis-set-id", required=True)
    parser.add_argument(
        "--membership-type",
        required=True,
        choices=["included_complete", "included_missing_aware"],
    )
    parser.add_argument("--output", required=True, help="Output Task-A CSV/Parquet")
    args = parser.parse_args()

    analysis_sets_path = Path(args.analysis_sets).expanduser().resolve()
    status_path = Path(args.probe_feature_status).expanduser().resolve()
    metadata_path = Path(args.probe_metadata).expanduser().resolve() if args.probe_metadata else None
    output_path = Path(args.output).expanduser().resolve()

    for label, path in (
        ("analysis_sets", analysis_sets_path),
        ("probe_feature_status", status_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if metadata_path is not None and not metadata_path.is_file():
        raise FileNotFoundError(f"probe_metadata not found: {metadata_path}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing Task-A input: {output_path}")

    analysis_sets = _read_table(analysis_sets_path)
    probe_feature_status = _read_table(status_path)
    probe_metadata = _read_table(metadata_path) if metadata_path is not None else None

    frame = materialize_supervised_input(
        analysis_sets,
        probe_feature_status,
        analysis_set_id=args.analysis_set_id,
        membership_type=args.membership_type,
        probe_metadata=probe_metadata,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_table(frame, output_path)
    summary = {
        "status": "complete",
        "analysis_set_id": str(args.analysis_set_id),
        "membership_type": str(args.membership_type),
        "rows": int(len(frame)),
        "participant_groups": int(frame["participant_group_id"].astype(str).nunique()),
        "sessions": int(frame["session_id"].astype(str).nunique()),
        "output": str(output_path),
        "analysis_sets": str(analysis_sets_path),
        "probe_feature_status": str(status_path),
        "probe_metadata": str(metadata_path) if metadata_path is not None else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
