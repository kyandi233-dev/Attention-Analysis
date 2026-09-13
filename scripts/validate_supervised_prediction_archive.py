"""Validate Task-A OOF predictions against Task-B analysis-set authority.

The input predictions may be the native Task-A ``probe_predictions.csv``. They
are normalized to the Task-B archive schema, checked against the requested
analysis-set membership and Behavior-authoritative Q1 labels, then written as an
audited archive. No model fitting or relabeling occurs here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from attention_pipeline.multimodal_formal.prediction_archive import (
    normalize_task_a_predictions,
    validate_prediction_archive,
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
    parser.add_argument("--predictions", required=True, help="Native Task-A predictions CSV/Parquet")
    parser.add_argument("--analysis-sets", required=True, help="Task-B analysis_sets CSV/Parquet")
    parser.add_argument("--analysis-set-id", required=True)
    parser.add_argument(
        "--membership-type",
        required=True,
        choices=["included_complete", "included_missing_aware"],
    )
    parser.add_argument("--archive-output", required=True, help="Normalized validated archive CSV/Parquet")
    parser.add_argument("--audit-output", required=True, help="Validation audit JSON")
    args = parser.parse_args()

    predictions_path = Path(args.predictions).expanduser().resolve()
    analysis_sets_path = Path(args.analysis_sets).expanduser().resolve()
    archive_output = Path(args.archive_output).expanduser().resolve()
    audit_output = Path(args.audit_output).expanduser().resolve()

    for label, path in (("predictions", predictions_path), ("analysis_sets", analysis_sets_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    for path in (archive_output, audit_output):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing validation output: {path}")

    native = _read_table(predictions_path)
    analysis_sets = _read_table(analysis_sets_path)
    normalized = normalize_task_a_predictions(native)
    audit = validate_prediction_archive(
        normalized,
        analysis_sets,
        membership_column=args.membership_type,
        requested_analysis_set_ids=[args.analysis_set_id],
    )

    archive_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    _write_table(normalized, archive_output)
    audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
