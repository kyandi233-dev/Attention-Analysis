"""RGB development pipeline entrypoint.

Implemented development stages:
    python scripts/rgb_analysis.py --stage audit
    python scripts/rgb_analysis.py --stage gaps

Neither stage runs face, pose, or motion models. ``audit`` verifies formal RGB
inputs and FocusWave coverage; ``gaps`` records every timestamp interval above
the development warning threshold with frame identity and experiment phase so
later temporal features can mark the affected sample missing rather than create
false movement.
"""
from __future__ import annotations

import argparse
import json
import sys

from attention_pipeline.config import load_config
from attention_pipeline.rgb.audit import run_audit
from attention_pipeline.rgb.gaps import run_gap_audit
from attention_pipeline.rgb.paths import RGBOutputLayout


def stage_audit(config) -> dict[str, object]:
    layout = RGBOutputLayout.from_config(config)
    inventory, duplicates = run_audit(config)

    inventory_path = layout.dataset_file("rgb_inventory.csv")
    summary_path = layout.dataset_file("audit_summary.json")
    inventory.to_csv(inventory_path, index=False, encoding="utf-8-sig")

    duplicate_path = None
    if not duplicates.empty:
        duplicate_path = layout.dataset_file("rgb_duplicate_subjects.csv")
        duplicates.to_csv(duplicate_path, index=False, encoding="utf-8-sig")

    basic_complete = int(inventory["basic_complete"].sum()) if "basic_complete" in inventory else 0
    excluded = int(inventory["analysis_excluded"].sum()) if "analysis_excluded" in inventory else 0
    eligible = int(inventory["analysis_eligible"].sum()) if "analysis_eligible" in inventory else 0
    incomplete_nonexcluded = int(
        ((~inventory["analysis_excluded"].astype(bool)) & (~inventory["basic_complete"].astype(bool))).sum()
    ) if {"analysis_excluded", "basic_complete"}.issubset(inventory.columns) else 0

    summary = {
        "subjects_discovered_unique": int(len(inventory)),
        "subjects_basic_complete_raw": basic_complete,
        "subjects_excluded_by_config": excluded,
        "subjects_analysis_eligible": eligible,
        "subjects_incomplete_nonexcluded": incomplete_nonexcluded,
        "duplicate_subjects": int(len(duplicates)),
        "inventory": str(inventory_path),
        "duplicates": str(duplicate_path) if duplicate_path else None,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def stage_gaps(config) -> dict[str, object]:
    layout = RGBOutputLayout.from_config(config)
    table = run_gap_audit(config)
    path = layout.dataset_file("rgb_timestamp_gaps.csv")
    table.to_csv(path, index=False, encoding="utf-8-sig")

    if table.empty:
        return {
            "gap_rows": 0,
            "subjects_with_gaps": 0,
            "subjects_with_nonexcluded_gaps": 0,
            "max_gap_ms": None,
            "output": str(path),
        }

    nonexcluded = table[~table["analysis_excluded"].astype(bool)]
    return {
        "gap_rows": int(len(table)),
        "subjects_with_gaps": int(table["subject"].nunique()),
        "subjects_with_nonexcluded_gaps": int(nonexcluded["subject"].nunique()),
        "max_gap_ms": int(table["gap_duration_ms"].max()),
        "warning_threshold_ms": int(config.section("qc").get("timestamp_gap_warning_ms", 100)),
        "output": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention-Analysis RGB development pipeline")
    parser.add_argument("--config", default="configs/rgb_analysis.yaml")
    parser.add_argument("--stage", choices=["audit", "gaps"], default="audit")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.stage == "audit":
        result = stage_audit(config)
    else:
        result = stage_gaps(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[rgb:{args.stage}] complete", file=sys.stderr)


if __name__ == "__main__":
    main()
