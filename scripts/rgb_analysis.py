"""RGB development pipeline entrypoint.

Current implemented stage:
    python scripts/rgb_analysis.py --stage audit

The audit stage does not run face, pose, or motion models. It only verifies
formal RGB files, frame timestamps, behavior files, and FocusWave timeline
coverage. Subjects configured under data.exclude remain visible in the audit
inventory for provenance, but are not counted as formally analysis-eligible.
"""
from __future__ import annotations

import argparse
import json
import sys

from attention_pipeline.config import load_config
from attention_pipeline.rgb.audit import run_audit
from attention_pipeline.rgb.paths import RGBOutputLayout


def stage_audit(config) -> dict[str, object]:
    layout = RGBOutputLayout.from_config(config)
    inventory, duplicates = run_audit(config)

    # Dataset-level audit files live directly under Beijing-RGB. There is no
    # extra audit/ wrapper directory.
    inventory_path = layout.dataset_file("rgb_inventory.csv")
    duplicate_path = layout.dataset_file("rgb_duplicate_subjects.csv")
    summary_path = layout.dataset_file("audit_summary.json")

    inventory.to_csv(inventory_path, index=False, encoding="utf-8-sig")
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
        "duplicates": str(duplicate_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention-Analysis RGB development pipeline")
    parser.add_argument("--config", default="configs/rgb_analysis.yaml")
    parser.add_argument("--stage", choices=["audit"], default="audit")
    args = parser.parse_args()

    config = load_config(args.config)
    result = stage_audit(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[rgb:{args.stage}] complete", file=sys.stderr)


if __name__ == "__main__":
    main()
