"""RGB development pipeline entrypoint."""
from __future__ import annotations

import argparse
import json
import sys

from attention_pipeline.config import load_config
from attention_pipeline.rgb.audit import run_audit
from attention_pipeline.rgb.face_benchmark import run_face_benchmark_sample
from attention_pipeline.rgb.face_libreface_qc import run_libreface_benchmark_qc
from attention_pipeline.rgb.face_pyfeat_qc import run_pyfeat_benchmark_qc
from attention_pipeline.rgb.gaps import run_gap_audit
from attention_pipeline.rgb.motion import run_motion_test
from attention_pipeline.rgb.motion_qc import run_motion_qc
from attention_pipeline.rgb.motion_review import run_motion_review
from attention_pipeline.rgb.pose import run_pose_test
from attention_pipeline.rgb.pose_qc import run_pose_qc
from attention_pipeline.rgb.pose_features import run_pose_features
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
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
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
            "gap_rows_inside_analysis_span": 0,
            "subjects_with_inside_analysis_span_gaps": 0,
            "gap_rows_inside_formal_block": 0,
            "subjects_with_inside_formal_block_gaps": 0,
            "max_gap_ms": None,
            "max_formal_block_gap_ms": None,
            "output": str(path),
        }
    nonexcluded = table[~table["analysis_excluded"].astype(bool)]
    inside_span = nonexcluded[nonexcluded["inside_analysis_span"].astype(bool)]
    inside_block = nonexcluded[nonexcluded["inside_formal_block"].astype(bool)]
    return {
        "gap_rows": int(len(table)),
        "subjects_with_gaps": int(table["subject"].nunique()),
        "subjects_with_nonexcluded_gaps": int(nonexcluded["subject"].nunique()),
        "gap_rows_inside_analysis_span": int(len(inside_span)),
        "subjects_with_inside_analysis_span_gaps": int(inside_span["subject"].nunique()),
        "gap_rows_inside_formal_block": int(len(inside_block)),
        "subjects_with_inside_formal_block_gaps": int(inside_block["subject"].nunique()),
        "max_gap_ms": int(table["gap_duration_ms"].max()),
        "max_formal_block_gap_ms": int(inside_block["gap_duration_ms"].max()) if not inside_block.empty else None,
        "warning_threshold_ms": int(config.section("qc").get("timestamp_gap_warning_ms", 100)),
        "output": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention-Analysis RGB development pipeline")
    parser.add_argument("--config", default="configs/rgb_analysis.yaml")
    parser.add_argument(
        "--stage",
        choices=[
            "audit", "gaps", "motion", "motion-qc", "motion-review",
            "pose", "pose-qc", "pose-features", "face-sample",
            "face-pyfeat-qc", "face-libreface-qc",
        ],
        default="audit",
    )
    parser.add_argument("--subject", help="Required for single-subject model/QC stages")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.stage == "audit":
        result = stage_audit(config)
    elif args.stage == "gaps":
        result = stage_gaps(config)
    else:
        if not args.subject:
            parser.error(f"--subject is required when --stage {args.stage}")
        if args.stage == "motion":
            result = run_motion_test(config, args.subject)
        elif args.stage == "motion-qc":
            result = run_motion_qc(config, args.subject)
        elif args.stage == "motion-review":
            result = run_motion_review(config, args.subject)
        elif args.stage == "pose":
            result = run_pose_test(config, args.subject)
        elif args.stage == "pose-qc":
            result = run_pose_qc(config, args.subject)
        elif args.stage == "pose-features":
            result = run_pose_features(config, args.subject)
        elif args.stage == "face-sample":
            result = run_face_benchmark_sample(config, args.subject)
        elif args.stage == "face-pyfeat-qc":
            result = run_pyfeat_benchmark_qc(config, args.subject)
        else:
            result = run_libreface_benchmark_qc(config, args.subject)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[rgb:{args.stage}] complete", file=sys.stderr)


if __name__ == "__main__":
    main()
