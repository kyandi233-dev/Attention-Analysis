"""Formal RGB session audit with sampled decode and behavior-file row evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd

from attention_pipeline.config import load_config
from attention_pipeline.rgb.audit import audit_subject
from attention_pipeline.rgb.discover import discover_rgb_subjects
from attention_pipeline.rgb.paths import RGBOutputLayout


def _sample_decode(path: Path, nominal_count: int) -> dict[str, object]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"decode_sample_ok": False, "decode_sample_positions": [], "decode_sample_successes": 0, "decode_error": "open_failed"}
    positions = sorted({0, max(0, nominal_count // 2), max(0, nominal_count - 1)})
    successes = 0
    failures: list[int] = []
    try:
        for position in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = cap.read()
            if ok and frame is not None and getattr(frame, "size", 0) > 0:
                successes += 1
            else:
                failures.append(position)
    finally:
        cap.release()
    return {
        "decode_sample_ok": not failures,
        "decode_sample_positions": positions,
        "decode_sample_successes": successes,
        "decode_sample_failures": failures,
        "decode_error": "" if not failures else "sample_read_failed",
    }


def _csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return max(0, sum(1 for _ in path.open("r", encoding="utf-8-sig", newline="")) - 1)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit accessible formal RGB sessions on J:\\Data")
    parser.add_argument("--config", default="configs/rgb_analysis.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    layout = RGBOutputLayout.from_config(config)
    records, duplicates = discover_rgb_subjects(config)
    rows: list[dict[str, object]] = []
    for record in records:
        row = audit_subject(record, config)
        nominal = int(row.get("video_frame_count_nominal") or 0)
        row.update(_sample_decode(record.video, nominal))
        row["block1_behavior_rows"] = _csv_rows(record.block1_behavior) if record.block1_behavior else None
        row["block2_behavior_rows"] = _csv_rows(record.block2_behavior) if record.block2_behavior else None
        row["formal_usable"] = bool(row.get("analysis_eligible") and row.get("decode_sample_ok"))
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)
    inventory_path = layout.dataset_file("rgb_formal_audit_v1.csv")
    summary_path = layout.dataset_file("rgb_formal_audit_v1_summary.json")
    table.to_csv(inventory_path, index=False, encoding="utf-8-sig")
    summary = {
        "schema_version": "rgb-formal-audit-v1",
        "data_root": "J:/Data",
        "sessions_discovered": int(len(table)),
        "basic_complete": int(table["basic_complete"].sum()),
        "decode_sample_ok": int(table["decode_sample_ok"].sum()),
        "formal_usable": int(table["formal_usable"].sum()),
        "not_formal_usable": table.loc[~table["formal_usable"], "subject"].tolist(),
        "duplicate_subjects": sorted(duplicates),
        "output": str(inventory_path),
        "decode_policy": "first/middle/last decoded frames; exact full decode remains a runner-level counter",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
