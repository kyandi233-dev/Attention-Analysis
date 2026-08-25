"""RGB development pipeline entrypoint.

Current implemented stage:
    python scripts/rgb_analysis.py --stage audit

The audit stage does not run face, pose, or motion models. It only verifies
formal RGB files, frame timestamps, behavior files, and FocusWave timeline
coverage.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from attention_pipeline.config import load_config
from attention_pipeline.rgb.audit import run_audit


def _output_root(config) -> Path:
    raw = config.section("output").get("root", "outputs/rgb-dev")
    path = Path(str(raw))
    if not path.is_absolute():
        path = (config.path.parent.parent / path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def stage_audit(config) -> dict[str, object]:
    root = _output_root(config) / "audit"
    root.mkdir(parents=True, exist_ok=True)
    inventory, duplicates = run_audit(config)
    inventory_path = root / "rgb_inventory.csv"
    duplicate_path = root / "rgb_duplicate_subjects.csv"
    inventory.to_csv(inventory_path, index=False, encoding="utf-8-sig")
    duplicates.to_csv(duplicate_path, index=False, encoding="utf-8-sig")

    complete = int(inventory["basic_complete"].sum()) if "basic_complete" in inventory else 0
    summary = {
        "subjects_discovered_unique": int(len(inventory)),
        "subjects_basic_complete": complete,
        "duplicate_subjects": int(len(duplicates)),
        "inventory": str(inventory_path),
        "duplicates": str(duplicate_path),
    }
    (root / "audit_summary.json").write_text(
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
