"""Build downstream NIR × FocusWave v3.1.3 formal-SART alignment tables.

Prototype:
    PYTHONPATH=src python scripts/nir_behavior_alignment.py --subjects sub-031

The script reads frozen behavior/full-class NIR artifacts and writes only to the
external analysis output root configured in configs/nir_behavior_alignment.yaml.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from attention_pipeline.config import load_config
from attention_pipeline.nir_behavior.alignment_v12 import run_subject_alignment_v12
from attention_pipeline.nir_behavior.discovery import (
    alignment_output_root,
    selected_subjects,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NIR × formal-SART alignment")
    parser.add_argument("--config", default="configs/nir_behavior_alignment.yaml")
    parser.add_argument(
        "--subjects",
        help="Optional comma-separated subject override, e.g. sub-031,sub-033",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-diagnostics", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    override = (
        [item.strip() for item in args.subjects.split(",") if item.strip()]
        if args.subjects
        else None
    )
    subjects = selected_subjects(config, override)
    if not subjects:
        raise RuntimeError("No completed full-class NIR subjects selected")

    results = []
    for index, subject in enumerate(subjects, start=1):
        print(f"[ALIGN {index}/{len(subjects)}] {subject}")
        result = run_subject_alignment_v12(
            config,
            subject,
            force=bool(args.force),
            make_diagnostics=not bool(args.no_diagnostics),
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, default=str))

    root = alignment_output_root(config)
    root.mkdir(parents=True, exist_ok=True)
    cohort_manifest = {
        "config": str(Path(args.config).resolve()),
        "config_digest": config.digest,
        "subjects": subjects,
        "results": results,
    }
    (root / "cohort_manifest.json").write_text(
        json.dumps(cohort_manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return 1 if any(item.get("status") == "failed" for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
