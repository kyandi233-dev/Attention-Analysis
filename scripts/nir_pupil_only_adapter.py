"""Portable CLI for the versioned fullclass-final pupil-only adapter."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from attention_pipeline.nir_pupil_only.adapter import (
    ADAPTER_VERSION,
    OUTPUT_SCHEMA_VERSION,
    adapt_session,
    attach_behavior_and_visual,
    file_sha256,
    load_json,
)


def _expand(value: str, config_dir: Path) -> Path:
    expanded = os.path.expandvars(value)
    path = Path(expanded)
    return path if path.is_absolute() else (config_dir / path).resolve()


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("adapter", {}).get("version") != ADAPTER_VERSION:
        raise ValueError("config adapter.version does not match installed adapter")
    return payload


def run(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    config_dir = config_path.parent
    behavior_path = _expand(config["inputs"]["behavior_trials_csv"], config_dir)
    visual_path = _expand(config["inputs"]["visual_properties_csv"], config_dir)
    behavior = pd.read_csv(behavior_path, low_memory=False)
    visual = pd.read_csv(visual_path, low_memory=False)
    outputs: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []

    for source in config["inputs"]["sources"]:
        manifest_path = _expand(source["manifest"], config_dir)
        manifest = load_json(manifest_path)
        eye_path = _expand(source.get("eye_metrics", manifest["source_path"]), config_dir)
        eyes = pd.read_csv(eye_path, low_memory=False)
        adapted = adapt_session(eyes, manifest, source_manifest_path=manifest_path)
        frames.append(attach_behavior_and_visual(adapted, behavior, visual))
        outputs.append(
            {
                "subject": manifest["subject"],
                "source_schema_version": manifest["source_schema_version"],
                "source_kind": manifest["source_kind"],
                "eye_metrics_path": str(eye_path),
                "eye_metrics_sha256": file_sha256(eye_path),
                "source_manifest_path": str(manifest_path),
                "source_manifest_sha256": file_sha256(manifest_path),
                "row_count": int(len(adapted)),
            }
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)
    output_root = _expand(config["output"]["root"], config_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_csv = output_root / config["output"].get("rows_csv", "pupil_only_rows.csv")
    combined.to_csv(output_csv, index=False, encoding="utf-8-sig")
    manifest = {
        "status": "validation_complete",
        "adapter_version": ADAPTER_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_evidence_commit": "171b081f3a3f9d06496c7b8d36915eebd4e2a3bb",
        "config_path": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "behavior_trials_path": str(behavior_path),
        "behavior_trials_sha256": file_sha256(behavior_path),
        "visual_properties_path": str(visual_path),
        "visual_properties_sha256": file_sha256(visual_path),
        "relative_luminance_semantics": (
            "linear-sRGB digital relative luminance; not physical cd/m²"
        ),
        "sources": outputs,
        "output": {
            "path": str(output_csv),
            "sha256": file_sha256(output_csv),
            "row_count": int(len(combined)),
        },
        "prohibitions": {
            "pir_or_oar_derived": False,
            "iris_fraction_used_as_geometry": False,
            "source_csv_modified": False,
            "interpolation_written_as_observation": False,
        },
    }
    manifest_path = output_root / config["output"].get("manifest_json", "manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
