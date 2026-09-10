"""Materialize the Task C Behavior -> B/A 30-second supervised interface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from attention_pipeline.config import load_config
from attention_pipeline.behavior_formal.behavior_supervised_interface import (
    materialize_behavior_supervised_interface,
)


SOURCE_RUN_MANIFEST_NAME = "run_manifest.json"


def _verified_source_config_digest(
    source: Path,
    *,
    current_config_digest: str,
    source_manifest_path: Path | None = None,
) -> str:
    """Verify that the authoritative probe table belongs to the loaded science config."""
    manifest_path = (
        source_manifest_path.resolve()
        if source_manifest_path is not None
        else (source.parent / SOURCE_RUN_MANIFEST_NAME).resolve()
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Behavior supervised handoff requires the source formal run manifest for provenance: "
            f"{manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Behavior source run manifest is unreadable: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Behavior source run manifest root must be a JSON object")
    source_digest = str(payload.get("config_digest", "")).strip()
    if not source_digest:
        raise ValueError("Behavior source run manifest is missing config_digest")
    if source_digest != str(current_config_digest).strip():
        raise ValueError(
            "Behavior source/config provenance mismatch: probe_primary_30s.csv was not produced "
            f"under the currently loaded science config (source={source_digest}, current={current_config_digest})"
        )
    return source_digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/behavior_formal_v2.yaml")
    parser.add_argument("--paths-config", default=None)
    parser.add_argument("--input", default=None, help="Optional explicit probe_primary_30s.csv path")
    parser.add_argument(
        "--source-run-manifest",
        default=None,
        help="Optional explicit source run_manifest.json; defaults to the probe table directory",
    )
    parser.add_argument("--output-root", default=None, help="Optional explicit derived interface directory")
    parser.add_argument("--force", action="store_true", help="Replace only the derived supervised_interface_v1 directory")
    args = parser.parse_args()

    config = load_config(args.config, paths_config=args.paths_config)
    behavior_root = config.path_value("output_root")
    source = Path(args.input).resolve() if args.input else behavior_root / "formal_v3" / "probe_primary_30s.csv"
    source_manifest = Path(args.source_run_manifest).resolve() if args.source_run_manifest else None
    verified_source_config_digest = _verified_source_config_digest(
        source,
        current_config_digest=config.digest,
        source_manifest_path=source_manifest,
    )
    output = Path(args.output_root).resolve() if args.output_root else behavior_root / "supervised_interface_v1"
    manifest = materialize_behavior_supervised_interface(
        source,
        output,
        config_digest=verified_source_config_digest,
        force=bool(args.force),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
