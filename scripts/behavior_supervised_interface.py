"""Materialize the Task C Behavior -> B/A 30-second supervised interface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from attention_pipeline.config import load_config
from attention_pipeline.behavior_formal.behavior_supervised_interface import (
    materialize_behavior_supervised_interface,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/behavior_formal_v2.yaml")
    parser.add_argument("--paths-config", default=None)
    parser.add_argument("--input", default=None, help="Optional explicit probe_primary_30s.csv path")
    parser.add_argument("--output-root", default=None, help="Optional explicit derived interface directory")
    parser.add_argument("--force", action="store_true", help="Replace only the derived supervised_interface_v1 directory")
    args = parser.parse_args()

    config = load_config(args.config, paths_config=args.paths_config)
    behavior_root = config.path_value("output_root")
    source = Path(args.input).resolve() if args.input else behavior_root / "formal_v3" / "probe_primary_30s.csv"
    output = Path(args.output_root).resolve() if args.output_root else behavior_root / "supervised_interface_v1"
    manifest = materialize_behavior_supervised_interface(
        source,
        output,
        config_digest=config.digest,
        force=bool(args.force),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
