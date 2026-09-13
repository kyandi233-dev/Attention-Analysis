"""Rebuild compact Behavior scientific outputs from existing formal_v3 tables."""
from __future__ import annotations

import argparse
import json

from attention_pipeline.config import load_config
from attention_pipeline.behavior_formal.science_output import build_behavior_science_output
from attention_pipeline.behavior_formal.publication_reporting import redraw_behavior_publication


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/behavior_formal_v2.yaml")
    parser.add_argument("--paths-config", default=None)
    parser.add_argument(
        "--legacy-publication-report",
        action="store_true",
        help="Reproduce the historical publication-report package instead of the current compact science layer.",
    )
    args = parser.parse_args()
    config = load_config(args.config, paths_config=args.paths_config)
    behavior_root = config.path_value("output_root")
    formal_root = behavior_root / "formal_v3"
    if args.legacy_publication_report:
        result = redraw_behavior_publication(formal_root)
    else:
        result = build_behavior_science_output(
            formal_root,
            behavior_root.parent / "FormalScience",
            authoritative=True,
            replace=True,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
