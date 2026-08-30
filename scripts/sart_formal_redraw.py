"""Redraw publication Behavior figures and rebuild the complete report from existing formal_v3 tables."""
from __future__ import annotations

import argparse
import json

from attention_pipeline.config import load_config
from attention_pipeline.behavior_formal.publication_reporting import redraw_behavior_publication


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/behavior_formal_v2.yaml")
    parser.add_argument("--paths-config", default=None)
    args = parser.parse_args()
    config = load_config(args.config, paths_config=args.paths_config)
    formal_root = config.path_value("output_root") / "formal_v3"
    result = redraw_behavior_publication(formal_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
