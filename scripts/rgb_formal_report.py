"""Build the RGB lightweight-route result report and publication/QC figures without rerunning producers."""
from __future__ import annotations

import argparse
import json

from attention_pipeline.config import load_config
from attention_pipeline.rgb_formal.reporting import build_rgb_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rgb_formal.yaml")
    parser.add_argument("--paths-config", default=None)
    args = parser.parse_args()
    config = load_config(args.config, paths_config=args.paths_config)
    result = build_rgb_report(config.path_value("output_root"), config.path_value("analysis_ready_root"))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
