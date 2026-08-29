"""Run the governed RGB formal downstream analysis without rerunning producers."""
from __future__ import annotations

import argparse
import json

from attention_pipeline.rgb_formal.runner import run_rgb_formal_v2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rgb_formal.yaml")
    parser.add_argument("--paths-config", default=None, help="Machine-local path registry; otherwise ATTENTION_ANALYSIS_PATHS_CONFIG is used.")
    parser.add_argument("--subjects", default=None, help="Optional comma-separated governed session subset for representative smoke.")
    args = parser.parse_args()
    subjects = [x.strip() for x in args.subjects.split(",") if x.strip()] if args.subjects else None
    result = run_rgb_formal_v2(args.config, paths_config=args.paths_config, subjects=subjects)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if not str(result.get("status", "")).startswith("blocked") else 2


if __name__ == "__main__":
    raise SystemExit(main())
