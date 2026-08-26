from __future__ import annotations

import argparse
import json

from attention_pipeline.config import load_config
from attention_pipeline.rgb.face_formal_dryrun_sample import run_face_formal_dryrun_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract timestamp-driven formal Face dry-run windows")
    parser.add_argument("--config", default="configs/rgb_analysis.yaml")
    parser.add_argument("--subject", required=True)
    args = parser.parse_args()
    result = run_face_formal_dryrun_sample(load_config(args.config), args.subject)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
