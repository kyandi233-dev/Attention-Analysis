from __future__ import annotations

import argparse
import json

from attention_pipeline.config import load_config
from attention_pipeline.rgb.face_formal import run_face_formal_prepare


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare full formal timestamp-driven Face frame manifest"
    )
    parser.add_argument("--config", default="configs/rgb_analysis.yaml")
    parser.add_argument("--subject", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_face_formal_prepare(config, args.subject)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[rgb:face-formal-prepare] complete {args.subject}")


if __name__ == "__main__":
    main()
