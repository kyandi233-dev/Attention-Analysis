from __future__ import annotations

import argparse
import json
from pathlib import Path

from attention_pipeline.config import load_config
from attention_pipeline.supervised_learning.time_legality import time_legality_audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the FocusWave frozen feature registry time-legality contract "
            "before any real Task-B/Task-A supervised run."
        )
    )
    parser.add_argument("--config", default="configs/supervised_learning_v1.yaml")
    parser.add_argument("--paths-config", default=None)
    parser.add_argument("--output", default=None, help="Optional JSON audit output path")
    args = parser.parse_args()

    config = load_config(args.config, paths_config=args.paths_config)
    section = config.data.get("feature_registry", {})
    audit = time_legality_audit(section)
    payload = json.dumps(audit, ensure_ascii=False, indent=2)
    print(payload)

    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
