"""Run the P5 Behavior + Ocular + Movement interface smoke without model fitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from attention_pipeline.multimodal_formal.p5_interface_smoke import run_p5_interface_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--science-root",
        required=True,
        help="Shared FormalScience root containing Behavior/, Ocular/, and Movement/.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional P5 output root. Default: <science-root>/P5_InterfaceSmoke.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Fail rather than replacing an existing P5_InterfaceSmoke output.",
    )
    args = parser.parse_args()
    manifest = run_p5_interface_smoke(
        Path(args.science_root),
        output_root=(Path(args.output_root) if args.output_root else None),
        replace=not args.keep_existing,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
