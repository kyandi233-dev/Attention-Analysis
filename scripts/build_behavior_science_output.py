"""Build the compact Behavior scientific-output package from an existing formal-v3 run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from attention_pipeline.behavior_formal.science_output import build_behavior_science_output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", required=True, help="Existing Behavior formal_v3 output directory.")
    parser.add_argument(
        "--science-root",
        default=None,
        help="Shared science-output root. Default: <Behavior parent>/../FormalScience.",
    )
    parser.add_argument("--non-authoritative", action="store_true", help="Mark subset/smoke output as non-authoritative.")
    parser.add_argument("--keep-existing", action="store_true", help="Do not replace an existing FormalScience/Behavior directory.")
    args = parser.parse_args()

    formal_root = Path(args.formal_root).expanduser().resolve()
    science_root = (
        Path(args.science_root).expanduser().resolve()
        if args.science_root
        else formal_root.parent.parent / "FormalScience"
    )
    manifest = build_behavior_science_output(
        formal_root,
        science_root,
        authoritative=not args.non_authoritative,
        replace=not args.keep_existing,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
