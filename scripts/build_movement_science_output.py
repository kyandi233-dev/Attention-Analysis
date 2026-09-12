"""Build P4 Movement scientific outputs from an existing completed RGB 5.5 run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from attention_pipeline.rgb_formal.movement_science_output import build_movement_science_output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rgb55-root",
        required=True,
        help="Existing RGB 5.5 output root containing tables/ and models/. Does not rerun video analysis.",
    )
    parser.add_argument(
        "--science-root",
        required=True,
        help="Shared FormalScience root; Movement/ is created underneath it.",
    )
    parser.add_argument(
        "--g1-probe-candidates",
        default=None,
        help=(
            "Optional local authoritative NIR G1 probe_measurement_candidates.csv. "
            "When supplied, P4 also writes the Ocular x Movement artifact-sensitivity audit."
        ),
    )
    parser.add_argument(
        "--non-authoritative",
        action="store_true",
        help="Mark subset/smoke output as non-authoritative.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Fail instead of replacing an existing FormalScience/Movement directory.",
    )
    args = parser.parse_args()

    manifest = build_movement_science_output(
        Path(args.rgb55_root),
        Path(args.science_root),
        g1_probe_candidates_path=(Path(args.g1_probe_candidates) if args.g1_probe_candidates else None),
        authoritative=not args.non_authoritative,
        replace=not args.keep_existing,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
