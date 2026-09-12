"""Build the post-G1 Ocular scientific handoff from existing audit products."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from attention_pipeline.nir_formal_analysis.ocular_science_output import build_ocular_science_output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-probe-candidates", required=True)
    parser.add_argument("--science-root", required=True)
    parser.add_argument("--rgb-probe-features", default=None)
    parser.add_argument("--movement-artifact-audit", default=None)
    parser.add_argument("--temporal-support-summary", default=None)
    parser.add_argument("--non-authoritative", action="store_true")
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    manifest = build_ocular_science_output(
        Path(args.g1_probe_candidates),
        Path(args.science_root),
        rgb_probe_features_path=Path(args.rgb_probe_features) if args.rgb_probe_features else None,
        movement_artifact_audit_path=Path(args.movement_artifact_audit) if args.movement_artifact_audit else None,
        temporal_support_summary_path=Path(args.temporal_support_summary) if args.temporal_support_summary else None,
        authoritative=not args.non_authoritative,
        replace=not args.keep_existing,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
