"""Build the frozen post-G1 Ocular handoff after P3 measurement decisions."""
from __future__ import annotations

import argparse
from pathlib import Path

from attention_pipeline.nir_formal_analysis.ocular_science_coverage import refresh_ocular_coverage
from attention_pipeline.nir_formal_analysis.ocular_science_freeze import build_frozen_ocular_science_output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-probe-candidates", required=True)
    parser.add_argument("--science-root", required=True)
    parser.add_argument("--rgb-probe-features", required=True)
    parser.add_argument("--movement-artifact-audit", default=None)
    parser.add_argument("--temporal-support-summary", required=True)
    args = parser.parse_args()
    root = Path(args.science_root)
    build_frozen_ocular_science_output(
        Path(args.g1_probe_candidates),
        root,
        rgb_probe_features_path=Path(args.rgb_probe_features),
        movement_artifact_audit_path=(Path(args.movement_artifact_audit) if args.movement_artifact_audit else None),
        temporal_support_summary_path=Path(args.temporal_support_summary),
        authoritative=True,
        replace=True,
    )
    refresh_ocular_coverage(root / "Ocular")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
