"""Build the frozen post-G1 Ocular handoff after P3 measurement decisions."""
from __future__ import annotations

import argparse
from pathlib import Path

from attention_pipeline.nir_formal_analysis.ocular_freeze_evidence import (
    archive_ocular_freeze_evidence,
)
from attention_pipeline.nir_formal_analysis.ocular_probe_identity import (
    ensure_ocular_canonical_probe_identity,
)
from attention_pipeline.nir_formal_analysis.ocular_science_coverage import (
    refresh_ocular_coverage,
)
from attention_pipeline.nir_formal_analysis.ocular_science_freeze import (
    build_frozen_ocular_science_output,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-probe-candidates", required=True)
    parser.add_argument("--science-root", required=True)
    parser.add_argument("--rgb-probe-features", required=True)
    parser.add_argument("--movement-artifact-audit", default=None)
    parser.add_argument("--temporal-support-summary", required=True)
    parser.add_argument("--cross-signal-summary", required=True)
    parser.add_argument("--sync-semantics-summary", required=True)
    parser.add_argument("--source-mode-summary", required=True)
    args = parser.parse_args()

    root = Path(args.science_root)
    build_frozen_ocular_science_output(
        Path(args.g1_probe_candidates),
        root,
        rgb_probe_features_path=Path(args.rgb_probe_features),
        movement_artifact_audit_path=(
            Path(args.movement_artifact_audit) if args.movement_artifact_audit else None
        ),
        temporal_support_summary_path=Path(args.temporal_support_summary),
        authoritative=True,
        replace=True,
    )
    ocular_root = root / "Ocular"
    ensure_ocular_canonical_probe_identity(
        Path(args.g1_probe_candidates),
        ocular_root / "tables/ocular_probe_features_wide.csv",
        supplemental_probe_identity_path=Path(args.rgb_probe_features),
    )
    refresh_ocular_coverage(ocular_root)
    archive_ocular_freeze_evidence(
        ocular_root,
        temporal_support=Path(args.temporal_support_summary),
        cross_signal_representation=Path(args.cross_signal_summary),
        sync_semantics=Path(args.sync_semantics_summary),
        source_mode_limit=Path(args.source_mode_summary),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
