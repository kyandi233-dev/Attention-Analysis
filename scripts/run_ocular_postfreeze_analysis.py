"""Run frozen-post Ocular explanatory analyses from already-materialized science tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from attention_pipeline.nir_formal_analysis.ocular_postfreeze_analysis import (
    AnalysisInputs,
    build_ocular_postfreeze_science,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocular-wide", required=True, help="Frozen FormalScience/Ocular ocular_probe_features_wide.csv")
    parser.add_argument("--behavior-probe", required=True, help="Behavior formal_v3 probe_primary_30s.csv")
    parser.add_argument("--ocular-root", required=True, help="Existing FormalScience/Ocular directory; P3 assets are preserved")
    parser.add_argument("--analysis-code-sha", required=True, help="Exact Attention-Analysis commit SHA used for the run")
    parser.add_argument("--non-authoritative", action="store_true", help="Allow subset/smoke key universes")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    manifest = build_ocular_postfreeze_science(
        AnalysisInputs(
            ocular_wide=Path(args.ocular_wide).expanduser().resolve(),
            behavior_probe=Path(args.behavior_probe).expanduser().resolve(),
            ocular_root=Path(args.ocular_root).expanduser().resolve(),
            analysis_code_sha=str(args.analysis_code_sha),
            authoritative=not args.non_authoritative,
        ),
        make_figures=not args.no_figures,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0 if str(manifest.get("status", "")).startswith("complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
