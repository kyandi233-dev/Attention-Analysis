"""Run frozen-post Ocular explanatory analyses from already-materialized science tables."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from attention_pipeline.nir_formal_analysis.ocular_postfreeze_analysis import (
    AnalysisInputs,
    build_ocular_postfreeze_science,
)

FAILURE_COLUMNS = (
    "analysis_family",
    "feature_id",
    "outcome",
    "model_family",
    "status",
    "reason",
    "n_rows",
    "participant_group_n",
    "session_n",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_empty_failure_schema(ocular_root: Path, manifest: dict) -> None:
    """Keep model_failures.csv machine-readable even when no model failed."""
    raw_path = manifest.get("outputs", {}).get("model_failures")
    if not raw_path:
        return
    path = Path(str(raw_path))
    if not path.is_file():
        return
    if path.read_text(encoding="utf-8-sig").strip():
        return
    path.write_text(",".join(FAILURE_COLUMNS) + "\n", encoding="utf-8-sig")

    # The builder creates run_manifest before this no-failure schema hardening;
    # refresh the recorded checksum so provenance remains internally consistent.
    run_manifest_path = ocular_root / "manifests" / "run_manifest.json"
    if run_manifest_path.is_file():
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        rel = str(path.relative_to(ocular_root))
        run_manifest.setdefault("output_sha256", {})[rel] = _sha256(path)
        run_manifest_path.write_text(
            json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
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

    ocular_root = Path(args.ocular_root).expanduser().resolve()
    manifest = build_ocular_postfreeze_science(
        AnalysisInputs(
            ocular_wide=Path(args.ocular_wide).expanduser().resolve(),
            behavior_probe=Path(args.behavior_probe).expanduser().resolve(),
            ocular_root=ocular_root,
            analysis_code_sha=str(args.analysis_code_sha),
            authoritative=not args.non_authoritative,
        ),
        make_figures=not args.no_figures,
    )
    _ensure_empty_failure_schema(ocular_root, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0 if str(manifest.get("status", "")).startswith("complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
