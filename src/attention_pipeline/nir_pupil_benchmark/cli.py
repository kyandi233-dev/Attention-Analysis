"""CLI for the seven-classical-algorithm native-resolution pupil benchmark.

Minimal usage (synthetic smoke):

    python -m attention_pipeline.nir_pupil_benchmark.cli \\
        --smoke --out smoke_results.csv --overlay-dir smoke_overlays

Manifest usage:

    python -m attention_pipeline.nir_pupil_benchmark.cli \\
        --manifest manifest.csv --crop-root crops --out results.csv \\
        --algorithms PuRe,PuReST,PupilLabs2D,ElSe,ExCuSe,Swirski2D,Starburst \\
        --run-confidence --mode independent
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

from .overlay import write_algorithm_montage
from .runner import ALGORITHM_SPECS, run_crop_list, scale_params
from .schema import ALGORITHMS, RESULT_COLUMNS
from .synthetic import write_smoke_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nir-pupil-benchmark",
        description=(
            "Compare the seven classical pupil detectors on native-resolution "
            "eye crops with a unified schema (docs/020-nir/030)."
        ),
    )
    parser.add_argument(
        "--algorithms",
        default=",".join(ALGORITHMS),
        help="comma-separated algorithm list (default: all seven)",
    )
    parser.add_argument(
        "--manifest",
        help="CSV manifest with crop_path plus identity columns",
    )
    parser.add_argument(
        "--crop-root",
        help="directory containing the crop images referenced by crop_path",
    )
    parser.add_argument("--out", required=True, help="output CSV path")
    parser.add_argument(
        "--run-confidence",
        action="store_true",
        help="also run the runWithConfidence pass and record outline_confidence "
             "with its own timing (separate from main runtime_ms)",
    )
    parser.add_argument(
        "--mode",
        choices=["independent", "continuous"],
        default="independent",
        help="independent = fresh detector per frame; continuous = shared detector "
             "per (subject, eye) processed in frame order",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="generate synthetic eye crops and run on them",
    )
    parser.add_argument("--smoke-n", type=int, default=4, help="synthetic frames for --smoke")
    parser.add_argument(
        "--overlay-dir",
        help="write per-algorithm ellipse overlay montages for manual QC",
    )
    parser.add_argument("--dry-import", action="store_true", help=argparse.SUPPRESS)
    return parser


def _resolve_algorithms(raw: str) -> list[str]:
    chosen = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = [name for name in chosen if name not in ALGORITHM_SPECS]
    if unknown:
        raise SystemExit(f"unknown algorithms: {', '.join(unknown)}; available: {', '.join(ALGORITHMS)}")
    return chosen


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    algorithms = _resolve_algorithms(args.algorithms)

    if args.smoke:
        work = Path(tempfile.mkdtemp(prefix="nir_benchmark_smoke_"))
        write_smoke_manifest(work, n_frames=args.smoke_n)
        manifest = work / "manifest.csv"
        crop_root = work
    else:
        if not args.manifest or not args.crop_root:
            build_parser().error("either --smoke or --manifest + --crop-root is required")
        manifest = Path(args.manifest)
        crop_root = Path(args.crop_root)

    rows = pd.read_csv(manifest).to_dict("records")
    result = run_crop_list(
        rows,
        algorithms,
        crop_root=crop_root,
        run_confidence=args.run_confidence,
        mode=args.mode,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, encoding="utf-8-sig")

    if args.overlay_dir:
        write_algorithm_montage(
            result,
            crop_root,
            args.overlay_dir,
            algorithms=algorithms,
        )

    summary = result.groupby("algorithm").agg(
        n_frames=("frame_idx", "count"),
        returned=("algorithm_returned", "sum"),
        official_valid=("official_valid", "sum"),
        geometry_sane=("geometry_sane", "sum"),
    ).reset_index()
    print(summary.to_string(index=False))
    print(f"\nwrote {out}")
    if args.overlay_dir:
        print(f"overlays -> {args.overlay_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
