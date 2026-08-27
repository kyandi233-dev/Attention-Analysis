"""CLI for the production-evidence seven-algorithm benchmark."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from .formal import (
    algorithm_pairwise_summary,
    atomic_write_csv,
    atomic_write_json,
    build_sample_plan,
    discover_production_run,
    execute_manifest,
    load_config,
    load_production_eyes,
    make_sample_manifest,
    materialize_crops,
    production_provenance,
    sha256_file,
    subject_algorithm_summary,
    temporal_summary,
    validate_result_contract,
    write_manual_qc_montages,
)
from .runner import VideoFrameSource
from .schema import ALGORITHMS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nir-pupil-benchmark-formal",
        description="Formal production eyes.csv -> native crop -> seven-algorithm benchmark",
    )
    parser.add_argument("--config", default="configs/nir_pypupilext_native_benchmark.yaml")
    parser.add_argument("--stage", choices=["plan", "prepare", "run", "all", "validate"], default="plan")
    parser.add_argument("--profile", choices=["smoke", "formal"], default="smoke")
    parser.add_argument("--subjects", help="comma-separated subjects; default from config")
    parser.add_argument("--algorithms", default=",".join(ALGORITHMS))
    parser.add_argument("--run-dir", help="required for prepare/run/all/validate")
    parser.add_argument("--run-confidence", action="store_true")
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="do not materialize crops to disk; decode source-video frames in "
             "memory during execution (manual QC montages are still written)",
    )
    parser.add_argument(
        "--approve-multi-subject",
        action="store_true",
        help="required when a writing stage targets more than one subject",
    )
    return parser


def _subjects(config: dict, raw: str | None) -> list[str]:
    values = raw.split(",") if raw else config["subjects"]["include"]
    from .core import normalize_subject

    subjects = [normalize_subject(value.strip()) for value in values if str(value).strip()]
    if not subjects:
        raise ValueError("no subjects selected")
    return subjects


def _algorithms(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    unknown = [value for value in values if value not in ALGORITHMS]
    if unknown:
        raise ValueError(f"unknown algorithms: {unknown}")
    return values


def _sampling(config: dict, profile: str) -> dict:
    source = config["smoke"] if profile == "smoke" else config["sampling"]
    return {
        "block_uniform_n": int(source["block_uniform_n"]),
        "ritnet_high_quality_n": int(source["ritnet_high_quality_n"]),
        "ritnet_difficult_n": int(source["ritnet_difficult_n"]),
        "temporal_n": int(source["temporal_n"]),
        "temporal_preferred_phase": str(config["sampling"]["temporal_preferred_phase"]),
        "full_video": bool(source.get("full_video", False)),
    }


def _resolve_run(config: dict, subject: str):
    production = config["production"]
    return discover_production_run(
        config["paths"]["production_root"],
        subject,
        required_status=production["required_completion_status"],
        preferred_name_token=production["prefer_run_name_contains"],
        required_files=production["required_files"],
    )


def _plan_one(config: dict, profile: str, subject: str):
    run = _resolve_run(config, subject)
    eyes = load_production_eyes(run)
    tight, temporal = build_sample_plan(eyes, **_sampling(config, profile))
    return run, eyes, tight, temporal


def _require_run_dir(args) -> Path:
    if not args.run_dir:
        raise ValueError(f"--run-dir is required for --stage {args.stage}")
    return Path(args.run_dir).resolve()


def _prepare(config_path: Path, config: dict, args, subjects: list[str], run_dir: Path) -> pd.DataFrame:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    provenance = []
    analysis_size = tuple(map(int, config["production"]["ritnet_analysis_size"]))
    for subject in subjects:
        run, eyes, tight, temporal = _plan_one(config, args.profile, subject)
        manifest = make_sample_manifest(
            run,
            eyes,
            tight,
            temporal,
            crop_root=run_dir,
            analysis_size=analysis_size,
            min_crop_width=int(config["input"]["min_crop_width"]),
            min_crop_height=int(config["input"]["min_crop_height"]),
        )
        if getattr(args, "in_memory", False):
            # Defer decoding to execution; no PNG crops are written to disk.
            manifest["input_status"] = manifest["input_status"].where(
                manifest["input_status"] != "pending", "ready"
            )
        else:
            manifest = materialize_crops(manifest, run_dir)
        atomic_write_csv(manifest, run_dir / "subjects" / subject / "sample_manifest.csv")
        manifests.append(manifest)
        provenance.append(
            production_provenance(
                run,
                config_path,
                bool(config["production"].get("hash_evidence_files", True)),
            )
        )
    combined = pd.concat(manifests, ignore_index=True)
    atomic_write_csv(combined, run_dir / "sample_manifest.csv")
    atomic_write_json(
        {
            "pipeline": config["pipeline"],
            "profile": args.profile,
            "subjects": subjects,
            "sampling": _sampling(config, args.profile),
            "production": provenance,
            "manifest_rows": int(len(combined)),
            "ready_rows": int(combined["input_status"].eq("ready").sum()),
        },
        run_dir / "prepare_manifest.json",
    )
    return combined


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _run(config: dict, args, algorithms: list[str], run_dir: Path) -> dict:
    manifest_path = run_dir / "sample_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing prepared manifest: {manifest_path}")
    if (run_dir / "frame_results.csv").exists():
        raise FileExistsError(f"refusing to overwrite existing results in {run_dir}")
    manifest = pd.read_csv(manifest_path, low_memory=False)
    image_source = "video" if args.in_memory else "disk"
    frame_source = VideoFrameSource() if args.in_memory else None
    try:
        results = execute_manifest(
            manifest,
            algorithms,
            run_dir=run_dir,
            run_confidence=args.run_confidence,
            image_source=image_source,
        )
        checks = validate_result_contract(manifest, results, algorithms)
        atomic_write_csv(results, run_dir / "frame_results.csv")
        _write_parquet_atomic(results, run_dir / "frame_results.parquet")
        summary = subject_algorithm_summary(results)
        pairwise = algorithm_pairwise_summary(results)
        temporal = temporal_summary(results)
        atomic_write_csv(summary, run_dir / "subject_algorithm_summary.csv")
        atomic_write_csv(pairwise, run_dir / "algorithm_pairwise_summary.csv")
        atomic_write_csv(temporal, run_dir / "temporal_window_summary.csv")
        manual_n = int(
            config["smoke"]["manual_qc_n"] if args.profile == "smoke"
            else config["sampling"]["manual_qc_n"]
        )
        manual = write_manual_qc_montages(
            results, run_dir, n_frames_per_subject=manual_n, frame_source=frame_source
        )
        atomic_write_csv(manual, run_dir / "manual_qc_labels.csv")
        atomic_write_json(checks, run_dir / "validation_summary.json")
        montage_paths = sorted((run_dir / "manual_qc").glob("**/*.png"))
        expected_montages = manual_n * int(results["subject"].nunique())
        if len(montage_paths) != expected_montages:
            raise AssertionError(
                f"manual QC montage count mismatch: expected {expected_montages}, "
                f"found {len(montage_paths)}"
            )
        completion = {
            "status": "complete",
            "pipeline": config["pipeline"],
            "profile": args.profile,
            "image_source": image_source,
            "subjects": sorted(results["subject"].dropna().unique().tolist()),
            "algorithms": algorithms,
            "result_rows": int(len(results)),
            "manual_qc_rows": int(len(manual)),
            "manual_qc_montage_count": int(len(montage_paths)),
            "manual_qc_montage_sha256": {
                path.relative_to(run_dir).as_posix(): sha256_file(path)
                for path in montage_paths
            },
            "validation": checks,
            "artifacts_sha256": {
                name: sha256_file(run_dir / name)
                for name in (
                    "sample_manifest.csv", "frame_results.csv", "frame_results.parquet",
                    "subject_algorithm_summary.csv", "algorithm_pairwise_summary.csv",
                    "temporal_window_summary.csv",
                    "manual_qc_labels.csv", "validation_summary.json",
                )
            },
            "scientific_boundary": "engineering benchmark and descriptive agreement only; no accuracy claim",
        }
        atomic_write_json(completion, run_dir / "completion.json")
        return completion
    finally:
        if frame_source is not None:
            frame_source.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    subjects = _subjects(config, args.subjects)
    algorithms = _algorithms(args.algorithms)
    if args.stage != "plan" and len(subjects) > 1 and not args.approve_multi_subject:
        raise SystemExit("multi-subject writing stages require --approve-multi-subject")

    if args.stage == "plan":
        for subject in subjects:
            run, _, tight, temporal = _plan_one(config, args.profile, subject)
            print(json.dumps(
                {
                    "subject": subject,
                    "production_run": str(run.path),
                    "source_video": str(run.source_video),
                    "tight_frames": int(len(tight)),
                    "temporal_frames": int(len(temporal)),
                    "profile": args.profile,
                },
                ensure_ascii=False,
            ))
        return 0

    run_dir = _require_run_dir(args)
    if args.stage in {"prepare", "all"}:
        _prepare(config_path, config, args, subjects, run_dir)
    if args.stage in {"run", "all"}:
        completion = _run(config, args, algorithms, run_dir)
        print(json.dumps(completion, ensure_ascii=False, indent=2))
    elif args.stage == "validate":
        manifest = pd.read_csv(run_dir / "sample_manifest.csv", low_memory=False)
        results = pd.read_csv(run_dir / "frame_results.csv", low_memory=False)
        print(json.dumps(validate_result_contract(manifest, results, algorithms), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
