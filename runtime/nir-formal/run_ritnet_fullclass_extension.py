"""Canonical single-subject final RITnet full-class runner.

This entrypoint is the only supported single-subject AMD/DirectML path. It
reuses the strictly validated historical YOLO formal source, runs the compact
final numeric core, produces bounded frame-level QC evidence, and publishes a
completion marker only after end-to-end integrity and <1 GiB checks pass.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from ritnet_fullclass_final_completion import (
    COMPLETION_NAME,
    finalize_subject,
    validate_final_completion,
)
from ritnet_fullclass_final_engine import (
    _work_identity,
    resolve_package_path,
    run_numeric_core,
)
from ritnet_fullclass_qc_producer import produce_qc_artifacts
from ritnet_fullclass_source import load_source_context


PACKAGE_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final <=1 GiB RITnet full-class analysis for one validated formal subject"
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed historical formal run directory")
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config.yaml")
    parser.add_argument("--device", default="0", help="DirectML device id")
    return parser.parse_args()


def _require_clean_git() -> None:
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PACKAGE_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception as exc:
        raise RuntimeError("final full-class run requires a readable Git worktree") from exc
    if dirty:
        raise RuntimeError(
            "final full-class run requires a clean Git worktree so the recorded commit fully "
            "identifies the executed code; commit/stash local changes first"
        )


def _subject_dir(context) -> Path:
    final_cfg = context.config.get("fullclass", {})
    output_dirname = str(final_cfg.get("output_dirname") or "ritnet-fullclass-final")
    return context.run_dir.parent / output_dirname / context.subject


def _expected_work_identity(context, config_path: Path) -> dict:
    model = resolve_package_path(context.config["models"]["ritnet_fullclass_final"]).resolve()
    external = resolve_package_path(
        context.config["models"]["ritnet_fullclass_final_external_data"]
    ).resolve()
    if not model.is_file() or not external.is_file():
        raise FileNotFoundError(
            "final RITnet ONNX/export data are missing; re-export the revised batch-16 final model first: "
            f"{model} / {external}"
        )
    return _work_identity(
        context=context,
        config_path=config_path,
        ritnet_model=model,
        ritnet_external_data=external,
    )


def _strict_skip_or_preflight(context, config_path: Path) -> tuple[Path, dict]:
    subject_dir = _subject_dir(context)
    expected_identity = _expected_work_identity(context, config_path)
    completion = subject_dir / COMPLETION_NAME
    if completion.exists():
        validation = validate_final_completion(
            subject_dir,
            expected_subject=context.subject,
            expected_work_identity=expected_identity,
        )
        if validation.valid:
            print(
                json.dumps(
                    {
                        "subject": context.subject,
                        "status": "skipped_valid_completion",
                        "subject_dir": str(subject_dir),
                        "completion": str(completion),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return subject_dir, expected_identity
        raise RuntimeError(
            "existing final completion is invalid or belongs to a different run identity; "
            "refusing automatic overwrite: " + validation.reason
        )

    # Numeric data may be regenerated transactionally from the SQLite workstore,
    # but partially published QC/final metadata are intentionally not overwritten.
    blockers = [
        subject_dir / "summary.json",
        subject_dir / "manifest.json",
        subject_dir / "qc" / "qc_index.csv",
    ]
    blockers += list((subject_dir / "qc" / "images").glob("*.png")) if (subject_dir / "qc" / "images").is_dir() else []
    existing = [path for path in blockers if path.exists()]
    if existing:
        raise RuntimeError(
            "incomplete final artifacts already exist without a valid completion marker; refusing "
            "automatic deletion/overwrite: " + ", ".join(str(path) for path in existing[:8])
        )
    return subject_dir, expected_identity


def main() -> int:
    args = parse_args()
    _require_clean_git()
    run_dir = args.run_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    context = load_source_context(run_dir, config_path)
    subject_dir, expected_identity = _strict_skip_or_preflight(context, config_path)
    completion_path = subject_dir / COMPLETION_NAME
    if completion_path.is_file():
        return 0

    core = run_numeric_core(
        run_dir=run_dir,
        config_path=config_path,
        device=str(args.device),
    )
    if core.work_identity != expected_identity:
        raise RuntimeError("preflight and numeric-core work identities diverged")

    qc = produce_qc_artifacts(
        subject=core.subject,
        subject_dir=core.subject_dir,
        source_video=core.source_context.video,
        config=core.source_context.config,
        eye_metrics_path=core.eye_metrics,
        frame_coverage_path=core.frame_coverage,
        device=str(args.device),
    )
    final_cfg = core.source_context.config.get("fullclass", {})
    output_limit = int(final_cfg.get("final_output_limit_bytes", 1073741824))
    completion = finalize_subject(
        core=core,
        qc=qc,
        output_limit_bytes=output_limit,
    )
    validation = validate_final_completion(
        core.subject_dir,
        expected_subject=core.subject,
        expected_work_identity=core.work_identity,
    )
    if not validation.valid:
        raise RuntimeError("final post-write validation failed: " + validation.reason)

    print(
        json.dumps(
            {
                "subject": core.subject,
                "status": "complete",
                "subject_dir": str(core.subject_dir),
                "eye_metric_rows": core.eye_row_count,
                "frame_coverage_rows": core.frame_row_count,
                "qc_images": qc.saved_image_count,
                "qc_bytes": qc.total_qc_bytes,
                "total_output_bytes": validation.completion["total_output_bytes"],
                "output_limit_bytes": output_limit,
                "completion": str(completion),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
