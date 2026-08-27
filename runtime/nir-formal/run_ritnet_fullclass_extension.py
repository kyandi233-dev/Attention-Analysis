"""Canonical single-subject final RITnet full-class runner.

This entrypoint is the only supported single-subject AMD/DirectML path. It
reuses the strictly validated historical YOLO formal source, runs the compact
final numeric core, produces bounded frame-level QC evidence, and publishes a
completion marker only after end-to-end integrity and <1 GiB checks pass.
"""
from __future__ import annotations

import argparse
import json
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
from ritnet_fullclass_git import require_clean_code_worktree
from ritnet_fullclass_qc_producer import QC_PIXEL_EVIDENCE_NAME, produce_qc_artifacts
from ritnet_fullclass_source import load_source_context


PACKAGE_ROOT = Path(__file__).resolve().parent
LEGACY_GZIP_DATA_NAMES = (
    "eye_metrics.csv.gz",
    "frame_coverage.csv.gz",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final <=1 GiB RITnet full-class analysis for one validated formal subject"
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed historical formal run directory")
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config.yaml")
    parser.add_argument("--device", default="0", help="DirectML device id")
    parser.add_argument(
        "--source-selection-reason",
        default="direct_single_subject_run",
        help="Provenance reason for choosing this historical formal source run",
    )
    parser.add_argument(
        "--source-alternative-run",
        action="append",
        default=[],
        help="Alternative validated historical run considered by the batch selector; repeatable",
    )
    return parser.parse_args()


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

    blockers = [
        subject_dir / "summary.json",
        subject_dir / "manifest.json",
        subject_dir / "qc" / "qc_index.csv",
        subject_dir / "qc" / QC_PIXEL_EVIDENCE_NAME,
        *(subject_dir / "data" / name for name in LEGACY_GZIP_DATA_NAMES),
    ]
    blockers += list((subject_dir / "qc" / "images").glob("*.png")) if (subject_dir / "qc" / "images").is_dir() else []
    existing = [path for path in blockers if path.exists()]
    if existing:
        raise RuntimeError(
            "incomplete or legacy final artifacts already exist without a valid completion marker; "
            "refusing automatic deletion/overwrite. Archive them outside the subject directory first: "
            + ", ".join(str(path) for path in existing[:8])
        )
    return subject_dir, expected_identity


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    context = load_source_context(run_dir, config_path)
    require_clean_code_worktree(context.config)
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
        eye_metric_rows=core.eye_metric_rows,
        frame_coverage_rows=core.frame_coverage_rows,
        device=str(args.device),
    )
    final_cfg = core.source_context.config.get("fullclass", {})
    output_limit = int(final_cfg.get("final_output_limit_bytes", 1073741824))
    source_selection = {
        "reason": str(args.source_selection_reason),
        "selected_run_dir": str(run_dir),
        "alternatives": [str(Path(value).expanduser()) for value in args.source_alternative_run],
    }
    completion = finalize_subject(
        core=core,
        qc=qc,
        output_limit_bytes=output_limit,
        source_selection=source_selection,
    )

    # finalize_subject already performs the one full artifact-integrity pass
    # before publishing completion.json. Avoid immediately rereading the entire
    # eye/frame tables a second time; future skip/validation calls still use the
    # strict public validator.
    completion_payload = json.loads(completion.read_text(encoding="utf-8"))

    print(
        json.dumps(
            {
                "subject": core.subject,
                "status": "complete",
                "subject_dir": str(core.subject_dir),
                "eye_metric_rows": core.eye_row_count,
                "frame_coverage_rows": core.frame_row_count,
                "qc_images": qc.saved_image_count,
                "qc_pixel_evidence_eyes": qc.pixel_evidence_saved_count,
                "qc_bytes": qc.total_qc_bytes,
                "total_output_bytes": completion_payload["total_output_bytes"],
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
