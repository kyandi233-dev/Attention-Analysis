"""Canonical single-subject final RITnet full-class runner.

This entrypoint is the supported single-subject NVIDIA/CUDA final path. It
reuses the strictly validated historical YOLO formal source, runs the compact
final numeric core, produces bounded frame-level QC evidence, and publishes a
completion marker only after end-to-end integrity and <1 GiB checks pass.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from ritnet_fullclass_final_completion import (
    COMPLETION_NAME,
    MANIFEST_NAME,
    finalize_subject,
    validate_final_completion,
)
from ritnet_fullclass_final_engine import (
    _work_identity,
    resolve_package_path,
    run_numeric_core,
)
from ritnet_fullclass_git import require_clean_code_worktree
from ritnet_fullclass_qc_producer import produce_qc_artifacts
from ritnet_fullclass_source import load_source_context


PACKAGE_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final <=1 GiB RITnet full-class analysis for one validated formal subject"
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed historical formal run directory")
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config.yaml")
    parser.add_argument("--device", default="0", help="CUDA device id")
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


def _manifest_work_identity(subject_dir: Path) -> tuple[dict, dict]:
    manifest_path = Path(subject_dir) / MANIFEST_NAME
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise RuntimeError("final completion manifest must be an object")
    work_identity = manifest.get("work_identity")
    if not isinstance(work_identity, dict):
        raise RuntimeError("final completion manifest has no work_identity object")
    return dict(work_identity), manifest


def _complete_manifest_identity(work_identity: dict, manifest: dict) -> bool:
    """Require explicit provenance and scientific identity before a skip.

    Legacy final outputs can pass the structural validator by reconstructing
    scientific identity from their recorded commit. They are still not
    evidence of the current formal rerun, so they must be archived and rerun.
    """
    commit = work_identity.get("git_commit")
    branch = work_identity.get("git_branch")
    scientific = manifest.get("scientific_identity")
    provenance = manifest.get("provenance_identity")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(char not in "0123456789abcdefABCDEF" for char in commit)
        or not isinstance(branch, str)
        or not branch.strip()
    ):
        return False
    if not isinstance(scientific, dict) or not scientific:
        return False
    if not isinstance(provenance, dict):
        return False
    return provenance.get("git_commit") == commit and provenance.get("git_branch") == branch


def _archive_reason_slug(reason: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(reason))
    text = "-".join(part for part in text.split("-") if part)
    return (text or "stale-state")[:64]


def _archive_subject_dir(subject_dir: Path, *, reason: str) -> Path | None:
    """Move stale final output aside without deleting or overwriting it."""
    subject_dir = Path(subject_dir)
    if not subject_dir.exists():
        return None
    try:
        has_contents = any(subject_dir.iterdir())
    except OSError as exc:
        raise RuntimeError(f"cannot inspect existing final subject directory: {subject_dir}: {exc}") from exc
    if not has_contents:
        return None

    output_root = subject_dir.parent.parent
    archive_parent = output_root / "_archive" / subject_dir.parent.name / subject_dir.name
    archive_parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = _archive_reason_slug(reason)
    destination = archive_parent / f"{stamp}__{slug}"
    suffix = 1
    while destination.exists():
        destination = archive_parent / f"{stamp}-{suffix:02d}__{slug}"
        suffix += 1

    try:
        subject_dir.replace(destination)
    except OSError as exc:
        raise RuntimeError(
            "existing final subject state needs archiving but Windows refused the directory move. "
            "Close any Explorer preview/image viewer/VS Code tab that has files open under this subject, "
            f"then rerun the same command. source={subject_dir} destination={destination} error={exc}"
        ) from exc

    archive_record = {
        "archived_at_local": datetime.now().isoformat(timespec="seconds"),
        "reason": str(reason),
        "source_subject_dir": str(subject_dir),
        "archive_dir": str(destination),
        "policy": "preserve-by-move-no-delete",
    }
    (destination / "_archive_reason.json").write_text(
        json.dumps(archive_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "subject": subject_dir.name,
                "status": "archived_stale_final_state",
                "reason": str(reason),
                "from": str(subject_dir),
                "to": str(destination),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return destination


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
            try:
                stored_identity, manifest = _manifest_work_identity(subject_dir)
            except Exception as exc:
                _archive_subject_dir(
                    subject_dir,
                    reason=f"invalid-completion__{_archive_reason_slug(str(exc))}",
                )
                return subject_dir, expected_identity
            if _complete_manifest_identity(stored_identity, manifest):
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
            _archive_subject_dir(subject_dir, reason="valid-completion-identity-incomplete")
            return subject_dir, expected_identity

        _archive_subject_dir(
            subject_dir,
            reason=f"invalid-completion__{_archive_reason_slug(validation.reason)}",
        )
        return subject_dir, expected_identity

    if subject_dir.exists():
        try:
            has_contents = any(subject_dir.iterdir())
        except OSError as exc:
            raise RuntimeError(f"cannot inspect existing final subject directory: {subject_dir}: {exc}") from exc
        if has_contents:
            _archive_subject_dir(
                subject_dir,
                reason="incomplete-or-legacy-without-valid-completion",
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
        eye_metrics_path=core.eye_metrics,
        frame_coverage_path=core.frame_coverage,
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
                "qc_pixel_evidence_eyes": qc.pixel_evidence_saved_count,
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
    from nir_python_contract import require_nir_gpu_python

    require_nir_gpu_python()
    raise SystemExit(main())
