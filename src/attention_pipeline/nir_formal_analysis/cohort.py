from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from attention_pipeline.config import Config, load_config
from attention_pipeline.nir_behavior.contract import normalize_subject

from .tables import (
    PIPELINE_VERSION,
    SCHEMA_VERSION,
    _analysis_ready_root,
    _atomic_json,
    _output_root,
    run_subject,
    selected_subjects,
)


def _result_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    completed = sum(item.get("status") == "complete" for item in results)
    skipped = sum(item.get("status") == "skipped" for item in results)
    failed = sum(item.get("status") == "failed" for item in results)
    return {
        "n_subjects_processed": len(results),
        "n_subjects_completed": int(completed),
        "n_subjects_skipped_validated": int(skipped),
        "n_subjects_failed": int(failed),
        "n_subjects_validated": int(completed + skipped),
    }


def _manifest_payload(
    config: Config,
    selected: list[str],
    results: list[dict[str, Any]],
    *,
    status: str,
    current_subject: str | None = None,
) -> dict[str, Any]:
    counts = _result_counts(results)
    return {
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config.path),
        "config_digest": config.digest,
        "analysis_ready_root": str(_analysis_ready_root(config)),
        "subjects": selected,
        "current_subject": current_subject,
        **counts,
        "n_subjects_requested": len(selected),
        "n_subjects_remaining_including_current": max(0, len(selected) - len(results)),
        "resume_safe": True,
        "resume_rule": (
            "Subjects with a matching complete identity are skipped; an interrupted "
            "subject without a valid completion is rebuilt on the next run."
        ),
        "results": results,
    }


def _write_progress_manifest(
    config: Config,
    selected: list[str],
    results: list[dict[str, Any]],
    *,
    status: str,
    current_subject: str | None = None,
) -> Path:
    root = _output_root(config)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "cohort_manifest.json"
    _atomic_json(
        path,
        _manifest_payload(
            config,
            selected,
            results,
            status=status,
            current_subject=current_subject,
        ),
    )
    return path


def run_cohort(
    config_path: str | Path,
    *,
    subjects: Iterable[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run analysis-table materialization with interruption-safe cohort progress.

    The cohort manifest is written before the first subject and after every subject.
    If the process is interrupted, the manifest is marked ``interrupted`` and the
    next run can safely resume: validated completions are skipped automatically,
    while an incomplete current subject is rebuilt.
    """

    config = load_config(config_path)
    selected = selected_subjects(config, subjects)
    if not selected:
        raise ValueError("No analysis-ready subjects selected")

    results: list[dict[str, Any]] = []
    manifest_path = _write_progress_manifest(
        config,
        selected,
        results,
        status="running",
        current_subject=selected[0],
    )

    for subject in selected:
        normalized = normalize_subject(subject)
        try:
            result = run_subject(config, normalized, force=force)
        except KeyboardInterrupt:
            _write_progress_manifest(
                config,
                selected,
                results,
                status="interrupted",
                current_subject=normalized,
            )
            raise
        except Exception as exc:
            result = {
                "subject": normalized,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

        results.append(result)
        running_status = (
            "running_with_failures"
            if any(item.get("status") == "failed" for item in results)
            else "running"
        )
        _write_progress_manifest(
            config,
            selected,
            results,
            status=running_status,
            current_subject=None,
        )

    failed = [item for item in results if item.get("status") == "failed"]
    final_status = "complete" if not failed else "partial"
    manifest_path = _write_progress_manifest(
        config,
        selected,
        results,
        status=final_status,
        current_subject=None,
    )
    counts = _result_counts(results)
    return {
        "status": final_status,
        "n_subjects_requested": len(selected),
        **counts,
        "results": results,
        "manifest": str(manifest_path),
    }
