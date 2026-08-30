"""Integrity-checked cohort runner for pupil-only NIR analysis tables.

The legacy table builder writes a per-session completion marker, but that marker
predates output-file digests and its cache check happens after the expensive
window construction.  This wrapper is the authoritative runner used by
``scripts/nir_formal_pipeline.py``.  It validates completion markers before any
heavy work, requires every declared output to exist with the recorded SHA-256,
and forces regeneration when a marker is stale, incomplete, or from an older
completion contract.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from attention_pipeline.config import load_config
from . import pupil_tables as base

COMPLETION_CONTRACT_VERSION = 2
OUTPUT_KEYS = (
    "trial_level",
    "trial_windows",
    "probe_windows",
    "time_on_task",
    "trial_coverage",
    "probe_coverage",
    "dependency_audit",
    "manifest",
)


def _read_completion(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError, AttributeError):
        return None
    return value if isinstance(value, dict) else None


def _completion_is_valid(config: Any, session_id: str) -> bool:
    paths = base._session_paths(config, session_id)
    completion = _read_completion(paths["completion"])
    if completion is None:
        return False
    if completion.get("status") != "complete":
        return False
    if completion.get("pipeline_version") != base.PIPELINE_VERSION:
        return False
    if int(completion.get("schema_version", -1)) != int(base.SCHEMA_VERSION):
        return False
    if int(completion.get("completion_contract_version", -1)) != COMPLETION_CONTRACT_VERSION:
        return False

    analysis_ready = base._session_frame_path(config, session_id)
    if not analysis_ready.is_file():
        return False
    if completion.get("analysis_ready_sha256") != base._digest_file(analysis_ready):
        return False

    recorded = completion.get("outputs_sha256")
    if not isinstance(recorded, dict) or set(recorded) != set(OUTPUT_KEYS):
        return False
    for key in OUTPUT_KEYS:
        path = paths[key]
        expected = recorded.get(key)
        if not isinstance(expected, str) or not expected or not path.is_file():
            return False
        if base._digest_file(path) != expected:
            return False
    return True


def _seal_completion(config: Any, session_id: str) -> None:
    paths = base._session_paths(config, session_id)
    analysis_ready = base._session_frame_path(config, session_id)
    if not analysis_ready.is_file():
        raise FileNotFoundError(analysis_ready)
    missing = [key for key in OUTPUT_KEYS if not paths[key].is_file()]
    if missing:
        raise RuntimeError(f"{session_id}: cannot seal completion; missing outputs {missing}")

    existing = _read_completion(paths["completion"]) or {}
    payload = {
        **existing,
        "status": "complete",
        "pipeline_version": base.PIPELINE_VERSION,
        "schema_version": base.SCHEMA_VERSION,
        "completion_contract_version": COMPLETION_CONTRACT_VERSION,
        "session_id": session_id,
        "analysis_ready_sha256": base._digest_file(analysis_ready),
        "outputs_sha256": {key: base._digest_file(paths[key]) for key in OUTPUT_KEYS},
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    base._write_json(paths["completion"], payload)


def run_cohort(
    config_path: str | Path,
    *,
    subjects: Iterable[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run tables with early, digest-checked completion reuse.

    Any pre-v2 completion marker is intentionally treated as stale once.  The
    corresponding derived tables are rebuilt without touching producer NIR, and
    the new marker is sealed only after all table outputs and the manifest exist.
    """
    config = load_config(config_path)
    selected = list(dict.fromkeys(base.selected_sessions(config, subjects)))
    if not selected:
        raise ValueError("No pupil analysis-ready sessions selected")

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for session_id in selected:
        if not force and _completion_is_valid(config, session_id):
            results.append(
                {
                    "session_id": session_id,
                    "status": "skipped",
                    "reason": "validated_completion_with_output_digests",
                }
            )
            continue
        try:
            # Force the legacy per-session builder so its old weak completion
            # shortcut cannot bypass this wrapper's integrity decision.
            result = base.run_session(config, session_id, force=True)
            if result.get("status") != "complete":
                raise RuntimeError(
                    f"{session_id}: table builder returned unexpected status {result.get('status')!r}"
                )
            _seal_completion(config, session_id)
        except Exception as exc:
            result = {
                "session_id": session_id,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(result)
        results.append(result)

    root = base._output_root(config)
    root.mkdir(parents=True, exist_ok=True)
    failure_path = root / "failure_tables" / "analysis_table_session_failures.csv"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        failures, columns=["session_id", "status", "error_type", "error"]
    ).to_csv(failure_path, index=False, encoding="utf-8-sig")

    n_failed = len(failures)
    summary = {
        "status": "complete" if n_failed == 0 else "partial",
        "pipeline_version": base.PIPELINE_VERSION,
        "schema_version": base.SCHEMA_VERSION,
        "completion_contract_version": COMPLETION_CONTRACT_VERSION,
        "n_sessions_requested": len(selected),
        "n_sessions_completed": sum(row.get("status") == "complete" for row in results),
        "n_sessions_skipped_validated": sum(row.get("status") == "skipped" for row in results),
        "n_sessions_failed": n_failed,
        # Compatibility key for the top-level CLI; this counts sessions.
        "n_subjects_failed": n_failed,
        "failure_table": str(failure_path),
        "results": results,
    }
    base._write_json(root / "cohort_manifest.json", summary)
    return summary


__all__ = [
    "COMPLETION_CONTRACT_VERSION",
    "OUTPUT_KEYS",
    "run_cohort",
]
