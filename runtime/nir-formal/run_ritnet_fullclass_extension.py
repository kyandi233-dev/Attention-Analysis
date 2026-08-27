"""Canonical single-subject RITnet full-class runner.

This entry point always executes the complete 640x400 evidence workflow:
lossless hard-label chunks, same-label pupil/iris geometry, probability-summary
checkpoints, sparse QC, provenance manifests, strict resume and final SHA256
verification. The former fast/320x160 extension is historical and is no longer
an executable production path.

Canonical production runs always hash the source video, require a clean Git
worktree, reject model-mismatch overrides, and guarantee an explicit ``subject``
column in every derived full-class CSV row even when a historical source
``eyes.csv`` predates the subject-column fix.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import run_ritnet_fullclass_native_extension as implementation

PACKAGE_ROOT = Path(__file__).resolve().parent


def _enforce_canonical_provenance() -> None:
    if "--allow-model-mismatch" in sys.argv:
        raise SystemExit(
            "Canonical full-class runs do not permit --allow-model-mismatch. "
            "Use the frozen model/config that matches the source evidence."
        )
    if "--hash-video" not in sys.argv:
        sys.argv.append("--hash-video")
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PACKAGE_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception as exc:
        raise SystemExit("Canonical full-class run requires a readable Git worktree") from exc
    if dirty:
        raise SystemExit(
            "Canonical full-class run requires a clean Git worktree so the recorded commit "
            "fully identifies the executed code. Commit/stash local changes first."
        )


def _install_subject_identity_guard() -> None:
    """Adapt historical source eyes.csv rows without modifying the source file."""
    original = implementation._source_rows

    def source_rows_with_subject(path: Path, subject: str):
        fields, rows = original(path, subject)
        if "subject" not in fields:
            fields = ["subject", *fields]
        normalized_rows = []
        for ordinal, row in enumerate(rows):
            copied = dict(row)
            row_subject = implementation.normalize_subject(copied.get("subject") or subject)
            if row_subject != subject:
                raise ValueError(
                    f"mixed subjects in source eyes.csv at row {ordinal}: "
                    f"{row_subject} != {subject}"
                )
            copied["subject"] = subject
            normalized_rows.append(copied)
        return fields, normalized_rows

    implementation._source_rows = source_rows_with_subject


if __name__ == "__main__":
    _enforce_canonical_provenance()
    _install_subject_identity_guard()
    raise SystemExit(implementation.main())
