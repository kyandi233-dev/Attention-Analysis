"""Canonical single-subject RITnet full-class runner.

This entry point always executes the complete 640x400 evidence workflow:
lossless hard-label chunks, same-label pupil/iris geometry, probability-summary
checkpoints, sparse QC, provenance manifests, strict resume and final SHA256
verification. The former fast/320x160 extension is historical and is no longer
an executable production path.

Canonical production runs always hash the source video and do not permit model
mismatch overrides.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from run_ritnet_fullclass_native_extension import main

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


if __name__ == "__main__":
    _enforce_canonical_provenance()
    raise SystemExit(main())
