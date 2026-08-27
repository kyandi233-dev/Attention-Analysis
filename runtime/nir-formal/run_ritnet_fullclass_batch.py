"""Canonical batch RITnet full-class runner.

All batch executions use the complete 640x400 evidence workflow. Canonical
production runs always hash source videos, reject model-mismatch overrides, and
require a clean Git worktree. There is no separate fast/legacy production batch.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from run_ritnet_fullclass_native_batch import main

PACKAGE_ROOT = Path(__file__).resolve().parent


def _enforce_canonical_provenance() -> None:
    if "--allow-model-mismatch" in sys.argv:
        raise SystemExit(
            "Canonical full-class batch runs do not permit --allow-model-mismatch."
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
        raise SystemExit("Canonical full-class batch requires a readable Git worktree") from exc
    if dirty:
        raise SystemExit(
            "Canonical full-class batch requires a clean Git worktree so the recorded commit "
            "fully identifies the executed code. Commit/stash local changes first."
        )


if __name__ == "__main__":
    _enforce_canonical_provenance()
    raise SystemExit(main())
