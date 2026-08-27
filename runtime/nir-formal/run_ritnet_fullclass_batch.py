"""Canonical batch RITnet full-class runner.

This outer gate requires a clean Git worktree, then delegates source discovery
and per-subject dispatch to the internal batch selector. Every child run uses the
same final <=1 GiB single-subject entrypoint; no legacy chunk/compression knobs
remain in the canonical CLI.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from run_ritnet_fullclass_native_batch import main

PACKAGE_ROOT = Path(__file__).resolve().parent


def _enforce_canonical_provenance() -> None:
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
