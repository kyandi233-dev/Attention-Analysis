"""Canonical batch RITnet full-class runner.

This outer gate allows only the locally generated final RITnet ONNX/.data files
to be dirty in Git; their SHA256 is recorded separately. All source code/config
must remain clean. Every child uses the same final <=1 GiB single-subject path.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ritnet_fullclass_git import require_clean_code_worktree
from run_ritnet_fullclass_native_batch import load_config, main

PACKAGE_ROOT = Path(__file__).resolve().parent


def _config_path_from_argv() -> Path:
    if "--config" in sys.argv:
        index = sys.argv.index("--config")
        if index + 1 >= len(sys.argv):
            raise SystemExit("--config requires a path value")
        return Path(sys.argv[index + 1]).expanduser().resolve()
    return (PACKAGE_ROOT / "config.yaml").resolve()


def _enforce_canonical_provenance() -> None:
    config_path = _config_path_from_argv()
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")
    config = load_config(config_path)
    try:
        require_clean_code_worktree(config)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    _enforce_canonical_provenance()
    raise SystemExit(main())
