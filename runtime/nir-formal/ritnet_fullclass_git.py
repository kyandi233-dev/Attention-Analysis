"""Git provenance gate for final full-class runs.

Generated final ONNX artifacts may be intentionally untracked/modified because
their byte identity is frozen separately by SHA256. All code/config/other files
must still be clean so the recorded Git commit identifies executed source code.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Mapping


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
FINAL_MODEL_KEYS = (
    "ritnet_fullclass_final",
    "ritnet_fullclass_final_external_data",
)


def _package_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PACKAGE_ROOT / path).resolve()


def allowed_generated_model_paths(config: Mapping[str, Any]) -> set[str]:
    models = config.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("config.models must be a mapping")
    allowed: set[str] = set()
    for key in FINAL_MODEL_KEYS:
        if key not in models:
            raise ValueError(f"config.models missing {key}")
        path = _package_path(str(models[key]))
        try:
            relative = path.relative_to(REPO_ROOT)
        except ValueError as exc:
            raise ValueError(f"final generated model must remain inside repository: {path}") from exc
        allowed.add(relative.as_posix())
    return allowed


def require_clean_code_worktree(config: Mapping[str, Any]) -> None:
    """Reject dirty worktree except the two generated final-model artifacts."""
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        raise RuntimeError("final full-class run requires a readable Git worktree") from exc

    allowed = allowed_generated_model_paths(config)
    unexpected: list[str] = []
    for raw in status.splitlines():
        if not raw.strip():
            continue
        # Porcelain v1 uses two status columns, one space, then the path.
        path_text = raw[3:].strip() if len(raw) >= 4 else raw.strip()
        # Renames are not expected for generated models; treat them as code-tree
        # changes even if one side happens to mention an allowed filename.
        if " -> " in path_text or path_text.replace("\\", "/") not in allowed:
            unexpected.append(raw)
    if unexpected:
        raise RuntimeError(
            "final full-class run requires all code/config files to be clean. The only allowed "
            "Git changes are the generated final RITnet ONNX and .onnx.data files, whose SHA256 "
            "is recorded separately. Unexpected status: " + " | ".join(unexpected[:12])
        )
