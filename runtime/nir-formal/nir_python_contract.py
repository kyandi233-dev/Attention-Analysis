"""Fail-closed interpreter contract for the formal NVIDIA NIR runtime."""
from __future__ import annotations

import sys
from pathlib import Path


EXPECTED_PYTHON = Path(r"D:\CondaEnvs\nir-nvidia\python.exe")


def require_nir_gpu_python() -> None:
    """Refuse formal execution if Python was resolved from another environment."""
    actual = Path(sys.executable).resolve()
    expected = EXPECTED_PYTHON.resolve()
    if actual != expected:
        raise SystemExit(
            "Formal NVIDIA NIR requires D:\\CondaEnvs\\nir-nvidia Python; "
            f"expected={expected}; actual={actual}"
        )
