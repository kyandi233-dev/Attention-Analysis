"""Formal RGB downstream analysis.

The single authoritative execution entry is :func:`run_rgb_formal_v2`.
"""
from __future__ import annotations

from typing import Any


def run_rgb_formal_v2(*args: Any, **kwargs: Any):
    from .runner import run_rgb_formal_v2 as _run

    return _run(*args, **kwargs)


__all__ = ["run_rgb_formal_v2"]
