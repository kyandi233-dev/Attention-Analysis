"""Canonical batch RITnet full-class runner.

All batch executions use the complete 640x400 evidence workflow. There is no
separate fast/legacy production batch path.
"""
from __future__ import annotations

from run_ritnet_fullclass_native_batch import main


if __name__ == "__main__":
    raise SystemExit(main())
