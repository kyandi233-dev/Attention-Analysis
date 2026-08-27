"""Canonical single-subject RITnet full-class runner.

This entry point always executes the complete 640x400 evidence workflow:
lossless hard-label chunks, same-label pupil/iris geometry, probability-summary
checkpoints, sparse QC, provenance manifests, strict resume and final SHA256
verification. The former fast/320x160 extension is historical and is no longer
an executable production path.
"""
from __future__ import annotations

from run_ritnet_fullclass_native_extension import main


if __name__ == "__main__":
    raise SystemExit(main())
