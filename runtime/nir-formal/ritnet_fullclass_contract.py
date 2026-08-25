"""Shared schema, naming, and frozen QC policy for the RITnet full-class extension."""
from __future__ import annotations

from pathlib import Path

EXTENSION_SCHEMA_VERSION = 1
EXTENSION_VERSION = "ritnet-fullclass-v1.2-fast-qc"

CLASS_BACKGROUND = 0
CLASS_SCLERA = 1
CLASS_IRIS = 2
CLASS_PUPIL = 3
CLASS_MAPPING = {
    CLASS_BACKGROUND: "background",
    CLASS_SCLERA: "sclera",
    CLASS_IRIS: "iris",
    CLASS_PUPIL: "pupil",
}

# Frozen deterministic QC sampling policy.
# 3000 frames ~= 100 s at the current 30 FPS NIR acquisition. Each phase/segment
# also gets first/middle/last anchors, so short phases remain represented.
QC_STRIDE_FRAMES = 3000
QC_ANOMALY_LIMIT_PER_REASON_PER_PHASE = 2
QC_OVERLAY_ALPHA = 0.45
QC_PALETTE_BGR = {
    CLASS_BACKGROUND: (0, 0, 0),
    CLASS_SCLERA: (255, 0, 0),   # blue
    CLASS_IRIS: (0, 255, 0),     # green
    CLASS_PUPIL: (0, 0, 255),    # red
}


def normalize_subject(value: str) -> str:
    text = str(value).strip().rstrip("_")
    if not text.startswith("sub-"):
        text = f"sub-{text}"
    number = text[4:]
    if not number.isdigit():
        raise ValueError(f"Invalid subject identifier: {value!r}")
    return f"sub-{int(number):03d}"


def subject_output_paths(run_dir: Path, subject: str) -> dict[str, Path]:
    """Every per-subject artifact filename/folder carries the normalized subject ID."""
    prefix = normalize_subject(subject)
    run_dir = Path(run_dir)
    stem = f"{prefix}_ritnet_fullclass_v1-2-fast-qc"
    return {
        "csv": run_dir / f"{stem}.csv",
        "summary": run_dir / f"{stem}_summary.json",
        "manifest": run_dir / f"{stem}_manifest.json",
        "completion": run_dir / f"{stem}_completion.json",
        "qc_index": run_dir / f"{stem}_qc_index.csv",
        "qc_dir": run_dir / f"{stem}_qc",
    }
