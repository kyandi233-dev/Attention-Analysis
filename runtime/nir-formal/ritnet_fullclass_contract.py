"""Shared schema and naming contract for the RITnet full-class extension."""
from __future__ import annotations

from pathlib import Path

EXTENSION_SCHEMA_VERSION = 1
EXTENSION_VERSION = "ritnet-fullclass-v1.1-fast"

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


def normalize_subject(value: str) -> str:
    text = str(value).strip().rstrip("_")
    if not text.startswith("sub-"):
        text = f"sub-{text}"
    number = text[4:]
    if not number.isdigit():
        raise ValueError(f"Invalid subject identifier: {value!r}")
    return f"sub-{int(number):03d}"


def subject_output_paths(run_dir: Path, subject: str) -> dict[str, Path]:
    """Every per-subject artifact filename must carry the normalized subject ID."""
    prefix = normalize_subject(subject)
    return {
        "csv": Path(run_dir) / f"{prefix}_ritnet_fullclass.csv",
        "summary": Path(run_dir) / f"{prefix}_ritnet_fullclass_summary.json",
        "manifest": Path(run_dir) / f"{prefix}_ritnet_fullclass_manifest.json",
        "completion": Path(run_dir) / f"{prefix}_ritnet_fullclass_completion.json",
    }
