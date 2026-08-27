"""Single canonical schema, naming and frozen QC policy for RITnet full-class evidence.

There is one supported production contract: the complete 640x400 evidence path.
Older v1.2 artifacts are historical data only and are not an active runner/schema.
"""
from __future__ import annotations

from pathlib import Path

FULLCLASS_SCHEMA_VERSION = 2
FULLCLASS_VERSION = "ritnet-fullclass-v2-native640"
FULLCLASS_OUTPUT_STEM_VERSION = "v2-native640"

# Internal compatibility aliases used by implementation modules. They all refer
# to the single canonical contract above; they do not define parallel versions.
EXTENSION_SCHEMA_VERSION = FULLCLASS_SCHEMA_VERSION
EXTENSION_VERSION = FULLCLASS_VERSION
NATIVE_EXTENSION_SCHEMA_VERSION = FULLCLASS_SCHEMA_VERSION
NATIVE_EXTENSION_VERSION = FULLCLASS_VERSION

NATIVE_LABEL_SCHEMA_VERSION = 1
NATIVE_LABEL_CLASS_MAPPING_VERSION = "ritnet-4class-v1"
NATIVE_PREPROCESSING_VERSION = "ritnet-upstream-preprocess-plus-project-roi-resize-v1"
NATIVE_GEOMETRY_ALGORITHM_VERSION = "opencv-largest-external-contour-fitellipse-native640-v1"
OFFICIAL_UPSTREAM_REPOSITORY = "AayushKrChaudhary/RITnet"
OFFICIAL_UPSTREAM_COMMIT = "6431c57ce7bf0eda935fb6178b926ae9440b50bf"
OFFICIAL_WEIGHTS_GIT_BLOB_SHA1 = "f0864e6651f578525a9101c7ca787e23d2d201d7"

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

# Deterministic sparse QC sampling. These are sampling rules only; they are not
# scientific validity thresholds and do not define blink/PERCLOS labels.
QC_STRIDE_FRAMES = 3000
QC_ANOMALY_LIMIT_PER_REASON_PER_PHASE = 2
QC_OVERLAY_ALPHA = 0.45
QC_PALETTE_BGR = {
    CLASS_BACKGROUND: (0, 0, 0),
    CLASS_SCLERA: (255, 0, 0),
    CLASS_IRIS: (0, 255, 0),
    CLASS_PUPIL: (0, 0, 255),
}


def normalize_subject(value: str) -> str:
    text = str(value).strip().rstrip("_")
    if not text.startswith("sub-"):
        text = f"sub-{text}"
    number = text[4:]
    if not number.isdigit():
        raise ValueError(f"Invalid subject identifier: {value!r}")
    return f"sub-{int(number):03d}"


def fullclass_subject_output_paths(run_dir: Path, subject: str) -> dict[str, Path]:
    """Canonical complete full-class artifact paths for one subject."""
    prefix = normalize_subject(subject)
    run_dir = Path(run_dir)
    stem = f"{prefix}_ritnet_fullclass_{FULLCLASS_OUTPUT_STEM_VERSION}"
    return {
        "csv": run_dir / f"{stem}.csv",
        "summary": run_dir / f"{stem}_summary.json",
        "manifest": run_dir / f"{stem}_manifest.json",
        "completion": run_dir / f"{stem}_completion.json",
        "qc_index": run_dir / f"{stem}_qc_index.csv",
        "qc_dir": run_dir / f"{stem}_qc",
        "labels_dir": run_dir / f"{stem}_labels",
    }


# Code-level aliases so implementation files and existing automation resolve to
# the same canonical output family instead of creating separate production data.
def subject_output_paths(run_dir: Path, subject: str) -> dict[str, Path]:
    return fullclass_subject_output_paths(run_dir, subject)


def native_subject_output_paths(run_dir: Path, subject: str) -> dict[str, Path]:
    return fullclass_subject_output_paths(run_dir, subject)
