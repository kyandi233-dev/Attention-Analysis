"""Shared schema, naming, and frozen QC policy for RITnet full-class extensions."""
from __future__ import annotations

from pathlib import Path

# Legacy v1.2 contract. These names/values remain frozen for historical outputs.
EXTENSION_SCHEMA_VERSION = 1
EXTENSION_VERSION = "ritnet-fullclass-v1.2-fast-qc"

# Native evidence v2 contract. This is a parallel output family and never
# replaces/overwrites v1.2 artifacts.
NATIVE_EXTENSION_SCHEMA_VERSION = 2
NATIVE_EXTENSION_VERSION = "ritnet-fullclass-v2-native640"
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

# Frozen deterministic QC sampling policy inherited from v1.2. v2 uses the same
# periodic anchors but interprets anomaly facts from native640 metrics.
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
    """Legacy v1.2 artifact paths; kept byte-for-byte compatible in naming."""
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


def native_subject_output_paths(run_dir: Path, subject: str) -> dict[str, Path]:
    """Versioned v2 native640 evidence artifacts; never collide with v1.2."""
    prefix = normalize_subject(subject)
    run_dir = Path(run_dir)
    stem = f"{prefix}_ritnet_fullclass_v2-native640"
    return {
        "csv": run_dir / f"{stem}.csv",
        "summary": run_dir / f"{stem}_summary.json",
        "manifest": run_dir / f"{stem}_manifest.json",
        "completion": run_dir / f"{stem}_completion.json",
        "qc_index": run_dir / f"{stem}_qc_index.csv",
        "qc_dir": run_dir / f"{stem}_qc",
        "labels_dir": run_dir / f"{stem}_labels",
    }
