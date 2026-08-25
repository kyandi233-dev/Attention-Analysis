from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ALIGNMENT_SCHEMA_VERSION = 1
ALIGNMENT_PIPELINE_VERSION = "nir-behavior-v1.1"
FULLCLASS_EXTENSION_VERSION = "ritnet-fullclass-v1.2-fast-qc"

PIR_COLUMN = "fullclass_pupil_to_iris_diameter_ratio"
PIR_VALID_COLUMN = "fullclass_normalization_valid"
OAR_COLUMN = "fullclass_ocular_aperture_ratio_median"
OAR_P90_COLUMN = "fullclass_ocular_aperture_ratio_p90"

REQUIRED_NIR_COLUMNS = {
    "subject",
    "phase",
    "phase_segment",
    "frame_idx",
    "video_time_ms",
    "unix_ms",
    "eye",
    PIR_COLUMN,
    PIR_VALID_COLUMN,
    OAR_COLUMN,
    OAR_P90_COLUMN,
}

OPTIONAL_NIR_QC_COLUMNS = (
    "roi_clipped",
    "ritnet_found",
    "fullclass_ocular_component_count",
    "fullclass_ocular_largest_component_fraction",
    "fullclass_ocular_fraction",
    "fullclass_iris_outer_fraction",
    "fullclass_pupil_fraction",
)

EYES = ("left", "right")


@dataclass(frozen=True)
class WindowSpec:
    name: str
    family: str
    start_offset_ms: int
    end_offset_ms: int

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("Window name must not be empty")
        if self.end_offset_ms <= self.start_offset_ms:
            raise ValueError(
                f"Window {self.name}: end_offset_ms must exceed start_offset_ms"
            )


def normalize_subject(value: str | int) -> str:
    text = str(value).strip().rstrip("_")
    if text.startswith("sub-"):
        text = text[4:]
    if not text.isdigit():
        raise ValueError(f"Invalid subject identifier: {value!r}")
    return f"sub-{int(text):03d}"


def parse_subject_list(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        subject = normalize_subject(value)
        if subject not in seen:
            seen.add(subject)
            result.append(subject)
    return sorted(result, key=lambda x: int(x.split("-")[1]))


def parse_window_specs(raw: Iterable[dict[str, Any]], *, family: str) -> list[WindowSpec]:
    specs: list[WindowSpec] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"Window specification must be a mapping: {item!r}")
        spec = WindowSpec(
            name=str(item["name"]),
            family=str(item.get("family", family)),
            start_offset_ms=int(item["start_offset_ms"]),
            end_offset_ms=int(item["end_offset_ms"]),
        )
        spec.validate()
        if spec.name in seen:
            raise ValueError(f"Duplicate window name: {spec.name}")
        seen.add(spec.name)
        specs.append(spec)
    if not specs:
        raise ValueError(f"No {family} windows configured")
    return specs


def subject_output_paths(output_root: Path, subject: str) -> dict[str, Path]:
    subject = normalize_subject(subject)
    base = Path(output_root) / "subjects" / subject
    qc_dir = base / f"{subject}_alignment_qc"
    return {
        "subject_dir": base,
        "trial_level": base / f"{subject}_trial_level.csv",
        "trial_windows": base / f"{subject}_trial_nir_windows.csv",
        "probe_windows": base / f"{subject}_probe_windows.csv",
        "trial_coverage": base / f"{subject}_trial_window_coverage.csv",
        "probe_coverage": base / f"{subject}_probe_window_coverage.csv",
        "manifest": base / f"{subject}_alignment_manifest.json",
        "summary": base / f"{subject}_alignment_summary.json",
        "completion": base / f"{subject}_alignment_completion.json",
        "qc_dir": qc_dir,
        "qc_timeline_pir": qc_dir / f"{subject}_qc_01_timeline_pir.png",
        "qc_timeline_oar": qc_dir / f"{subject}_qc_02_timeline_oar.png",
        "qc_probe_pir": qc_dir / f"{subject}_qc_03_probe_centered_pir.png",
    }
