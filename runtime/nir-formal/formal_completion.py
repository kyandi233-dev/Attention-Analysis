"""Atomic completion markers and strict validation for formal NIR runs."""
from __future__ import annotations

import csv
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
MARKER_NAME = "completion.json"
REQUIRED_ARTIFACTS = (
    "phase_windows.json",
    "frames.csv",
    "eyes.csv",
    "summary.json",
    "run_manifest.json",
)


@dataclass(frozen=True)
class CompletionValidation:
    valid: bool
    reason: str
    marker: dict[str, Any] | None = None


def normalize_path(value: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(value)))


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON in the destination directory and atomically replace ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_completion(run_dir: Path, payload: Mapping[str, Any]) -> Path:
    marker = Path(run_dir) / MARKER_NAME
    atomic_write_json(marker, payload)
    return marker


def expected_frame_keys(windows: Iterable[Mapping[str, Any]]) -> set[tuple[str, int, int]]:
    keys: set[tuple[str, int, int]] = set()
    for window in windows:
        phase = str(window["phase"])
        segment = int(window["segment"])
        start = int(window["start_frame_idx"])
        end = int(window["end_frame_idx"])
        if end < start:
            raise ValueError(f"Invalid phase window {phase} segment {segment}: {start}..{end}")
        for frame_idx in range(start, end + 1):
            key = (phase, segment, frame_idx)
            if key in keys:
                raise ValueError(f"Duplicate expected frame key: {key}")
            keys.add(key)
    return keys


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _identity_matches(marker: Mapping[str, Any], expected: Mapping[str, Any]) -> str | None:
    for key, value in expected.items():
        actual = marker.get(key)
        if key == "video":
            if normalize_path(str(actual or "")) != normalize_path(str(value)):
                return f"identity mismatch for {key}: {actual!r} != {value!r}"
        elif actual != value:
            return f"identity mismatch for {key}: {actual!r} != {value!r}"
    return None


def _read_frame_keys(path: Path) -> tuple[int, set[tuple[str, int, int]], int]:
    count = 0
    failures = 0
    keys: set[tuple[str, int, int]] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"phase", "phase_segment", "frame_idx", "status"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"frames.csv is missing columns: {sorted(required)}")
        for row in reader:
            count += 1
            key = (str(row["phase"]), int(row["phase_segment"]), int(row["frame_idx"]))
            if key in keys:
                raise ValueError(f"duplicate frame key in frames.csv: {key}")
            keys.add(key)
            failures += int(str(row.get("status", "")) == "video_read_failed")
    return count, keys, failures


def validate_completion(
    run_dir: Path,
    expected_identity: Mapping[str, Any] | None = None,
    *,
    accepted_statuses: tuple[str, ...] = ("complete",),
) -> CompletionValidation:
    """Validate a formal run without trusting process exit code or summary existence."""
    run_dir = Path(run_dir)
    marker_path = run_dir / MARKER_NAME
    if not marker_path.is_file():
        return CompletionValidation(False, f"missing {MARKER_NAME}")

    try:
        marker = _load_json(marker_path)
    except Exception as exc:
        return CompletionValidation(False, f"unreadable {MARKER_NAME}: {exc}")
    if not isinstance(marker, dict):
        return CompletionValidation(False, f"{MARKER_NAME} must contain a JSON object")
    if marker.get("schema_version") != SCHEMA_VERSION:
        return CompletionValidation(False, "unsupported completion schema", marker)
    if marker.get("status") not in accepted_statuses:
        return CompletionValidation(
            False,
            f"status {marker.get('status')!r} is not accepted ({', '.join(accepted_statuses)})",
            marker,
        )
    if expected_identity:
        mismatch = _identity_matches(marker, expected_identity)
        if mismatch:
            return CompletionValidation(False, mismatch, marker)

    required_artifacts = tuple(marker.get("required_artifacts") or ())
    if required_artifacts != REQUIRED_ARTIFACTS:
        return CompletionValidation(False, "required_artifacts contract mismatch", marker)
    for name in required_artifacts:
        artifact = run_dir / name
        if not artifact.is_file():
            return CompletionValidation(False, f"missing required artifact: {name}", marker)
        if marker_path.stat().st_mtime_ns < artifact.stat().st_mtime_ns:
            return CompletionValidation(False, f"completion marker predates {name}", marker)

    try:
        phase_windows = _load_json(run_dir / "phase_windows.json")
        summary = _load_json(run_dir / "summary.json")
        manifest = _load_json(run_dir / "run_manifest.json")
        if not isinstance(phase_windows, list) or not isinstance(summary, dict) or not isinstance(manifest, dict):
            raise ValueError("formal JSON artifacts have unexpected top-level types")
        expected_keys = expected_frame_keys(phase_windows)
        frame_count, actual_keys, failure_count = _read_frame_keys(run_dir / "frames.csv")
        with (run_dir / "eyes.csv").open(newline="", encoding="utf-8-sig") as handle:
            list(csv.reader(handle))
    except Exception as exc:
        return CompletionValidation(False, f"artifact validation failed: {exc}", marker)

    missing = expected_keys - actual_keys
    unexpected = actual_keys - expected_keys
    checks = {
        "expected_frames": len(expected_keys),
        "processed_frames": frame_count,
        "missing_expected_frame_count": len(missing),
        "unexpected_frame_count": len(unexpected),
        "video_read_failure_count": failure_count,
    }
    for key, actual in checks.items():
        if int(marker.get(key, -1)) != int(actual):
            return CompletionValidation(
                False,
                f"marker {key}={marker.get(key)!r} but artifact value is {actual}",
                marker,
            )

    if int(summary.get("processed_frames", -1)) != frame_count:
        return CompletionValidation(False, "summary processed_frames mismatch", marker)
    if summary.get("subject") != marker.get("subject"):
        return CompletionValidation(False, "summary subject mismatch", marker)
    if normalize_path(str(summary.get("video", ""))) != normalize_path(str(marker.get("video", ""))):
        return CompletionValidation(False, "summary video mismatch", marker)
    if summary.get("phases") != marker.get("phases"):
        return CompletionValidation(False, "summary phases mismatch", marker)
    if bool(summary.get("truncated_for_smoke_test")) != bool(marker.get("truncated_for_smoke_test")):
        return CompletionValidation(False, "summary truncation flag mismatch", marker)

    effective = manifest.get("effective_parameters") or {}
    package = manifest.get("package") or {}
    if package.get("version") != marker.get("package_version"):
        return CompletionValidation(False, "manifest package version mismatch", marker)
    if effective.get("phases") != marker.get("phases"):
        return CompletionValidation(False, "manifest phases mismatch", marker)
    if effective.get("max_frames") != marker.get("max_frames"):
        return CompletionValidation(False, "manifest max_frames mismatch", marker)

    if marker.get("status") == "complete":
        if (
            marker.get("max_frames") is not None
            or marker.get("truncated_for_smoke_test")
            or marker.get("partial_phase_selection")
        ):
            return CompletionValidation(False, "complete run cannot be truncated", marker)
        if int(marker.get("decoded_frames", -1)) != len(expected_keys):
            return CompletionValidation(False, "decoded_frames mismatch", marker)
        if missing or unexpected or failure_count:
            return CompletionValidation(False, "complete run has missing/unexpected/failed frames", marker)

    return CompletionValidation(True, "valid completion marker", marker)
