from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from attention_pipeline.config import load_config
from attention_pipeline.rgb.paths import RGBOutputLayout


SCHEMA_VERSION = "rgb-formal-subject-manifest-v1.0"
REQUIRED_PHASES = {"baseline", "block1", "block2"}
IDENTITY_COLUMNS = {
    "subject",
    "video_frame_position",
    "capture_frame_idx",
    "unix_ms",
    "phase",
}


def _read_json(path: Path, issues: list[str]) -> dict[str, Any]:
    if not path.is_file():
        issues.append(f"missing file: {path.name}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        issues.append(f"invalid json {path.name}: {exc}")
        return {}


def _read_parquet(path: Path, issues: list[str]) -> pd.DataFrame:
    if not path.is_file():
        issues.append(f"missing file: {path.name}")
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        issues.append(f"cannot read parquet {path.name}: {exc}")
        return pd.DataFrame()


def _check_identity(
    table: pd.DataFrame,
    *,
    label: str,
    subject: str,
    issues: list[str],
) -> None:
    if table.empty:
        issues.append(f"{label}: empty output")
        return
    missing = sorted(IDENTITY_COLUMNS - set(table.columns))
    if missing:
        issues.append(f"{label}: missing identity columns {missing}")
        return
    subjects = set(table["subject"].dropna().astype(str).unique())
    if subjects != {subject}:
        issues.append(f"{label}: subject column mismatch {sorted(subjects)}")
    phases = set(table["phase"].dropna().astype(str).unique())
    missing_phases = sorted(REQUIRED_PHASES - phases)
    if missing_phases:
        issues.append(f"{label}: missing required phases {missing_phases}")


def _file_record(path: Path, rows: int | None = None) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": int(path.stat().st_size) if path.is_file() else None,
        "rows": rows,
    }


def validate_subject(config_path: str, subject: str) -> dict[str, Any]:
    config = load_config(config_path)
    layout = RGBOutputLayout.from_config(config)
    issues: list[str] = []
    warnings: list[str] = []

    paths = {
        "face_frames": layout.subject_file(subject, "face_frames.csv"),
        "face_prepare_manifest": layout.subject_file(subject, "face_prepare_manifest.json"),
        "face_raw": layout.subject_file(subject, "face_raw.parquet"),
        "face_raw_manifest": layout.subject_file(subject, "face_raw_manifest.json"),
        "face_tracks": layout.subject_file(subject, "face_tracks.parquet"),
        "eye_features": layout.subject_file(subject, "eye_features.parquet"),
        "face_derived_manifest": layout.subject_file(subject, "face_derived_manifest.json"),
        "motion_raw": layout.subject_file(subject, "motion_raw.parquet"),
        "motion_manifest": layout.subject_file(subject, "motion_manifest.json"),
        "pose_landmarks": layout.subject_file(subject, "pose_landmarks.parquet"),
        "pose_manifest": layout.subject_file(subject, "pose_manifest.json"),
        "pose_features": layout.subject_file(subject, "pose_features.parquet"),
        "pose_features_manifest": layout.subject_file(subject, "pose_features_manifest.json"),
    }
    final_manifest = layout.subject_file(subject, "manifest.json")

    prepare_manifest = _read_json(paths["face_prepare_manifest"], issues)
    face_manifest = _read_json(paths["face_raw_manifest"], issues)
    face_derived_manifest = _read_json(paths["face_derived_manifest"], issues)
    motion_manifest = _read_json(paths["motion_manifest"], issues)
    pose_manifest = _read_json(paths["pose_manifest"], issues)
    pose_features_manifest = _read_json(paths["pose_features_manifest"], issues)

    for label, manifest in {
        "face_raw": face_manifest,
        "face_derived": face_derived_manifest,
        "motion": motion_manifest,
        "pose": pose_manifest,
        "pose_features": pose_features_manifest,
    }.items():
        if manifest and manifest.get("completion_status") != "complete":
            issues.append(f"{label}: manifest completion_status is not complete")

    if paths["face_frames"].is_file():
        try:
            face_frames = pd.read_csv(paths["face_frames"])
        except Exception as exc:
            issues.append(f"cannot read csv {paths['face_frames'].name}: {exc}")
            face_frames = pd.DataFrame()
    else:
        issues.append(f"missing file: {paths['face_frames'].name}")
        face_frames = pd.DataFrame()

    face_raw = _read_parquet(paths["face_raw"], issues)
    face_tracks = _read_parquet(paths["face_tracks"], issues)
    eye_features = _read_parquet(paths["eye_features"], issues)
    motion_raw = _read_parquet(paths["motion_raw"], issues)
    pose_landmarks = _read_parquet(paths["pose_landmarks"], issues)
    pose_features = _read_parquet(paths["pose_features"], issues)

    for label, table in {
        "face_frames": face_frames,
        "face_raw": face_raw,
        "motion_raw": motion_raw,
        "pose_landmarks": pose_landmarks,
    }.items():
        _check_identity(table, label=label, subject=subject, issues=issues)

    # Face formal prepare selects one row for every planned 15 Hz sample. The raw
    # runner also writes one row when no face is detected, so every planned sample
    # must still be represented in face_raw.
    if not face_frames.empty and not face_raw.empty:
        if "benchmark_index" not in face_frames.columns or "benchmark_index" not in face_raw.columns:
            issues.append("face: benchmark_index missing from frames/raw")
        else:
            expected_face_frames = int(face_frames["benchmark_index"].nunique())
            output_face_frames = int(face_raw["benchmark_index"].nunique())
            if expected_face_frames != output_face_frames:
                issues.append(
                    f"face: planned/output frame mismatch {expected_face_frames} != {output_face_frames}"
                )
            expected_ids = set(pd.to_numeric(face_frames["benchmark_index"], errors="coerce").dropna().astype(int))
            output_ids = set(pd.to_numeric(face_raw["benchmark_index"], errors="coerce").dropna().astype(int))
            missing_ids = expected_ids - output_ids
            if missing_ids:
                issues.append(f"face: {len(missing_ids)} planned sample indices missing from raw")

    # Motion is full-fps and must contain exactly one row for every AVI position
    # inside the formal analysis span.
    if not motion_raw.empty and motion_manifest:
        span = motion_manifest.get("analysis_span", {})
        first_pos = span.get("first_video_frame_position")
        last_pos = span.get("last_video_frame_position")
        if isinstance(first_pos, int) and isinstance(last_pos, int) and last_pos >= first_pos:
            expected_motion_rows = last_pos - first_pos + 1
            if len(motion_raw) != expected_motion_rows:
                issues.append(
                    f"motion: expected/output row mismatch {expected_motion_rows} != {len(motion_raw)}"
                )
        if "video_frame_position" in motion_raw.columns and motion_raw["video_frame_position"].duplicated().any():
            issues.append("motion: duplicate video_frame_position rows")

    # Pose retains one placeholder row for a sampled frame when no pose is found,
    # and 33 rows per returned pose otherwise. Unique frame count must therefore
    # match the manifest sampled-frame count.
    if not pose_landmarks.empty and pose_manifest:
        sampled = pose_manifest.get("output", {}).get("sampled_frames")
        if isinstance(sampled, int) and "video_frame_position" in pose_landmarks.columns:
            unique_pose_frames = int(pose_landmarks["video_frame_position"].nunique())
            if unique_pose_frames != sampled:
                issues.append(
                    f"pose: sampled/output frame mismatch {sampled} != {unique_pose_frames}"
                )

    # Derived files are intentionally recalculable. They must exist and be readable,
    # but QC thresholds are not part of extraction completion.
    for label, table in {
        "face_tracks": face_tracks,
        "eye_features": eye_features,
        "pose_features": pose_features,
    }.items():
        if table.empty:
            warnings.append(f"{label}: empty derived output")

    if not eye_features.empty and "subject" in eye_features.columns:
        subjects = set(eye_features["subject"].dropna().astype(str).unique())
        if subjects and subjects != {subject}:
            issues.append(f"eye_features: subject column mismatch {sorted(subjects)}")

    provenance_commits = {
        str(m.get("attention_analysis_git_commit"))
        for m in (motion_manifest, pose_manifest)
        if m.get("attention_analysis_git_commit")
    }
    face_commit = face_manifest.get("attention_analysis_git_commit")
    if face_commit:
        provenance_commits.add(str(face_commit))
    if len(provenance_commits) > 1:
        warnings.append(f"stage manifests contain multiple git commits: {sorted(provenance_commits)}")

    row_counts = {
        "face_frames": int(len(face_frames)),
        "face_raw": int(len(face_raw)),
        "face_tracks": int(len(face_tracks)),
        "eye_features": int(len(eye_features)),
        "motion_raw": int(len(motion_raw)),
        "pose_landmarks": int(len(pose_landmarks)),
        "pose_features": int(len(pose_features)),
    }

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "subject": subject,
        "completion_status": "complete" if not issues else "incomplete",
        "extraction_complete": not issues,
        "qc_pass": None,
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_span_definition": "continuous from baseline start through Block2 end",
        "config_path": str(config.path),
        "config_digest": config.digest,
        "issues": issues,
        "warnings": warnings,
        "row_counts": row_counts,
        "stage_manifests": {
            "face_prepare": str(paths["face_prepare_manifest"]),
            "face_raw": str(paths["face_raw_manifest"]),
            "face_derived": str(paths["face_derived_manifest"]),
            "motion": str(paths["motion_manifest"]),
            "pose": str(paths["pose_manifest"]),
            "pose_features": str(paths["pose_features_manifest"]),
        },
        "files": {
            key: _file_record(path, row_counts.get(key))
            for key, path in paths.items()
        },
        "notes": [
            "Extraction completion is separate from downstream QC eligibility.",
            "QC thresholds, blink-event rules and PERCLOS rules do not block extraction completion.",
            "Face uses the frozen Py-Feat/RetinaFace default formal threshold already implemented by the formal runner.",
        ],
    }
    final_manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["manifest"] = str(final_manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one completed formal RGB subject")
    parser.add_argument("--config", default="configs/rgb_analysis.yaml")
    parser.add_argument("--subject", required=True)
    args = parser.parse_args()

    result = validate_subject(args.config, args.subject)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["extraction_complete"]:
        raise SystemExit(2)
    print(f"[rgb:formal-validate] complete {args.subject}")


if __name__ == "__main__":
    main()
