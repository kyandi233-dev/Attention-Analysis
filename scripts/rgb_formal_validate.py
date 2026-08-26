from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from attention_pipeline.config import load_config
from attention_pipeline.rgb.paths import RGBOutputLayout


SCHEMA_VERSION = "rgb-formal-subject-manifest-v1.1"
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


def _read_parquet_projection(
    path: Path,
    *,
    wanted_columns: set[str],
    issues: list[str],
) -> tuple[pd.DataFrame, int, set[str]]:
    """Validate parquet readability without loading the very wide scientific raw table."""
    if not path.is_file():
        issues.append(f"missing file: {path.name}")
        return pd.DataFrame(), 0, set()
    try:
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema.names)
        selected = sorted(wanted_columns & available)
        table = parquet.read(columns=selected).to_pandas() if selected else pd.DataFrame()
        rows = int(parquet.metadata.num_rows)
        return table, rows, available
    except Exception as exc:
        issues.append(f"cannot read parquet {path.name}: {exc}")
        return pd.DataFrame(), 0, set()


def _check_identity(
    table: pd.DataFrame,
    *,
    available_columns: set[str],
    label: str,
    subject: str,
    issues: list[str],
) -> None:
    missing = sorted(IDENTITY_COLUMNS - available_columns)
    if missing:
        issues.append(f"{label}: missing identity columns {missing}")
        return
    if table.empty:
        issues.append(f"{label}: empty output")
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
        "motion_raw": layout.subject_file(subject, "motion_raw.parquet"),
        "motion_manifest": layout.subject_file(subject, "motion_manifest.json"),
        "pose_landmarks": layout.subject_file(subject, "pose_landmarks.parquet"),
        "pose_manifest": layout.subject_file(subject, "pose_manifest.json"),
    }
    optional_derived = {
        "face_tracks": layout.subject_file(subject, "face_tracks.parquet"),
        "eye_features": layout.subject_file(subject, "eye_features.parquet"),
        "face_derived_manifest": layout.subject_file(subject, "face_derived_manifest.json"),
        "pose_features": layout.subject_file(subject, "pose_features.parquet"),
        "pose_features_manifest": layout.subject_file(subject, "pose_features_manifest.json"),
    }
    final_manifest = layout.subject_file(subject, "manifest.json")

    prepare_manifest = _read_json(paths["face_prepare_manifest"], issues)
    face_manifest = _read_json(paths["face_raw_manifest"], issues)
    motion_manifest = _read_json(paths["motion_manifest"], issues)
    pose_manifest = _read_json(paths["pose_manifest"], issues)

    for label, manifest in {
        "face_raw": face_manifest,
        "motion": motion_manifest,
        "pose": pose_manifest,
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

    projection = set(IDENTITY_COLUMNS) | {"benchmark_index"}
    face_raw, face_raw_rows, face_cols = _read_parquet_projection(
        paths["face_raw"], wanted_columns=projection, issues=issues
    )
    motion_raw, motion_raw_rows, motion_cols = _read_parquet_projection(
        paths["motion_raw"], wanted_columns=set(IDENTITY_COLUMNS), issues=issues
    )
    pose_landmarks, pose_rows, pose_cols = _read_parquet_projection(
        paths["pose_landmarks"], wanted_columns=set(IDENTITY_COLUMNS), issues=issues
    )

    if not face_frames.empty:
        missing = sorted(IDENTITY_COLUMNS - set(face_frames.columns))
        if missing:
            issues.append(f"face_frames: missing identity columns {missing}")
        else:
            _check_identity(
                face_frames,
                available_columns=set(face_frames.columns),
                label="face_frames",
                subject=subject,
                issues=issues,
            )

    _check_identity(
        face_raw,
        available_columns=face_cols,
        label="face_raw",
        subject=subject,
        issues=issues,
    )
    _check_identity(
        motion_raw,
        available_columns=motion_cols,
        label="motion_raw",
        subject=subject,
        issues=issues,
    )
    _check_identity(
        pose_landmarks,
        available_columns=pose_cols,
        label="pose_landmarks",
        subject=subject,
        issues=issues,
    )

    # Face formal prepare selects one row per planned 15 Hz sample. Face raw keeps
    # a placeholder row for no-detection, so every planned sample must be present.
    if not face_frames.empty and not face_raw.empty:
        if "benchmark_index" not in face_frames.columns or "benchmark_index" not in face_cols:
            issues.append("face: benchmark_index missing from frames/raw")
        else:
            expected_ids = set(
                pd.to_numeric(face_frames["benchmark_index"], errors="coerce").dropna().astype(int)
            )
            output_ids = set(
                pd.to_numeric(face_raw["benchmark_index"], errors="coerce").dropna().astype(int)
            )
            if len(expected_ids) != len(output_ids):
                issues.append(
                    f"face: planned/output frame mismatch {len(expected_ids)} != {len(output_ids)}"
                )
            missing_ids = expected_ids - output_ids
            if missing_ids:
                issues.append(f"face: {len(missing_ids)} planned sample indices missing from raw")

    # Motion is full-fps and must contain one row for every AVI position in span.
    if motion_manifest:
        span = motion_manifest.get("analysis_span", {})
        first_pos = span.get("first_video_frame_position")
        last_pos = span.get("last_video_frame_position")
        if isinstance(first_pos, int) and isinstance(last_pos, int) and last_pos >= first_pos:
            expected_motion_rows = last_pos - first_pos + 1
            if motion_raw_rows != expected_motion_rows:
                issues.append(
                    f"motion: expected/output row mismatch {expected_motion_rows} != {motion_raw_rows}"
                )
        if "video_frame_position" in motion_raw.columns and motion_raw["video_frame_position"].duplicated().any():
            issues.append("motion: duplicate video_frame_position rows")

    # Pose stores one placeholder row for no-pose frames and 33 rows per returned pose.
    # Therefore unique sampled frame count, not total parquet row count, is the invariant.
    if pose_manifest:
        sampled = pose_manifest.get("output", {}).get("sampled_frames")
        if isinstance(sampled, int) and "video_frame_position" in pose_landmarks.columns:
            unique_pose_frames = int(pose_landmarks["video_frame_position"].nunique())
            if unique_pose_frames != sampled:
                issues.append(
                    f"pose: sampled/output frame mismatch {sampled} != {unique_pose_frames}"
                )

    provenance_commits = {
        str(m.get("attention_analysis_git_commit"))
        for m in (motion_manifest, pose_manifest, face_manifest)
        if m.get("attention_analysis_git_commit")
    }
    if len(provenance_commits) > 1:
        warnings.append(f"stage manifests contain multiple git commits: {sorted(provenance_commits)}")

    row_counts = {
        "face_frames": int(len(face_frames)),
        "face_raw": face_raw_rows,
        "motion_raw": motion_raw_rows,
        "pose_landmarks": pose_rows,
    }

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "subject": subject,
        "completion_status": "complete" if not issues else "incomplete",
        "extraction_complete": not issues,
        "qc_pass": None,
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_span_definition": "continuous from baseline start through Block2 end",
        "extraction_definition": "raw-first: Motion raw + Pose landmarks raw + Py-Feat Face raw",
        "config_path": str(config.path),
        "config_digest": config.digest,
        "issues": issues,
        "warnings": warnings,
        "row_counts": row_counts,
        "stage_manifests": {
            "face_prepare": str(paths["face_prepare_manifest"]),
            "face_raw": str(paths["face_raw_manifest"]),
            "motion": str(paths["motion_manifest"]),
            "pose": str(paths["pose_manifest"]),
        },
        "files": {
            key: _file_record(path, row_counts.get(key))
            for key, path in paths.items()
        },
        "optional_downstream_files_present": {
            key: path.is_file() for key, path in optional_derived.items()
        },
        "notes": [
            "Extraction completion is separate from downstream QC eligibility.",
            "Tracking, primary-face selection, eyelid features, Pose features, blink and PERCLOS are reconstructable downstream products and do not block extraction completion.",
            "The validator reads only identity/index columns from wide parquet files, avoiding a full 478-landmark Face table load during completion checks.",
        ],
    }
    final_manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["manifest"] = str(final_manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one completed formal RGB raw extraction")
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
