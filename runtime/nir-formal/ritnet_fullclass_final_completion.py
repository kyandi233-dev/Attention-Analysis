"""Integrity contract and atomic completion for final RITnet full-class outputs."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from math import log
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ritnet_fullclass_final_engine import CoreArtifacts
from ritnet_fullclass_io import (
    atomic_write_json,
    csv_gz_fieldnames,
    directory_size,
    iter_csv_gz,
)
from ritnet_fullclass_qc_producer import (
    QCArtifacts,
    QC_INDEX_FIELDS,
    QC_PIXEL_EVIDENCE_NAME,
    QC_PIXEL_EVIDENCE_VERSION,
)
from ritnet_fullclass_schema import (
    EYE_METRIC_FIELDS,
    EYE_METRICS_SCHEMA_VERSION,
    FRAME_COVERAGE_FIELDS,
    FRAME_COVERAGE_SCHEMA_VERSION,
    validate_exact_schema,
)
from ritnet_label_store import sha256_file


FINAL_MANIFEST_SCHEMA_VERSION = 1
FINAL_SUMMARY_SCHEMA_VERSION = 1
FINAL_COMPLETION_SCHEMA_VERSION = 1
FINAL_COMPLETION_STATUS = "complete"
COMPLETION_NAME = "completion.json"
SUMMARY_NAME = "summary.json"
MANIFEST_NAME = "manifest.json"
REQUIRED_DATA_ARTIFACTS = (
    "data/eye_metrics.csv.gz",
    "data/frame_coverage.csv.gz",
    "qc/qc_index.csv",
    f"qc/{QC_PIXEL_EVIDENCE_NAME}",
)
PIXEL_EVIDENCE_KEYS = frozenset(
    {
        "version",
        "subject",
        "labels",
        "entropy",
        "valid_source_mask",
        "phase",
        "phase_segment",
        "frame_idx",
        "eye",
        "reasons",
    }
)


@dataclass(frozen=True)
class FinalCompletionValidation:
    valid: bool
    reason: str
    completion: dict[str, Any] | None = None


def _load_json(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _json_size(payload: Mapping[str, Any]) -> int:
    text = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    return len(text.encode("utf-8"))


def _count_and_validate_csv_gz(
    path: Path,
    *,
    expected_fields: tuple[str, ...],
    expected_subject: str,
    schema_field: str,
    schema_version: int,
) -> int:
    validate_exact_schema(csv_gz_fieldnames(path), expected_fields)
    count = 0
    for row in iter_csv_gz(path):
        count += 1
        if str(row.get("subject") or "") != expected_subject:
            raise ValueError(f"subject mismatch in {path.name} row {count}")
        if int(float(row.get(schema_field) or -1)) != int(schema_version):
            raise ValueError(
                f"schema version mismatch in {path.name} row {count}: "
                f"{row.get(schema_field)!r} != {schema_version}"
            )
    return count


def _read_qc_index(subject_dir: Path, expected_subject: str) -> tuple[int, int]:
    index_path = Path(subject_dir) / "qc" / "qc_index.csv"
    with index_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != QC_INDEX_FIELDS:
            raise ValueError("qc_index.csv schema mismatch")
        count = 0
        image_bytes = 0
        seen_paths: set[str] = set()
        for row in reader:
            count += 1
            if str(row.get("subject") or "") != expected_subject:
                raise ValueError(f"qc_index subject mismatch at row {count}")
            relative = str(row.get("image_path") or "")
            if not relative or relative in seen_paths:
                raise ValueError(f"invalid/duplicate qc image path at row {count}: {relative!r}")
            seen_paths.add(relative)
            image = (Path(subject_dir) / relative).resolve()
            try:
                image.relative_to(Path(subject_dir).resolve())
            except ValueError as exc:
                raise ValueError(f"qc image escapes subject directory: {relative!r}") from exc
            if not image.is_file():
                raise FileNotFoundError(f"missing QC image: {relative}")
            actual_size = int(image.stat().st_size)
            expected_size = int(row.get("image_size_bytes") or -1)
            if actual_size != expected_size:
                raise ValueError(
                    f"QC image size mismatch for {relative}: {actual_size} != {expected_size}"
                )
            actual_hash = sha256_file(image)
            if actual_hash != str(row.get("image_sha256") or ""):
                raise ValueError(f"QC image SHA256 mismatch: {relative}")
            image_bytes += actual_size
    return count, image_bytes


def _validate_qc_pixel_evidence(subject_dir: Path, expected_subject: str) -> tuple[int, int]:
    path = Path(subject_dir) / "qc" / QC_PIXEL_EVIDENCE_NAME
    if not path.is_file():
        raise FileNotFoundError(f"missing sparse QC pixel evidence: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if frozenset(payload.files) != PIXEL_EVIDENCE_KEYS:
            raise ValueError(
                "QC pixel evidence key contract mismatch: "
                f"expected={sorted(PIXEL_EVIDENCE_KEYS)}, got={sorted(payload.files)}"
            )
        version = str(np.asarray(payload["version"]).item())
        subject = str(np.asarray(payload["subject"]).item())
        if version != QC_PIXEL_EVIDENCE_VERSION:
            raise ValueError(f"QC pixel evidence version mismatch: {version!r}")
        if subject != expected_subject:
            raise ValueError(f"QC pixel evidence subject mismatch: {subject!r} != {expected_subject!r}")

        labels = np.asarray(payload["labels"])
        entropy = np.asarray(payload["entropy"])
        valid = np.asarray(payload["valid_source_mask"])
        if labels.ndim != 3 or labels.shape[1:] != (400, 640) or labels.dtype != np.uint8:
            raise ValueError(f"QC pixel labels contract mismatch: {labels.shape} {labels.dtype}")
        count = int(labels.shape[0])
        expected_shape = (count, 400, 640)
        if entropy.shape != expected_shape or entropy.dtype != np.float16:
            raise ValueError(f"QC pixel entropy contract mismatch: {entropy.shape} {entropy.dtype}")
        if valid.shape != expected_shape or valid.dtype != np.bool_:
            raise ValueError(f"QC pixel valid-source mask mismatch: {valid.shape} {valid.dtype}")
        if count and int(labels.max()) > 3:
            raise ValueError("QC pixel labels contain values outside {0,1,2,3}")
        if not np.isfinite(entropy).all():
            raise ValueError("QC pixel entropy contains non-finite values")
        if entropy.size:
            minimum = float(entropy.min())
            maximum = float(entropy.max())
            if minimum < -1e-3 or maximum > log(4.0) + 2e-3:
                raise ValueError(f"QC pixel entropy outside [0,log(4)]: {minimum}..{maximum}")
        if count and not np.all(valid.reshape(count, -1).any(axis=1)):
            raise ValueError("QC pixel evidence contains an eye with no source-backed pixels")

        for name in ("phase", "eye", "reasons"):
            values = np.asarray(payload[name])
            if values.shape != (count,) or values.dtype.kind not in {"U", "S"}:
                raise ValueError(f"QC pixel metadata {name} contract mismatch: {values.shape} {values.dtype}")
        phase_segment = np.asarray(payload["phase_segment"])
        frame_idx = np.asarray(payload["frame_idx"])
        if phase_segment.shape != (count,) or phase_segment.dtype != np.int32:
            raise ValueError("QC pixel phase_segment contract mismatch")
        if frame_idx.shape != (count,) or frame_idx.dtype != np.int64:
            raise ValueError("QC pixel frame_idx contract mismatch")
        eyes = {str(value) for value in np.asarray(payload["eye"]).tolist()}
        if not eyes.issubset({"frame_left", "frame_right"}):
            raise ValueError(f"QC pixel evidence contains unsupported eye labels: {sorted(eyes)}")
    return count, int(path.stat().st_size)


def _artifact_record(subject_dir: Path, relative: str) -> dict[str, Any]:
    path = Path(subject_dir) / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def build_summary(
    *,
    core: CoreArtifacts,
    qc: QCArtifacts,
    output_limit_bytes: int,
) -> dict[str, Any]:
    return {
        "schema_version": FINAL_SUMMARY_SCHEMA_VERSION,
        "subject": core.subject,
        "status": FINAL_COMPLETION_STATUS,
        "source_video": str(core.source_context.video),
        "source_video_sha256": core.source_context.source_identity["source_video_sha256"],
        "source_eye_row_count": len(core.source_context.eye_rows),
        "source_frame_row_count": len(core.source_context.frame_rows),
        "eye_metric_row_count": core.eye_row_count,
        "frame_coverage_row_count": core.frame_row_count,
        "qc_selected_frame_count": qc.selected_count,
        "qc_saved_image_count": qc.saved_image_count,
        "qc_skipped_for_budget_count": qc.skipped_for_budget_count,
        "qc_pixel_evidence_requested_count": qc.pixel_evidence_requested_count,
        "qc_pixel_evidence_saved_count": qc.pixel_evidence_saved_count,
        "qc_pixel_evidence_skipped_for_budget_count": qc.pixel_evidence_skipped_for_budget_count,
        "qc_pixel_evidence_bytes": qc.pixel_evidence_bytes,
        "qc_total_bytes": qc.total_qc_bytes,
        "output_limit_bytes": int(output_limit_bytes),
    }


def build_manifest(*, core: CoreArtifacts, qc: QCArtifacts) -> dict[str, Any]:
    subject_dir = core.subject_dir
    artifacts = {
        relative: _artifact_record(subject_dir, relative)
        for relative in REQUIRED_DATA_ARTIFACTS
    }
    return {
        "schema_version": FINAL_MANIFEST_SCHEMA_VERSION,
        "subject": core.subject,
        "source_identity": dict(core.source_context.source_identity),
        "source_video_resolution": dict(core.source_context.video_resolution),
        "work_identity": dict(core.work_identity),
        "artifacts": artifacts,
        "qc": {
            "selected_count": qc.selected_count,
            "saved_image_count": qc.saved_image_count,
            "skipped_for_budget_count": qc.skipped_for_budget_count,
            "pixel_evidence_requested_count": qc.pixel_evidence_requested_count,
            "pixel_evidence_saved_count": qc.pixel_evidence_saved_count,
            "pixel_evidence_skipped_for_budget_count": qc.pixel_evidence_skipped_for_budget_count,
            "image_bytes": qc.image_bytes,
            "index_bytes": qc.index_bytes,
            "pixel_evidence_bytes": qc.pixel_evidence_bytes,
            "total_qc_bytes": qc.total_qc_bytes,
        },
    }


def validate_final_completion(
    subject_dir: Path,
    *,
    expected_subject: str | None = None,
    expected_work_identity: Mapping[str, Any] | None = None,
) -> FinalCompletionValidation:
    subject_dir = Path(subject_dir)
    completion_path = subject_dir / COMPLETION_NAME
    if not completion_path.is_file():
        return FinalCompletionValidation(False, f"missing {COMPLETION_NAME}")
    try:
        completion = _load_json(completion_path)
        if completion.get("schema_version") != FINAL_COMPLETION_SCHEMA_VERSION:
            raise ValueError("unsupported final completion schema")
        if completion.get("status") != FINAL_COMPLETION_STATUS:
            raise ValueError(f"completion status is not {FINAL_COMPLETION_STATUS!r}")
        subject = str(completion.get("subject") or "")
        if not subject:
            raise ValueError("completion has no subject")
        if expected_subject is not None and subject != expected_subject:
            raise ValueError(f"completion subject mismatch: {subject} != {expected_subject}")

        summary_path = subject_dir / SUMMARY_NAME
        manifest_path = subject_dir / MANIFEST_NAME
        if not summary_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError("summary.json or manifest.json is missing")
        if sha256_file(summary_path) != completion.get("summary_sha256"):
            raise ValueError("summary.json SHA256 mismatch")
        if sha256_file(manifest_path) != completion.get("manifest_sha256"):
            raise ValueError("manifest.json SHA256 mismatch")
        summary = _load_json(summary_path)
        manifest = _load_json(manifest_path)
        if summary.get("schema_version") != FINAL_SUMMARY_SCHEMA_VERSION:
            raise ValueError("summary schema mismatch")
        if manifest.get("schema_version") != FINAL_MANIFEST_SCHEMA_VERSION:
            raise ValueError("manifest schema mismatch")
        if summary.get("subject") != subject or manifest.get("subject") != subject:
            raise ValueError("summary/manifest subject mismatch")
        work_identity = manifest.get("work_identity")
        if not isinstance(work_identity, dict):
            raise ValueError("manifest work_identity must be an object")
        if expected_work_identity is not None and work_identity != dict(expected_work_identity):
            raise ValueError("manifest work_identity does not match requested run identity")

        eye_path = subject_dir / "data" / "eye_metrics.csv.gz"
        coverage_path = subject_dir / "data" / "frame_coverage.csv.gz"
        eye_count = _count_and_validate_csv_gz(
            eye_path,
            expected_fields=EYE_METRIC_FIELDS,
            expected_subject=subject,
            schema_field="eye_metrics_schema_version",
            schema_version=EYE_METRICS_SCHEMA_VERSION,
        )
        frame_count = _count_and_validate_csv_gz(
            coverage_path,
            expected_fields=FRAME_COVERAGE_FIELDS,
            expected_subject=subject,
            schema_field="frame_coverage_schema_version",
            schema_version=FRAME_COVERAGE_SCHEMA_VERSION,
        )
        qc_count, qc_image_bytes = _read_qc_index(subject_dir, subject)
        pixel_count, pixel_bytes = _validate_qc_pixel_evidence(subject_dir, subject)
        if eye_count != int(summary.get("eye_metric_row_count", -1)):
            raise ValueError("summary eye metric row count mismatch")
        if frame_count != int(summary.get("frame_coverage_row_count", -1)):
            raise ValueError("summary frame coverage row count mismatch")
        if qc_count != int(summary.get("qc_saved_image_count", -1)):
            raise ValueError("summary QC image count mismatch")
        if pixel_count != int(summary.get("qc_pixel_evidence_saved_count", -1)):
            raise ValueError("summary QC pixel evidence count mismatch")
        if pixel_bytes != int(summary.get("qc_pixel_evidence_bytes", -1)):
            raise ValueError("summary QC pixel evidence byte count mismatch")
        if qc_image_bytes + pixel_bytes > int(summary.get("qc_total_bytes", -1)):
            raise ValueError("summary QC byte count is smaller than indexed QC evidence")

        artifact_records = manifest.get("artifacts")
        if not isinstance(artifact_records, dict) or tuple(artifact_records) != REQUIRED_DATA_ARTIFACTS:
            raise ValueError("manifest required artifact contract mismatch")
        for relative in REQUIRED_DATA_ARTIFACTS:
            expected = artifact_records[relative]
            if not isinstance(expected, dict):
                raise ValueError(f"manifest artifact record invalid: {relative}")
            actual = _artifact_record(subject_dir, relative)
            if actual != expected:
                raise ValueError(f"manifest artifact mismatch: {relative}")

        actual_total = directory_size(subject_dir)
        limit = int(completion.get("output_limit_bytes", -1))
        if limit <= 0 or actual_total > limit:
            raise ValueError(f"final subject output exceeds size limit: {actual_total} > {limit}")
        if actual_total != int(completion.get("total_output_bytes", -1)):
            raise ValueError("completion total_output_bytes mismatch")
        return FinalCompletionValidation(True, "valid final completion", completion)
    except Exception as exc:
        return FinalCompletionValidation(False, str(exc), locals().get("completion"))


def finalize_subject(
    *,
    core: CoreArtifacts,
    qc: QCArtifacts,
    output_limit_bytes: int,
) -> Path:
    subject_dir = Path(core.subject_dir)
    limit = int(output_limit_bytes)
    if limit <= 0:
        raise ValueError("final output limit must be positive")
    completion_path = subject_dir / COMPLETION_NAME
    if completion_path.exists():
        existing = validate_final_completion(
            subject_dir,
            expected_subject=core.subject,
            expected_work_identity=core.work_identity,
        )
        if existing.valid:
            return completion_path
        raise RuntimeError(
            "completion.json exists but is invalid; refusing to overwrite ambiguous final output: "
            + existing.reason
        )

    summary = build_summary(core=core, qc=qc, output_limit_bytes=limit)
    manifest = build_manifest(core=core, qc=qc)
    summary_path = subject_dir / SUMMARY_NAME
    manifest_path = subject_dir / MANIFEST_NAME
    if summary_path.exists() or manifest_path.exists():
        raise RuntimeError(
            "summary.json/manifest.json already exists without a valid completion marker; "
            "refusing to overwrite incomplete output automatically"
        )
    atomic_write_json(summary_path, summary)
    atomic_write_json(manifest_path, manifest)

    eye_count = _count_and_validate_csv_gz(
        core.eye_metrics,
        expected_fields=EYE_METRIC_FIELDS,
        expected_subject=core.subject,
        schema_field="eye_metrics_schema_version",
        schema_version=EYE_METRICS_SCHEMA_VERSION,
    )
    frame_count = _count_and_validate_csv_gz(
        core.frame_coverage,
        expected_fields=FRAME_COVERAGE_FIELDS,
        expected_subject=core.subject,
        schema_field="frame_coverage_schema_version",
        schema_version=FRAME_COVERAGE_SCHEMA_VERSION,
    )
    qc_count, _qc_image_bytes = _read_qc_index(subject_dir, core.subject)
    pixel_count, pixel_bytes = _validate_qc_pixel_evidence(subject_dir, core.subject)
    if eye_count != core.eye_row_count or frame_count != core.frame_row_count:
        raise RuntimeError("final table row counts changed before completion")
    if qc_count != qc.saved_image_count:
        raise RuntimeError("QC index row count changed before completion")
    if pixel_count != qc.pixel_evidence_saved_count or pixel_bytes != qc.pixel_evidence_bytes:
        raise RuntimeError("QC pixel evidence changed before completion")

    completion = {
        "schema_version": FINAL_COMPLETION_SCHEMA_VERSION,
        "status": FINAL_COMPLETION_STATUS,
        "subject": core.subject,
        "summary_sha256": sha256_file(summary_path),
        "manifest_sha256": sha256_file(manifest_path),
        "eye_metric_row_count": eye_count,
        "frame_coverage_row_count": frame_count,
        "qc_saved_image_count": qc_count,
        "qc_pixel_evidence_saved_count": pixel_count,
        "output_limit_bytes": limit,
        "work_identity_sha256": __import__("hashlib").sha256(
            json.dumps(core.work_identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    existing_bytes = directory_size(subject_dir)
    completion["total_output_bytes"] = existing_bytes + _json_size(completion)
    for _ in range(4):
        total = existing_bytes + _json_size(completion)
        if total == completion["total_output_bytes"]:
            break
        completion["total_output_bytes"] = total
    predicted_total = existing_bytes + _json_size(completion)
    completion["total_output_bytes"] = predicted_total
    if predicted_total > limit:
        raise RuntimeError(
            f"final subject output would exceed hard limit: {predicted_total} > {limit}"
        )
    atomic_write_json(completion_path, completion)

    validation = validate_final_completion(
        subject_dir,
        expected_subject=core.subject,
        expected_work_identity=core.work_identity,
    )
    if not validation.valid:
        raise RuntimeError("final completion self-validation failed: " + validation.reason)
    return completion_path