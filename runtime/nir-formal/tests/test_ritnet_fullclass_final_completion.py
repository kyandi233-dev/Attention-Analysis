from __future__ import annotations

import csv
import gzip
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ritnet_fullclass_final_completion import finalize_subject, validate_final_completion
from ritnet_fullclass_qc_producer import (
    QC_INDEX_FIELDS,
    QC_PIXEL_EVIDENCE_NAME,
    QC_PIXEL_EVIDENCE_VERSION,
)
from ritnet_fullclass_schema import (
    EYE_METRIC_FIELDS,
    EYE_METRICS_SCHEMA_VERSION,
    FRAME_COVERAGE_FIELDS,
    FRAME_COVERAGE_SCHEMA_VERSION,
)


def _write_gz(path: Path, fields, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerow(row)


def _write_pixel_evidence(path: Path, subject: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.zeros((1, 400, 640), dtype=np.uint8)
    labels[:, 100:300, 100:540] = 1
    entropy = np.full((1, 400, 640), 0.25, dtype=np.float16)
    valid = np.ones((1, 400, 640), dtype=bool)
    np.savez_compressed(
        path,
        version=np.asarray(QC_PIXEL_EVIDENCE_VERSION),
        subject=np.asarray(subject),
        labels=labels,
        entropy=entropy,
        valid_source_mask=valid,
        phase=np.asarray(["block1"], dtype=np.str_),
        phase_segment=np.asarray([1], dtype=np.int32),
        frame_idx=np.asarray([10], dtype=np.int64),
        eye=np.asarray(["frame_left"], dtype=np.str_),
        reasons=np.asarray(["fixed_anchor"], dtype=np.str_),
    )


def _fixture(tmp_path: Path, *, output_limit=1_000_000):
    subject = "sub-031"
    subject_dir = tmp_path / subject
    eye_path = subject_dir / "data" / "eye_metrics.csv.gz"
    coverage_path = subject_dir / "data" / "frame_coverage.csv.gz"
    eye_row = {field: "" for field in EYE_METRIC_FIELDS}
    eye_row.update(
        eye_metrics_schema_version=EYE_METRICS_SCHEMA_VERSION,
        subject=subject,
        frame_idx=10,
        eye="frame_left",
        phase="block1",
        phase_segment=1,
    )
    coverage_row = {field: "" for field in FRAME_COVERAGE_FIELDS}
    coverage_row.update(
        frame_coverage_schema_version=FRAME_COVERAGE_SCHEMA_VERSION,
        subject=subject,
        phase="block1",
        phase_segment=1,
        frame_idx=10,
        coverage_status="single_eye_success",
        fixed_qc_anchor=True,
    )
    _write_gz(eye_path, EYE_METRIC_FIELDS, eye_row)
    _write_gz(coverage_path, FRAME_COVERAGE_FIELDS, coverage_row)

    image = subject_dir / "qc" / "images" / "sample.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"small-lossless-qc-evidence")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    index = subject_dir / "qc" / "qc_index.csv"
    with index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(QC_INDEX_FIELDS))
        writer.writeheader()
        writer.writerow(
            {
                "qc_index_schema_version": 1,
                "qc_selection_version": "test-selection",
                "qc_composite_version": "test-composite",
                "subject": subject,
                "phase": "block1",
                "phase_segment": 1,
                "frame_idx": 10,
                "coverage_status": "single_eye_success",
                "reasons": "fixed_anchor",
                "eyes": "frame_left",
                "source_frame_available": True,
                "left_overlay_available": True,
                "right_overlay_available": False,
                "image_path": "qc/images/sample.png",
                "image_sha256": digest,
                "image_size_bytes": image.stat().st_size,
            }
        )

    pixel_evidence = subject_dir / "qc" / QC_PIXEL_EVIDENCE_NAME
    _write_pixel_evidence(pixel_evidence, subject)

    work_identity = {"core_version": "test-core", "git_commit": "a" * 40}
    source_context = SimpleNamespace(
        run_dir=tmp_path / "source_formal_run",
        video=tmp_path / "source.avi",
        eye_rows=({},),
        frame_rows=({},),
        source_identity={"source_video_sha256": "b" * 64},
        video_resolution={"content_sha256": "b" * 64},
    )
    core = SimpleNamespace(
        subject=subject,
        subject_dir=subject_dir,
        eye_metrics=eye_path,
        frame_coverage=coverage_path,
        eye_row_count=1,
        frame_row_count=1,
        source_context=source_context,
        work_identity=work_identity,
    )
    qc = SimpleNamespace(
        selected_count=1,
        saved_image_count=1,
        skipped_for_budget_count=0,
        pixel_evidence_requested_count=1,
        pixel_evidence_saved_count=1,
        pixel_evidence_skipped_for_budget_count=0,
        image_bytes=image.stat().st_size,
        index_bytes=index.stat().st_size,
        pixel_evidence_bytes=pixel_evidence.stat().st_size,
        total_qc_bytes=image.stat().st_size + index.stat().st_size + pixel_evidence.stat().st_size,
    )
    return core, qc, output_limit, image, pixel_evidence


def test_finalize_writes_marker_last_and_self_validates(tmp_path):
    core, qc, limit, _image, _pixel = _fixture(tmp_path)
    marker = finalize_subject(core=core, qc=qc, output_limit_bytes=limit)
    assert marker.name == "completion.json"
    assert (core.subject_dir / "summary.json").is_file()
    assert (core.subject_dir / "manifest.json").is_file()
    result = validate_final_completion(
        core.subject_dir,
        expected_subject=core.subject,
        expected_work_identity=core.work_identity,
    )
    assert result.valid, result.reason
    assert result.completion["total_output_bytes"] <= limit
    assert result.completion["qc_pixel_evidence_saved_count"] == 1


def test_validator_detects_qc_image_tampering(tmp_path):
    core, qc, limit, image, _pixel = _fixture(tmp_path)
    finalize_subject(core=core, qc=qc, output_limit_bytes=limit)
    image.write_bytes(b"tampered")
    result = validate_final_completion(core.subject_dir, expected_subject=core.subject)
    assert not result.valid
    assert "QC image" in result.reason


def test_validator_detects_pixel_evidence_tampering(tmp_path):
    core, qc, limit, _image, pixel = _fixture(tmp_path)
    finalize_subject(core=core, qc=qc, output_limit_bytes=limit)
    pixel.write_bytes(b"tampered")
    result = validate_final_completion(core.subject_dir, expected_subject=core.subject)
    assert not result.valid


def test_invalid_existing_completion_is_not_overwritten(tmp_path):
    core, qc, limit, _image, _pixel = _fixture(tmp_path)
    marker = core.subject_dir / "completion.json"
    marker.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid"):
        finalize_subject(core=core, qc=qc, output_limit_bytes=limit)
    assert marker.read_text(encoding="utf-8") == "{}"


def test_hard_output_limit_prevents_completion_marker(tmp_path):
    core, qc, _limit, _image, _pixel = _fixture(tmp_path, output_limit=64)
    with pytest.raises(RuntimeError, match="hard limit"):
        finalize_subject(core=core, qc=qc, output_limit_bytes=64)
    assert not (core.subject_dir / "completion.json").exists()
