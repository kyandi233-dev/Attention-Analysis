from __future__ import annotations

import csv
from pathlib import Path

import pytest

import run_ritnet_fullclass_native_extension as implementation
from run_ritnet_fullclass_extension import (
    _install_subject_identity_guard,
    _resolve_source_video,
)


FIELDS = [
    "frame_idx",
    "eye",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "anchor_yolo_confidence",
    "roi_x1",
    "roi_y1",
    "roi_x2",
    "roi_y2",
]


def write_eye_rows(path: Path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def valid_row(frame_idx=100, eye="frame_left", x1=10, x2=50):
    return {
        "frame_idx": frame_idx,
        "eye": eye,
        "bbox_x1": x1,
        "bbox_y1": 20,
        "bbox_x2": x2,
        "bbox_y2": 50,
        "anchor_yolo_confidence": 0.8,
        "roi_x1": 5,
        "roi_y1": 10,
        "roi_x2": 60,
        "roi_y2": 60,
    }


def test_canonical_entry_backfills_subject_without_modifying_source(tmp_path):
    source = tmp_path / "eyes.csv"
    write_eye_rows(source, [valid_row()])

    before = source.read_bytes()
    original = implementation._source_rows
    try:
        _install_subject_identity_guard()
        output_fields, rows = implementation._source_rows(source, "sub-031")
    finally:
        implementation._source_rows = original

    assert output_fields[0] == "subject"
    assert rows[0]["subject"] == "sub-031"
    assert source.read_bytes() == before


def test_canonical_entry_rejects_unknown_eye_label(tmp_path):
    source = tmp_path / "eyes.csv"
    write_eye_rows(source, [valid_row(eye="left_eye_typo")])
    original = implementation._source_rows
    try:
        _install_subject_identity_guard()
        with pytest.raises(ValueError, match="unsupported eye label"):
            implementation._source_rows(source, "sub-031")
    finally:
        implementation._source_rows = original


def test_canonical_entry_rejects_reversed_pair_identity(tmp_path):
    source = tmp_path / "eyes.csv"
    write_eye_rows(
        source,
        [
            valid_row(eye="frame_left", x1=100, x2=140),
            valid_row(eye="frame_right", x1=10, x2=50),
        ],
    )
    original = implementation._source_rows
    try:
        _install_subject_identity_guard()
        with pytest.raises(ValueError, match="identity/order violation"):
            implementation._source_rows(source, "sub-031")
    finally:
        implementation._source_rows = original


def test_source_video_rediscovery_accepts_identical_drive_candidates(monkeypatch, tmp_path):
    root_a = tmp_path / "E"
    root_b = tmp_path / "F"
    for root in (root_a, root_b):
        video = root / "sub-031_" / "nir" / "sub-031_nir.avi"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"same-video")

    marker = {"video": "Z:/old/sub-031_/nir/sub-031_nir.avi"}
    config = {"data": {"roots": [str(root_a), str(root_b)]}}
    resolved, info = _resolve_source_video(marker=marker, config=config, subject="sub-031")

    assert resolved in {
        (root_a / "sub-031_" / "nir" / "sub-031_nir.avi").resolve(),
        (root_b / "sub-031_" / "nir" / "sub-031_nir.avi").resolve(),
    }
    assert info["candidate_count"] == 2
    assert info["resolution_reason"] == "rediscovered_by_subject_filename_and_identical_sha256"


def test_source_video_rediscovery_rejects_different_content(tmp_path):
    root_a = tmp_path / "E"
    root_b = tmp_path / "F"
    video_a = root_a / "sub-031_" / "nir" / "sub-031_nir.avi"
    video_b = root_b / "sub-031_" / "nir" / "sub-031_nir.avi"
    video_a.parent.mkdir(parents=True)
    video_b.parent.mkdir(parents=True)
    video_a.write_bytes(b"video-a")
    video_b.write_bytes(b"video-b")

    marker = {"video": "Z:/old/sub-031_/nir/sub-031_nir.avi"}
    config = {"data": {"roots": [str(root_a), str(root_b)]}}
    with pytest.raises(SystemExit, match="different content"):
        _resolve_source_video(marker=marker, config=config, subject="sub-031")
