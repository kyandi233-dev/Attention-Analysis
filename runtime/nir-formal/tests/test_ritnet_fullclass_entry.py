from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ritnet_fullclass_source import load_source_eye_rows, resolve_source_video


FIELDS = [
    "subject",
    "phase",
    "phase_segment",
    "frame_idx",
    "video_time_ms",
    "unix_ms",
    "phase_time_ms",
    "eye",
    "source",
    "redetect_reason",
    "frame_status",
    "status",
    "anchor_yolo_confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "yolo_batch_size",
]


def write_eye_rows(path: Path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def valid_row(frame_idx=100, eye="frame_left", x1=10, x2=50, subject=""):
    return {
        "subject": subject,
        "phase": "block1",
        "phase_segment": 1,
        "frame_idx": frame_idx,
        "video_time_ms": 1000,
        "unix_ms": 2000,
        "phase_time_ms": 500,
        "eye": eye,
        "source": "yolo",
        "redetect_reason": "tracker_disabled",
        "frame_status": "two_eyes",
        "status": "observed",
        "anchor_yolo_confidence": 0.8,
        "bbox_x1": x1,
        "bbox_y1": 20,
        "bbox_x2": x2,
        "bbox_y2": 50,
        "yolo_batch_size": 8,
    }


def test_source_loader_accepts_missing_subject_column_value_without_modifying_source(tmp_path):
    source = tmp_path / "eyes.csv"
    write_eye_rows(source, [valid_row()])
    before = source.read_bytes()

    fields, rows = load_source_eye_rows(source, "sub-031")

    assert "subject" in fields
    assert rows[0]["eye"] == "frame_left"
    assert source.read_bytes() == before


def test_source_loader_rejects_unknown_eye_label(tmp_path):
    source = tmp_path / "eyes.csv"
    write_eye_rows(source, [valid_row(eye="left_eye_typo")])
    with pytest.raises(ValueError, match="unsupported eye label"):
        load_source_eye_rows(source, "sub-031")


def test_source_loader_rejects_reversed_pair_identity(tmp_path):
    source = tmp_path / "eyes.csv"
    write_eye_rows(
        source,
        [
            valid_row(eye="frame_left", x1=100, x2=140),
            valid_row(eye="frame_right", x1=10, x2=50),
        ],
    )
    with pytest.raises(ValueError, match="left/right identity violation"):
        load_source_eye_rows(source, "sub-031")


def test_source_video_rediscovery_accepts_identical_drive_candidates(tmp_path):
    root_a = tmp_path / "E"
    root_b = tmp_path / "F"
    for root in (root_a, root_b):
        video = root / "sub-031_" / "nir" / "sub-031_nir.avi"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"same-video")

    completion = {"video": "Z:/old/sub-031_/nir/sub-031_nir.avi"}
    config = {"data": {"roots": [str(root_a), str(root_b)]}}
    resolved, info = resolve_source_video(
        completion=completion,
        config=config,
        subject="sub-031",
    )

    assert resolved in {
        (root_a / "sub-031_" / "nir" / "sub-031_nir.avi").resolve(),
        (root_b / "sub-031_" / "nir" / "sub-031_nir.avi").resolve(),
    }
    assert info["candidate_count"] == 2
    assert info["resolution_reason"] == "rediscovered_identical_content"


def test_source_video_rediscovery_rejects_different_content(tmp_path):
    root_a = tmp_path / "E"
    root_b = tmp_path / "F"
    video_a = root_a / "sub-031_" / "nir" / "sub-031_nir.avi"
    video_b = root_b / "sub-031_" / "nir" / "sub-031_nir.avi"
    video_a.parent.mkdir(parents=True)
    video_b.parent.mkdir(parents=True)
    video_a.write_bytes(b"video-a")
    video_b.write_bytes(b"video-b")

    completion = {"video": "Z:/old/sub-031_/nir/sub-031_nir.avi"}
    config = {"data": {"roots": [str(root_a), str(root_b)]}}
    with pytest.raises(RuntimeError, match="different SHA256"):
        resolve_source_video(
            completion=completion,
            config=config,
            subject="sub-031",
        )
