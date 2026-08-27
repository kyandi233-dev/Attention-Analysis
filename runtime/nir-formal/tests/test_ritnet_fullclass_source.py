from __future__ import annotations

import csv

import pytest

from ritnet_fullclass_source import load_source_eye_rows


FIELDS = [
    "subject", "phase", "phase_segment", "frame_idx", "video_time_ms", "unix_ms", "phase_time_ms",
    "eye", "source", "redetect_reason", "frame_status", "status", "anchor_yolo_confidence",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "yolo_batch_size",
]


def row(eye, x1, x2, frame=100):
    return {
        "subject": "sub-031", "phase": "block1", "phase_segment": 1, "frame_idx": frame,
        "video_time_ms": 1000, "unix_ms": 2000, "phase_time_ms": 500,
        "eye": eye, "source": "yolo", "redetect_reason": "tracker_disabled",
        "frame_status": "two_eyes", "status": "observed", "anchor_yolo_confidence": 0.9,
        "bbox_x1": x1, "bbox_y1": 20, "bbox_x2": x2, "bbox_y2": 60,
        "yolo_batch_size": 8,
    }


def write(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_source_eye_loader_accepts_ordered_pair(tmp_path):
    path = tmp_path / "eyes.csv"
    write(path, [row("frame_left", 10, 50), row("frame_right", 100, 140)])
    fields, rows = load_source_eye_rows(path, "sub-031")
    assert tuple(fields) == tuple(FIELDS)
    assert len(rows) == 2


def test_source_eye_loader_rejects_reversed_identity(tmp_path):
    path = tmp_path / "eyes.csv"
    write(path, [row("frame_left", 100, 140), row("frame_right", 10, 50)])
    with pytest.raises(ValueError, match="identity violation"):
        load_source_eye_rows(path, "sub-031")


def test_source_eye_loader_requires_final_source_columns(tmp_path):
    path = tmp_path / "eyes.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_idx", "eye"])
        writer.writeheader()
        writer.writerow({"frame_idx": 1, "eye": "frame_left"})
    with pytest.raises(ValueError, match="missing final-source columns"):
        load_source_eye_rows(path, "sub-031")
