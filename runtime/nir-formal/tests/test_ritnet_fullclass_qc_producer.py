from __future__ import annotations

import csv
import gzip
import hashlib

import cv2
import numpy as np
import pytest

import ritnet_fullclass_qc_producer as producer
from ritnet_fullclass_qc_producer import (
    QC_INDEX_FIELDS,
    _prepare_eye_overlays,
    produce_qc_artifacts,
)
from ritnet_fullclass_roi import fixed_aspect_roi_geometry


def write_csv_gz(path, fieldnames, rows):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def config(*, budget=10_000_000):
    return {
        "models": {"ritnet_fullclass_final": "models/missing-test.onnx"},
        "fullclass": {
            "roi": {
                "expand_horizontal_each_side": 0.30,
                "expand_vertical_each_side": 0.45,
                "padding_mode": "replicate",
            },
            "qc_image_max_count": 10,
            "qc_anomaly_max_per_reason": 5,
            "qc_artifact_budget_bytes": budget,
        },
    }


class FakeCapture:
    def __init__(self, *_):
        self.position = 0

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 160
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 100
        return 0

    def set(self, prop, value):
        assert prop == cv2.CAP_PROP_POS_FRAMES
        self.position = int(value)
        return True

    def read(self):
        frame = np.full((100, 160, 3), 90, dtype=np.uint8)
        self.position += 1
        return True, frame

    def release(self):
        pass


def coverage(frame=10, *, fixed=True, status="yolo_no_eye"):
    return {
        "phase": "block1",
        "phase_segment": 1,
        "frame_idx": frame,
        "coverage_status": status,
        "fixed_qc_anchor": fixed,
    }


def test_producer_saves_fixed_yolo_miss_without_needing_ritnet(monkeypatch, tmp_path):
    monkeypatch.setattr(producer.cv2, "VideoCapture", FakeCapture)
    coverage_path = tmp_path / "coverage.csv.gz"
    eyes_path = tmp_path / "eyes.csv.gz"
    write_csv_gz(coverage_path, list(coverage().keys()), [coverage()])
    write_csv_gz(eyes_path, ["phase", "phase_segment", "frame_idx", "eye"], [])

    subject_dir = tmp_path / "ritnet-fullclass-final" / "sub-031"
    artifacts = produce_qc_artifacts(
        subject="sub-031",
        subject_dir=subject_dir,
        source_video=tmp_path / "dummy.avi",
        config=config(),
        eye_metrics_path=eyes_path,
        frame_coverage_path=coverage_path,
    )

    assert artifacts.selected_count == 1
    assert artifacts.saved_image_count == 1
    assert artifacts.skipped_for_budget_count == 0
    assert artifacts.total_qc_bytes <= 10_000_000
    images = list(artifacts.images_dir.glob("*.png"))
    assert len(images) == 1
    assert artifacts.index_path.is_file()

    with artifacts.index_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == QC_INDEX_FIELDS
    assert rows[0]["coverage_status"] == "yolo_no_eye"
    assert "fixed_anchor" in rows[0]["reasons"]
    assert rows[0]["source_frame_available"] == "True"
    assert rows[0]["image_sha256"] == hashlib.sha256(images[0].read_bytes()).hexdigest()
    assert int(rows[0]["image_size_bytes"]) == images[0].stat().st_size


def test_nonfixed_anomaly_is_skipped_when_byte_budget_is_too_small(monkeypatch, tmp_path):
    monkeypatch.setattr(producer.cv2, "VideoCapture", FakeCapture)
    coverage_path = tmp_path / "coverage.csv.gz"
    eyes_path = tmp_path / "eyes.csv.gz"
    row = coverage(fixed=False, status="yolo_no_eye")
    write_csv_gz(coverage_path, list(row.keys()), [row])
    write_csv_gz(eyes_path, ["phase", "phase_segment", "frame_idx", "eye"], [])

    subject_dir = tmp_path / "out" / "sub-031"
    artifacts = produce_qc_artifacts(
        subject="sub-031",
        subject_dir=subject_dir,
        source_video=tmp_path / "dummy.avi",
        config=config(budget=2000),
        eye_metrics_path=eyes_path,
        frame_coverage_path=coverage_path,
    )
    assert artifacts.selected_count == 1
    assert artifacts.saved_image_count == 0
    assert artifacts.skipped_for_budget_count == 1
    assert artifacts.total_qc_bytes <= 2000
    assert list(artifacts.images_dir.glob("*.png")) == []


def test_fixed_anchor_budget_overflow_fails_before_publishing_files(monkeypatch, tmp_path):
    monkeypatch.setattr(producer.cv2, "VideoCapture", FakeCapture)
    coverage_path = tmp_path / "coverage.csv.gz"
    eyes_path = tmp_path / "eyes.csv.gz"
    row = coverage(fixed=True, status="yolo_no_eye")
    write_csv_gz(coverage_path, list(row.keys()), [row])
    write_csv_gz(eyes_path, ["phase", "phase_segment", "frame_idx", "eye"], [])
    subject_dir = tmp_path / "out" / "sub-031"

    with pytest.raises(RuntimeError, match="mandatory fixed QC"):
        produce_qc_artifacts(
            subject="sub-031",
            subject_dir=subject_dir,
            source_video=tmp_path / "dummy.avi",
            config=config(budget=2000),
            eye_metrics_path=eyes_path,
            frame_coverage_path=coverage_path,
        )
    assert not (subject_dir / "qc" / "qc_index.csv").exists()
    assert list((subject_dir / "qc" / "images").glob("*.png")) == []


class FakeRuntime:
    def infer_batch(self, rois):
        labels = np.zeros((len(rois), 400, 640), dtype=np.uint8)
        labels[:, 100:300, 100:540] = 1
        labels[:, 150:250, 220:420] = 2
        labels[:, 180:220, 290:350] = 3
        return {"labels": labels}, {"valid_batch_size": len(rois)}


def metric_row():
    geometry = fixed_aspect_roi_geometry(
        bbox=(20, 20, 80, 60),
        frame_width=160,
        frame_height=100,
        expand_horizontal_each_side=0.30,
        expand_vertical_each_side=0.45,
        padding_mode="replicate",
    )
    row = {
        "eye": "frame_left",
        "ritnet_status": "success",
        "yolo_bbox_x1": 20,
        "yolo_bbox_y1": 20,
        "yolo_bbox_x2": 80,
        "yolo_bbox_y2": 60,
    }
    row.update(geometry.as_dict())
    return row


def test_sparse_qc_rerun_must_reproduce_saved_roi_geometry_exactly():
    frame = np.full((100, 160, 3), 90, dtype=np.uint8)
    row = metric_row()
    overlays = _prepare_eye_overlays(
        frame=frame,
        eye_rows={"frame_left": row},
        config=config(),
        runtime=FakeRuntime(),
    )
    assert overlays["frame_left"].shape == (400, 640, 3)

    bad = dict(row)
    bad["roi_requested_x1"] = int(bad["roi_requested_x1"]) + 1
    with pytest.raises(RuntimeError, match="QC ROI provenance mismatch"):
        _prepare_eye_overlays(
            frame=frame,
            eye_rows={"frame_left": bad},
            config=config(),
            runtime=FakeRuntime(),
        )
