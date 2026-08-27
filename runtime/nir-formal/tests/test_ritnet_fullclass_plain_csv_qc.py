from __future__ import annotations

import csv

import cv2
import numpy as np

import ritnet_fullclass_qc_producer as producer
from ritnet_fullclass_qc_producer import produce_qc_artifacts


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
        self.position = int(value)
        return True

    def read(self):
        self.position += 1
        return True, np.full((100, 160, 3), 90, dtype=np.uint8)

    def release(self):
        pass


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def test_qc_consumer_accepts_current_plain_csv_contract(monkeypatch, tmp_path):
    monkeypatch.setattr(producer.cv2, "VideoCapture", FakeCapture)
    coverage = tmp_path / "frame_coverage.csv"
    eyes = tmp_path / "eye_metrics.csv"
    write_csv(
        coverage,
        ("phase", "phase_segment", "frame_idx", "coverage_status", "fixed_qc_anchor"),
        [
            {
                "phase": "block1",
                "phase_segment": 1,
                "frame_idx": 10,
                "coverage_status": "yolo_no_eye",
                "fixed_qc_anchor": True,
            }
        ],
    )
    write_csv(eyes, ("phase", "phase_segment", "frame_idx", "eye"), [])

    config = {
        "models": {"ritnet_fullclass_final": "models/missing-test.onnx"},
        "fullclass": {
            "roi": {
                "expand_horizontal_each_side": 0.30,
                "expand_vertical_each_side": 0.45,
                "padding_mode": "replicate",
            },
            "qc_image_max_count": 10,
            "qc_anomaly_max_per_reason": 5,
            "qc_pixel_evidence_max_eyes": 16,
            "qc_artifact_budget_bytes": 10_000_000,
        },
    }
    subject_dir = tmp_path / "out" / "sub-031"
    artifacts = produce_qc_artifacts(
        subject="sub-031",
        subject_dir=subject_dir,
        source_video=tmp_path / "source.avi",
        config=config,
        eye_metrics_path=eyes,
        frame_coverage_path=coverage,
    )

    assert artifacts.saved_image_count == 1
    assert artifacts.index_path.is_file()
    assert artifacts.pixel_evidence_path.is_file()
    assert coverage.suffix == ".csv"
    assert eyes.suffix == ".csv"
