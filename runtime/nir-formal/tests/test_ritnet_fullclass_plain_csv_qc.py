from __future__ import annotations

import cv2
import numpy as np

import ritnet_fullclass_qc_producer as producer
from ritnet_fullclass_io import atomic_write_csv, csv_fieldnames, iter_csv
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


def test_plain_csv_io_roundtrips_without_gzip(tmp_path):
    path = tmp_path / "eye_metrics.csv"
    fields = ("subject", "frame_idx", "eye")
    rows = [
        {"subject": "sub-031", "frame_idx": 10, "eye": "frame_left"},
        {"subject": "sub-031", "frame_idx": 10, "eye": "frame_right"},
    ]

    count = atomic_write_csv(path, rows, fields)

    assert count == 2
    assert path.suffix == ".csv"
    assert csv_fieldnames(path) == fields
    assert list(iter_csv(path)) == [
        {"subject": "sub-031", "frame_idx": "10", "eye": "frame_left"},
        {"subject": "sub-031", "frame_idx": "10", "eye": "frame_right"},
    ]


def test_qc_consumes_rows_without_reopening_final_csv(monkeypatch, tmp_path):
    monkeypatch.setattr(producer.cv2, "VideoCapture", FakeCapture)
    coverage_rows = [
        {
            "phase": "block1",
            "phase_segment": 1,
            "frame_idx": 10,
            "coverage_status": "yolo_no_eye",
            "fixed_qc_anchor": True,
        }
    ]
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

    artifacts = produce_qc_artifacts(
        subject="sub-031",
        subject_dir=tmp_path / "out" / "sub-031",
        source_video=tmp_path / "source.avi",
        config=config,
        eye_metric_rows=[],
        frame_coverage_rows=coverage_rows,
    )

    assert artifacts.saved_image_count == 1
    assert artifacts.index_path.is_file()
    assert artifacts.pixel_evidence_path.is_file()
