from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


RUNTIME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME))

import directml_runtime
from directml_runtime import YoloDirectMLRuntime, create_directml_session
from ritnet_runtime import RitnetRuntime


def test_directml_unavailable_fails_before_cpu_session(monkeypatch, tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"placeholder")

    class FakeOrt:
        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(directml_runtime, "_import_onnxruntime", lambda: FakeOrt)
    with pytest.raises(RuntimeError, match="refusing CPU fallback"):
        create_directml_session(model)


def test_ritnet_tail_batch_is_padded_to_16_and_padding_outputs_are_discarded():
    captured = {}

    class FakeSession:
        def run(self, output_names, feeds):
            captured["output_names"] = output_names
            captured["tensor"] = feeds["images"].copy()
            labels = np.full((16, 2, 3), 3, dtype=np.uint8)
            pupil_prob = np.full((16, 2, 3), 0.98, dtype=np.float32)
            return [labels, pupil_prob]

    runtime = RitnetRuntime.__new__(RitnetRuntime)
    runtime.session = FakeSession()
    runtime.input_name = "images"
    runtime.output_names = ["labels_u8", "pupil_prob"]
    runtime.precision = "fp32"
    runtime.device = "dml:0"
    runtime.last_timing = {}
    runtime._preprocess_one = lambda roi: np.full((2, 3), int(roi[0, 0]), dtype=np.uint8)
    runtime._postprocess_one = lambda pred, prob: {
        "found": bool((pred == 3).all()),
        "mean_probability": float(prob.mean()),
    }

    rois = [np.full((1, 1), value, dtype=np.uint8) for value in (10, 20, 30)]
    results = runtime.infer_batch(rois)

    tensor = captured["tensor"]
    assert tensor.shape == (16, 1, 2, 3)
    assert tensor.dtype == np.float32
    assert np.all(tensor[3:] == tensor[2])
    assert len(results) == 3
    assert all(result["found"] for result in results)
    assert all(result["mean_probability"] == pytest.approx(0.98) for result in results)
    assert runtime.last_timing["batch_size"] == 16
    assert runtime.last_timing["valid_batch_size"] == 3
    assert runtime.last_timing["padded_count"] == 13


def test_yolo_end_to_end_rows_keep_confidence_gate_and_restore_coordinates():
    class FakeSession:
        def run(self, output_names, feeds):
            assert output_names == ["output0"]
            assert feeds["images"].shape == (1, 3, 640, 640)
            return [
                np.array(
                    [[
                        [10.0, 20.0, 110.0, 70.0, 0.90, 0.0],
                        [20.0, 30.0, 40.0, 50.0, 0.39, 0.0],
                        [30.0, 40.0, 60.0, 80.0, 0.95, 1.0],
                    ]],
                    dtype=np.float32,
                )
            ]

    runtime = YoloDirectMLRuntime.__new__(YoloDirectMLRuntime)
    runtime.session = FakeSession()
    runtime.input_name = "images"
    runtime.output_name = "output0"
    runtime._letterbox = lambda frame: (
        np.zeros((1, 3, 640, 640), dtype=np.float32),
        0.5,
        (10.0, 20.0),
    )

    detections = runtime.detect(
        np.zeros((100, 240, 3), dtype=np.uint8),
        confidence=0.40,
        max_det=20,
    )

    assert len(detections) == 2
    assert detections[0] == ((40.0, 40.0, 100.0, 100.0), pytest.approx(0.95), 1)
    assert detections[1] == ((0.0, 0.0, 200.0, 100.0), pytest.approx(0.90), 0)
