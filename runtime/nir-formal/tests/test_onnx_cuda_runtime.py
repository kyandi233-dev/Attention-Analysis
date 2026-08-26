from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import cv2


RUNTIME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME))

import onnx_cuda_runtime
from onnx_cuda_runtime import YoloCudaRuntime, create_cuda_session
from ritnet_onnx_runtime import RitnetOnnxRuntime
from ritnet_fullclass_qc import save_qc_pair


def test_cuda_provider_unavailable_refuses_cpu_fallback(monkeypatch, tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"placeholder")

    class FakeOrt:
        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(onnx_cuda_runtime, "_import_onnxruntime", lambda: FakeOrt)
    with pytest.raises(RuntimeError, match="refusing CPU fallback"):
        create_cuda_session(model)


def test_cuda_provider_priority_allows_registered_cpu_provider(monkeypatch, tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"placeholder")

    class FakeSession:
        def disable_fallback(self):
            pass

        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    class FakeOrt:
        SessionOptions = type(
            "SessionOptions",
            (),
            {
                "__init__": lambda self: None,
                "add_session_config_entry": lambda self, key, value: None,
            },
        )
        GraphOptimizationLevel = type("GraphOptimizationLevel", (), {"ORT_ENABLE_ALL": 99})

        @staticmethod
        def get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        @staticmethod
        def InferenceSession(*args, **kwargs):
            return FakeSession()

    monkeypatch.setattr(onnx_cuda_runtime, "_import_onnxruntime", lambda: FakeOrt)
    session = create_cuda_session(model)
    assert session.get_providers()[0] == "CUDAExecutionProvider"


def test_qc_png_writer_supports_non_ascii_windows_path(tmp_path):
    qc_dir = tmp_path / "中文" / "qc"
    roi = np.zeros((8, 12), dtype=np.uint8)
    labels = np.zeros((8, 12), dtype=np.uint8)
    labels[2:6, 3:9] = 3
    labels_path, overlay_path = save_qc_pair(
        qc_dir,
        "sub-100",
        {"phase": "baseline", "phase_segment": 1, "frame_idx": 7, "eye": "left"},
        roi,
        labels,
    )
    assert labels_path.is_file() and labels_path.stat().st_size > 0
    assert overlay_path.is_file() and overlay_path.stat().st_size > 0
    encoded = np.fromfile(labels_path, dtype=np.uint8)
    assert cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED) is not None


def test_pytorch_formal_mode_refuses_cpu_fallback(monkeypatch):
    torch = pytest.importorskip("torch")
    from run_pipeline import _require_pytorch_cuda

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="refusing silent CPU fallback"):
        _require_pytorch_cuda("0")


def test_ritnet_tail_batch_is_padded_to_16_and_padding_outputs_are_discarded():
    captured = {}

    class FakeSession:
        def run(self, output_names, feeds):
            captured["tensor"] = feeds["images"].copy()
            return [
                np.full((16, 2, 3), 3, dtype=np.uint8),
                np.full((16, 2, 3), 0.98, dtype=np.float32),
            ]

    runtime = RitnetOnnxRuntime.__new__(RitnetOnnxRuntime)
    runtime.session = FakeSession()
    runtime.input_name = "images"
    runtime.output_names = ["labels_u8", "pupil_prob"]
    runtime.precision = "fp32"
    runtime.last_timing = {}
    runtime._preprocess_one = lambda roi: np.full((2, 3), int(roi[0, 0]), dtype=np.uint8)
    runtime._postprocess_one = lambda pred, prob: {"found": bool((pred == 3).all())}

    results = runtime.infer_batch(
        [np.full((1, 1), value, dtype=np.uint8) for value in (10, 20, 30)]
    )

    tensor = captured["tensor"]
    assert tensor.shape == (16, 1, 2, 3)
    assert tensor.dtype == np.float32
    assert np.all(tensor[3:] == tensor[2])
    assert len(results) == 3
    assert runtime.last_timing["valid_batch_size"] == 3
    assert runtime.last_timing["padded_count"] == 13


def test_yolo_fp32_rows_keep_confidence_gate_and_restore_coordinates():
    class FakeSession:
        def run(self, output_names, feeds):
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

    runtime = YoloCudaRuntime.__new__(YoloCudaRuntime)
    runtime.session = FakeSession()
    runtime.input_name = "images"
    runtime.output_name = "output0"
    runtime._letterbox = lambda frame: (
        np.zeros((1, 3, 640, 640), dtype=np.float32),
        0.5,
        (10.0, 20.0),
    )

    detections = runtime.detect(
        np.zeros((100, 240, 3), dtype=np.uint8), confidence=0.40, max_det=20
    )

    assert len(detections) == 1
    assert detections[0] == ((0.0, 0.0, 200.0, 100.0), pytest.approx(0.90), 0)
