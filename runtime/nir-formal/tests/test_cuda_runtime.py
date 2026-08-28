from __future__ import annotations

import numpy as np
import pytest

import cuda_runtime
from cuda_runtime import YoloCudaRuntime, create_cuda_session, parse_device_id


def test_cuda_device_parser_accepts_index_and_cuda_prefix():
    assert parse_device_id(0) == 0
    assert parse_device_id("1") == 1
    assert parse_device_id("cuda:2") == 2
    with pytest.raises(ValueError, match="CUDA device"):
        parse_device_id("dml:0")
    with pytest.raises(ValueError, match="CUDA device"):
        parse_device_id("-1")


def test_cuda_unavailable_fails_before_cpu_session(monkeypatch, tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"placeholder")

    class FakeOrt:
        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(cuda_runtime, "_import_onnxruntime", lambda: FakeOrt)
    with pytest.raises(RuntimeError, match="refusing CPU fallback"):
        create_cuda_session(model)


def test_cuda_session_is_provider_only_tf32_off_and_fallback_disabled(monkeypatch, tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"placeholder")
    captured = {}

    class FakeOptions:
        def __init__(self):
            self.graph_optimization_level = None

        def add_session_config_entry(self, key, value):
            captured["session_config"] = (key, value)

    class FakeSession:
        def __init__(self, path, *, sess_options, providers):
            captured["path"] = path
            captured["providers"] = providers
            captured["options"] = sess_options
            self.fallback_disabled = False

        def disable_fallback(self):
            self.fallback_disabled = True
            captured["fallback_disabled"] = True

        def get_providers(self):
            return ["CUDAExecutionProvider"]

    class FakeGraphOptimizationLevel:
        ORT_ENABLE_ALL = "all"

    class FakeOrt:
        SessionOptions = FakeOptions
        InferenceSession = FakeSession
        GraphOptimizationLevel = FakeGraphOptimizationLevel

        @staticmethod
        def get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    monkeypatch.setattr(cuda_runtime, "_import_onnxruntime", lambda: FakeOrt)
    session = create_cuda_session(model, "cuda:1")

    assert session.get_providers()[0] == "CUDAExecutionProvider"
    assert captured["session_config"] == ("session.disable_cpu_ep_fallback", "1")
    assert captured["providers"] == [
        ("CUDAExecutionProvider", {"device_id": "1", "use_tf32": "0"})
    ]
    assert captured["fallback_disabled"] is True


def test_yolo_batched_tail_is_padded_and_only_real_frames_are_returned():
    captured = {}

    class FakeSession:
        def run(self, output_names, feeds):
            captured["tensor"] = feeds["images"].copy()
            output = np.zeros((8, 1, 6), dtype=np.float32)
            for index in range(8):
                output[index, 0] = [10, 10, 20, 20, 0.9, 0]
            return [output]

    runtime = YoloCudaRuntime.__new__(YoloCudaRuntime)
    runtime.session = FakeSession()
    runtime.input_name = "images"
    runtime.output_name = "output0"
    runtime.batch_size = 8
    runtime._letterbox = lambda frame: (
        np.full((3, 640, 640), int(frame[0, 0, 0]), dtype=np.float32),
        1.0,
        (0.0, 0.0),
    )

    frames = [
        np.full((100, 100, 3), 1, dtype=np.uint8),
        np.full((100, 100, 3), 2, dtype=np.uint8),
    ]
    results = runtime.detect_batch(frames, confidence=0.4, max_det=20)

    tensor = captured["tensor"]
    assert tensor.shape == (8, 3, 640, 640)
    assert np.all(tensor[0] == 1)
    assert np.all(tensor[1:] == 2)
    assert len(results) == 2
    assert all(len(item) == 1 for item in results)
