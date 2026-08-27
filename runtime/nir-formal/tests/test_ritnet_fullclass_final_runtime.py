from __future__ import annotations

import numpy as np
import pytest

import ritnet_fullclass_final_runtime as final_runtime
from ritnet_fullclass_final_runtime import (
    FIXED_BATCH_SIZE,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    OUTPUT_NAMES,
    RitnetFullClassFinalRuntime,
)


class Node:
    def __init__(self, name, node_type, shape):
        self.name = name
        self.type = node_type
        self.shape = list(shape)


class ContractSession:
    def __init__(self, outputs=None):
        self._inputs = [Node("image", "tensor(float)", (16, 1, 400, 640))]
        self._outputs = outputs or [
            Node("labels", "tensor(uint8)", (16, 400, 640)),
            Node("class_probability", "tensor(float)", (16, 4, 400, 640)),
            Node("max_probability", "tensor(float)", (16, 400, 640)),
            Node("top1_top2_margin", "tensor(float)", (16, 400, 640)),
            Node("entropy", "tensor(float)", (16, 400, 640)),
        ]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_providers(self):
        return ["DmlExecutionProvider", "CPUExecutionProvider"]


def test_constructor_locks_final_five_output_contract(monkeypatch, tmp_path):
    monkeypatch.setattr(final_runtime, "create_directml_session", lambda *_: ContractSession())
    runtime = RitnetFullClassFinalRuntime(tmp_path / "ritnet-b16-fp32-uncertainty.onnx")
    assert runtime.input_size == (640, 400)
    assert runtime.FIXED_BATCH_SIZE == 16
    assert runtime.providers[0] == "DmlExecutionProvider"


def test_constructor_rejects_old_aggregate_soft_fraction_contract(monkeypatch, tmp_path):
    old_outputs = [
        Node("labels", "tensor(uint8)", (16, 400, 640)),
        Node("soft_class_fraction", "tensor(float)", (16, 4)),
        Node("max_probability", "tensor(float)", (16, 400, 640)),
        Node("top1_top2_margin", "tensor(float)", (16, 400, 640)),
        Node("entropy", "tensor(float)", (16, 400, 640)),
    ]
    monkeypatch.setattr(final_runtime, "create_directml_session", lambda *_: ContractSession(old_outputs))
    with pytest.raises(ValueError, match="output names/order mismatch"):
        RitnetFullClassFinalRuntime(tmp_path / "old-aggregate.onnx")


def test_constructor_rejects_wrong_class_probability_shape(monkeypatch, tmp_path):
    outputs = [
        Node("labels", "tensor(uint8)", (16, 400, 640)),
        Node("class_probability", "tensor(float)", (16, 4)),
        Node("max_probability", "tensor(float)", (16, 400, 640)),
        Node("top1_top2_margin", "tensor(float)", (16, 400, 640)),
        Node("entropy", "tensor(float)", (16, 400, 640)),
    ]
    monkeypatch.setattr(final_runtime, "create_directml_session", lambda *_: ContractSession(outputs))
    with pytest.raises(ValueError, match="contract mismatch"):
        RitnetFullClassFinalRuntime(tmp_path / "bad-shape.onnx")


def _valid_class_probability():
    probability = np.zeros((16, 4, 400, 640), dtype=np.float32)
    probability[:, 0] = 0.5
    probability[:, 1] = 0.3
    probability[:, 2] = 0.15
    probability[:, 3] = 0.05
    return probability


def test_infer_prepared_discards_batch_padding_and_returns_transient_probability_maps():
    class Session:
        def run(self, output_names, feeds):
            assert tuple(output_names) == OUTPUT_NAMES
            assert feeds["image"].shape == (16, 1, 400, 640)
            labels = np.zeros((16, 400, 640), dtype=np.uint8)
            labels[:, 100:300, 100:540] = 1
            class_probability = _valid_class_probability()
            maxprob = np.full((16, 400, 640), 0.9, dtype=np.float32)
            margin = np.full((16, 400, 640), 0.7, dtype=np.float32)
            entropy = np.full((16, 400, 640), 0.3, dtype=np.float32)
            return [labels, class_probability, maxprob, margin, entropy]

    runtime = RitnetFullClassFinalRuntime.__new__(RitnetFullClassFinalRuntime)
    runtime.session = Session()
    runtime.input_name = "image"
    runtime.precision = "fp32"
    tensor = np.zeros((FIXED_BATCH_SIZE, 1, INPUT_HEIGHT, INPUT_WIDTH), dtype=np.float32)

    outputs, timing = runtime.infer_prepared(tensor, 3)
    assert outputs["labels"].shape == (3, 400, 640)
    assert outputs["class_probability"].shape == (3, 4, 400, 640)
    assert outputs["max_probability"].shape == (3, 400, 640)
    assert outputs["top1_top2_margin"].shape == (3, 400, 640)
    assert outputs["entropy"].shape == (3, 400, 640)
    assert timing["valid_batch_size"] == 3


def test_labels_only_qc_inference_requests_no_probability_outputs():
    class Session:
        def run(self, output_names, feeds):
            assert output_names == ["labels"]
            assert feeds["image"].shape == (16, 1, 400, 640)
            labels = np.zeros((16, 400, 640), dtype=np.uint8)
            labels[:, 100:300, 100:540] = 1
            return [labels]

    runtime = RitnetFullClassFinalRuntime.__new__(RitnetFullClassFinalRuntime)
    runtime.session = Session()
    runtime.input_name = "image"
    runtime.precision = "fp32"
    tensor = np.zeros((FIXED_BATCH_SIZE, 1, INPUT_HEIGHT, INPUT_WIDTH), dtype=np.float32)

    labels, timing = runtime.infer_labels_prepared(tensor, 3)
    assert labels.shape == (3, 400, 640)
    assert labels.dtype == np.uint8
    assert timing["valid_batch_size"] == 3
    assert timing["output_contract"] == "labels-only-qc"


def test_infer_prepared_rejects_probability_mass_not_summing_to_one():
    class Session:
        def run(self, output_names, feeds):
            labels = np.zeros((16, 400, 640), dtype=np.uint8)
            probability = np.full((16, 4, 400, 640), 0.2, dtype=np.float32)
            maxprob = np.full((16, 400, 640), 0.9, dtype=np.float32)
            margin = np.full((16, 400, 640), 0.7, dtype=np.float32)
            entropy = np.full((16, 400, 640), 0.3, dtype=np.float32)
            return [labels, probability, maxprob, margin, entropy]

    runtime = RitnetFullClassFinalRuntime.__new__(RitnetFullClassFinalRuntime)
    runtime.session = Session()
    runtime.input_name = "image"
    runtime.precision = "fp32"
    tensor = np.zeros((16, 1, 400, 640), dtype=np.float32)
    with pytest.raises(RuntimeError, match="per-pixel class mass"):
        runtime.infer_prepared(tensor, 1)
