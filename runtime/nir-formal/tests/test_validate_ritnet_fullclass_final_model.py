from __future__ import annotations

import numpy as np

import validate_ritnet_fullclass_final_model as validator


class FakeRuntime:
    FIXED_BATCH_SIZE = 16

    def __init__(self, model, *, device="0"):
        self.device = f"dml:{device}"
        self.providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        self.input_size = (640, 400)

    def infer_batch(self, rois):
        count = len(rois)
        labels = np.zeros((count, 400, 640), dtype=np.uint8)
        probability = np.zeros((count, 4, 400, 640), dtype=np.float32)
        probability[:, 0] = 0.5
        probability[:, 1] = 0.3
        probability[:, 2] = 0.15
        probability[:, 3] = 0.05
        return {
            "labels": labels,
            "class_probability": probability,
            "max_probability": np.full((count, 400, 640), 0.5, dtype=np.float32),
            "top1_top2_margin": np.full((count, 400, 640), 0.2, dtype=np.float32),
            "entropy": np.full((count, 400, 640), 0.8, dtype=np.float32),
        }, {"valid_batch_size": count}


def test_validation_gate_reports_final_contract(monkeypatch, tmp_path):
    model = tmp_path / "ritnet-b16-fp32-uncertainty.onnx"
    external = tmp_path / "ritnet-b16-fp32-uncertainty.onnx.data"
    model.write_bytes(b"onnx")
    external.write_bytes(b"data")
    monkeypatch.setattr(validator, "RitnetFullClassFinalRuntime", FakeRuntime)

    result = validator.validate_model(model, device="0")
    assert result["status"] == "pass"
    assert result["providers"][0] == "DmlExecutionProvider"
    assert result["outputs"]["class_probability"] == [2, 4, 400, 640]
    assert result["class_probability_mass_max_abs_deviation"] <= 1e-5
