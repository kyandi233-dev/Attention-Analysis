"""Fixed-batch FP32 RITnet inference through ONNX Runtime CUDA."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from onnx_cuda_runtime import create_cuda_session, fixed_shape, parse_device_id


class RitnetOnnxRuntime:
    FIXED_BATCH_SIZE = 16

    def __init__(
        self,
        package_root: Path,
        weights: Path,
        input_size: tuple[int, int] = (640, 400),
        device: str = "0",
        analysis_size: tuple[int, int] = (320, 160),
        precision: str = "fp32",
    ):
        del package_root
        self.device_id = parse_device_id(device)
        self.device = f"cuda:{self.device_id}"
        self.precision = str(precision).strip().lower()
        if self.precision != "fp32":
            raise ValueError("ORT CUDA RITnet is frozen to FP32")
        self.input_size = tuple(map(int, input_size))
        self.analysis_size = tuple(map(int, analysis_size))
        self.weights = Path(weights)
        self.session = create_cuda_session(self.weights, self.device_id)
        self.providers = list(self.session.get_providers())
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        expected_input = (16, 1, self.input_size[1], self.input_size[0])
        expected_output = (16, self.input_size[1], self.input_size[0])
        if len(inputs) != 1 or len(outputs) != 2:
            raise ValueError("RITnet ONNX must expose one input plus two outputs")
        self.input_name = inputs[0].name
        self.output_names = [output.name for output in outputs]
        if inputs[0].type != "tensor(float)" or fixed_shape(inputs[0]) != expected_input:
            raise ValueError(f"Unexpected RITnet ONNX input: {inputs[0].type} {inputs[0].shape}")
        if outputs[0].type != "tensor(uint8)" or fixed_shape(outputs[0]) != expected_output:
            raise ValueError(f"Unexpected RITnet label output: {outputs[0].type} {outputs[0].shape}")
        if outputs[1].type != "tensor(float)" or fixed_shape(outputs[1]) != expected_output:
            raise ValueError(f"Unexpected RITnet probability output: {outputs[1].type} {outputs[1].shape}")
        self.gamma_table = (255.0 * (np.linspace(0, 1, 256) ** 0.8)).astype(np.uint8)
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        self.last_timing: dict[str, float | int | str] = {}

    def _preprocess_one(self, roi_gray: np.ndarray) -> np.ndarray:
        if roi_gray.ndim == 3:
            roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(roi_gray, self.input_size, interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(self.clahe.apply(cv2.LUT(image, self.gamma_table)))

    def _postprocess_one(self, pred: np.ndarray, pupil_prob: np.ndarray) -> dict:
        pupil = (pred == 3).astype(np.uint8)
        mask = cv2.resize(pupil, self.analysis_size, interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"found": False}
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 5 or cv2.contourArea(contour) < 5:
            return {"found": False}
        (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
        area = float(cv2.contourArea(contour))
        return {
            "found": True,
            "center_x": float(cx),
            "center_y": float(cy),
            "axis_a": float(axis_a),
            "axis_b": float(axis_b),
            "angle_deg": float(angle),
            "mask_area": area,
            "equiv_diameter": float(2 * np.sqrt(area / np.pi)),
            "pupil_confidence": float(pupil_prob[pupil.astype(bool)].mean()) if pupil.any() else 0.0,
        }

    def infer_batch(self, roi_grays: list[np.ndarray]) -> list[dict]:
        if not roi_grays:
            self.last_timing = {"batch_size": 0, "total_ms": 0.0}
            return []
        if len(roi_grays) > self.FIXED_BATCH_SIZE:
            raise ValueError(f"RITnet accepts at most 16 ROIs per call; got {len(roi_grays)}")
        total_started = time.perf_counter()
        preprocess_started = time.perf_counter()
        valid_size = len(roi_grays)
        images = [self._preprocess_one(roi) for roi in roi_grays]
        padded_count = 16 - valid_size
        images.extend([images[-1]] * padded_count)
        tensor = np.stack(images).astype(np.float32, copy=False)
        tensor = ((tensor / np.float32(255.0) - np.float32(0.5)) / np.float32(0.5))[:, None]
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0
        gpu_started = time.perf_counter()
        labels, probabilities = self.session.run(self.output_names, {self.input_name: tensor})
        gpu_ms = (time.perf_counter() - gpu_started) * 1000.0
        post_started = time.perf_counter()
        results = [
            self._postprocess_one(label, probability)
            for label, probability in zip(labels[:valid_size], probabilities[:valid_size])
        ]
        post_ms = (time.perf_counter() - post_started) * 1000.0
        self.last_timing = {
            "batch_size": 16,
            "valid_batch_size": valid_size,
            "padded_count": padded_count,
            "precision": "fp32",
            "preprocess_ms": preprocess_ms,
            "gpu_and_transfer_ms": gpu_ms,
            "postprocess_ms": post_ms,
            "total_ms": (time.perf_counter() - total_started) * 1000.0,
        }
        return results

    def infer(self, roi_gray: np.ndarray) -> dict:
        return self.infer_batch([roi_gray])[0]
