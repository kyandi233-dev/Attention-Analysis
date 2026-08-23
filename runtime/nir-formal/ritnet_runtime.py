"""Fixed-batch FP32 RITnet inference through ONNX Runtime DirectML."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from directml_runtime import _fixed_shape, create_directml_session, parse_device_id


class RitnetRuntime:
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
        del package_root  # Kept in the signature for the existing pipeline adapter.
        self.device_id = parse_device_id(device)
        self.device = f"dml:{self.device_id}"
        self.precision = str(precision).strip().lower()
        if self.precision != "fp32":
            raise ValueError("AMD/DirectML RITnet is frozen to FP32")

        self.input_size = tuple(map(int, input_size))
        self.analysis_size = tuple(map(int, analysis_size))
        self.weights = Path(weights)
        self.session = create_directml_session(self.weights, self.device_id)
        self.providers = list(self.session.get_providers())
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 2:
            raise ValueError(
                "RITnet ONNX must expose one input plus label/probability outputs"
            )
        self.input_name = inputs[0].name
        self.output_names = [output.name for output in outputs]
        expected_input = (
            self.FIXED_BATCH_SIZE,
            1,
            int(self.input_size[1]),
            int(self.input_size[0]),
        )
        expected_output = (
            self.FIXED_BATCH_SIZE,
            int(self.input_size[1]),
            int(self.input_size[0]),
        )
        if inputs[0].type != "tensor(float)" or _fixed_shape(inputs[0]) != expected_input:
            raise ValueError(
                f"RITnet ONNX must use FP32 input shape {expected_input}, got "
                f"{inputs[0].type} {_fixed_shape(inputs[0])}"
            )
        if (
            outputs[0].type != "tensor(uint8)"
            or _fixed_shape(outputs[0]) != expected_output
        ):
            raise ValueError(
                f"RITnet label output must use UINT8 shape {expected_output}, got "
                f"{outputs[0].type} {_fixed_shape(outputs[0])}"
            )
        if (
            outputs[1].type != "tensor(float)"
            or _fixed_shape(outputs[1]) != expected_output
        ):
            raise ValueError(
                f"RITnet pupil-probability output must use FP32 shape {expected_output}, got "
                f"{outputs[1].type} {_fixed_shape(outputs[1])}"
            )

        self.gamma_table = (255.0 * (np.linspace(0, 1, 256) ** 0.8)).astype(np.uint8)
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        self.last_timing: dict[str, float | int | str] = {}

    def _preprocess_one(self, roi_gray: np.ndarray) -> np.ndarray:
        if roi_gray is None or roi_gray.size == 0:
            raise ValueError("Empty RITnet ROI")
        if roi_gray.ndim == 3:
            roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(roi_gray, self.input_size, interpolation=cv2.INTER_LINEAR)
        image = cv2.LUT(image, self.gamma_table)
        image = self.clahe.apply(image)
        return np.ascontiguousarray(image)

    def _postprocess_one(self, pred: np.ndarray, pupil_prob: np.ndarray) -> dict:
        pupil = (pred == 3).astype(np.uint8)
        analysis_w, analysis_h = self.analysis_size
        mask = cv2.resize(
            pupil,
            (analysis_w, analysis_h),
            interpolation=cv2.INTER_NEAREST,
        )
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
            "pupil_confidence": (
                float(pupil_prob[pupil.astype(bool)].mean()) if pupil.any() else 0.0
            ),
        }

    def infer_batch(self, roi_grays: list[np.ndarray]) -> list[dict]:
        """Infer one or more raw eye crops as one RITnet batch.

        Returned pupil geometry is always expressed in ``analysis_size`` coordinates
        (320x160 by default), regardless of the raw crop size or model input size.
        """
        if not roi_grays:
            self.last_timing = {
                "batch_size": 0,
                "precision": self.precision,
                "preprocess_ms": 0.0,
                "gpu_and_transfer_ms": 0.0,
                "postprocess_ms": 0.0,
                "total_ms": 0.0,
            }
            return []

        total_started = time.perf_counter()
        preprocess_started = time.perf_counter()
        valid_batch_size = len(roi_grays)
        if valid_batch_size > self.FIXED_BATCH_SIZE:
            raise ValueError(
                f"RITnet accepts at most {self.FIXED_BATCH_SIZE} ROIs per call; "
                f"got {valid_batch_size}"
            )
        images = [self._preprocess_one(roi) for roi in roi_grays]
        padded_count = self.FIXED_BATCH_SIZE - valid_batch_size
        if padded_count:
            # Reuse the final real ROI instead of inventing an out-of-domain blank
            # image. Outputs for these padding slots are discarded below.
            images.extend([images[-1]] * padded_count)
        tensor = np.stack(images, axis=0).astype(np.float32, copy=False)
        tensor = ((tensor / np.float32(255.0) - np.float32(0.5)) / np.float32(0.5))[
            :, None, :, :
        ]
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0

        gpu_started = time.perf_counter()
        pred_batch, pupil_prob_batch = self.session.run(
            self.output_names,
            {self.input_name: tensor},
        )
        pred_batch = np.asarray(pred_batch[:valid_batch_size], dtype=np.uint8)
        pupil_prob_batch = np.asarray(
            pupil_prob_batch[:valid_batch_size],
            dtype=np.float32,
        )
        gpu_and_transfer_ms = (time.perf_counter() - gpu_started) * 1000.0

        post_started = time.perf_counter()
        results = [
            self._postprocess_one(pred, pupil_prob)
            for pred, pupil_prob in zip(pred_batch, pupil_prob_batch)
        ]
        postprocess_ms = (time.perf_counter() - post_started) * 1000.0

        self.last_timing = {
            "batch_size": self.FIXED_BATCH_SIZE,
            "valid_batch_size": valid_batch_size,
            "padded_count": padded_count,
            "precision": self.precision,
            "preprocess_ms": preprocess_ms,
            "gpu_and_transfer_ms": gpu_and_transfer_ms,
            "postprocess_ms": postprocess_ms,
            "total_ms": (time.perf_counter() - total_started) * 1000.0,
        }
        return results

    def infer(self, roi_gray: np.ndarray) -> dict:
        """Run one real ROI in a padded fixed batch and return only its output."""
        return self.infer_batch([roi_gray])[0]
