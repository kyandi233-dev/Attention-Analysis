"""ONNX Runtime DirectML helpers and YOLO26n inference.

The runtime deliberately refuses to create a CPU-only session. ONNX Runtime may
still keep ``CPUExecutionProvider`` after DirectML for lightweight graph nodes,
but DirectML must be installed, requested first, and remain the active primary
provider. Runtime provider failover is disabled after session creation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


DML_PROVIDER = "DmlExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


def _import_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime-directml is required. Activate the nir-amd environment "
            "and install runtime/nir-formal/requirements.txt."
        ) from exc
    return ort


def parse_device_id(device: str | int) -> int:
    value = str(device).strip().lower()
    if value.startswith("dml:"):
        value = value[4:]
    if not value.isdigit():
        raise ValueError(
            f"DirectML device must be a non-negative adapter index, got {device!r}"
        )
    return int(value)


def create_directml_session(model_path: Path, device: str | int = "0"):
    """Create a DirectML-first session and reject full-session CPU fallback."""
    ort = _import_onnxruntime()
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    available = list(ort.get_available_providers())
    if DML_PROVIDER not in available:
        raise RuntimeError(
            "DirectML is unavailable; refusing CPU fallback. "
            f"Available ONNX Runtime providers: {available}"
        )

    options = ort.SessionOptions()
    # Required by the DirectML execution provider.
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    device_id = parse_device_id(device)
    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=[
                (DML_PROVIDER, {"device_id": str(device_id)}),
                CPU_PROVIDER,
            ],
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize DirectML for {model_path}; refusing CPU fallback"
        ) from exc

    # Prevent ONNX Runtime's Python wrapper from replacing the provider list with
    # CPUExecutionProvider if a later execution-provider error occurs.
    session.disable_fallback()
    active = list(session.get_providers())
    if not active or active[0] != DML_PROVIDER:
        raise RuntimeError(
            "DirectML did not become the primary execution provider; refusing CPU "
            f"fallback. Active providers: {active}"
        )
    return session


def _fixed_shape(node: Any) -> tuple[int, ...]:
    shape = tuple(node.shape)
    if not shape or any(not isinstance(value, int) for value in shape):
        raise ValueError(f"Expected a fixed ONNX tensor shape, got {shape}")
    return tuple(int(value) for value in shape)


class YoloDirectMLRuntime:
    """Fixed-shape YOLO26n end-to-end detector backed by DirectML."""

    names = {0: "eye"}

    def __init__(self, weights: Path, device: str = "0"):
        self.weights = Path(weights)
        self.device_id = parse_device_id(device)
        self.device = f"dml:{self.device_id}"
        self.precision = "fp32"
        self.session = create_directml_session(self.weights, self.device_id)
        self.providers = list(self.session.get_providers())

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("YOLO ONNX must expose exactly one input and one output")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.input_shape = _fixed_shape(inputs[0])
        self.output_shape = _fixed_shape(outputs[0])
        if inputs[0].type != "tensor(float)" or self.input_shape != (1, 3, 640, 640):
            raise ValueError(
                "YOLO ONNX must use FP32 input shape [1,3,640,640], got "
                f"{inputs[0].type} {self.input_shape}"
            )
        if outputs[0].type != "tensor(float)" or (
            len(self.output_shape) != 3
            or self.output_shape[0] != 1
            or self.output_shape[2] != 6
        ):
            raise ValueError(
                "YOLO ONNX must return FP32 end-to-end rows [1,N,6], got "
                f"{outputs[0].type} {self.output_shape}"
            )

    @staticmethod
    def _letterbox(frame: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
        if frame is None or frame.size == 0:
            raise ValueError("Empty YOLO frame")
        source_h, source_w = frame.shape[:2]
        target_h = target_w = 640
        scale = min(target_w / source_w, target_h / source_h)
        resized_w = int(round(source_w * scale))
        resized_h = int(round(source_h * scale))
        if (resized_w, resized_h) != (source_w, source_h):
            resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = frame

        pad_w = (target_w - resized_w) / 2.0
        pad_h = (target_h - resized_h) / 2.0
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
        tensor /= np.float32(255.0)
        return tensor, scale, (float(left), float(top))

    def detect(
        self,
        frame: np.ndarray,
        *,
        confidence: float,
        max_det: int,
    ) -> list[tuple[tuple[float, float, float, float], float, int]]:
        tensor, scale, (pad_x, pad_y) = self._letterbox(frame)
        output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        rows = np.asarray(output, dtype=np.float32)[0]
        if rows.ndim != 2 or rows.shape[1] != 6:
            raise RuntimeError(f"Unexpected YOLO output shape: {output.shape}")

        rows = rows[np.isfinite(rows).all(axis=1)]
        rows = rows[rows[:, 4] >= np.float32(confidence)]
        if not len(rows):
            return []
        rows = rows[np.argsort(-rows[:, 4], kind="stable")[: int(max_det)]]

        frame_h, frame_w = frame.shape[:2]
        detections = []
        for x1, y1, x2, y2, score, class_id in rows:
            box = np.array(
                [x1 - pad_x, y1 - pad_y, x2 - pad_x, y2 - pad_y],
                dtype=np.float32,
            ) / np.float32(scale)
            box[[0, 2]] = np.clip(box[[0, 2]], 0, frame_w)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, frame_h)
            detections.append(
                (tuple(float(value) for value in box), float(score), int(class_id))
            )
        return detections
