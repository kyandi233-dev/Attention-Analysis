"""Strict ONNX Runtime CUDA helpers and YOLO26n inference."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


CUDA_PROVIDER = "CUDAExecutionProvider"


def _import_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime-gpu is required for the ort-cuda profile; refusing CPU fallback"
        ) from exc
    return ort


def parse_device_id(device: str | int) -> int:
    value = str(device).strip().lower()
    if value.startswith("cuda:"):
        value = value[5:]
    if not value.isdigit():
        raise ValueError(f"CUDA device must be a non-negative adapter index, got {device!r}")
    return int(value)


def create_cuda_session(model_path: Path, device: str | int = "0"):
    """Create a CUDA-only session and reject session or runtime CPU fallback."""
    ort = _import_onnxruntime()
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    available = list(ort.get_available_providers())
    if CUDA_PROVIDER not in available:
        raise RuntimeError(
            "ONNX Runtime CUDA is unavailable; refusing CPU fallback. "
            f"Available providers: {available}"
        )

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=[
                (
                    CUDA_PROVIDER,
                    {"device_id": str(parse_device_id(device)), "use_tf32": "0"},
                )
            ],
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize CUDAExecutionProvider for {model_path}; refusing CPU fallback"
        ) from exc
    session.disable_fallback()
    active = list(session.get_providers())
    # ONNX Runtime may report CPUExecutionProvider as an installed provider even
    # when the session is configured CUDA-only. The session config above
    # disables CPU EP fallback; the meaningful runtime invariant here is that
    # CUDA is active and has priority, not that CPU is absent from the registry.
    if not active or active[0] != CUDA_PROVIDER:
        raise RuntimeError(
            "CUDA did not become the primary ONNX Runtime provider; refusing CPU fallback. "
            f"Active providers: {active}"
        )
    return session


def fixed_shape(node: Any) -> tuple[int, ...]:
    shape = tuple(node.shape)
    if not shape or any(not isinstance(value, int) for value in shape):
        raise ValueError(f"Expected a fixed ONNX tensor shape, got {shape}")
    return tuple(int(value) for value in shape)


class YoloCudaRuntime:
    names = {0: "eye"}

    def __init__(self, weights: Path, device: str = "0"):
        self.weights = Path(weights)
        self.device_id = parse_device_id(device)
        self.device = f"cuda:{self.device_id}"
        self.precision = "fp32"
        self.session = create_cuda_session(self.weights, self.device_id)
        self.providers = list(self.session.get_providers())
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("YOLO ONNX must expose exactly one input and one output")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.input_shape = fixed_shape(inputs[0])
        self.output_shape = fixed_shape(outputs[0])
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
        source_h, source_w = frame.shape[:2]
        scale = min(640 / source_w, 640 / source_h)
        resized_w, resized_h = int(round(source_w * scale)), int(round(source_h * scale))
        resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        pad_w, pad_h = (640 - resized_w) / 2.0, (640 - resized_h) / 2.0
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
        tensor /= np.float32(255.0)
        return tensor, scale, (float(left), float(top))

    def detect(self, frame: np.ndarray, *, confidence: float, max_det: int):
        tensor, scale, (pad_x, pad_y) = self._letterbox(frame)
        output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        rows = np.asarray(output, dtype=np.float32)[0]
        rows = rows[np.isfinite(rows).all(axis=1)]
        rows = rows[(rows[:, 4] >= np.float32(confidence)) & (rows[:, 5].astype(int) == 0)]
        rows = rows[np.argsort(-rows[:, 4])[:max_det]]
        height, width = frame.shape[:2]
        detections = []
        for x1, y1, x2, y2, score, class_id in rows:
            box = (
                float(np.clip((x1 - pad_x) / scale, 0, width)),
                float(np.clip((y1 - pad_y) / scale, 0, height)),
                float(np.clip((x2 - pad_x) / scale, 0, width)),
                float(np.clip((y2 - pad_y) / scale, 0, height)),
            )
            if box[2] > box[0] and box[3] > box[1]:
                detections.append((box, float(score), int(class_id)))
        return detections
