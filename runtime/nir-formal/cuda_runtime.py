"""Strict ONNX Runtime CUDA helpers and fixed-batch YOLO26n inference.

This module is the NVIDIA execution-layer counterpart of the AMD DirectML
runtime. It preserves the same preprocessing/postprocessing contract while
requiring CUDAExecutionProvider to be installed, selected first, and protected
against runtime CPU execution-provider fallback.
"""
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
            "onnxruntime-gpu is required for the NVIDIA CUDA runtime. "
            "Install runtime/nir-formal/requirements.txt in the NVIDIA environment."
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
    """Create a CUDA-only ONNX Runtime session and fail closed on CPU fallback."""
    ort = _import_onnxruntime()
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    available = list(ort.get_available_providers())
    if CUDA_PROVIDER not in available:
        raise RuntimeError(
            "CUDAExecutionProvider is unavailable; refusing CPU fallback. "
            f"Available ONNX Runtime providers: {available}"
        )

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    device_id = parse_device_id(device)
    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=[
                (
                    CUDA_PROVIDER,
                    {"device_id": str(device_id), "use_tf32": "0"},
                )
            ],
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize CUDAExecutionProvider for {model_path}; refusing CPU fallback"
        ) from exc

    session.disable_fallback()
    active = list(session.get_providers())
    if not active or active[0] != CUDA_PROVIDER:
        raise RuntimeError(
            "CUDAExecutionProvider did not become the primary provider; refusing CPU fallback. "
            f"Active providers: {active}"
        )
    return session


def _fixed_shape(node: Any) -> tuple[int, ...]:
    shape = tuple(node.shape)
    if not shape or any(not isinstance(value, int) for value in shape):
        raise ValueError(f"Expected a fixed ONNX tensor shape, got {shape}")
    return tuple(int(value) for value in shape)


class YoloCudaRuntime:
    """Fixed-batch YOLO26n end-to-end detector backed by CUDAExecutionProvider.

    The ONNX model may use any positive fixed batch size. ``detect_batch`` pads
    only the final partial call by repeating the last real tensor; padded outputs
    are discarded. ``detect`` remains a compatibility wrapper for diagnostic code.
    """

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
        self.input_shape = _fixed_shape(inputs[0])
        self.output_shape = _fixed_shape(outputs[0])
        if (
            inputs[0].type != "tensor(float)"
            or len(self.input_shape) != 4
            or self.input_shape[0] <= 0
            or self.input_shape[1:] != (3, 640, 640)
        ):
            raise ValueError(
                "YOLO ONNX must use fixed FP32 input shape [B,3,640,640], got "
                f"{inputs[0].type} {self.input_shape}"
            )
        self.batch_size = int(self.input_shape[0])
        if outputs[0].type != "tensor(float)" or (
            len(self.output_shape) != 3
            or self.output_shape[0] != self.batch_size
            or self.output_shape[2] != 6
        ):
            raise ValueError(
                "YOLO ONNX must return FP32 end-to-end rows [B,N,6] with matching B, got "
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
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)
        tensor /= np.float32(255.0)
        return tensor, scale, (float(left), float(top))

    @staticmethod
    def _postprocess(
        rows: np.ndarray,
        *,
        frame_shape: tuple[int, ...],
        scale: float,
        pad: tuple[float, float],
        confidence: float,
        max_det: int,
    ) -> list[tuple[tuple[float, float, float, float], float, int]]:
        rows = np.asarray(rows, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != 6:
            raise RuntimeError(f"Unexpected YOLO output rows shape: {rows.shape}")
        rows = rows[np.isfinite(rows).all(axis=1)]
        rows = rows[rows[:, 4] >= np.float32(confidence)]
        if not len(rows):
            return []
        rows = rows[np.argsort(-rows[:, 4], kind="stable")[: int(max_det)]]

        pad_x, pad_y = pad
        frame_h, frame_w = frame_shape[:2]
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

    def detect_batch(
        self,
        frames: list[np.ndarray],
        *,
        confidence: float,
        max_det: int,
    ) -> list[list[tuple[tuple[float, float, float, float], float, int]]]:
        if not frames:
            return []
        if len(frames) > self.batch_size:
            raise ValueError(
                f"YOLO model accepts at most {self.batch_size} frames per call; got {len(frames)}"
            )

        prepared = [self._letterbox(frame) for frame in frames]
        tensors = [item[0] for item in prepared]
        while len(tensors) < self.batch_size:
            tensors.append(tensors[-1])
        batch = np.ascontiguousarray(np.stack(tensors, axis=0), dtype=np.float32)
        output = self.session.run([self.output_name], {self.input_name: batch})[0]
        output = np.asarray(output, dtype=np.float32)
        if output.ndim != 3 or output.shape[0] != self.batch_size or output.shape[2] != 6:
            raise RuntimeError(f"Unexpected YOLO output shape: {output.shape}")

        results = []
        for index, frame in enumerate(frames):
            _, scale, pad = prepared[index]
            results.append(
                self._postprocess(
                    output[index],
                    frame_shape=frame.shape,
                    scale=scale,
                    pad=pad,
                    confidence=confidence,
                    max_det=max_det,
                )
            )
        return results

    def detect(
        self,
        frame: np.ndarray,
        *,
        confidence: float,
        max_det: int,
    ) -> list[tuple[tuple[float, float, float, float], float, int]]:
        return self.detect_batch(
            [frame],
            confidence=confidence,
            max_det=max_det,
        )[0]
