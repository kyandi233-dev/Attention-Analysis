"""In-process RITnet inference for the portable NIR GPU pipeline.

Formal-mode changes relative to the original trial runtime:
- raw expanded eye crops can be passed directly to RITnet;
- multiple crops are preprocessed and inferred as one GPU batch;
- FP32 is the default reference precision; CUDA FP16 autocast is optional;
- pupil ellipse outputs remain in the stable 320x160 analysis coordinate system.
"""
from __future__ import annotations

import sys
import time
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch


class RitnetRuntime:
    def __init__(
        self,
        package_root: Path,
        weights: Path,
        input_size: tuple[int, int] = (640, 400),
        device: str = "0",
        analysis_size: tuple[int, int] = (320, 160),
        precision: str = "fp32",
    ):
        module_dir = package_root / "ritnet"
        sys.path.insert(0, str(module_dir))
        from densenet import DenseNet2D

        requested = str(device).strip().lower()
        if requested == "cpu" or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif requested.isdigit():
            self.device = torch.device(f"cuda:{requested}")
        else:
            self.device = torch.device(requested)

        self.precision = str(precision).strip().lower()
        if self.precision not in {"fp32", "fp16"}:
            raise ValueError(f"Unsupported RITnet precision: {precision!r}; use fp32 or fp16")
        if self.precision == "fp16" and self.device.type != "cuda":
            raise ValueError(
                "RITnet fp16 is enabled only for CUDA in this pipeline. "
                "Use --device <GPU index> or select --ritnet-precision fp32."
            )

        self.input_size = tuple(map(int, input_size))
        self.analysis_size = tuple(map(int, analysis_size))
        self.model = DenseNet2D(dropout=True, prob=0.2)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

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
        confidence = float(pupil_prob[pupil.astype(bool)].mean()) if pupil.any() else 0.0
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
            "pupil_confidence": confidence,
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
        images = np.stack([self._preprocess_one(roi) for roi in roi_grays], axis=0)
        tensor = torch.from_numpy(images).to(self.device, dtype=torch.float32)
        tensor = ((tensor / 255.0 - 0.5) / 0.5)[:, None, :, :]
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0

        gpu_started = time.perf_counter()
        use_fp16 = self.precision == "fp16"
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_fp16
            else nullcontext()
        )
        with torch.inference_mode():
            with autocast_context:
                logits = self.model(tensor)
                pred_tensor = logits.argmax(dim=1)
                pupil_prob_tensor = torch.softmax(logits, dim=1)[:, 3]

            # One batched device->host transfer per output type, instead of one
            # synchronization/transfer for every individual eye.
            pred_batch = pred_tensor.cpu().numpy()
            pupil_prob_batch = pupil_prob_tensor.float().cpu().numpy()
        gpu_and_transfer_ms = (time.perf_counter() - gpu_started) * 1000.0

        post_started = time.perf_counter()
        results = [
            self._postprocess_one(pred, pupil_prob)
            for pred, pupil_prob in zip(pred_batch, pupil_prob_batch)
        ]
        postprocess_ms = (time.perf_counter() - post_started) * 1000.0

        self.last_timing = {
            "batch_size": len(roi_grays),
            "precision": self.precision,
            "preprocess_ms": preprocess_ms,
            "gpu_and_transfer_ms": gpu_and_transfer_ms,
            "postprocess_ms": postprocess_ms,
            "total_ms": (time.perf_counter() - total_started) * 1000.0,
        }
        return results

    def infer(self, roi_gray: np.ndarray) -> dict:
        """Compatibility wrapper for the original diagnostic runner."""
        return self.infer_batch([roi_gray])[0]
