"""RITnet full-class DirectML adapter for post-hoc ocular extension.

This module intentionally does not change the frozen production pupil-only
runtime. It reuses the same ONNX, preprocessing, fixed batch=16 and analysis
geometry, but retains all four hard segmentation classes for downstream
metrics.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ritnet_fullclass_metrics import summarize_fullclass
from ritnet_runtime import RitnetRuntime


class RitnetFullClassRuntime:
    FIXED_BATCH_SIZE = RitnetRuntime.FIXED_BATCH_SIZE

    def __init__(
        self,
        package_root: Path,
        weights: Path,
        input_size: tuple[int, int] = (640, 400),
        device: str = "0",
        analysis_size: tuple[int, int] = (320, 160),
        precision: str = "fp32",
    ) -> None:
        self.base = RitnetRuntime(
            package_root,
            weights,
            input_size=input_size,
            device=device,
            analysis_size=analysis_size,
            precision=precision,
        )
        self.analysis_size = tuple(map(int, analysis_size))
        self.input_size = tuple(map(int, input_size))
        self.device = self.base.device
        self.device_id = self.base.device_id
        self.precision = self.base.precision
        self.providers = list(self.base.providers)
        self.weights = Path(weights)
        self.last_timing: dict[str, float | int | str] = {}

    def infer_batch(self, roi_grays: list[np.ndarray]) -> list[dict[str, Any]]:
        if not roi_grays:
            self.last_timing = {
                "batch_size": 0,
                "valid_batch_size": 0,
                "padded_count": 0,
                "precision": self.precision,
                "preprocess_ms": 0.0,
                "gpu_and_transfer_ms": 0.0,
                "postprocess_ms": 0.0,
                "total_ms": 0.0,
            }
            return []

        valid_batch_size = len(roi_grays)
        if valid_batch_size > self.FIXED_BATCH_SIZE:
            raise ValueError(
                f"RITnet accepts at most {self.FIXED_BATCH_SIZE} ROIs per call; "
                f"got {valid_batch_size}"
            )

        total_started = time.perf_counter()
        preprocess_started = time.perf_counter()
        images = [self.base._preprocess_one(roi) for roi in roi_grays]
        padded_count = self.FIXED_BATCH_SIZE - valid_batch_size
        if padded_count:
            images.extend([images[-1]] * padded_count)
        tensor = np.stack(images, axis=0).astype(np.float32, copy=False)
        tensor = ((tensor / np.float32(255.0) - np.float32(0.5)) / np.float32(0.5))[
            :, None, :, :
        ]
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0

        gpu_started = time.perf_counter()
        labels_batch, pupil_prob_batch = self.base.session.run(
            self.base.output_names,
            {self.base.input_name: tensor},
        )
        labels_batch = np.asarray(labels_batch[:valid_batch_size], dtype=np.uint8)
        pupil_prob_batch = np.asarray(
            pupil_prob_batch[:valid_batch_size],
            dtype=np.float32,
        )
        gpu_and_transfer_ms = (time.perf_counter() - gpu_started) * 1000.0

        post_started = time.perf_counter()
        results = [
            summarize_fullclass(labels, pupil_prob, self.analysis_size)
            for labels, pupil_prob in zip(labels_batch, pupil_prob_batch)
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
