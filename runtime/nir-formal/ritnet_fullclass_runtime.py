"""Fast RITnet full-class DirectML adapter for post-hoc ocular extensions.

The runtime keeps the frozen project ONNX interface: native 640x400 uint8 hard
labels plus optional class-3 pupil probability. v1.2 may request labels only;
v2-native640 requests both so probability summaries can be checkpointed with
native evidence. Neither interface should be described as an upstream RITnet
CSV variable: the network/weights are upstream, while these ONNX outputs are
project-added deterministic post-network nodes.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ritnet_runtime import RitnetRuntime


PREPROCESSING_VERSION = "ritnet-upstream-preprocess-plus-project-roi-resize-v1"
PREPROCESSING_SPEC = {
    "grayscale": {
        "operation": "eye ROI converted to grayscale",
        "provenance": "project ROI adapter; upstream RITnet test data are grayscale",
    },
    "resize": {
        "interpolation": "cv2.INTER_LINEAR",
        "provenance": "project ROI adapter",
    },
    "gamma": {"factor": 0.8, "provenance": "upstream RITnet required preprocessing"},
    "clahe": {
        "clip_limit": 1.5,
        "tile_grid_size": [8, 8],
        "provenance": "upstream RITnet required preprocessing",
    },
    "normalization": {
        "formula": "(x/255 - 0.5) / 0.5",
        "provenance": "equivalent to upstream Normalize([0.5],[0.5])",
    },
}


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
        self.label_output_name = self.base.output_names[0]
        self.pupil_probability_output_name = self.base.output_names[1]
        self.preprocessing_version = PREPROCESSING_VERSION
        self.preprocessing_spec = {
            **PREPROCESSING_SPEC,
            "resize": {
                **PREPROCESSING_SPEC["resize"],
                "size": list(self.input_size),
            },
        }

    def prepare_batch(
        self,
        roi_grays: list[np.ndarray],
    ) -> tuple[np.ndarray, int, dict[str, float | int | str]]:
        """CPU-only preprocessing, safe to run while previous batch is on DirectML."""
        if not roi_grays:
            raise ValueError("prepare_batch requires at least one ROI")
        valid_batch_size = len(roi_grays)
        if valid_batch_size > self.FIXED_BATCH_SIZE:
            raise ValueError(
                f"RITnet accepts at most {self.FIXED_BATCH_SIZE} ROIs per call; "
                f"got {valid_batch_size}"
            )

        started = time.perf_counter()
        images = [self.base._preprocess_one(roi) for roi in roi_grays]
        padded_count = self.FIXED_BATCH_SIZE - valid_batch_size
        if padded_count:
            images.extend([images[-1]] * padded_count)

        tensor = np.stack(images, axis=0).astype(np.float32, copy=False)
        tensor = ((tensor / np.float32(255.0) - np.float32(0.5)) / np.float32(0.5))[
            :, None, :, :
        ]
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        timing = {
            "valid_batch_size": valid_batch_size,
            "padded_count": padded_count,
            "preprocess_ms": (time.perf_counter() - started) * 1000.0,
        }
        return tensor, valid_batch_size, timing

    def infer_prepared(
        self,
        tensor: np.ndarray,
        valid_batch_size: int,
        *,
        include_pupil_probability: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None, dict[str, float | int | str]]:
        """Run one prepared fixed-b16 tensor; optionally return class-3 probability."""
        output_names = (
            [self.label_output_name, self.pupil_probability_output_name]
            if include_pupil_probability
            else [self.label_output_name]
        )
        started = time.perf_counter()
        outputs = self.base.session.run(
            output_names,
            {self.base.input_name: tensor},
        )
        labels_batch = np.asarray(
            outputs[0][:valid_batch_size],
            dtype=np.uint8,
        )
        pupil_prob_batch = None
        if include_pupil_probability:
            pupil_prob_batch = np.asarray(
                outputs[1][:valid_batch_size],
                dtype=np.float32,
            )

        timing = {
            "batch_size": self.FIXED_BATCH_SIZE,
            "valid_batch_size": int(valid_batch_size),
            "precision": self.precision,
            "labels_only": not include_pupil_probability,
            "gpu_and_transfer_ms": (time.perf_counter() - started) * 1000.0,
        }
        return labels_batch, pupil_prob_batch, timing

    def infer_batch(
        self,
        roi_grays: list[np.ndarray],
        *,
        include_pupil_probability: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None, dict[str, float | int | str]]:
        """Compatibility wrapper for benchmarks/tests and native evidence production."""
        total_started = time.perf_counter()
        tensor, valid, prep = self.prepare_batch(roi_grays)
        labels, probs, gpu = self.infer_prepared(
            tensor,
            valid,
            include_pupil_probability=include_pupil_probability,
        )
        timing: dict[str, float | int | str] = {
            **prep,
            **gpu,
            "total_ms": (time.perf_counter() - total_started) * 1000.0,
        }
        return labels, probs, timing
