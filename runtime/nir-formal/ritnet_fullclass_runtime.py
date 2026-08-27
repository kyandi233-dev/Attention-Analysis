"""Canonical RITnet full-class DirectML adapter.

The frozen project ONNX interface exposes native 640x400 uint8 hard labels plus
class-3 pupil probability. These are Attention-Analysis deterministic adapter
outputs derived from the upstream RITnet logits; they are not upstream RITnet
CSV variables. The current complete full-class workflow requests both outputs so
probability summaries can be checkpointed beside the hard-label evidence.
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
        """CPU preprocessing for one fixed-b16 DirectML call."""
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
        expected_shape = (
            self.FIXED_BATCH_SIZE,
            1,
            int(self.input_size[1]),
            int(self.input_size[0]),
        )
        if tensor.shape != expected_shape or tensor.dtype != np.float32:
            raise RuntimeError(
                f"Prepared RITnet tensor contract mismatch: expected {expected_shape} float32, "
                f"got {tensor.shape} {tensor.dtype}"
            )
        if not np.isfinite(tensor).all():
            raise RuntimeError("Prepared RITnet tensor contains non-finite values")

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
        """Run one prepared fixed-b16 tensor and validate returned evidence."""
        expected_tensor_shape = (
            self.FIXED_BATCH_SIZE,
            1,
            int(self.input_size[1]),
            int(self.input_size[0]),
        )
        if tensor.shape != expected_tensor_shape or tensor.dtype != np.float32:
            raise ValueError(
                f"infer_prepared requires {expected_tensor_shape} float32 tensor; "
                f"got {tensor.shape} {tensor.dtype}"
            )
        if not (1 <= int(valid_batch_size) <= self.FIXED_BATCH_SIZE):
            raise ValueError(
                f"valid_batch_size must be 1..{self.FIXED_BATCH_SIZE}, got {valid_batch_size}"
            )
        if not np.isfinite(tensor).all():
            raise ValueError("infer_prepared tensor contains non-finite values")

        output_names = (
            [self.label_output_name, self.pupil_probability_output_name]
            if include_pupil_probability
            else [self.label_output_name]
        )
        started = time.perf_counter()
        outputs = self.base.session.run(output_names, {self.base.input_name: tensor})
        expected_output_shape = (
            self.FIXED_BATCH_SIZE,
            int(self.input_size[1]),
            int(self.input_size[0]),
        )

        raw_labels = np.asarray(outputs[0])
        if raw_labels.shape != expected_output_shape or raw_labels.dtype != np.uint8:
            raise RuntimeError(
                f"RITnet label runtime output contract mismatch: expected {expected_output_shape} "
                f"uint8, got {raw_labels.shape} {raw_labels.dtype}"
            )
        labels_batch = np.ascontiguousarray(raw_labels[:valid_batch_size])
        if not np.isin(np.unique(labels_batch), (0, 1, 2, 3)).all():
            raise RuntimeError("RITnet label runtime output contains values outside {0,1,2,3}")

        pupil_prob_batch = None
        if include_pupil_probability:
            raw_probs = np.asarray(outputs[1])
            if raw_probs.shape != expected_output_shape or raw_probs.dtype != np.float32:
                raise RuntimeError(
                    f"RITnet probability runtime output contract mismatch: expected {expected_output_shape} "
                    f"float32, got {raw_probs.shape} {raw_probs.dtype}"
                )
            pupil_prob_batch = np.ascontiguousarray(raw_probs[:valid_batch_size])
            if not np.isfinite(pupil_prob_batch).all():
                raise RuntimeError("RITnet pupil probability contains non-finite values")
            if pupil_prob_batch.size and (
                float(pupil_prob_batch.min()) < -1e-6
                or float(pupil_prob_batch.max()) > 1.0 + 1e-6
            ):
                raise RuntimeError("RITnet pupil probability is outside the expected [0,1] range")

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
        """Prepare, infer and return only real (unpadded) evidence rows."""
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
