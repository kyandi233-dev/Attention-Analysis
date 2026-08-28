"""Final fixed-b16 RITnet CUDA runtime for compact full-class analysis.

The frozen ONNX graph still exposes the qualified five-output contract. Cohort
production requests only hard labels plus four-class probabilities. The three
deterministic uncertainty maps are represented by zero-copy descriptors and are
reduced directly from ocular probability pixels by CPU summary workers, rather
than materialized as full 400x640 maps. Full five-output inference remains
available for model qualification and sparse QC.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cuda_runtime import _fixed_shape, create_cuda_session, parse_device_id


FIXED_BATCH_SIZE = 16
INPUT_WIDTH = 640
INPUT_HEIGHT = 400
OUTPUT_NAMES = (
    "labels",
    "class_probability",
    "max_probability",
    "top1_top2_margin",
    "entropy",
)
COHORT_OUTPUT_NAMES = ("labels", "class_probability")
COHORT_FULL_VALIDATION_INTERVAL = 100
PREPROCESSING_VERSION = "ritnet-upstream-preprocess-fixed-aspect-roi-v2"


class _DerivedUncertaintyEye:
    """Zero-copy marker for one deterministic uncertainty metric of one eye."""

    def __init__(self, class_probability: np.ndarray, metric: str) -> None:
        self.class_probability = np.asarray(class_probability, dtype=np.float32)
        self.ritnet_derived_uncertainty_metric = str(metric)


class _DerivedUncertaintyBatch:
    """Zero-copy descriptors derived from four-class probability.

    Production only needs compact ocular means. Returning a descriptor here
    avoids materializing full max/margin/entropy maps on CPU for every eye.
    ``summarize_uncertainty(inputs_validated=True)`` recognizes the descriptor
    and reduces directly over ocular probability pixels.
    """

    def __init__(self, class_probability: np.ndarray, metric: str) -> None:
        probability = np.asarray(class_probability)
        expected_tail = (4, INPUT_HEIGHT, INPUT_WIDTH)
        if probability.ndim != 4 or tuple(probability.shape[1:]) != expected_tail:
            raise ValueError(
                "derived uncertainty requires [B,4,400,640] class_probability; "
                f"got {probability.shape}"
            )
        if probability.dtype != np.float32:
            raise TypeError(f"class_probability must be float32; got {probability.dtype}")
        if metric not in {"max_probability", "top1_top2_margin", "entropy"}:
            raise ValueError(f"unsupported derived uncertainty metric: {metric}")
        self._probability = probability
        self._metric = metric

    def __len__(self) -> int:
        return int(self._probability.shape[0])

    def __getitem__(self, index: int) -> _DerivedUncertaintyEye:
        return _DerivedUncertaintyEye(self._probability[index], self._metric)


class RitnetFullClassFinalRuntime:
    FIXED_BATCH_SIZE = FIXED_BATCH_SIZE

    def __init__(self, weights: Path, *, device: str = "0") -> None:
        self.weights = Path(weights)
        self.device_id = parse_device_id(device)
        self.device = f"cuda:{self.device_id}"
        self.precision = "fp32"
        self.input_size = (INPUT_WIDTH, INPUT_HEIGHT)
        self.preprocessing_version = PREPROCESSING_VERSION
        self.session = create_cuda_session(self.weights, self.device_id)
        self.providers = list(self.session.get_providers())
        self.cohort_compact_outputs = True
        self._cohort_call_count = 0

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1:
            raise ValueError(f"final RITnet ONNX must expose exactly one input; got {len(inputs)}")
        if len(outputs) != 5:
            raise ValueError(f"final RITnet ONNX must expose exactly five outputs; got {len(outputs)}")

        self.input_name = inputs[0].name
        actual_names = tuple(output.name for output in outputs)
        if actual_names != OUTPUT_NAMES:
            raise ValueError(
                f"final RITnet output names/order mismatch: expected={OUTPUT_NAMES}, got={actual_names}"
            )

        expected_input_shape = (FIXED_BATCH_SIZE, 1, INPUT_HEIGHT, INPUT_WIDTH)
        if inputs[0].type != "tensor(float)" or _fixed_shape(inputs[0]) != expected_input_shape:
            raise ValueError(
                "final RITnet input must be FP32 fixed-b16 [16,1,400,640], got "
                f"{inputs[0].type} {_fixed_shape(inputs[0])}"
            )

        pixel_shape = (FIXED_BATCH_SIZE, INPUT_HEIGHT, INPUT_WIDTH)
        expected = (
            ("tensor(uint8)", pixel_shape),
            ("tensor(float)", (FIXED_BATCH_SIZE, 4, INPUT_HEIGHT, INPUT_WIDTH)),
            ("tensor(float)", pixel_shape),
            ("tensor(float)", pixel_shape),
            ("tensor(float)", pixel_shape),
        )
        for output, (expected_type, expected_shape) in zip(outputs, expected):
            actual_shape = _fixed_shape(output)
            if output.type != expected_type or actual_shape != expected_shape:
                raise ValueError(
                    f"final RITnet output {output.name!r} contract mismatch: expected "
                    f"{expected_type} {expected_shape}, got {output.type} {actual_shape}"
                )

        self.gamma_table = (255.0 * (np.linspace(0, 1, 256) ** 0.8)).astype(np.uint8)
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

    def _preprocess_one(self, roi_gray: np.ndarray) -> np.ndarray:
        if roi_gray is None or np.asarray(roi_gray).size == 0:
            raise ValueError("Empty RITnet ROI")
        image = np.asarray(roi_gray)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim != 2:
            raise ValueError(f"RITnet ROI must be 2-D grayscale or BGR, got {image.shape}")
        if image.dtype != np.uint8:
            image = np.asarray(np.clip(image, 0, 255), dtype=np.uint8)
        if image.shape != (INPUT_HEIGHT, INPUT_WIDTH):
            image = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        else:
            image = np.ascontiguousarray(image)
        image = cv2.LUT(image, self.gamma_table)
        image = self.clahe.apply(image)
        return np.ascontiguousarray(image, dtype=np.uint8)

    def prepare_batch(self, roi_grays: list[np.ndarray]) -> tuple[np.ndarray, int, dict[str, Any]]:
        if not roi_grays:
            raise ValueError("prepare_batch requires at least one ROI")
        valid = len(roi_grays)
        if valid > FIXED_BATCH_SIZE:
            raise ValueError(f"RITnet accepts at most {FIXED_BATCH_SIZE} ROIs per call; got {valid}")

        started = time.perf_counter()
        images = [self._preprocess_one(roi) for roi in roi_grays]
        padded_count = FIXED_BATCH_SIZE - valid
        if padded_count:
            images.extend([images[-1]] * padded_count)
        tensor = np.stack(images, axis=0).astype(np.float32, copy=False)
        tensor /= np.float32(255.0)
        tensor -= np.float32(0.5)
        tensor /= np.float32(0.5)
        tensor = tensor[:, None, :, :]
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        expected = (FIXED_BATCH_SIZE, 1, INPUT_HEIGHT, INPUT_WIDTH)
        if tensor.shape != expected or tensor.dtype != np.float32:
            raise RuntimeError(f"prepared RITnet tensor mismatch: {tensor.shape} {tensor.dtype}")
        return tensor, valid, {
            "valid_batch_size": valid,
            "padded_count": padded_count,
            "preprocess_ms": (time.perf_counter() - started) * 1000.0,
        }

    @staticmethod
    def _labels_output_structure(value: np.ndarray, valid_batch_size: int) -> np.ndarray:
        labels = np.asarray(value)
        expected = (FIXED_BATCH_SIZE, INPUT_HEIGHT, INPUT_WIDTH)
        if labels.shape != expected or labels.dtype != np.uint8:
            raise RuntimeError(
                f"RITnet labels output must be {expected} uint8, got {labels.shape} {labels.dtype}"
            )
        return np.ascontiguousarray(labels[: int(valid_batch_size)])

    @classmethod
    def _validate_labels_output(cls, value: np.ndarray, valid_batch_size: int) -> np.ndarray:
        real = cls._labels_output_structure(value, valid_batch_size)
        if not np.isin(np.unique(real), (0, 1, 2, 3)).all():
            raise RuntimeError("RITnet labels contain values outside {0,1,2,3}")
        return real

    @staticmethod
    def _validate_float_map(name: str, value: np.ndarray, *, lower: float, upper: float) -> np.ndarray:
        array = np.asarray(value)
        expected = (FIXED_BATCH_SIZE, INPUT_HEIGHT, INPUT_WIDTH)
        if array.shape != expected or array.dtype != np.float32:
            raise RuntimeError(
                f"RITnet {name} output must be {expected} float32, got {array.shape} {array.dtype}"
            )
        if not np.isfinite(array).all():
            raise RuntimeError(f"RITnet {name} output contains non-finite values")
        if array.size:
            minimum = float(array.min())
            maximum = float(array.max())
            if minimum < lower - 1e-6 or maximum > upper + 1e-6:
                raise RuntimeError(
                    f"RITnet {name} output outside [{lower},{upper}]: {minimum}..{maximum}"
                )
        return array

    @staticmethod
    def _class_probability_structure(value: np.ndarray) -> np.ndarray:
        array = np.asarray(value)
        expected = (FIXED_BATCH_SIZE, 4, INPUT_HEIGHT, INPUT_WIDTH)
        if array.shape != expected or array.dtype != np.float32:
            raise RuntimeError(
                f"RITnet class_probability output must be {expected} float32, got "
                f"{array.shape} {array.dtype}"
            )
        return array

    @classmethod
    def _validate_class_probability(cls, value: np.ndarray) -> np.ndarray:
        array = cls._class_probability_structure(value)
        if not np.isfinite(array).all():
            raise RuntimeError("RITnet class_probability contains non-finite values")
        if array.size:
            minimum = float(array.min())
            maximum = float(array.max())
            if minimum < -1e-6 or maximum > 1.0 + 1e-6:
                raise RuntimeError(
                    f"RITnet class_probability outside [0,1]: {minimum}..{maximum}"
                )
        class_mass = array.sum(axis=1)
        if not np.allclose(class_mass, 1.0, rtol=0.0, atol=1e-5):
            deviation = float(np.max(np.abs(class_mass - 1.0)))
            raise RuntimeError(
                "RITnet class_probability per-pixel class mass does not sum to 1; "
                f"max_abs_deviation={deviation}"
            )
        return array

    @staticmethod
    def _prepared_input_structure(tensor: np.ndarray, valid_batch_size: int) -> None:
        expected_tensor = (FIXED_BATCH_SIZE, 1, INPUT_HEIGHT, INPUT_WIDTH)
        if tensor.shape != expected_tensor or tensor.dtype != np.float32:
            raise ValueError(
                f"prepared inference requires {expected_tensor} float32 tensor; got "
                f"{tensor.shape} {tensor.dtype}"
            )
        if not 1 <= int(valid_batch_size) <= FIXED_BATCH_SIZE:
            raise ValueError(f"valid_batch_size must be 1..{FIXED_BATCH_SIZE}")

    @classmethod
    def _validate_prepared_input(cls, tensor: np.ndarray, valid_batch_size: int) -> None:
        cls._prepared_input_structure(tensor, valid_batch_size)
        if not np.isfinite(tensor).all():
            raise ValueError("prepared RITnet tensor contains non-finite values")

    def _infer_full_prepared(
        self,
        tensor: np.ndarray,
        valid_batch_size: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Qualified five-output path used by validator and sparse pixel QC."""
        self._validate_prepared_input(tensor, valid_batch_size)

        total_started = time.perf_counter()
        session_started = time.perf_counter()
        raw = self.session.run(list(OUTPUT_NAMES), {self.input_name: tensor})
        session_run_ms = (time.perf_counter() - session_started) * 1000.0
        if len(raw) != len(OUTPUT_NAMES):
            raise RuntimeError(f"RITnet returned {len(raw)} outputs, expected {len(OUTPUT_NAMES)}")

        validation_started = time.perf_counter()
        labels = self._validate_labels_output(raw[0], valid_batch_size)
        class_probability = self._validate_class_probability(raw[1])
        max_probability = self._validate_float_map("max_probability", raw[2], lower=0.0, upper=1.0)
        margin = self._validate_float_map("top1_top2_margin", raw[3], lower=0.0, upper=1.0)
        entropy = self._validate_float_map("entropy", raw[4], lower=0.0, upper=math.log(4.0))

        real = slice(0, int(valid_batch_size))
        outputs: dict[str, Any] = {
            "labels": labels,
            "class_probability": np.ascontiguousarray(class_probability[real]),
            "max_probability": np.ascontiguousarray(max_probability[real]),
            "top1_top2_margin": np.ascontiguousarray(margin[real]),
            "entropy": np.ascontiguousarray(entropy[real]),
        }
        output_validation_ms = (time.perf_counter() - validation_started) * 1000.0
        gpu_and_transfer_ms = (time.perf_counter() - total_started) * 1000.0
        return outputs, {
            "batch_size": FIXED_BATCH_SIZE,
            "valid_batch_size": int(valid_batch_size),
            "precision": self.precision,
            "session_run_ms": float(session_run_ms),
            "output_validation_ms": float(output_validation_ms),
            "gpu_and_transfer_ms": float(gpu_and_transfer_ms),
            "output_contract": "five-output-full",
            "full_output_validation": True,
        }

    def _infer_cohort_prepared(
        self,
        tensor: np.ndarray,
        valid_batch_size: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Information-complete cohort path with sampled full pixel validation."""
        call_count = int(getattr(self, "_cohort_call_count", 0)) + 1
        self._cohort_call_count = call_count
        full_validation = (
            call_count == 1
            or call_count % COHORT_FULL_VALIDATION_INTERVAL == 0
            or int(valid_batch_size) < FIXED_BATCH_SIZE
        )
        if full_validation:
            self._validate_prepared_input(tensor, valid_batch_size)
        else:
            self._prepared_input_structure(tensor, valid_batch_size)

        total_started = time.perf_counter()
        session_started = time.perf_counter()
        raw = self.session.run(list(COHORT_OUTPUT_NAMES), {self.input_name: tensor})
        session_run_ms = (time.perf_counter() - session_started) * 1000.0
        if len(raw) != len(COHORT_OUTPUT_NAMES):
            raise RuntimeError(
                f"RITnet cohort call returned {len(raw)} outputs, expected {len(COHORT_OUTPUT_NAMES)}"
            )

        validation_started = time.perf_counter()
        if full_validation:
            labels = self._validate_labels_output(raw[0], valid_batch_size)
            class_probability_full = self._validate_class_probability(raw[1])
        else:
            labels = self._labels_output_structure(raw[0], valid_batch_size)
            class_probability_full = self._class_probability_structure(raw[1])
        real = slice(0, int(valid_batch_size))
        class_probability = np.ascontiguousarray(class_probability_full[real])
        outputs: dict[str, Any] = {
            "labels": labels,
            "class_probability": class_probability,
            "max_probability": _DerivedUncertaintyBatch(class_probability, "max_probability"),
            "top1_top2_margin": _DerivedUncertaintyBatch(
                class_probability, "top1_top2_margin"
            ),
            "entropy": _DerivedUncertaintyBatch(class_probability, "entropy"),
        }
        output_validation_ms = (time.perf_counter() - validation_started) * 1000.0
        gpu_and_transfer_ms = (time.perf_counter() - total_started) * 1000.0
        return outputs, {
            "batch_size": FIXED_BATCH_SIZE,
            "valid_batch_size": int(valid_batch_size),
            "precision": self.precision,
            "session_run_ms": float(session_run_ms),
            "output_validation_ms": float(output_validation_ms),
            "gpu_and_transfer_ms": float(gpu_and_transfer_ms),
            "output_contract": "labels+class_probability-cohort",
            "full_output_validation": bool(full_validation),
            "cohort_call_count": int(call_count),
        }

    def infer_prepared(
        self,
        tensor: np.ndarray,
        valid_batch_size: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if getattr(self, "cohort_compact_outputs", False):
            return self._infer_cohort_prepared(tensor, valid_batch_size)
        return self._infer_full_prepared(tensor, valid_batch_size)

    def infer_labels_prepared(
        self,
        tensor: np.ndarray,
        valid_batch_size: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._validate_prepared_input(tensor, valid_batch_size)
        total_started = time.perf_counter()
        session_started = time.perf_counter()
        raw = self.session.run(["labels"], {self.input_name: tensor})
        session_run_ms = (time.perf_counter() - session_started) * 1000.0
        if len(raw) != 1:
            raise RuntimeError(f"RITnet labels-only inference returned {len(raw)} outputs")
        validation_started = time.perf_counter()
        labels = self._validate_labels_output(raw[0], valid_batch_size)
        output_validation_ms = (time.perf_counter() - validation_started) * 1000.0
        return labels, {
            "batch_size": FIXED_BATCH_SIZE,
            "valid_batch_size": int(valid_batch_size),
            "precision": self.precision,
            "session_run_ms": float(session_run_ms),
            "output_validation_ms": float(output_validation_ms),
            "gpu_and_transfer_ms": float((time.perf_counter() - total_started) * 1000.0),
            "output_contract": "labels-only-qc",
            "full_output_validation": True,
        }

    def infer_batch(self, roi_grays: list[np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Full five-output convenience path retained for validator/sparse QC."""
        total_started = time.perf_counter()
        tensor, valid, prep = self.prepare_batch(roi_grays)
        outputs, infer = self._infer_full_prepared(tensor, valid)
        return outputs, {
            **prep,
            **infer,
            "total_ms": (time.perf_counter() - total_started) * 1000.0,
        }

    def infer_labels_batch(self, roi_grays: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
        total_started = time.perf_counter()
        tensor, valid, prep = self.prepare_batch(roi_grays)
        labels, infer = self.infer_labels_prepared(tensor, valid)
        return labels, {
            **prep,
            **infer,
            "total_ms": (time.perf_counter() - total_started) * 1000.0,
        }
