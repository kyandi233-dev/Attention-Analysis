from __future__ import annotations

import numpy as np

from ritnet_fullclass_final_runtime import (
    COHORT_OUTPUT_NAMES,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    OUTPUT_NAMES,
    RitnetFullClassFinalRuntime,
    _DerivedUncertaintyBatch,
)


def _valid_probability(batch: int = 16) -> np.ndarray:
    probability = np.empty((batch, 4, INPUT_HEIGHT, INPUT_WIDTH), dtype=np.float32)
    probability[:, 0] = np.float32(0.50)
    probability[:, 1] = np.float32(0.30)
    probability[:, 2] = np.float32(0.15)
    probability[:, 3] = np.float32(0.05)
    return probability


def test_cohort_prepared_requests_only_labels_and_four_class_probability():
    class Session:
        def run(self, output_names, feeds):
            assert tuple(output_names) == COHORT_OUTPUT_NAMES
            assert feeds["image"].shape == (16, 1, 400, 640)
            return [np.zeros((16, 400, 640), dtype=np.uint8), _valid_probability()]

    runtime = RitnetFullClassFinalRuntime.__new__(RitnetFullClassFinalRuntime)
    runtime.session = Session()
    runtime.input_name = "image"
    runtime.precision = "fp32"
    runtime.cohort_compact_outputs = True
    tensor = np.zeros((16, 1, 400, 640), dtype=np.float32)

    outputs, timing = runtime.infer_prepared(tensor, 3)
    assert outputs["labels"].shape == (3, 400, 640)
    assert outputs["class_probability"].shape == (3, 4, 400, 640)
    assert len(outputs["max_probability"]) == 3
    assert len(outputs["top1_top2_margin"]) == 3
    assert len(outputs["entropy"]) == 3
    assert timing["output_contract"] == "labels+class_probability-cohort"

    np.testing.assert_allclose(outputs["max_probability"][0], 0.50, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(outputs["top1_top2_margin"][0], 0.20, rtol=0.0, atol=1e-7)
    expected_entropy = -sum(
        float(value) * np.log(max(float(value), 1e-12)) for value in (0.50, 0.30, 0.15, 0.05)
    )
    np.testing.assert_allclose(outputs["entropy"][0], expected_entropy, rtol=0.0, atol=1e-6)


def test_cpu_derived_uncertainty_matches_frozen_export_formulas():
    rng = np.random.default_rng(20260828)
    probability = rng.random((2, 4, 400, 640), dtype=np.float32)
    probability /= probability.sum(axis=1, keepdims=True)

    max_map = _DerivedUncertaintyBatch(probability, "max_probability")
    margin_map = _DerivedUncertaintyBatch(probability, "top1_top2_margin")
    entropy_map = _DerivedUncertaintyBatch(probability, "entropy")

    for index in range(2):
        eye = probability[index]
        sorted_probability = np.sort(eye, axis=0)
        expected_max = sorted_probability[3]
        expected_margin = sorted_probability[3] - sorted_probability[2]
        expected_entropy = -np.sum(
            eye * np.log(np.maximum(eye, np.float32(1e-12))), axis=0
        )
        np.testing.assert_array_equal(max_map[index], expected_max)
        np.testing.assert_allclose(margin_map[index], expected_margin, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(entropy_map[index], expected_entropy, rtol=0.0, atol=0.0)


def test_infer_batch_keeps_full_five_output_contract_for_validator_and_sparse_qc():
    class Session:
        def run(self, output_names, feeds):
            assert tuple(output_names) == OUTPUT_NAMES
            probability = _valid_probability()
            entropy_value = -sum(
                float(value) * np.log(max(float(value), 1e-12))
                for value in (0.50, 0.30, 0.15, 0.05)
            )
            return [
                np.zeros((16, 400, 640), dtype=np.uint8),
                probability,
                np.full((16, 400, 640), 0.50, dtype=np.float32),
                np.full((16, 400, 640), 0.20, dtype=np.float32),
                np.full((16, 400, 640), entropy_value, dtype=np.float32),
            ]

    runtime = RitnetFullClassFinalRuntime.__new__(RitnetFullClassFinalRuntime)
    runtime.session = Session()
    runtime.input_name = "image"
    runtime.precision = "fp32"
    runtime.cohort_compact_outputs = True
    runtime.prepare_batch = lambda _rois: (
        np.zeros((16, 1, 400, 640), dtype=np.float32),
        2,
        {"preprocess_ms": 0.0},
    )

    outputs, timing = runtime.infer_batch(
        [np.zeros((400, 640), dtype=np.uint8), np.zeros((400, 640), dtype=np.uint8)]
    )
    assert timing["output_contract"] == "five-output-full"
    assert outputs["labels"].shape == (2, 400, 640)
    assert outputs["class_probability"].shape == (2, 4, 400, 640)
    assert outputs["max_probability"].shape == (2, 400, 640)
    assert outputs["top1_top2_margin"].shape == (2, 400, 640)
    assert outputs["entropy"].shape == (2, 400, 640)
