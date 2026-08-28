from __future__ import annotations

import numpy as np
import pytest

from ritnet_fullclass_final_runtime import _DerivedUncertaintyEye
from ritnet_fullclass_uncertainty import summarize_uncertainty


EXPECTED_FIELDS = {
    "uncertainty_algorithm_version",
    "uncertainty_domain_version",
    "soft_class_fraction_domain_version",
    "soft_background_fraction",
    "soft_sclera_fraction",
    "soft_iris_fraction",
    "soft_pupil_fraction",
    "uncertainty_ocular_pixel_count",
    "ocular_max_probability_mean",
    "ocular_top1_top2_margin_mean",
    "ocular_entropy_mean",
}


def _probabilities(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.random((4, 400, 640), dtype=np.float32)
    raw /= raw.sum(axis=0, keepdims=True)
    return np.ascontiguousarray(raw, dtype=np.float32)


def _reference(probability: np.ndarray, labels: np.ndarray, valid: np.ndarray):
    soft = np.asarray(
        [np.mean(probability[class_id][valid]) for class_id in range(4)],
        dtype=np.float64,
    )
    ocular = (labels != 0) & valid
    selected = np.ascontiguousarray(probability[:, ocular], dtype=np.float32)
    if selected.shape[1] == 0:
        means = (None, None, None)
    else:
        top = np.partition(selected, kth=2, axis=0)
        top1 = top[3]
        margin = top[3] - top[2]
        safe = np.maximum(selected, np.float32(1e-12))
        entropy = -np.sum(selected * np.log(safe), axis=0)
        means = (
            float(np.mean(top1.astype(np.float64, copy=False))),
            float(np.mean(margin.astype(np.float64, copy=False))),
            float(np.mean(entropy.astype(np.float64, copy=False))),
        )
    return soft, ocular, means


@pytest.mark.parametrize("padded", [False, True])
def test_cohort_fastpath_matches_direct_formula_for_full_and_padded_domains(padded):
    probability = _probabilities(20260828 + int(padded))
    labels = np.argmax(probability, axis=0).astype(np.uint8)
    valid = np.ones((400, 640), dtype=bool)
    if padded:
        valid[:37, :] = False
        valid[:, :23] = False
        valid[-19:, :] = False

    result = summarize_uncertainty(
        labels=labels,
        valid_source_mask=valid,
        class_probability=probability,
        max_probability=_DerivedUncertaintyEye(probability, "max_probability"),
        top1_top2_margin=_DerivedUncertaintyEye(probability, "top1_top2_margin"),
        entropy=_DerivedUncertaintyEye(probability, "entropy"),
        inputs_validated=True,
    )
    soft, ocular, means = _reference(probability, labels, valid)

    assert set(result) == EXPECTED_FIELDS
    for index, name in enumerate(("background", "sclera", "iris", "pupil")):
        assert result[f"soft_{name}_fraction"] == pytest.approx(float(soft[index]), rel=0, abs=1e-12)
    assert result["uncertainty_ocular_pixel_count"] == int(ocular.sum())
    assert result["ocular_max_probability_mean"] == pytest.approx(means[0], rel=0, abs=1e-12)
    assert result["ocular_top1_top2_margin_mean"] == pytest.approx(means[1], rel=0, abs=1e-12)
    assert result["ocular_entropy_mean"] == pytest.approx(means[2], rel=0, abs=1e-12)
