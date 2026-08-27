from __future__ import annotations

import cv2
import numpy as np
import pytest

from ritnet_fullclass_uncertainty import summarize_uncertainty


def _inputs():
    labels = np.zeros((400, 640), dtype=np.uint8)
    cv2.ellipse(labels, (320, 200), (220, 90), 0, 0, 360, 1, -1)
    cv2.ellipse(labels, (320, 200), (90, 60), 0, 0, 360, 2, -1)
    cv2.ellipse(labels, (320, 200), (35, 25), 0, 0, 360, 3, -1)

    probability = np.empty((4, 400, 640), dtype=np.float32)
    probability[0] = 0.50
    probability[1] = 0.30
    probability[2] = 0.15
    probability[3] = 0.05
    max_probability = np.full((400, 640), 0.9, dtype=np.float32)
    margin = np.full((400, 640), 0.7, dtype=np.float32)
    entropy = np.full((400, 640), 0.3, dtype=np.float32)
    valid = np.ones((400, 640), dtype=bool)
    valid[:, :37] = False
    return labels, probability, max_probability, margin, entropy, valid


def test_validated_cohort_path_preserves_only_final_scientific_uncertainty_contract():
    labels, probability, max_probability, margin, entropy, valid = _inputs()
    common = dict(
        labels=labels,
        valid_source_mask=valid,
        class_probability=probability,
        max_probability=max_probability,
        top1_top2_margin=margin,
        entropy=entropy,
    )

    checked = summarize_uncertainty(
        **common,
        boundary_band_px=5,
        low_max_probability_threshold=0.60,
        inputs_validated=False,
    )
    cohort = summarize_uncertainty(**common, inputs_validated=True)

    for name in ("background", "sclera", "iris", "pupil"):
        field = f"soft_{name}_fraction"
        assert cohort[field] == pytest.approx(checked[field])

    for field in (
        "ocular_max_probability_mean",
        "ocular_top1_top2_margin_mean",
        "ocular_entropy_mean",
    ):
        assert cohort[field] == pytest.approx(checked[field])

    # Full/QC mode may compute these expensive diagnostics. Cohort mode must not
    # compute them and must not serialize null placeholders for them.
    assert "whole_max_probability_mean" in checked
    assert "boundary_entropy_p95" in checked
    assert "low_max_probability_threshold" in checked
    for field in (
        "whole_max_probability_mean",
        "boundary_entropy_p95",
        "uncertainty_boundary_band_px",
        "uncertainty_boundary_pixel_count",
        "low_max_probability_threshold",
        "whole_low_max_probability_fraction",
        "ocular_low_max_probability_fraction",
        "boundary_low_max_probability_fraction",
    ):
        assert field not in cohort

    assert set(cohort) == {
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
