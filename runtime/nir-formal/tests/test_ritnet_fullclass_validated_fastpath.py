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


def test_validated_cohort_path_preserves_primary_class_outputs_and_temporal_uncertainty_means():
    labels, probability, max_probability, margin, entropy, valid = _inputs()
    kwargs = dict(
        labels=labels,
        valid_source_mask=valid,
        class_probability=probability,
        max_probability=max_probability,
        top1_top2_margin=margin,
        entropy=entropy,
        boundary_band_px=5,
        low_max_probability_threshold=0.60,
    )

    checked = summarize_uncertainty(**kwargs, inputs_validated=False)
    cohort = summarize_uncertainty(**kwargs, inputs_validated=True)

    # Production keeps all four soft classes exactly; these are scientific outputs.
    for name in ("background", "sclera", "iris", "pupil"):
        field = f"soft_{name}_fraction"
        assert cohort[field] == pytest.approx(checked[field])

    # The three ocular means consumed by temporal QC remain numerically identical.
    for field in (
        "ocular_max_probability_mean",
        "ocular_top1_top2_margin_mean",
        "ocular_entropy_mean",
    ):
        assert cohort[field] == pytest.approx(checked[field])

    # Cohort production deliberately skips the expensive QC-only percentile/boundary scope.
    assert "whole_max_probability_mean" in checked
    assert "whole_max_probability_mean" not in cohort
    assert "boundary_entropy_p95" in checked
    assert "boundary_entropy_p95" not in cohort
    assert cohort["uncertainty_boundary_band_px"] is None
    assert cohort["uncertainty_boundary_pixel_count"] is None

    # The configured low-confidence threshold is retained only for the ocular domain used in cohort QC.
    assert cohort["low_max_probability_threshold"] == pytest.approx(0.60)
    assert cohort["ocular_low_max_probability_fraction"] == pytest.approx(
        checked["ocular_low_max_probability_fraction"]
    )
    assert cohort["whole_low_max_probability_fraction"] is None
    assert cohort["boundary_low_max_probability_fraction"] is None
