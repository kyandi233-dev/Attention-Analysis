from __future__ import annotations

import numpy as np
import pytest

from ritnet_fullclass_uncertainty import (
    LABEL_HEIGHT,
    LABEL_WIDTH,
    SOFT_CLASS_FRACTION_DOMAIN_VERSION,
    boundary_band_mask,
    distribution_summary,
    summarize_uncertainty,
)


def base_inputs():
    labels = np.zeros((LABEL_HEIGHT, LABEL_WIDTH), dtype=np.uint8)
    labels[100:300, 100:540] = 1
    labels[140:260, 220:420] = 2
    labels[170:230, 280:360] = 3

    maxprob = np.full(labels.shape, 0.90, dtype=np.float32)
    margin = np.full(labels.shape, 0.70, dtype=np.float32)
    entropy = np.full(labels.shape, 0.30, dtype=np.float32)
    probability = np.zeros((4, LABEL_HEIGHT, LABEL_WIDTH), dtype=np.float32)
    probability[0] = 0.55
    probability[1] = 0.25
    probability[2] = 0.15
    probability[3] = 0.05
    valid = np.ones(labels.shape, dtype=bool)
    return labels, probability, maxprob, margin, entropy, valid


def test_distribution_summary_uses_symmetric_tail_quartile_grid():
    values = np.arange(100, dtype=np.float32).reshape(10, 10)
    summary = distribution_summary(values, np.ones((10, 10), dtype=bool))
    assert set(summary) == {"mean", "p05", "p25", "p50", "p75", "p95"}
    assert summary["mean"] == pytest.approx(49.5)
    assert summary["p50"] == pytest.approx(49.5)
    assert summary["p05"] < summary["p25"] < summary["p50"] < summary["p75"] < summary["p95"]


def test_boundary_band_contains_class_transitions_but_not_far_interior():
    labels, *_ = base_inputs()
    boundary = boundary_band_mask(labels, band_px=2)
    assert boundary[100, 200]
    assert boundary[170, 300]
    assert not boundary[50, 50]
    assert not boundary[200, 320]


def test_boundary_band_does_not_create_transition_from_padding_neighbor():
    labels = np.zeros((LABEL_HEIGHT, LABEL_WIDTH), dtype=np.uint8)
    valid = np.ones(labels.shape, dtype=bool)
    valid[:, :80] = False
    labels[:, :80] = 3

    boundary = boundary_band_mask(labels, band_px=0, valid_source_mask=valid)

    assert not boundary[:, 80].any()
    assert not boundary[:, 79].any()
    assert not boundary[~valid].any()


def test_uncertainty_summary_uses_source_valid_soft_fractions_and_domains():
    labels, probability, maxprob, margin, entropy, valid = base_inputs()
    maxprob[170:230, 280:360] = 0.55
    margin[170:230, 280:360] = 0.10
    entropy[170:230, 280:360] = 1.10

    result = summarize_uncertainty(
        labels=labels,
        valid_source_mask=valid,
        class_probability=probability,
        max_probability=maxprob,
        top1_top2_margin=margin,
        entropy=entropy,
        boundary_band_px=5,
        low_max_probability_threshold=0.60,
    )

    assert result["soft_class_fraction_domain_version"] == SOFT_CLASS_FRACTION_DOMAIN_VERSION
    assert result["soft_background_fraction"] == pytest.approx(0.55)
    assert result["soft_sclera_fraction"] == pytest.approx(0.25)
    assert result["soft_iris_fraction"] == pytest.approx(0.15)
    assert result["soft_pupil_fraction"] == pytest.approx(0.05)
    assert sum(result[f"soft_{name}_fraction"] for name in ("background", "sclera", "iris", "pupil")) == pytest.approx(1.0)
    assert result["uncertainty_ocular_pixel_count"] > 0
    assert result["uncertainty_boundary_pixel_count"] > 0
    assert result["ocular_max_probability_p05"] <= result["ocular_max_probability_p50"]
    assert result["ocular_entropy_p95"] >= result["ocular_entropy_p50"]
    assert 0.0 <= result["ocular_low_max_probability_fraction"] <= 1.0


def test_padding_probability_cannot_change_soft_fractions_or_uncertainty_domains():
    labels, probability, maxprob, margin, entropy, valid = base_inputs()
    valid[:, :80] = False

    labels[:, :80] = 3
    probability[:, :, :80] = 0.0
    probability[3, :, :80] = 1.0
    maxprob[:, :80] = 0.01
    margin[:, :80] = 0.0
    entropy[:, :80] = 1.30

    result = summarize_uncertainty(
        labels=labels,
        valid_source_mask=valid,
        class_probability=probability,
        max_probability=maxprob,
        top1_top2_margin=margin,
        entropy=entropy,
        boundary_band_px=5,
        low_max_probability_threshold=0.60,
    )

    assert result["soft_background_fraction"] == pytest.approx(0.55)
    assert result["soft_sclera_fraction"] == pytest.approx(0.25)
    assert result["soft_iris_fraction"] == pytest.approx(0.15)
    assert result["soft_pupil_fraction"] == pytest.approx(0.05)
    assert result["whole_max_probability_mean"] == pytest.approx(0.90)
    assert result["whole_top1_top2_margin_mean"] == pytest.approx(0.70)
    assert result["whole_entropy_mean"] == pytest.approx(0.30)
    assert result["whole_low_max_probability_fraction"] == pytest.approx(0.0)
    assert result["uncertainty_ocular_pixel_count"] == int(((labels != 0) & valid).sum())


def test_uncertainty_threshold_can_remain_unfrozen():
    labels, probability, maxprob, margin, entropy, valid = base_inputs()
    result = summarize_uncertainty(
        labels=labels,
        valid_source_mask=valid,
        class_probability=probability,
        max_probability=maxprob,
        top1_top2_margin=margin,
        entropy=entropy,
        low_max_probability_threshold=None,
    )
    assert result["low_max_probability_threshold"] is None
    assert result["whole_low_max_probability_fraction"] is None


def test_class_probability_contract_rejects_bad_per_pixel_mass():
    labels, probability, maxprob, margin, entropy, valid = base_inputs()
    bad = probability.copy()
    bad[:, 0, 0] = 0.2
    with pytest.raises(ValueError, match="per-pixel class mass"):
        summarize_uncertainty(
            labels=labels,
            valid_source_mask=valid,
            class_probability=bad,
            max_probability=maxprob,
            top1_top2_margin=margin,
            entropy=entropy,
        )


def test_uncertainty_maps_reject_nonfinite_or_out_of_range():
    labels, probability, maxprob, margin, entropy, valid = base_inputs()
    bad = maxprob.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        summarize_uncertainty(
            labels=labels,
            valid_source_mask=valid,
            class_probability=probability,
            max_probability=bad,
            top1_top2_margin=margin,
            entropy=entropy,
        )

    bad_margin = margin.copy()
    bad_margin[0, 0] = 1.2
    with pytest.raises(ValueError, match="expected range"):
        summarize_uncertainty(
            labels=labels,
            valid_source_mask=valid,
            class_probability=probability,
            max_probability=maxprob,
            top1_top2_margin=bad_margin,
            entropy=entropy,
        )


def test_uncertainty_requires_boolean_source_mask_with_real_pixels():
    labels, probability, maxprob, margin, entropy, _ = base_inputs()
    with pytest.raises(TypeError, match="must be bool"):
        summarize_uncertainty(
            labels=labels,
            valid_source_mask=np.ones(labels.shape, dtype=np.uint8),
            class_probability=probability,
            max_probability=maxprob,
            top1_top2_margin=margin,
            entropy=entropy,
        )

    with pytest.raises(ValueError, match="no source-backed pixels"):
        summarize_uncertainty(
            labels=labels,
            valid_source_mask=np.zeros(labels.shape, dtype=bool),
            class_probability=probability,
            max_probability=maxprob,
            top1_top2_margin=margin,
            entropy=entropy,
        )
