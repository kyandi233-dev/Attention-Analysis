from __future__ import annotations

import numpy as np
import pytest

from ritnet_fullclass_uncertainty import (
    LABEL_HEIGHT,
    LABEL_WIDTH,
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
    soft = np.asarray([0.55, 0.25, 0.15, 0.05], dtype=np.float32)
    return labels, soft, maxprob, margin, entropy


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
    assert not boundary[200, 320]  # pupil interior, farther than 2 px from boundary


def test_uncertainty_summary_preserves_soft_fractions_and_domains():
    labels, soft, maxprob, margin, entropy = base_inputs()
    maxprob[170:230, 280:360] = 0.55
    margin[170:230, 280:360] = 0.10
    entropy[170:230, 280:360] = 1.10

    result = summarize_uncertainty(
        labels=labels,
        soft_class_fraction=soft,
        max_probability=maxprob,
        top1_top2_margin=margin,
        entropy=entropy,
        boundary_band_px=5,
        low_max_probability_threshold=0.60,
    )

    assert result["soft_background_fraction"] == pytest.approx(float(soft[0]))
    assert sum(result[f"soft_{name}_fraction"] for name in ("background", "sclera", "iris", "pupil")) == pytest.approx(1.0)
    assert result["uncertainty_ocular_pixel_count"] > 0
    assert result["uncertainty_boundary_pixel_count"] > 0
    assert result["ocular_max_probability_p05"] <= result["ocular_max_probability_p50"]
    assert result["ocular_entropy_p95"] >= result["ocular_entropy_p50"]
    assert 0.0 <= result["ocular_low_max_probability_fraction"] <= 1.0


def test_uncertainty_threshold_can_remain_unfrozen():
    labels, soft, maxprob, margin, entropy = base_inputs()
    result = summarize_uncertainty(
        labels=labels,
        soft_class_fraction=soft,
        max_probability=maxprob,
        top1_top2_margin=margin,
        entropy=entropy,
        low_max_probability_threshold=None,
    )
    assert result["low_max_probability_threshold"] is None
    assert result["whole_low_max_probability_fraction"] is None


def test_soft_fraction_contract_rejects_bad_sum():
    labels, _, maxprob, margin, entropy = base_inputs()
    with pytest.raises(ValueError, match="sum to 1"):
        summarize_uncertainty(
            labels=labels,
            soft_class_fraction=np.asarray([0.2, 0.2, 0.2, 0.2], dtype=np.float32),
            max_probability=maxprob,
            top1_top2_margin=margin,
            entropy=entropy,
        )


def test_uncertainty_maps_reject_nonfinite_or_out_of_range():
    labels, soft, maxprob, margin, entropy = base_inputs()
    bad = maxprob.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        summarize_uncertainty(
            labels=labels,
            soft_class_fraction=soft,
            max_probability=bad,
            top1_top2_margin=margin,
            entropy=entropy,
        )

    bad_margin = margin.copy()
    bad_margin[0, 0] = 1.2
    with pytest.raises(ValueError, match="expected range"):
        summarize_uncertainty(
            labels=labels,
            soft_class_fraction=soft,
            max_probability=maxprob,
            top1_top2_margin=bad_margin,
            entropy=entropy,
        )
