"""Compact online uncertainty summaries for final RITnet full-class analysis.

Pixelwise four-class probabilities and uncertainty maps exist only for the
current inference batch. This module reduces them to small, versioned per-eye
records and does not persist the maps. The hard segmentation remains primary;
uncertainty summaries are QC facts, not calibrated probabilities of correctness.
"""
from __future__ import annotations

from math import log
from typing import Any

import cv2
import numpy as np

LABEL_HEIGHT = 400
LABEL_WIDTH = 640
CLASS_IDS = (0, 1, 2, 3)
STAT_PERCENTILES = (5, 25, 50, 75, 95)
STAT_SUFFIXES = ("mean", "p05", "p25", "p50", "p75", "p95")
UNCERTAINTY_ALGORITHM_VERSION = "allclass-online-summary-v3-source-valid-softclass"
UNCERTAINTY_DOMAIN_VERSION = "source-valid-allclass-whole-ocular-boundary-v3"
SOFT_CLASS_FRACTION_DOMAIN_VERSION = "source-valid-class-probability-mean-v1"
DEFAULT_BOUNDARY_BAND_PX = 5


def _validate_labels(labels: np.ndarray) -> np.ndarray:
    array = np.asarray(labels)
    if array.shape != (LABEL_HEIGHT, LABEL_WIDTH) or array.dtype != np.uint8:
        raise ValueError(
            f"labels must be uint8 {(LABEL_HEIGHT, LABEL_WIDTH)}, got {array.shape} {array.dtype}"
        )
    if not np.isin(np.unique(array), CLASS_IDS).all():
        raise ValueError("labels contain values outside {0,1,2,3}")
    return np.ascontiguousarray(array)


def _validate_valid_source_mask(valid_source_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(valid_source_mask)
    if mask.shape != (LABEL_HEIGHT, LABEL_WIDTH):
        raise ValueError(
            f"valid_source_mask must have shape {(LABEL_HEIGHT, LABEL_WIDTH)}, got {mask.shape}"
        )
    if mask.dtype != np.bool_:
        raise TypeError(f"valid_source_mask must be bool, got {mask.dtype}")
    if not mask.any():
        raise ValueError("valid_source_mask contains no source-backed pixels")
    return np.ascontiguousarray(mask)


def _validate_map(name: str, values: np.ndarray, *, lower: float, upper: float) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (LABEL_HEIGHT, LABEL_WIDTH):
        raise ValueError(f"{name} shape mismatch: {array.shape}")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must be floating point, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    minimum = float(array.min()) if array.size else lower
    maximum = float(array.max()) if array.size else upper
    if minimum < lower - 1e-6 or maximum > upper + 1e-6:
        raise ValueError(f"{name} outside expected range [{lower},{upper}]: {minimum}..{maximum}")
    return np.asarray(array, dtype=np.float32)


def _validate_class_probability(class_probability: np.ndarray) -> np.ndarray:
    array = np.asarray(class_probability)
    expected = (4, LABEL_HEIGHT, LABEL_WIDTH)
    if array.shape != expected:
        raise ValueError(f"class_probability shape mismatch: expected={expected}, got={array.shape}")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"class_probability must be floating point, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError("class_probability contains non-finite values")
    minimum = float(array.min())
    maximum = float(array.max())
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise ValueError(f"class_probability outside [0,1]: {minimum}..{maximum}")
    mass = array.sum(axis=0)
    if not np.allclose(mass, 1.0, rtol=0.0, atol=1e-5):
        deviation = float(np.max(np.abs(mass - 1.0)))
        raise ValueError(
            "class_probability per-pixel class mass must sum to 1; "
            f"max_abs_deviation={deviation}"
        )
    return np.asarray(array, dtype=np.float32)


def boundary_band_mask(
    labels: np.ndarray,
    band_px: int = DEFAULT_BOUNDARY_BAND_PX,
    valid_source_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return a class-boundary band without letting synthetic padding create boundaries.

    When a source-valid mask is supplied, a class transition counts as a real
    segmentation boundary only if both adjacent pixels are backed by the AVI.
    The dilated band is then intersected with that same source-valid domain.
    """
    labels = _validate_labels(labels)
    band_px = int(band_px)
    if band_px < 0:
        raise ValueError("boundary band must be non-negative")
    valid = (
        np.ones(labels.shape, dtype=bool)
        if valid_source_mask is None
        else _validate_valid_source_mask(valid_source_mask)
    )

    boundary = np.zeros(labels.shape, dtype=np.uint8)
    horizontal = (labels[:, 1:] != labels[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    vertical = (labels[1:, :] != labels[:-1, :]) & valid[1:, :] & valid[:-1, :]
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    if band_px > 0 and boundary.any():
        kernel_size = 2 * band_px + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        boundary = cv2.dilate(boundary, kernel, iterations=1)
    return boundary.astype(bool) & valid


def distribution_summary(values: np.ndarray, mask: np.ndarray) -> dict[str, float | None]:
    selected = np.asarray(values)[np.asarray(mask, dtype=bool)]
    if selected.size == 0:
        return {suffix: None for suffix in STAT_SUFFIXES}
    selected = selected.astype(np.float64, copy=False)
    percentiles = np.percentile(selected, STAT_PERCENTILES)
    return {
        "mean": float(np.mean(selected)),
        "p05": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p95": float(percentiles[4]),
    }


def summarize_uncertainty(
    *,
    labels: np.ndarray,
    valid_source_mask: np.ndarray,
    class_probability: np.ndarray,
    max_probability: np.ndarray,
    top1_top2_margin: np.ndarray,
    entropy: np.ndarray,
    boundary_band_px: int = DEFAULT_BOUNDARY_BAND_PX,
    low_max_probability_threshold: float | None = None,
) -> dict[str, Any]:
    """Reduce one eye's temporary probability/uncertainty maps to scalar evidence.

    The four soft class fractions and all whole/ocular/boundary uncertainty
    summaries are computed only from source-backed AVI pixels. Replicate padding
    may influence the neural network context, but its pixels never enter these
    final numeric summaries.
    """
    labels = _validate_labels(labels)
    valid = _validate_valid_source_mask(valid_source_mask)
    probabilities = _validate_class_probability(class_probability)
    max_probability = _validate_map("max_probability", max_probability, lower=0.0, upper=1.0)
    top1_top2_margin = _validate_map("top1_top2_margin", top1_top2_margin, lower=0.0, upper=1.0)
    entropy = _validate_map("entropy", entropy, lower=0.0, upper=log(4.0))

    valid_count = int(valid.sum())
    soft = np.asarray(
        [float(np.mean(probabilities[class_id][valid])) for class_id in CLASS_IDS],
        dtype=np.float64,
    )
    if not np.isclose(float(soft.sum()), 1.0, rtol=0.0, atol=1e-5):
        raise AssertionError(f"source-valid soft class fractions do not sum to 1: {soft.sum()}")

    domains = {
        "whole": valid,
        "ocular": (labels != 0) & valid,
        "boundary": boundary_band_mask(
            labels,
            boundary_band_px,
            valid_source_mask=valid,
        ),
    }
    metrics = {
        "max_probability": max_probability,
        "top1_top2_margin": top1_top2_margin,
        "entropy": entropy,
    }

    result: dict[str, Any] = {
        "uncertainty_algorithm_version": UNCERTAINTY_ALGORITHM_VERSION,
        "uncertainty_domain_version": UNCERTAINTY_DOMAIN_VERSION,
        "soft_class_fraction_domain_version": SOFT_CLASS_FRACTION_DOMAIN_VERSION,
        "uncertainty_boundary_band_px": int(boundary_band_px),
        "soft_background_fraction": float(soft[0]),
        "soft_sclera_fraction": float(soft[1]),
        "soft_iris_fraction": float(soft[2]),
        "soft_pupil_fraction": float(soft[3]),
        "uncertainty_ocular_pixel_count": int(domains["ocular"].sum()),
        "uncertainty_boundary_pixel_count": int(domains["boundary"].sum()),
    }
    if valid_count <= 0:
        raise AssertionError("validated source domain unexpectedly has no pixels")

    for domain_name, mask in domains.items():
        for metric_name, values in metrics.items():
            summary = distribution_summary(values, mask)
            for suffix, value in summary.items():
                result[f"{domain_name}_{metric_name}_{suffix}"] = value

    if low_max_probability_threshold is None:
        result["low_max_probability_threshold"] = None
        result["whole_low_max_probability_fraction"] = None
        result["ocular_low_max_probability_fraction"] = None
        result["boundary_low_max_probability_fraction"] = None
    else:
        threshold = float(low_max_probability_threshold)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("low max-probability threshold must be in [0,1]")
        result["low_max_probability_threshold"] = threshold
        for domain_name, mask in domains.items():
            count = int(mask.sum())
            result[f"{domain_name}_low_max_probability_fraction"] = (
                float(np.mean(max_probability[mask] < threshold)) if count else None
            )
    return result
