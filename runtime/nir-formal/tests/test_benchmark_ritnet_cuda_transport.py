from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


RUNTIME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME))

from benchmark_ritnet_cuda_transport import _parity, _stats


def test_parity_requires_exact_labels_and_bounded_probability_delta():
    expected_labels = np.zeros((2, 2), dtype=np.uint8)
    expected_probability = np.zeros((2, 2), dtype=np.float32)

    exact = _parity(
        (expected_labels.copy(), expected_probability.copy()),
        (expected_labels, expected_probability),
        probability_atol=1e-6,
    )
    assert exact["pass"] is True
    assert exact["labels_exact"] is True
    assert exact["probability_max_abs_diff"] == 0.0

    changed_probability = expected_probability.copy()
    changed_probability[0, 0] = np.float32(2e-6)
    soft_fail = _parity(
        (expected_labels.copy(), changed_probability),
        (expected_labels, expected_probability),
        probability_atol=1e-6,
    )
    assert soft_fail["pass"] is False
    assert soft_fail["labels_exact"] is True

    changed_labels = expected_labels.copy()
    changed_labels[0, 0] = 1
    label_fail = _parity(
        (changed_labels, expected_probability.copy()),
        (expected_labels, expected_probability),
        probability_atol=1e-6,
    )
    assert label_fail["pass"] is False
    assert label_fail["labels_exact"] is False


def test_stats_reports_fixed_b16_eye_rate():
    result = _stats([100.0, 100.0], batches=2)
    assert result["timed_batches"] == 2
    assert result["timed_eyes"] == 32
    assert result["total_ms"] == 200.0
    assert result["mean_ms_per_batch"] == 100.0
    assert result["median_ms_per_batch"] == 100.0
    assert result["p95_ms_per_batch"] == 100.0
    assert result["eyes_per_second"] == 160.0
