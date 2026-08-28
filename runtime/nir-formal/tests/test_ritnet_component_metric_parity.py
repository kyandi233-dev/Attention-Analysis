from __future__ import annotations

import cv2
import numpy as np
import pytest

from ritnet_native_metrics import _component_metrics


def _reference(mask: np.ndarray) -> tuple[int, float | None]:
    binary = np.ascontiguousarray(mask.astype(np.uint8, copy=False))
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    component_count = max(0, int(count) - 1)
    total = int(binary.sum())
    if component_count == 0 or total <= 0:
        return component_count, None
    areas = stats[1:, cv2.CC_STAT_AREA]
    return component_count, float(np.max(areas) / total)


@pytest.mark.parametrize("seed", [1, 7, 20260828])
def test_component_metrics_match_connected_component_area_definition(seed):
    rng = np.random.default_rng(seed)
    mask = rng.random((400, 640)) > 0.985
    expected_count, expected_fraction = _reference(mask)
    count, fraction = _component_metrics(mask)
    assert count == expected_count
    assert fraction == pytest.approx(expected_fraction, rel=0, abs=1e-15)


def test_component_metrics_empty_mask_contract_is_unchanged():
    count, fraction = _component_metrics(np.zeros((400, 640), dtype=bool))
    assert count == 0
    assert fraction is None
