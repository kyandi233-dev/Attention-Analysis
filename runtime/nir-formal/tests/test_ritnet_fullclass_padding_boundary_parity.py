from __future__ import annotations

import cv2
import numpy as np

from ritnet_fullclass_metric_adapter import summarize_final_hard_metrics


def _reference_touch(structure: np.ndarray, valid: np.ndarray) -> bool:
    adjacent = cv2.dilate(
        (~valid).astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    return bool((structure & valid & adjacent).any())


def test_padded_boundary_flags_match_direct_definition_for_all_structures():
    rng = np.random.default_rng(20260828)
    labels = rng.integers(0, 4, size=(400, 640), dtype=np.uint8)
    valid = np.ones((400, 640), dtype=bool)
    valid[:31, :] = False
    valid[:, :17] = False
    valid[-13:, :] = False
    valid[:, -9:] = False

    result = summarize_final_hard_metrics(labels, valid)
    pupil = labels == 3
    iris_outer = labels >= 2
    ocular = labels != 0

    assert result["pupil_touches_valid_domain_edge"] == _reference_touch(pupil, valid)
    assert result["iris_outer_touches_valid_domain_edge"] == _reference_touch(iris_outer, valid)
    assert result["ocular_touches_valid_domain_edge"] == _reference_touch(ocular, valid)
    assert result["pupil_predicted_in_padding_pixels"] == int((pupil & ~valid).sum())
    assert result["iris_outer_predicted_in_padding_pixels"] == int((iris_outer & ~valid).sum())
    assert result["ocular_predicted_in_padding_pixels"] == int((ocular & ~valid).sum())
