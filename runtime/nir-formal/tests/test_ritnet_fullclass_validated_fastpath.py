from __future__ import annotations

import cv2
import numpy as np

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


def test_runtime_validated_fast_path_matches_fail_closed_path_exactly():
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
    trusted = summarize_uncertainty(**kwargs, inputs_validated=True)

    assert trusted == checked
