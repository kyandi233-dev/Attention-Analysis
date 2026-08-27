from __future__ import annotations

import numpy as np
import pytest

from ritnet_fixed_aspect_roi import (
    ROI_ALGORITHM_VERSION,
    TARGET_ASPECT_RATIO,
    build_fixed_aspect_geometry,
    crop_fixed_aspect_gray,
    uniform_resize_scale,
)
from ritnet_fullclass_roi import fixed_aspect_roi_geometry


FRAME_W = 1920
FRAME_H = 1080


def build(box):
    return build_fixed_aspect_geometry(
        box,
        frame_width=FRAME_W,
        frame_height=FRAME_H,
        expand_horizontal_each_side=0.30,
        expand_vertical_each_side=0.45,
    )


def test_compatibility_facade_returns_canonical_geometry():
    box = (700.25, 430.5, 900.75, 530.5)
    facade = build(box)
    canonical = fixed_aspect_roi_geometry(
        bbox=box,
        frame_width=FRAME_W,
        frame_height=FRAME_H,
        expand_horizontal_each_side=0.30,
        expand_vertical_each_side=0.45,
    )
    assert facade == canonical
    assert facade.algorithm_version == ROI_ALGORITHM_VERSION
    assert facade.aspect_ratio == TARGET_ASPECT_RATIO


def test_compatibility_facade_keeps_uniform_resize_and_crop():
    geometry = build((0, 0, 70, 50))
    assert geometry.requested_x1 <= geometry.expanded_x1
    assert geometry.requested_y1 <= geometry.expanded_y1
    assert geometry.requested_x2 >= geometry.expanded_x2
    assert geometry.requested_y2 >= geometry.expanded_y2
    assert geometry.pad_left > 0
    assert geometry.pad_top > 0
    assert np.isclose(uniform_resize_scale(geometry), geometry.resize_scale)

    frame = np.arange(FRAME_H * FRAME_W, dtype=np.uint32).reshape(FRAME_H, FRAME_W)
    frame = (frame % 256).astype(np.uint8)
    roi = crop_fixed_aspect_gray(frame, geometry)
    assert roi.shape == (geometry.height, geometry.width)


def test_compatibility_facade_rejects_non_8_by_5_output():
    geometry = build((700, 430, 900, 530))
    with pytest.raises(ValueError):
        uniform_resize_scale(geometry, (640, 401))
