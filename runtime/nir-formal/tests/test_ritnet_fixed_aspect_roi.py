from __future__ import annotations

import numpy as np
import pytest

from ritnet_fixed_aspect_roi import (
    TARGET_ASPECT_RATIO,
    build_fixed_aspect_geometry,
    crop_fixed_aspect_gray,
    uniform_resize_scale,
)


FRAME_W = 1920
FRAME_H = 1080
H_EXPAND = 0.30
V_EXPAND = 0.45


def build(box, *, padding_mode="reflect101"):
    return build_fixed_aspect_geometry(
        box,
        frame_width=FRAME_W,
        frame_height=FRAME_H,
        expand_horizontal_each_side=H_EXPAND,
        expand_vertical_each_side=V_EXPAND,
        padding_mode=padding_mode,
    )


def assert_contract(geometry):
    assert geometry.virtual_width * 5 == geometry.virtual_height * 8
    assert geometry.aspect_ratio == TARGET_ASPECT_RATIO
    assert geometry.virtual_x1 <= geometry.expanded_x1
    assert geometry.virtual_y1 <= geometry.expanded_y1
    assert geometry.virtual_x2 >= geometry.expanded_x2
    assert geometry.virtual_y2 >= geometry.expanded_y2
    assert (
        geometry.source_width
        + geometry.padding_left
        + geometry.padding_right
        == geometry.virtual_width
    )
    assert (
        geometry.source_height
        + geometry.padding_top
        + geometry.padding_bottom
        == geometry.virtual_height
    )
    assert np.isclose(uniform_resize_scale(geometry), 640 / geometry.virtual_width)


def test_centered_normal_box_is_exact_8_by_5_and_unpadded():
    geometry = build((700.25, 430.5, 900.75, 530.5))
    assert_contract(geometry)
    assert geometry.padded is False
    assert geometry.source_width == geometry.virtual_width
    assert geometry.source_height == geometry.virtual_height


def test_fractional_boundary_case_never_loses_expanded_context():
    geometry = build_fixed_aspect_geometry(
        (10.3, 20.2, 60.3, 50.2),
        frame_width=500,
        frame_height=300,
        expand_horizontal_each_side=0.30,
        expand_vertical_each_side=0.45,
    )
    assert_contract(geometry)
    assert geometry.virtual_x1 <= geometry.expanded_x1
    assert geometry.virtual_x2 >= geometry.expanded_x2


@pytest.mark.parametrize(
    "box",
    [
        (500, 500, 900, 540),   # extremely wide
        (800, 200, 840, 600),   # extremely tall
        (700, 430, 701, 431),   # tiny but valid
        (701, 431, 812, 498),   # odd dimensions / noninteger target k
    ],
)
def test_extreme_shapes_still_produce_uniform_resize(box):
    geometry = build(box)
    assert_contract(geometry)
    assert uniform_resize_scale(geometry) > 0


@pytest.mark.parametrize(
    "box, expected_side",
    [
        ((0, 400, 80, 470), "left"),
        ((1840, 400, 1920, 470), "right"),
        ((800, 0, 900, 60), "top"),
        ((800, 1020, 900, 1080), "bottom"),
    ],
)
def test_frame_edges_use_padding_not_aspect_clipping(box, expected_side):
    geometry = build(box)
    assert_contract(geometry)
    assert geometry.padded is True
    if expected_side == "left":
        assert geometry.padding_left > 0
    elif expected_side == "right":
        assert geometry.padding_right > 0
    elif expected_side == "top":
        assert geometry.padding_top > 0
    else:
        assert geometry.padding_bottom > 0


def test_corner_can_pad_two_axes_and_crop_shape_remains_exact():
    geometry = build((0, 0, 70, 50))
    assert_contract(geometry)
    assert geometry.padding_left > 0
    assert geometry.padding_top > 0

    frame = np.arange(FRAME_H * FRAME_W, dtype=np.uint32).reshape(FRAME_H, FRAME_W)
    frame = (frame % 256).astype(np.uint8)
    roi = crop_fixed_aspect_gray(frame, geometry)
    assert roi.shape == (geometry.virtual_height, geometry.virtual_width)
    assert roi.dtype == np.uint8


def test_bgr_crop_is_converted_to_gray_without_resize():
    geometry = build((700, 430, 900, 530))
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    frame[..., 1] = 100
    roi = crop_fixed_aspect_gray(frame, geometry)
    assert roi.ndim == 2
    assert roi.shape == (geometry.virtual_height, geometry.virtual_width)


def test_invalid_bbox_and_padding_mode_fail_closed():
    with pytest.raises(ValueError):
        build((-1, 20, 40, 50))
    with pytest.raises(ValueError):
        build((10, 20, 10, 50))
    with pytest.raises(ValueError):
        build((10, 20, 40, 50), padding_mode="unknown")


def test_output_size_must_keep_same_aspect_ratio():
    geometry = build((700, 430, 900, 530))
    with pytest.raises(ValueError):
        uniform_resize_scale(geometry, (640, 401))
