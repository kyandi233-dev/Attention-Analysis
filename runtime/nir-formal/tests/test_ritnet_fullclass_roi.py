from __future__ import annotations

import numpy as np
import pytest

from ritnet_fullclass_roi import (
    TARGET_ASPECT_RATIO,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    crop_fixed_aspect_gray,
    fixed_aspect_roi_geometry,
    valid_source_analysis_mask,
)


def make_geometry(bbox, frame_width=1920, frame_height=1080):
    return fixed_aspect_roi_geometry(
        bbox=bbox,
        frame_width=frame_width,
        frame_height=frame_height,
        expand_horizontal_each_side=0.30,
        expand_vertical_each_side=0.45,
    )


def assert_contract(geometry, bbox):
    x1, y1, x2, y2 = bbox
    assert geometry.width / geometry.height == TARGET_ASPECT_RATIO
    assert geometry.width % 8 == 0
    assert geometry.height % 5 == 0
    assert geometry.requested_x1 <= geometry.expanded_x1 <= x1
    assert geometry.requested_y1 <= geometry.expanded_y1 <= y1
    assert geometry.requested_x2 >= geometry.expanded_x2 >= x2
    assert geometry.requested_y2 >= geometry.expanded_y2 >= y2
    assert TARGET_WIDTH / geometry.width == pytest.approx(TARGET_HEIGHT / geometry.height)
    assert geometry.resize_scale == pytest.approx(TARGET_WIDTH / geometry.width)
    assert 0.0 < geometry.valid_content_fraction <= 1.0
    assert (
        geometry.source_x2 - geometry.source_x1 + geometry.pad_left + geometry.pad_right
        == geometry.width
    )
    assert (
        geometry.source_y2 - geometry.source_y1 + geometry.pad_top + geometry.pad_bottom
        == geometry.height
    )


def test_regular_bbox_expands_to_exact_1p6_without_padding():
    bbox = (800.0, 400.0, 1000.0, 500.0)
    geometry = make_geometry(bbox)
    assert_contract(geometry, bbox)
    assert (geometry.pad_left, geometry.pad_top, geometry.pad_right, geometry.pad_bottom) == (0, 0, 0, 0)


def test_fractional_expanded_context_is_never_lost_by_integer_origin_rounding():
    bbox = (101.2, 203.4, 156.7, 238.9)
    geometry = make_geometry(bbox, frame_width=500, frame_height=400)
    assert_contract(geometry, bbox)
    assert geometry.requested_x1 <= geometry.expanded_x1
    assert geometry.requested_x2 >= geometry.expanded_x2
    assert geometry.requested_y1 <= geometry.expanded_y1
    assert geometry.requested_y2 >= geometry.expanded_y2


def test_tall_bbox_expands_width_without_losing_vertical_context():
    bbox = (900.0, 300.0, 980.0, 600.0)
    geometry = make_geometry(bbox)
    assert_contract(geometry, bbox)
    assert geometry.width >= (bbox[2] - bbox[0]) * 1.60
    assert geometry.height >= (bbox[3] - bbox[1]) * 1.90


def test_wide_bbox_expands_height_without_losing_horizontal_context():
    bbox = (600.0, 450.0, 1200.0, 520.0)
    geometry = make_geometry(bbox)
    assert_contract(geometry, bbox)
    assert geometry.width >= (bbox[2] - bbox[0]) * 1.60
    assert geometry.height >= (bbox[3] - bbox[1]) * 1.90


@pytest.mark.parametrize(
    "bbox,pad_field",
    [
        ((0.0, 300.0, 60.0, 360.0), "pad_left"),
        ((1860.0, 300.0, 1920.0, 360.0), "pad_right"),
        ((800.0, 0.0, 860.0, 40.0), "pad_top"),
        ((800.0, 1040.0, 860.0, 1080.0), "pad_bottom"),
    ],
)
def test_touching_each_frame_edge_uses_padding_without_cropping_expanded_context(bbox, pad_field):
    geometry = make_geometry(bbox)
    assert_contract(geometry, bbox)
    assert getattr(geometry, pad_field) > 0
    assert geometry.valid_content_fraction < 1.0


def test_corner_can_pad_two_axes_without_breaking_aspect_ratio():
    bbox = (0.0, 0.0, 70.0, 50.0)
    geometry = make_geometry(bbox)
    assert_contract(geometry, bbox)
    assert geometry.pad_left > 0
    assert geometry.pad_top > 0


def test_odd_and_small_bbox_still_yields_exact_integer_8_by_5_geometry():
    bbox = (101.2, 203.4, 106.7, 208.9)
    geometry = make_geometry(bbox)
    assert_contract(geometry, bbox)
    assert geometry.width >= 8
    assert geometry.height >= 5


def test_crop_padding_returns_exact_geometry_and_replicates_edge():
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[:, :, 0] = np.arange(30, dtype=np.uint8)[None, :]
    geometry = fixed_aspect_roi_geometry(
        bbox=(0.0, 5.0, 6.0, 12.0),
        frame_width=30,
        frame_height=20,
        expand_horizontal_each_side=0.30,
        expand_vertical_each_side=0.45,
    )
    roi = crop_fixed_aspect_gray(frame, geometry)
    assert roi.shape == (geometry.height, geometry.width)
    assert roi.dtype == np.uint8
    if geometry.pad_left:
        assert np.array_equal(roi[:, 0], roi[:, geometry.pad_left])


def test_valid_source_mask_is_all_true_without_padding():
    geometry = make_geometry((800.0, 400.0, 1000.0, 500.0))
    mask = valid_source_analysis_mask(geometry)
    assert mask.shape == (TARGET_HEIGHT, TARGET_WIDTH)
    assert mask.dtype == np.bool_
    assert mask.all()


@pytest.mark.parametrize(
    "bbox,invalid_edge",
    [
        ((0.0, 300.0, 60.0, 360.0), "left"),
        ((1860.0, 300.0, 1920.0, 360.0), "right"),
        ((800.0, 0.0, 860.0, 40.0), "top"),
        ((800.0, 1040.0, 860.0, 1080.0), "bottom"),
    ],
)
def test_valid_source_mask_excludes_each_padding_edge(bbox, invalid_edge):
    geometry = make_geometry(bbox)
    mask = valid_source_analysis_mask(geometry)
    assert mask.any()
    assert not mask.all()
    if invalid_edge == "left":
        assert not mask[:, 0].any()
    elif invalid_edge == "right":
        assert not mask[:, -1].any()
    elif invalid_edge == "top":
        assert not mask[0, :].any()
    else:
        assert not mask[-1, :].any()


def test_valid_source_mask_corner_excludes_two_padding_edges():
    geometry = make_geometry((0.0, 0.0, 70.0, 50.0))
    mask = valid_source_analysis_mask(geometry)
    assert not mask[:, 0].any()
    assert not mask[0, :].any()
    assert mask[-1, -1]


def test_valid_source_mask_preserves_8_by_5_output_contract():
    geometry = make_geometry((0.0, 300.0, 60.0, 360.0))
    mask = valid_source_analysis_mask(geometry, output_width=320, output_height=200)
    assert mask.shape == (200, 320)
    with pytest.raises(ValueError):
        valid_source_analysis_mask(geometry, output_width=640, output_height=401)


def test_invalid_or_outside_bbox_is_rejected():
    with pytest.raises(ValueError):
        make_geometry((100.0, 100.0, 100.0, 120.0))
    with pytest.raises(ValueError):
        make_geometry((-1.0, 100.0, 20.0, 120.0))
    with pytest.raises(ValueError):
        make_geometry((100.0, 100.0, 2000.0, 120.0))
