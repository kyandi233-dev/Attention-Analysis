"""Compatibility facade for the canonical final full-class ROI implementation.

The active algorithm lives only in :mod:`ritnet_fullclass_roi`.  This module is
kept because it was introduced during the audit and may already be referenced by
local work/tests; it deliberately contains no independent ROI geometry logic.
"""
from __future__ import annotations

from typing import Sequence

from ritnet_fullclass_roi import (
    ASPECT_HEIGHT_UNIT as TARGET_ASPECT_DEN,
    ASPECT_WIDTH_UNIT as TARGET_ASPECT_NUM,
    PADDING_MODE_REPLICATE as DEFAULT_PADDING_MODE,
    ROI_ALGORITHM_VERSION,
    TARGET_ASPECT_RATIO,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    FixedAspectRoi as FixedAspectRoiGeometry,
    crop_fixed_aspect_gray,
    fixed_aspect_roi_geometry,
)


def build_fixed_aspect_geometry(
    box: Sequence[float],
    *,
    frame_width: int,
    frame_height: int,
    expand_horizontal_each_side: float,
    expand_vertical_each_side: float,
    padding_mode: str = DEFAULT_PADDING_MODE,
) -> FixedAspectRoiGeometry:
    """Forward to the one canonical fixed-aspect ROI implementation."""
    return fixed_aspect_roi_geometry(
        bbox=box,
        frame_width=frame_width,
        frame_height=frame_height,
        expand_horizontal_each_side=expand_horizontal_each_side,
        expand_vertical_each_side=expand_vertical_each_side,
        padding_mode=padding_mode,
    )


def uniform_resize_scale(
    geometry: FixedAspectRoiGeometry,
    output_size: tuple[int, int] = (TARGET_WIDTH, TARGET_HEIGHT),
) -> float:
    """Return the isotropic resize scale, rejecting any non-8:5 output size."""
    output_width, output_height = map(int, output_size)
    if output_width <= 0 or output_height <= 0:
        raise ValueError(f"invalid output size: {output_size}")
    if output_width * TARGET_ASPECT_DEN != output_height * TARGET_ASPECT_NUM:
        raise ValueError(f"output size must use 8:5 aspect ratio, got {output_size}")
    scale_x = output_width / geometry.width
    scale_y = output_height / geometry.height
    if abs(scale_x - scale_y) > 1e-12:
        raise AssertionError(
            f"non-uniform resize would occur: scale_x={scale_x}, scale_y={scale_y}"
        )
    return float(scale_x)
