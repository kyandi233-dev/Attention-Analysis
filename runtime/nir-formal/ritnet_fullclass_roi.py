"""Fixed-aspect ROI construction for the final RITnet full-class workflow.

The historical YOLO bounding box is the immutable source. Context expansion and
1.6 aspect-ratio normalization may only EXPAND around that box; they never crop
away any part of the YOLO detection. When the desired rectangle extends beyond
the source frame, explicit padding supplies the missing pixels.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, isfinite
from typing import Sequence

import cv2
import numpy as np

TARGET_WIDTH = 640
TARGET_HEIGHT = 400
TARGET_ASPECT_RATIO = TARGET_WIDTH / TARGET_HEIGHT
ASPECT_WIDTH_UNIT = 8
ASPECT_HEIGHT_UNIT = 5
PADDING_MODE_REPLICATE = "replicate"
SUPPORTED_PADDING_MODES = frozenset({PADDING_MODE_REPLICATE})
ROI_ALGORITHM_VERSION = "fixed-aspect-1p6-expand-pad-v1"


@dataclass(frozen=True)
class FixedAspectRoi:
    requested_x1: int
    requested_y1: int
    requested_x2: int
    requested_y2: int
    source_x1: int
    source_y1: int
    source_x2: int
    source_y2: int
    width: int
    height: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    padding_mode: str
    valid_content_fraction: float
    resize_scale: float
    algorithm_version: str = ROI_ALGORITHM_VERSION

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "roi_requested_x1": self.requested_x1,
            "roi_requested_y1": self.requested_y1,
            "roi_requested_x2": self.requested_x2,
            "roi_requested_y2": self.requested_y2,
            "roi_source_x1": self.source_x1,
            "roi_source_y1": self.source_y1,
            "roi_source_x2": self.source_x2,
            "roi_source_y2": self.source_y2,
            "roi_width": self.width,
            "roi_height": self.height,
            "roi_pad_left": self.pad_left,
            "roi_pad_top": self.pad_top,
            "roi_pad_right": self.pad_right,
            "roi_pad_bottom": self.pad_bottom,
            "roi_padding_mode": self.padding_mode,
            "roi_valid_content_fraction": self.valid_content_fraction,
            "roi_resize_scale": self.resize_scale,
            "roi_algorithm_version": self.algorithm_version,
        }


def _validate_bbox(
    bbox: Sequence[float],
    frame_width: int,
    frame_height: int,
) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError(f"bbox must contain four values; got {bbox!r}")
    x1, y1, x2, y2 = map(float, bbox)
    if not all(isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError(f"bbox contains non-finite value: {bbox!r}")
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(f"invalid frame dimensions: {frame_width}x{frame_height}")
    if not (0.0 <= x1 < x2 <= frame_width and 0.0 <= y1 < y2 <= frame_height):
        raise ValueError(
            f"bbox must lie inside source frame; bbox={bbox!r}, frame={frame_width}x{frame_height}"
        )
    return x1, y1, x2, y2


def fixed_aspect_roi_geometry(
    *,
    bbox: Sequence[float],
    frame_width: int,
    frame_height: int,
    expand_horizontal_each_side: float,
    expand_vertical_each_side: float,
    padding_mode: str = PADDING_MODE_REPLICATE,
) -> FixedAspectRoi:
    """Construct an exact 8:5 ROI that contains the expanded YOLO eye box."""
    x1, y1, x2, y2 = _validate_bbox(bbox, frame_width, frame_height)
    horizontal = float(expand_horizontal_each_side)
    vertical = float(expand_vertical_each_side)
    if not (isfinite(horizontal) and isfinite(vertical) and horizontal >= 0 and vertical >= 0):
        raise ValueError("ROI expansion fractions must be finite and non-negative")
    if padding_mode not in SUPPORTED_PADDING_MODES:
        raise ValueError(
            f"unsupported padding_mode={padding_mode!r}; supported={sorted(SUPPORTED_PADDING_MODES)}"
        )

    bbox_width = x2 - x1
    bbox_height = y2 - y1
    desired_width = bbox_width * (1.0 + 2.0 * horizontal)
    desired_height = bbox_height * (1.0 + 2.0 * vertical)

    # Exact integer 8:5 dimensions avoid later x/y scale mismatch. Taking the
    # maximum unit count means aspect-ratio normalization only expands context.
    units = max(
        1,
        int(ceil(desired_width / ASPECT_WIDTH_UNIT)),
        int(ceil(desired_height / ASPECT_HEIGHT_UNIT)),
    )
    width = ASPECT_WIDTH_UNIT * units
    height = ASPECT_HEIGHT_UNIT * units

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    requested_x1 = int(floor(center_x - width / 2.0))
    requested_y1 = int(floor(center_y - height / 2.0))
    requested_x2 = requested_x1 + width
    requested_y2 = requested_y1 + height

    # Because width/height are >= desired expansion dimensions and centered on
    # the same YOLO box, this is a fail-fast invariant rather than a best effort.
    if not (
        requested_x1 <= x1
        and requested_y1 <= y1
        and requested_x2 >= x2
        and requested_y2 >= y2
    ):
        raise AssertionError("fixed-aspect ROI unexpectedly failed to contain YOLO bbox")

    source_x1 = max(0, requested_x1)
    source_y1 = max(0, requested_y1)
    source_x2 = min(frame_width, requested_x2)
    source_y2 = min(frame_height, requested_y2)
    if source_x2 <= source_x1 or source_y2 <= source_y1:
        raise ValueError("fixed-aspect ROI has no overlap with source frame")

    pad_left = source_x1 - requested_x1
    pad_top = source_y1 - requested_y1
    pad_right = requested_x2 - source_x2
    pad_bottom = requested_y2 - source_y2
    if min(pad_left, pad_top, pad_right, pad_bottom) < 0:
        raise AssertionError("ROI padding cannot be negative")

    source_area = (source_x2 - source_x1) * (source_y2 - source_y1)
    requested_area = width * height
    valid_content_fraction = float(source_area / requested_area)
    resize_scale_x = TARGET_WIDTH / width
    resize_scale_y = TARGET_HEIGHT / height
    if abs(resize_scale_x - resize_scale_y) > 1e-12:
        raise AssertionError("fixed-aspect ROI does not yield uniform RITnet resize")

    return FixedAspectRoi(
        requested_x1=requested_x1,
        requested_y1=requested_y1,
        requested_x2=requested_x2,
        requested_y2=requested_y2,
        source_x1=source_x1,
        source_y1=source_y1,
        source_x2=source_x2,
        source_y2=source_y2,
        width=width,
        height=height,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        padding_mode=padding_mode,
        valid_content_fraction=valid_content_fraction,
        resize_scale=float(resize_scale_x),
    )


def crop_fixed_aspect_gray(
    frame: np.ndarray,
    geometry: FixedAspectRoi,
) -> np.ndarray:
    """Crop/pad one geometry to its exact pre-resize 8:5 grayscale ROI."""
    array = np.asarray(frame)
    if array.ndim not in (2, 3) or array.size == 0:
        raise ValueError(f"invalid source frame shape: {array.shape}")
    frame_height, frame_width = array.shape[:2]
    if not (
        0 <= geometry.source_x1 < geometry.source_x2 <= frame_width
        and 0 <= geometry.source_y1 < geometry.source_y2 <= frame_height
    ):
        raise ValueError("ROI source intersection is outside the supplied frame")

    crop = array[
        geometry.source_y1 : geometry.source_y2,
        geometry.source_x1 : geometry.source_x2,
    ]
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop = np.ascontiguousarray(crop, dtype=np.uint8)

    if geometry.padding_mode == PADDING_MODE_REPLICATE:
        crop = cv2.copyMakeBorder(
            crop,
            geometry.pad_top,
            geometry.pad_bottom,
            geometry.pad_left,
            geometry.pad_right,
            borderType=cv2.BORDER_REPLICATE,
        )
    else:  # Defensive even though geometry creation already validates the mode.
        raise ValueError(f"unsupported padding mode: {geometry.padding_mode}")

    expected_shape = (geometry.height, geometry.width)
    if crop.shape != expected_shape:
        raise RuntimeError(
            f"fixed-aspect crop shape mismatch: expected={expected_shape}, got={crop.shape}"
        )
    return np.ascontiguousarray(crop)
