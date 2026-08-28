"""Fixed-aspect ROI construction for the final RITnet full-class workflow.

The historical YOLO bounding box is the immutable source. Context expansion and
1.6 aspect-ratio normalization may only EXPAND around that box; they never crop
away any part of the requested eye context. When the desired rectangle extends
beyond the source frame, explicit padding supplies the missing pixels.
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
ROI_ALGORITHM_VERSION = "fixed-aspect-1p6-expanded-context-replicate-v2"
VALID_SOURCE_MASK_VERSION = "pre-resize-source-domain-nearest-v1"

# Canonical immutable output-space mask for the overwhelmingly common case in
# which the requested ROI lies fully inside the source frame. Downstream metric
# reducers can recognize this object by identity and skip repeated 400x640
# any()/all() scans without changing a single scientific pixel or denominator.
FULL_SOURCE_VALID_MASK = np.ones((TARGET_HEIGHT, TARGET_WIDTH), dtype=bool)
FULL_SOURCE_VALID_MASK.setflags(write=False)


@dataclass(frozen=True)
class FixedAspectRoi:
    expanded_x1: float
    expanded_y1: float
    expanded_x2: float
    expanded_y2: float
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
            "roi_expanded_x1": self.expanded_x1,
            "roi_expanded_y1": self.expanded_y1,
            "roi_expanded_x2": self.expanded_x2,
            "roi_expanded_y2": self.expanded_y2,
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


def _centered_origin_that_encloses(low: float, high: float, size: int) -> int:
    """Pick the centered integer origin among all origins enclosing [low, high]."""
    if not (isfinite(low) and isfinite(high) and high > low and size > 0):
        raise ValueError("invalid interval for fixed-aspect ROI")
    minimum_origin = int(ceil(high - size))
    maximum_origin = int(floor(low))
    if minimum_origin > maximum_origin:
        raise AssertionError(
            f"ROI size={size} cannot enclose interval {low}..{high}"
        )
    desired = int(round((low + high - size) / 2.0))
    return min(max(desired, minimum_origin), maximum_origin)


def fixed_aspect_roi_geometry(
    *,
    bbox: Sequence[float],
    frame_width: int,
    frame_height: int,
    expand_horizontal_each_side: float,
    expand_vertical_each_side: float,
    padding_mode: str = PADDING_MODE_REPLICATE,
) -> FixedAspectRoi:
    """Construct an exact integer 8:5 ROI containing all expanded eye context.

    The historical YOLO bbox is expanded using the configured horizontal and
    vertical fractions in floating source-frame coordinates. The smallest
    integer 8*k by 5*k rectangle capable of enclosing that entire expanded
    region is then chosen. Its origin is as centered as possible subject to the
    enclosure constraint. Out-of-frame parts are represented by padding, not by
    clipping the requested rectangle and stretching what remains.
    """
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
    expanded_x1 = x1 - horizontal * bbox_width
    expanded_x2 = x2 + horizontal * bbox_width
    expanded_y1 = y1 - vertical * bbox_height
    expanded_y2 = y2 + vertical * bbox_height

    required_width = int(ceil(expanded_x2) - floor(expanded_x1))
    required_height = int(ceil(expanded_y2) - floor(expanded_y1))
    units = max(
        1,
        int(ceil(required_width / ASPECT_WIDTH_UNIT)),
        int(ceil(required_height / ASPECT_HEIGHT_UNIT)),
    )
    width = ASPECT_WIDTH_UNIT * units
    height = ASPECT_HEIGHT_UNIT * units

    requested_x1 = _centered_origin_that_encloses(expanded_x1, expanded_x2, width)
    requested_y1 = _centered_origin_that_encloses(expanded_y1, expanded_y2, height)
    requested_x2 = requested_x1 + width
    requested_y2 = requested_y1 + height

    if not (
        requested_x1 <= expanded_x1
        and requested_y1 <= expanded_y1
        and requested_x2 >= expanded_x2
        and requested_y2 >= expanded_y2
    ):
        raise AssertionError("fixed-aspect ROI failed to contain full expanded context")
    if width * ASPECT_HEIGHT_UNIT != height * ASPECT_WIDTH_UNIT:
        raise AssertionError("fixed-aspect ROI is not exactly 8:5")

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
    if (source_x2 - source_x1) + pad_left + pad_right != width:
        raise AssertionError("horizontal ROI accounting mismatch")
    if (source_y2 - source_y1) + pad_top + pad_bottom != height:
        raise AssertionError("vertical ROI accounting mismatch")

    source_area = (source_x2 - source_x1) * (source_y2 - source_y1)
    requested_area = width * height
    valid_content_fraction = float(source_area / requested_area)
    resize_scale_x = TARGET_WIDTH / width
    resize_scale_y = TARGET_HEIGHT / height
    if abs(resize_scale_x - resize_scale_y) > 1e-12:
        raise AssertionError("fixed-aspect ROI does not yield uniform RITnet resize")

    return FixedAspectRoi(
        expanded_x1=float(expanded_x1),
        expanded_y1=float(expanded_y1),
        expanded_x2=float(expanded_x2),
        expanded_y2=float(expanded_y2),
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

    padding = (
        geometry.pad_top,
        geometry.pad_bottom,
        geometry.pad_left,
        geometry.pad_right,
    )
    if any(padding):
        if geometry.padding_mode != PADDING_MODE_REPLICATE:
            raise ValueError(f"unsupported padding mode: {geometry.padding_mode}")
        crop = cv2.copyMakeBorder(
            crop,
            geometry.pad_top,
            geometry.pad_bottom,
            geometry.pad_left,
            geometry.pad_right,
            borderType=cv2.BORDER_REPLICATE,
        )
    elif geometry.padding_mode != PADDING_MODE_REPLICATE:
        raise ValueError(f"unsupported padding mode: {geometry.padding_mode}")

    expected_shape = (geometry.height, geometry.width)
    if crop.shape != expected_shape:
        raise RuntimeError(
            f"fixed-aspect crop shape mismatch: expected={expected_shape}, got={crop.shape}"
        )
    return np.ascontiguousarray(crop)


def valid_source_analysis_mask(
    geometry: FixedAspectRoi,
    *,
    output_width: int = TARGET_WIDTH,
    output_height: int = TARGET_HEIGHT,
) -> np.ndarray:
    """Return a bool output-space mask for pixels backed by real AVI content.

    The pre-resize ROI contains real source pixels inside the four explicit
    padding widths and synthetic replicate padding outside that rectangle. A
    binary membership mask is resized with nearest-neighbour interpolation so
    it never creates fractional/artificial validity weights. This mask is for
    downstream analysis-domain selection only; it does not alter the image sent
    to RITnet.
    """
    output_width = int(output_width)
    output_height = int(output_height)
    if output_width <= 0 or output_height <= 0:
        raise ValueError("analysis mask output size must be positive")
    if output_width * ASPECT_HEIGHT_UNIT != output_height * ASPECT_WIDTH_UNIT:
        raise ValueError("analysis mask output must preserve exact 8:5 aspect ratio")

    if not any((geometry.pad_left, geometry.pad_top, geometry.pad_right, geometry.pad_bottom)):
        if (output_width, output_height) == (TARGET_WIDTH, TARGET_HEIGHT):
            return FULL_SOURCE_VALID_MASK
        return np.ones((output_height, output_width), dtype=bool)

    pre = np.zeros((geometry.height, geometry.width), dtype=np.uint8)
    y1 = int(geometry.pad_top)
    y2 = int(geometry.height - geometry.pad_bottom)
    x1 = int(geometry.pad_left)
    x2 = int(geometry.width - geometry.pad_right)
    if not (0 <= x1 < x2 <= geometry.width and 0 <= y1 < y2 <= geometry.height):
        raise ValueError("geometry has no valid source-backed content")
    pre[y1:y2, x1:x2] = 1

    if (geometry.width, geometry.height) == (output_width, output_height):
        resized = pre
    else:
        resized = cv2.resize(pre, (output_width, output_height), interpolation=cv2.INTER_NEAREST)
    mask = np.ascontiguousarray(resized.astype(bool, copy=False))
    if not mask.any():
        raise RuntimeError("analysis mask lost all valid source pixels during resize")
    return mask
