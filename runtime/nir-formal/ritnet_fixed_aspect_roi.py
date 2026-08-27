"""Deterministic fixed-aspect ROI construction for final RITnet analysis.

The historical formal pipeline expanded each YOLO eye box and then resized the
result to 640x400 even when the crop aspect ratio differed from 1.6. That
introduces anisotropic stretch. The final full-class path instead builds a
virtual crop whose integer pixel size is exactly 8:5 (=1.6), pads at frame
boundaries when necessary, and then resizes uniformly to 640x400.

This module contains only geometry/cropping logic. It does not run YOLO or
RITnet and does not modify historical source files.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, isfinite
from typing import Sequence

import cv2
import numpy as np

TARGET_ASPECT_NUM = 8
TARGET_ASPECT_DEN = 5
TARGET_ASPECT_RATIO = TARGET_ASPECT_NUM / TARGET_ASPECT_DEN
DEFAULT_PADDING_MODE = "reflect101"
ROI_ALGORITHM_VERSION = "fixed-aspect-8x5-expanded-yolo-reflect101-v1"


@dataclass(frozen=True)
class FixedAspectRoiGeometry:
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    expanded_x1: float
    expanded_y1: float
    expanded_x2: float
    expanded_y2: float
    virtual_x1: int
    virtual_y1: int
    virtual_x2: int
    virtual_y2: int
    source_x1: int
    source_y1: int
    source_x2: int
    source_y2: int
    padding_left: int
    padding_top: int
    padding_right: int
    padding_bottom: int
    virtual_width: int
    virtual_height: int
    source_width: int
    source_height: int
    padding_mode: str
    algorithm_version: str = ROI_ALGORITHM_VERSION

    @property
    def aspect_ratio(self) -> float:
        return self.virtual_width / self.virtual_height

    @property
    def padded(self) -> bool:
        return any(
            value > 0
            for value in (
                self.padding_left,
                self.padding_top,
                self.padding_right,
                self.padding_bottom,
            )
        )

    def as_dict(self) -> dict[str, int | float | bool | str]:
        return {
            "source_yolo_bbox_x1": self.bbox_x1,
            "source_yolo_bbox_y1": self.bbox_y1,
            "source_yolo_bbox_x2": self.bbox_x2,
            "source_yolo_bbox_y2": self.bbox_y2,
            "expanded_context_x1": self.expanded_x1,
            "expanded_context_y1": self.expanded_y1,
            "expanded_context_x2": self.expanded_x2,
            "expanded_context_y2": self.expanded_y2,
            "final_roi_virtual_x1": self.virtual_x1,
            "final_roi_virtual_y1": self.virtual_y1,
            "final_roi_virtual_x2": self.virtual_x2,
            "final_roi_virtual_y2": self.virtual_y2,
            "final_roi_source_x1": self.source_x1,
            "final_roi_source_y1": self.source_y1,
            "final_roi_source_x2": self.source_x2,
            "final_roi_source_y2": self.source_y2,
            "final_roi_padding_left": self.padding_left,
            "final_roi_padding_top": self.padding_top,
            "final_roi_padding_right": self.padding_right,
            "final_roi_padding_bottom": self.padding_bottom,
            "final_roi_width": self.virtual_width,
            "final_roi_height": self.virtual_height,
            "final_roi_aspect_ratio": self.aspect_ratio,
            "final_roi_padded": self.padded,
            "final_roi_padding_mode": self.padding_mode,
            "final_roi_algorithm_version": self.algorithm_version,
        }


def _validate_bbox(
    box: Sequence[float], frame_width: int, frame_height: int
) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError(f"bbox must have four values, got {box!r}")
    x1, y1, x2, y2 = map(float, box)
    if not all(isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError(f"bbox contains non-finite values: {box!r}")
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(f"invalid frame size: {(frame_width, frame_height)}")
    if not (0.0 <= x1 < x2 <= frame_width and 0.0 <= y1 < y2 <= frame_height):
        raise ValueError(
            f"bbox must lie inside frame and have positive area: box={(x1, y1, x2, y2)}, "
            f"frame={(frame_width, frame_height)}"
        )
    return x1, y1, x2, y2


def _choose_integer_origin(
    *, low_edge: float, high_edge: float, size: int
) -> int:
    """Choose the most centered integer origin that still encloses [low, high]."""
    if size <= 0 or high_edge <= low_edge:
        raise ValueError("invalid interval/size for integer ROI origin")
    minimum_origin = int(ceil(high_edge - size))
    maximum_origin = int(floor(low_edge))
    if minimum_origin > maximum_origin:
        raise AssertionError(
            f"integer ROI size {size} cannot enclose interval {low_edge}..{high_edge}"
        )
    desired = int(round((low_edge + high_edge - size) / 2.0))
    return min(max(desired, minimum_origin), maximum_origin)


def build_fixed_aspect_geometry(
    box: Sequence[float],
    *,
    frame_width: int,
    frame_height: int,
    expand_horizontal_each_side: float,
    expand_vertical_each_side: float,
    padding_mode: str = DEFAULT_PADDING_MODE,
) -> FixedAspectRoiGeometry:
    """Build one deterministic 8:5 virtual crop around a YOLO eye box.

    Expansion happens in floating-point source-frame coordinates. The expanded
    context is fully enclosed by the smallest integer 8*k by 5*k crop that can
    contain it. Among all legal origins, the crop closest to the expanded
    context center is selected. The virtual crop may extend outside the frame;
    those pixels are explicit padding rather than asymmetric clipping.
    """
    x1, y1, x2, y2 = _validate_bbox(box, int(frame_width), int(frame_height))
    horizontal = float(expand_horizontal_each_side)
    vertical = float(expand_vertical_each_side)
    if (
        not isfinite(horizontal)
        or not isfinite(vertical)
        or horizontal < 0
        or vertical < 0
    ):
        raise ValueError("ROI expansion fractions must be finite and non-negative")
    if padding_mode not in {"reflect101", "replicate", "constant"}:
        raise ValueError(f"unsupported padding_mode: {padding_mode!r}")

    bbox_w = x2 - x1
    bbox_h = y2 - y1
    expanded_x1 = x1 - horizontal * bbox_w
    expanded_x2 = x2 + horizontal * bbox_w
    expanded_y1 = y1 - vertical * bbox_h
    expanded_y2 = y2 + vertical * bbox_h

    # First convert the floating context to the exact integer span required to
    # cover every source pixel touched by that context. Then choose the smallest
    # 8*k by 5*k rectangle that can contain both required spans.
    required_w = int(ceil(expanded_x2) - floor(expanded_x1))
    required_h = int(ceil(expanded_y2) - floor(expanded_y1))
    k = max(
        1,
        int(
            ceil(
                max(
                    required_w / TARGET_ASPECT_NUM,
                    required_h / TARGET_ASPECT_DEN,
                )
            )
        ),
    )
    virtual_w = TARGET_ASPECT_NUM * k
    virtual_h = TARGET_ASPECT_DEN * k
    virtual_x1 = _choose_integer_origin(
        low_edge=expanded_x1, high_edge=expanded_x2, size=virtual_w
    )
    virtual_y1 = _choose_integer_origin(
        low_edge=expanded_y1, high_edge=expanded_y2, size=virtual_h
    )
    virtual_x2 = virtual_x1 + virtual_w
    virtual_y2 = virtual_y1 + virtual_h

    # These are invariant checks, not approximate QC conditions.
    if virtual_x1 > expanded_x1 or virtual_x2 < expanded_x2:
        raise AssertionError("fixed-aspect ROI failed to enclose expanded x context")
    if virtual_y1 > expanded_y1 or virtual_y2 < expanded_y2:
        raise AssertionError("fixed-aspect ROI failed to enclose expanded y context")
    if virtual_w * TARGET_ASPECT_DEN != virtual_h * TARGET_ASPECT_NUM:
        raise AssertionError("fixed-aspect ROI is not exactly 8:5")

    source_x1 = max(0, virtual_x1)
    source_y1 = max(0, virtual_y1)
    source_x2 = min(int(frame_width), virtual_x2)
    source_y2 = min(int(frame_height), virtual_y2)
    if source_x2 <= source_x1 or source_y2 <= source_y1:
        raise ValueError(
            "fixed-aspect ROI does not intersect the source frame: "
            f"virtual={(virtual_x1, virtual_y1, virtual_x2, virtual_y2)}, "
            f"frame={(frame_width, frame_height)}"
        )

    padding_left = source_x1 - virtual_x1
    padding_top = source_y1 - virtual_y1
    padding_right = virtual_x2 - source_x2
    padding_bottom = virtual_y2 - source_y2
    if min(padding_left, padding_top, padding_right, padding_bottom) < 0:
        raise AssertionError("computed ROI padding cannot be negative")
    if (source_x2 - source_x1) + padding_left + padding_right != virtual_w:
        raise AssertionError("horizontal ROI accounting mismatch")
    if (source_y2 - source_y1) + padding_top + padding_bottom != virtual_h:
        raise AssertionError("vertical ROI accounting mismatch")

    return FixedAspectRoiGeometry(
        bbox_x1=x1,
        bbox_y1=y1,
        bbox_x2=x2,
        bbox_y2=y2,
        expanded_x1=float(expanded_x1),
        expanded_y1=float(expanded_y1),
        expanded_x2=float(expanded_x2),
        expanded_y2=float(expanded_y2),
        virtual_x1=virtual_x1,
        virtual_y1=virtual_y1,
        virtual_x2=virtual_x2,
        virtual_y2=virtual_y2,
        source_x1=source_x1,
        source_y1=source_y1,
        source_x2=source_x2,
        source_y2=source_y2,
        padding_left=padding_left,
        padding_top=padding_top,
        padding_right=padding_right,
        padding_bottom=padding_bottom,
        virtual_width=virtual_w,
        virtual_height=virtual_h,
        source_width=source_x2 - source_x1,
        source_height=source_y2 - source_y1,
        padding_mode=padding_mode,
    )


def crop_fixed_aspect_gray(
    frame: np.ndarray,
    geometry: FixedAspectRoiGeometry,
    *,
    constant_value: int = 0,
) -> np.ndarray:
    """Crop and pad one ROI without resizing it.

    The returned array is always exactly ``virtual_height x virtual_width``.
    RITnet preprocessing may then resize this array to 640x400 with one uniform
    scale factor because both source and model input have the same 8:5 ratio.
    """
    array = np.asarray(frame)
    if array.ndim not in {2, 3}:
        raise ValueError(f"frame must be grayscale or BGR, got shape={array.shape}")
    frame_h, frame_w = array.shape[:2]
    if frame_w < geometry.source_x2 or frame_h < geometry.source_y2:
        raise ValueError(
            f"geometry source bounds exceed frame: frame={array.shape}, "
            f"bounds={(geometry.source_x1, geometry.source_y1, geometry.source_x2, geometry.source_y2)}"
        )
    crop = array[
        geometry.source_y1 : geometry.source_y2,
        geometry.source_x1 : geometry.source_x2,
    ]
    if array.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        crop = np.ascontiguousarray(crop)

    border_types = {
        "reflect101": cv2.BORDER_REFLECT_101,
        "replicate": cv2.BORDER_REPLICATE,
        "constant": cv2.BORDER_CONSTANT,
    }
    if geometry.padded:
        crop = cv2.copyMakeBorder(
            crop,
            geometry.padding_top,
            geometry.padding_bottom,
            geometry.padding_left,
            geometry.padding_right,
            border_types[geometry.padding_mode],
            value=int(constant_value),
        )
    crop = np.ascontiguousarray(crop)
    expected = (geometry.virtual_height, geometry.virtual_width)
    if crop.shape != expected:
        raise RuntimeError(
            f"fixed-aspect crop shape mismatch: expected={expected}, got={crop.shape}"
        )
    return crop


def uniform_resize_scale(
    geometry: FixedAspectRoiGeometry, output_size: tuple[int, int] = (640, 400)
) -> float:
    output_w, output_h = map(int, output_size)
    if output_w <= 0 or output_h <= 0:
        raise ValueError(f"invalid output size: {output_size}")
    if output_w * TARGET_ASPECT_DEN != output_h * TARGET_ASPECT_NUM:
        raise ValueError(f"output size must use 8:5 aspect ratio, got {output_size}")
    scale_x = output_w / geometry.virtual_width
    scale_y = output_h / geometry.virtual_height
    if not np.isclose(scale_x, scale_y, rtol=0.0, atol=1e-12):
        raise AssertionError(
            f"non-uniform resize would occur: scale_x={scale_x}, scale_y={scale_y}"
        )
    return float(scale_x)
