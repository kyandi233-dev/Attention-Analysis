from __future__ import annotations

import numpy as np
import pytest

from ritnet_fullclass_qc import QCSelection
from ritnet_fullclass_qc_producer import _selection_groups
from ritnet_native_metrics import _ocular_aperture_metrics


def _reference_ocular_aperture(mask: np.ndarray) -> dict[str, float | int | None]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return {
            "bbox_width": None,
            "bbox_height": None,
            "aperture_height_median": None,
            "aperture_height_p90": None,
            "aperture_ratio_median": None,
            "aperture_ratio_p90": None,
        }
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    left = x_min + int(round(0.10 * max(0, width - 1)))
    right = x_min + int(round(0.90 * max(0, width - 1)))
    heights: list[int] = []
    for x in range(left, right + 1):
        column_y = np.flatnonzero(mask[:, x])
        if column_y.size:
            heights.append(int(column_y[-1] - column_y[0] + 1))
    if not heights:
        median = p90 = ratio_median = ratio_p90 = None
    else:
        median = float(np.median(heights))
        p90 = float(np.percentile(heights, 90))
        ratio_median = float(median / width)
        ratio_p90 = float(p90 / width)
    return {
        "bbox_width": width,
        "bbox_height": height,
        "aperture_height_median": median,
        "aperture_height_p90": p90,
        "aperture_ratio_median": ratio_median,
        "aperture_ratio_p90": ratio_p90,
    }


def test_vectorized_ocular_aperture_matches_previous_column_formula():
    mask = np.zeros((400, 640), dtype=bool)
    # Deliberately irregular ocular opening with empty columns and asymmetric
    # upper/lower borders so this checks the exact old per-column definition.
    for x in range(120, 521):
        if x % 37 == 0:
            continue
        top = 120 + (x % 19)
        bottom = 270 - (x % 23)
        mask[top : bottom + 1, x] = True

    expected = _reference_ocular_aperture(mask)
    actual = _ocular_aperture_metrics(mask)
    for field in expected:
        if isinstance(expected[field], float):
            assert actual[field] == pytest.approx(expected[field], abs=1e-12, rel=0)
        else:
            assert actual[field] == expected[field]


def _selection(frame: int) -> QCSelection:
    return QCSelection(
        phase="block1",
        phase_segment=1,
        frame_idx=frame,
        reasons=("fixed_anchor",),
        eyes=("frame_left", "frame_right"),
    )


def _success_row() -> dict[str, str]:
    return {"ritnet_status": "success"}


def test_qc_selection_groups_fill_fixed_b16_across_multiple_frames():
    selections = [_selection(frame) for frame in range(100, 110)]
    eyes_by_key = {
        selection.key: {
            "frame_left": _success_row(),
            "frame_right": _success_row(),
        }
        for selection in selections
    }

    groups = list(_selection_groups(selections, eyes_by_key))

    # Ten two-eye frames require only two fixed-b16 calls: 8 frames / 16 eyes,
    # then 2 frames / 4 eyes. The old QC producer would have made ten b16 calls.
    assert [len(group) for group in groups] == [8, 2]
    for group in groups:
        eye_slots = sum(
            len(eyes_by_key[selection.key])
            for selection in group
        )
        assert 1 <= eye_slots <= 16


def test_qc_selection_groups_do_not_waste_slots_on_yolo_miss_frames():
    selections = [_selection(frame) for frame in range(100, 117)]
    eyes_by_key = {selection.key: {} for selection in selections}
    groups = list(_selection_groups(selections, eyes_by_key))
    # Frame cap prevents unbounded image buffering even when no RITnet inference
    # is required. This is a memory bound, not a GPU-batch requirement.
    assert [len(group) for group in groups] == [16, 1]
