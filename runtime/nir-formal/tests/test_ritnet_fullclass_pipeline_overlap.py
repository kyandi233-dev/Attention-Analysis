from __future__ import annotations

import cv2
import numpy as np
import pytest

from ritnet_fullclass_final_engine import PreparedBatch, _next_prepared_batch, _summarize_outputs


class FakePrepRuntime:
    FIXED_BATCH_SIZE = 16

    def prepare_batch(self, rois):
        valid = len(rois)
        tensor = np.zeros((16, 1, 400, 640), dtype=np.float32)
        return tensor, valid, {"preprocess_ms": 1.25}


def _prepared_item(ordinal: int, *, success: bool = True):
    return {
        "ordinal": ordinal,
        "base": {"row": ordinal, "ritnet_status": None, "ritnet_failure_reason": None},
        "roi": np.zeros((400, 640), dtype=np.uint8) if success else None,
        "valid_source_mask": np.ones((400, 640), dtype=bool) if success else None,
    }


def _labels():
    labels = np.zeros((400, 640), dtype=np.uint8)
    cv2.ellipse(labels, (320, 200), (220, 90), 0, 0, 360, 1, -1)
    cv2.ellipse(labels, (320, 200), (90, 60), 0, 0, 360, 2, -1)
    cv2.ellipse(labels, (320, 200), (35, 25), 0, 0, 360, 3, -1)
    return labels


def test_next_prepared_batch_keeps_fixed_16_row_order_and_skips_failed_roi_from_inference():
    items = [_prepared_item(index, success=index != 5) for index in range(20)]
    iterator = iter(items)
    runtime = FakePrepRuntime()

    first = _next_prepared_batch(item_iterator=iterator, runtime=runtime)
    second = _next_prepared_batch(item_iterator=iterator, runtime=runtime)
    end = _next_prepared_batch(item_iterator=iterator, runtime=runtime)

    assert first is not None
    assert [item["ordinal"] for item in first.items] == list(range(16))
    assert first.successful_indices == tuple(index for index in range(16) if index != 5)
    assert first.valid_batch_size == 15
    assert first.tensor.shape == (16, 1, 400, 640)
    assert first.timing["preprocess_ms"] == pytest.approx(1.25)

    assert second is not None
    assert [item["ordinal"] for item in second.items] == [16, 17, 18, 19]
    assert second.valid_batch_size == 4
    assert end is None


def test_summarize_outputs_preserves_failed_rows_and_scientific_metrics():
    labels = _labels()
    probability = np.zeros((1, 4, 400, 640), dtype=np.float32)
    probability[:, 0] = 0.50
    probability[:, 1] = 0.30
    probability[:, 2] = 0.15
    probability[:, 3] = 0.05
    outputs = {
        "labels": labels[None, ...],
        "class_probability": probability,
        "max_probability": np.full((1, 400, 640), 0.9, dtype=np.float32),
        "top1_top2_margin": np.full((1, 400, 640), 0.7, dtype=np.float32),
        "entropy": np.full((1, 400, 640), 0.3, dtype=np.float32),
    }
    prepared = PreparedBatch(
        items=[_prepared_item(0), _prepared_item(1, success=False)],
        successful_indices=(0,),
        tensor=None,
        valid_batch_size=1,
        timing={},
    )
    prepared.items[1]["base"]["ritnet_status"] = "failed"
    prepared.items[1]["base"]["ritnet_failure_reason"] = "roi_invalid:test"

    completed, timing = _summarize_outputs(
        prepared=prepared,
        outputs=outputs,
        boundary_band_px=5,
        low_max_probability_threshold=None,
    )

    assert [ordinal for ordinal, _ in completed] == [0, 1]
    assert completed[0][1]["ritnet_status"] == "success"
    assert completed[0][1]["hard_pupil_pixels"] > 0
    assert completed[0][1]["soft_pupil_fraction"] == pytest.approx(0.05)
    assert completed[0][1]["whole_max_probability_mean"] == pytest.approx(0.9)
    assert completed[1][1]["ritnet_status"] == "failed"
    assert completed[1][1]["ritnet_failure_reason"] == "roi_invalid:test"
    assert timing["hard_metric_ms"] >= 0.0
    assert timing["uncertainty_ms"] >= 0.0
    assert timing["summary_total_ms"] >= 0.0
