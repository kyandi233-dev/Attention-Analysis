import math

import numpy as np
import pandas as pd

from attention_pipeline.nir.sequence import compute_ear
from attention_pipeline.nir.sequence_eval import apply_interpolation, classify_status, continuity_metrics


def test_ear_formula_matches_v1():
    # v1 公式：(|p2-p3| + |p4-p5|) / (2*|p0-p1|)，pts=[33,133,160,144,158,153]
    pts = np.zeros((200, 2), dtype=np.float32)
    pts[33], pts[133] = (100.0, 100.0), (200.0, 100.0)   # 眼角，距 100
    pts[160], pts[144] = (170.0, 90.0), (130.0, 90.0)     # 上睑对，距 40
    pts[158], pts[153] = (170.0, 110.0), (130.0, 110.0)   # 下睑对，距 40
    assert math.isclose(compute_ear(pts, [33, 133, 160, 144, 158, 153]), (40 + 40) / (2 * 100))


def test_classify_status_all_branches():
    assert classify_status(0, 0.7, 0.55, True, True, False) == "no_face"
    assert classify_status(1, float("nan"), 0.55, True, True, False) == "openness_missing"
    assert classify_status(1, 0.3, 0.55, True, True, False) == "closed_gate"
    assert classify_status(1, 0.7, 0.55, True, True, False) == "accepted"
    assert classify_status(1, 0.7, 0.55, True, True, True) == "interpolated"
    assert classify_status(1, 0.7, 0.55, True, False, False) == "low_outline"
    assert classify_status(1, 0.7, 0.55, False, False, False) == "detector_missing"


def test_apply_interpolation_fills_bounded_gap():
    frames = pd.DataFrame({
        "frame_offset": range(0, 8),
        "unix_ms": [1000.0 + i * 33.0 for i in range(8)],
        "observed": [1, 1, 0, 0, 0, 1, 1, 1],           # 缺口 3 帧 ≈99ms
        "major_diameter": [20.0, 20.0, np.nan, np.nan, np.nan, 22.0, 22.0, 22.0],
        "minor_diameter": [10.0, 10.0, np.nan, np.nan, np.nan, 11.0, 11.0, 11.0],
        "center_x": [100.0, 100.0, np.nan, np.nan, np.nan, 104.0, 104.0, 104.0],
        "center_y": [80.0, 80.0, np.nan, np.nan, np.nan, 80.0, 80.0, 80.0],
    })
    frames = frames.assign(
        sequence_id="seq", eye="eye_right", face_detected=1, roi_path="roi.png",
        visible_proxy=1, p80_closed_proxy=0, openness=0.8,
    )
    filled = apply_interpolation(frames, max_gap_ms=200, endpoint_diameter_tol=0.20)
    assert int(filled["is_interpolated"].sum()) == 3
    assert np.isnan(filled.loc[3, "major_diameter"])
    assert 20.0 < filled.loc[3, "major_diameter_interpolated"] < 22.0
    # 缺口两端保持 observed
    assert filled.loc[1, "is_interpolated"] == 0 and filled.loc[5, "is_interpolated"] == 0


def test_apply_interpolation_respects_max_gap_and_boundary():
    frames = pd.DataFrame({
        "frame_offset": range(0, 6),
        "unix_ms": [0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
        "observed": [1, 0, 0, 0, 0, 1],                  # 缺口 400ms > 200ms
        "major_diameter": [20.0, np.nan, np.nan, np.nan, np.nan, 22.0],
        "minor_diameter": [10.0, np.nan, np.nan, np.nan, np.nan, 11.0],
        "center_x": [100.0, np.nan, np.nan, np.nan, np.nan, 104.0],
        "center_y": [80.0, np.nan, np.nan, np.nan, np.nan, 80.0],
    })
    frames = frames.assign(
        sequence_id="seq", eye="eye_right", face_detected=1, roi_path="roi.png",
        visible_proxy=1, p80_closed_proxy=0, openness=0.8,
    )
    filled = apply_interpolation(frames, max_gap_ms=200, endpoint_diameter_tol=0.20)
    assert int(filled["is_interpolated"].sum()) == 0     # 超 max_gap，不插

    # 缺口触及序列首（左侧无 observed）→ 不插
    frames2 = pd.DataFrame({
        "frame_offset": range(0, 4),
        "unix_ms": [0.0, 33.0, 66.0, 99.0],
        "observed": [0, 0, 1, 1],
        "major_diameter": [np.nan, np.nan, 20.0, 20.0],
        "minor_diameter": [np.nan, np.nan, 10.0, 10.0],
        "center_x": [np.nan, np.nan, 100.0, 100.0],
        "center_y": [np.nan, np.nan, 80.0, 80.0],
    })
    frames2 = frames2.assign(
        sequence_id="seq", eye="eye_right", face_detected=1, roi_path="roi.png",
        visible_proxy=1, p80_closed_proxy=0, openness=0.8,
    )
    assert int(apply_interpolation(frames2, 200, 0.20)["is_interpolated"].sum()) == 0


def test_continuity_metrics_on_synthetic():
    frames = pd.DataFrame({
        "frame_offset": range(0, 6),
        "unix_ms": [0.0, 33.0, 66.0, 99.0, 132.0, 165.0],
        "observed": [1, 1, 1, 1, 1, 1],
        "visible_proxy": [1, 1, 1, 1, 1, 1],
        "p80_closed_proxy": [0, 0, 0, 0, 0, 0],
        "major_diameter": [20.0, 22.0, 24.2, 26.62, 29.28, 32.21],   # 每帧 ×1.1 → log 跳变恒定
        "center_x": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "center_y": [80.0, 80.0, 80.0, 80.0, 80.0, 80.0],
    })
    frames = frames.assign(sequence_id="seq", eye="eye_right")
    cm = continuity_metrics(frames)
    expect_jump = abs(math.log(1.1))
    assert math.isclose(cm["diameter_log_jump_median"], expect_jump, rel_tol=1e-3)
    assert cm["center_jump_norm_median"] == 0.0
    assert cm["n_recovery_events"] == 0


def test_interpolation_rejects_closed_or_missing_face_gap():
    base = pd.DataFrame({
        "sequence_id": ["seq"] * 4,
        "eye": ["eye_right"] * 4,
        "frame_offset": range(4),
        "unix_ms": [0.0, 33.0, 66.0, 99.0],
        "observed": [1, 0, 0, 1],
        "major_diameter": [20.0, np.nan, np.nan, 21.0],
        "minor_diameter": [10.0, np.nan, np.nan, 10.5],
        "center_x": [100.0, np.nan, np.nan, 101.0],
        "center_y": [80.0, np.nan, np.nan, 80.0],
        "face_detected": [1, 1, 1, 1],
        "roi_path": ["roi.png"] * 4,
        "visible_proxy": [1, 1, 1, 1],
        "p80_closed_proxy": [0, 1, 1, 0],
        "openness": [0.8, 0.1, 0.1, 0.8],
    })
    assert apply_interpolation(base, 200, 0.20)["is_interpolated"].sum() == 0
    no_face = base.copy()
    no_face["p80_closed_proxy"] = 0
    no_face["openness"] = 0.8
    no_face.loc[1:2, "face_detected"] = 0
    assert apply_interpolation(no_face, 200, 0.20)["is_interpolated"].sum() == 0
