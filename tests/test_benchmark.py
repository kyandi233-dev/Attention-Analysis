import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

from attention_pipeline.nir.benchmark import (
    admission_thresholds,
    equivalent_diameter,
    geometry_plausible,
    rate_metrics,
    usable_fit,
)


def _load_adapter():
    path = Path(__file__).resolve().parents[1] / "scripts" / "nir_detect_batch.py"
    spec = importlib.util.spec_from_file_location("nir_detect_batch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_geometry_plausible_gates_returns():
    assert geometry_plausible((100, 80), 20, 10) is True
    assert geometry_plausible((340, 80), 20, 10) is False          # 中心出画
    assert geometry_plausible((100, 80), 2, 1) is False            # 过小
    assert geometry_plausible((100, 80), 200, 100) is False        # 过大（>0.65*min=104）
    assert geometry_plausible((100, 80), 40, 5) is False           # aspect 0.125 < 0.25


def test_equivalent_diameter_contract():
    assert math.isclose(equivalent_diameter(4, 9), 6.0)
    assert math.isclose(equivalent_diameter(20, 10), math.sqrt(200))


def test_usable_fit_against_truth():
    truth = {"center_x": 100.0, "center_y": 80.0, "major_diameter": 20.0, "minor_diameter": 10.0, "angle_deg": 0.0}
    exact = {"center_x": 100.0, "center_y": 80.0, "major_diameter": 20.0, "minor_diameter": 10.0, "angle_deg": 0.0}
    assert usable_fit(truth, exact, iou_min=0.7, center_error_max_px=16, diameter_relative_error_max=0.2) is True
    off_center = {**exact, "center_x": 130.0}                     # 中心差 30px > 16
    assert usable_fit(truth, off_center, 0.7, 16, 0.2) is False
    wrong_size = {**exact, "major_diameter": 60.0, "minor_diameter": 30.0}  # 等径相对误差 ~2.0
    assert usable_fit(truth, wrong_size, 0.7, 16, 0.2) is False


def _synthetic_det():
    return pd.DataFrame([
        {"sample_id": "a", "returned": 1, "center_x": 100.0, "center_y": 80.0, "major_diameter": 20.0, "minor_diameter": 10.0, "angle_deg": 0.0},
        {"sample_id": "b", "returned": 1, "center_x": 100.0, "center_y": 80.0, "major_diameter": 20.0, "minor_diameter": 10.0, "angle_deg": 0.0},
        {"sample_id": "c", "returned": 1, "center_x": 100.0, "center_y": 80.0, "major_diameter": 60.0, "minor_diameter": 30.0, "angle_deg": 0.0},
        {"sample_id": "d", "returned": 0, "center_x": np.nan, "center_y": np.nan, "major_diameter": np.nan, "minor_diameter": np.nan, "angle_deg": np.nan},
        {"sample_id": "e", "returned": 0, "center_x": np.nan, "center_y": np.nan, "major_diameter": np.nan, "minor_diameter": np.nan, "angle_deg": np.nan},
    ])


def test_rate_metrics_layers_and_denominators():
    det = _synthetic_det()
    truth_lookup = {
        s: {"center_x": 100.0, "center_y": 80.0, "major_diameter": 20.0, "minor_diameter": 10.0, "angle_deg": 0.0}
        for s in ("a", "b", "c", "d")
    }
    metrics = rate_metrics(
        det,
        truth_lookup,
        visible_truth_ids={"a", "b", "c", "d"},
        visible_all_ids={"a", "b", "c", "d", "face1"},   # face1 = face 失败，无检测行
        invisible_roi_ids={"e"},
        invisible_all_ids={"e", "face2"},
        subject_of={"a": "s1", "b": "s1", "c": "s2", "d": "s2", "e": "s1", "face1": "s3", "face2": "s3"},
        thresholds={"iou_min": 0.7, "center_error_max_px": 16.0, "diameter_relative_error_max": 0.2, "per_subject_usable_min": 0.75},
    )
    assert metrics["returned_n"] == 3
    assert metrics["usable_n"] == 2
    assert math.isclose(metrics["algorithm_layer_visible_rate"], 2 / 4)
    assert math.isclose(metrics["end_to_end_visible_rate"], 2 / 5)   # face1 进分母但不贡献
    assert math.isclose(metrics["wrong_among_returned_rate"], 1 / 3)
    assert metrics["fp_rate_algorithm_layer"] == 0.0
    assert metrics["fp_rate_end_to_end"] == 0.0
    assert metrics["subjects_passing_usable"] == 1   # s1=1.0 通过, s2=0.5 不通过


def test_admission_thresholds_load(config):
    thresholds = admission_thresholds(config)
    assert thresholds["iou_min"] == 0.7
    assert thresholds["center_error_max_px"] == 16.0
    assert thresholds["diameter_relative_error_max"] == 0.2
    assert thresholds["visible_end_to_end_rate_min"] == 0.85
    assert thresholds["invisible_fp_rate_max"] == 0.05


def test_photometric_gate_filters_low_contrast_returned():
    det = _synthetic_det()
    det["photometric_contrast"] = [0.10, 0.01, 0.08, np.nan, np.nan]  # b 低对比应被门控掉
    truth_lookup = {
        s: {"center_x": 100.0, "center_y": 80.0, "major_diameter": 20.0, "minor_diameter": 10.0, "angle_deg": 0.0}
        for s in ("a", "b", "c", "d")
    }
    thresholds = {"iou_min": 0.7, "center_error_max_px": 16.0, "diameter_relative_error_max": 0.2, "per_subject_usable_min": 0.75}
    metrics = rate_metrics(
        det, truth_lookup, visible_truth_ids={"a", "b", "c", "d"}, visible_all_ids={"a", "b", "c", "d", "face1"},
        invisible_roi_ids={"e"}, invisible_all_ids={"e", "face2"},
        subject_of={"a": "s1", "b": "s1", "c": "s2", "d": "s2", "e": "s1", "face1": "s3", "face2": "s3"},
        thresholds=thresholds, photometric_threshold=0.02,
    )
    # 无门控时 a、b 都 usable；门控后 b(0.01) 被滤，a(0.10) 保留，c 本就不符合 usable_fit
    assert metrics["returned_n"] == 2     # a、c（b 被光度门滤掉）
    assert metrics["usable_n"] == 1       # 只有 a
    assert math.isclose(metrics["end_to_end_visible_rate"], 1 / 5)


def test_apply_params_targets_swirski_params_object():
    class FakeParams:
        pass

    class FakeDetector:
        def __init__(self):
            self.params = FakeParams()

    module = _load_adapter()
    swirski = FakeDetector()
    module._apply_params(swirski, "Swirski2D", {"Radius_Min": 15, "Radius_Max": 45})
    assert swirski.params.Radius_Min == 15
    assert swirski.params.Radius_Max == 45

    starburst = FakeDetector()
    module._apply_params(starburst, "Starburst", {"edge_threshold": 20, "rays": 24})
    assert starburst.edge_threshold == 20
    assert starburst.rays == 24
    assert not hasattr(starburst.params, "edge_threshold")


def test_tuning_grid_has_three_configs_per_algorithm(config):
    tuning = config.section("nir")["benchmark"]["tuning"]
    assert len(tuning) == 18
    algorithms = [item["algorithm"] for item in tuning]
    assert all(algorithms.count(alg) == 3 for alg in set(algorithms))
    for item in tuning:
        assert item["config"] and "params" in item


def test_canonical_axes_preserve_ellipse_orientation():
    import cv2

    module = _load_adapter()
    for axis_w, axis_h, angle in [(40.0, 20.0, 20.0), (20.0, 40.0, 70.0), (18.0, 42.0, 110.0)]:
        major, minor, major_angle = module.canonicalize_axes(axis_w, axis_h, angle)
        raw = np.zeros((160, 320), dtype=np.uint8)
        canonical = np.zeros_like(raw)
        cv2.ellipse(raw, (160, 80), (round(axis_w / 2), round(axis_h / 2)), angle, 0, 360, 1, -1)
        cv2.ellipse(canonical, (160, 80), (round(major / 2), round(minor / 2)), major_angle, 0, 360, 1, -1)
        intersection = np.logical_and(raw, canonical).sum()
        union = np.logical_or(raw, canonical).sum()
        assert intersection / union > 0.98
