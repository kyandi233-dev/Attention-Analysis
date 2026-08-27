"""Unit tests for the seven-algorithm native-resolution pupil benchmark.

These tests exercise the three-layer semantics (algorithm_returned /
official_valid / geometry_sane), the unified schema, scale-rule parameter
freezing, and row assembly. They do not require the compiled detector
libraries; algorithm results are simulated with fakes.
"""
import importlib.util
import math
import sys

import numpy as np
import pandas as pd
import pytest

from attention_pipeline.nir_pupil_benchmark import (
    ALGORITHMS,
    ALGORITHM_SPECS,
    RESULT_COLUMNS,
    Ellipse,
    assemble_row,
    detect_crop,
    draw_detection,
    geometry_sane,
    make_synthetic_eye,
    normalize_result,
    parse_pupil_result,
    pupil_diameter_bounds,
    run_crop_list,
)
from attention_pipeline.nir_pupil_benchmark.adapters import (
    DetectionOutput,
    frozen_pupil_labs_properties,
    frozen_swirski_params,
)
from attention_pipeline.nir_pupil_benchmark import runner as runner_mod


class FakePupil:
    """Simulates a PyPupilEXT Pupil (center/size as tuples, valid(), hasOutline())."""

    def __init__(
        self,
        center=(100.0, 80.0),
        size=(8.0, 14.0),
        angle=30.0,
        confidence=0.8,
        outline_confidence=-1.0,
        has_outline=True,
        valid_result=True,
    ):
        self.center = center
        self.size = size
        self.angle = angle
        self.confidence = confidence
        self.outline_confidence = outline_confidence
        self._has_outline = has_outline
        self._valid_result = valid_result

    def hasOutline(self):
        return self._has_outline

    def valid(self, threshold):
        return self._valid_result


# ---------- three-layer parse semantics ----------


def test_parse_pupil_result_three_layers_kept_apart():
    parsed = parse_pupil_result(
        FakePupil(center=(200.0, 90.0), size=(8.0, 14.0), angle=12.0),
        width=424.0,
        height=187.0,
    )
    assert parsed["algorithm_returned"] is True
    assert parsed["official_valid"] is True
    assert parsed["geometry_sane"] is True
    assert parsed["major_axis"] == 14.0
    assert parsed["minor_axis"] == 8.0
    assert parsed["center_x"] == 200.0
    assert parsed["center_y"] == 90.0
    assert math.isclose(parsed["diameter_geom"], math.sqrt(14.0 * 8.0))
    assert math.isclose(parsed["area"], math.pi * 14.0 * 8.0 / 4.0)


def test_parse_pupil_result_sentinel_no_outline():
    # cleared / failure sentinel: center=(-1,-1), size=(-1,-1)
    parsed = parse_pupil_result(
        FakePupil(center=(-1.0, -1.0), size=(-1.0, -1.0), has_outline=False, valid_result=False),
        width=424.0,
        height=187.0,
    )
    assert parsed["algorithm_returned"] is False
    assert parsed["official_valid"] is False
    assert parsed["geometry_sane"] is False
    assert parsed["ellipse"] is None


def test_parse_pupil_result_confidence_less_algorithm_official_valid_false():
    # ElSe/ExCuSe/Swirski2D/Starburst: has outline but no native confidence ->
    # official valid() is False by official semantics.
    parsed = parse_pupil_result(
        FakePupil(center=(200.0, 90.0), size=(8.0, 14.0), confidence=-1.0, valid_result=False),
        width=424.0,
        height=187.0,
    )
    assert parsed["algorithm_returned"] is True
    assert parsed["official_valid"] is False
    assert parsed["geometry_sane"] is True  # geometry independent of confidence


def test_parse_pupil_result_pupil_labs_2d_dict():
    result = {
        "ellipse": {"center": (212.0, 93.0), "axes": (8.0, 14.0), "angle": 12.0},
        "diameter": 14.0,
        "location": (212.0, 93.0),
        "confidence": 0.91,
    }
    parsed = parse_pupil_result(result, width=424.0, height=187.0)
    assert parsed["algorithm_returned"] is True
    assert parsed["official_valid"] is True
    assert parsed["major_axis"] == 14.0
    assert parsed["minor_axis"] == 8.0


def test_parse_pupil_result_pupil_labs_2d_failure_sentinel():
    result = {
        "ellipse": {"center": (0.0, 0.0), "axes": (0.0, 0.0), "angle": -90.0},
        "diameter": 0.0,
        "location": (0.0, 0.0),
        "confidence": 0.0,
    }
    parsed = parse_pupil_result(result, width=424.0, height=187.0)
    assert parsed["algorithm_returned"] is False
    assert parsed["official_valid"] is False


def test_parse_pupil_result_none():
    parsed = parse_pupil_result(None, width=424.0, height=187.0)
    assert parsed["algorithm_returned"] is False
    assert parsed["geometry_sane"] is False


def test_normalize_result_axis_order_is_max_major():
    # PyPupilEXT size is (w,h) unordered; major must be max.
    norm = normalize_result(FakePupil(center=(1.0, 1.0), size=(20.0, 6.0), angle=45.0))
    assert norm["size"] == (20.0, 6.0)
    parsed = parse_pupil_result(FakePupil(center=(1.0, 1.0), size=(20.0, 6.0), angle=45.0))
    assert parsed["major_axis"] == 20.0
    assert parsed["minor_axis"] == 6.0


# ---------- geometry_sane ----------


def test_geometry_sane_gates():
    base = Ellipse(cx=212.0, cy=93.0, axis_a=14.0, axis_b=8.0, angle_deg=10.0)
    assert geometry_sane(base, 424.0, 187.0) is True

    # center out of crop
    assert geometry_sane(Ellipse(500.0, 93.0, 14.0, 8.0), 424.0, 187.0) is False
    # axis too large
    assert geometry_sane(Ellipse(212.0, 93.0, 200.0, 8.0), 424.0, 187.0) is False
    # axis too small
    assert geometry_sane(Ellipse(212.0, 93.0, 14.0, 1.0), 424.0, 187.0) is False
    # bad aspect
    assert geometry_sane(Ellipse(212.0, 93.0, 30.0, 3.0), 424.0, 187.0) is False
    # non-finite
    assert geometry_sane(Ellipse(212.0, 93.0, math.nan, 8.0), 424.0, 187.0) is False
    assert geometry_sane(None, 424.0, 187.0) is False


# ---------- scale rule ----------


def test_pupil_diameter_bounds_scale_rule():
    radius_min, radius_max = pupil_diameter_bounds(424, 187)
    assert radius_min == max(2, round(0.02 * 187))
    assert radius_max >= radius_min + 4
    # 424x187 -> Radius_Min=4, Radius_Max=19 per the frozen rule
    assert (radius_min, radius_max) == (4, 19)


def test_frozen_swirski_and_pupil_labs_params():
    sw = frozen_swirski_params(424, 187)
    assert sw["Radius_Min"] == 4
    assert sw["Radius_Max"] == 19
    assert sw["Seed"] == 0  # fixed RANSAC seed for reproducibility

    pl = frozen_pupil_labs_properties(424, 187)
    assert pl["pupil_size_min"] == 2 * 4
    assert pl["pupil_size_max"] == 2 * 19
    assert pl["coarse_filter_min"] == 2 * 4


def test_all_seven_algorithms_specs_present():
    assert set(ALGORITHMS) == {
        "PuRe", "PuReST", "PupilLabs2D", "ElSe", "ExCuSe", "Swirski2D", "Starburst",
    }
    for name in ALGORITHMS:
        spec = ALGORITHM_SPECS[name]
        assert spec.name == name
        assert spec.package in ("pypupilext", "pupil_detectors")
    assert ALGORITHM_SPECS["Swirski2D"].has_native_confidence is False
    assert ALGORITHM_SPECS["PuRe"].has_native_confidence is True
    assert ALGORITHM_SPECS["PuReST"].has_state is True
    assert ALGORITHM_SPECS["Starburst"].has_state is True
    assert ALGORITHM_SPECS["PupilLabs2D"].has_state is True


# ---------- row assembly + runner ----------


def test_detect_crop_row_assembly(monkeypatch):
    fake_result = FakePupil(center=(212.0, 93.0), size=(14.0, 8.0), angle=12.0)
    fake_output = DetectionOutput(
        algorithm="PuRe", result=fake_result, runtime_ms=3.5, failure=None,
    )
    monkeypatch.setattr(runner_mod, "make_detector", lambda spec, params=None: object())
    monkeypatch.setattr(runner_mod, "run_detection", lambda *a, **k: fake_output)

    image = np.full((187, 424), 120, dtype=np.uint8)
    row = detect_crop(image, "PuRe")
    assert row["algorithm"] == "PuRe"
    assert row["algorithm_returned"] is True
    assert row["official_valid"] is True
    assert row["geometry_sane"] is True
    assert row["runtime_ms"] == 3.5
    assert row["input_width"] == 424
    assert row["input_height"] == 187
    assert "scale_rule" in row["params_provenance"]


def test_detect_crop_failure_recorded(monkeypatch):
    fake_output = DetectionOutput(
        algorithm="Swirski2D", result=None, runtime_ms=1.0, failure="RuntimeError: boom",
    )
    monkeypatch.setattr(runner_mod, "make_detector", lambda spec, params=None: object())
    monkeypatch.setattr(runner_mod, "run_detection", lambda *a, **k: fake_output)
    row = detect_crop(np.zeros((187, 424), dtype=np.uint8), "Swirski2D")
    assert row["algorithm_returned"] is False
    assert row["failure"] == "RuntimeError: boom"


def test_assemble_row_merge():
    identity = {
        "subject": "sub-031", "phase": "block1", "frame_idx": 5,
        "eye": "eye_left", "bbox_x1": 10, "bbox_y1": 20,
    }
    detection = {"algorithm": "ElSe", "algorithm_returned": True, "center_x": 5.0}
    row = assemble_row(identity, detection)
    assert row["subject"] == "sub-031"
    assert row["algorithm"] == "ElSe"
    assert row["center_x"] == 5.0
    assert row["frame_idx"] == 5
    assert row["bbox_x1"] == 10
    # unspecified schema columns default to None
    assert row["confidence_runtime_ms"] is None


def test_run_crop_list_independent(monkeypatch, tmp_path):
    pytest.importorskip("cv2")

    def fake_detect(image, algorithm, **kwargs):
        return {"algorithm": algorithm, "algorithm_returned": True, "center_x": 1.0}

    monkeypatch.setattr(runner_mod, "detect_crop", fake_detect)
    (tmp_path / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    rows = [
        {"subject": "sub-031", "phase": "block1", "frame_idx": 0, "eye": "eye_left",
         "crop_path": "a.png", "sample_role": "smoke"},
        {"subject": "sub-031", "phase": "block1", "frame_idx": 1, "eye": "eye_right",
         "crop_path": "a.png", "sample_role": "smoke"},
    ]
    frame = run_crop_list(rows, ["ElSe", "ExCuSe"], crop_root=tmp_path)
    assert len(frame) == 2 * 2
    assert set(frame["algorithm"]) == {"ElSe", "ExCuSe"}
    assert set(RESULT_COLUMNS).issubset(frame.columns)


def test_ellipse_helpers():
    ellipse = Ellipse(cx=10.0, cy=20.0, axis_a=14.0, axis_b=8.0)
    assert math.isclose(ellipse.diameter_geom, math.sqrt(14.0 * 8.0))
    assert math.isclose(ellipse.area, math.pi * 14.0 * 8.0 / 4.0)
    assert ellipse.major_axis == 14.0
    assert ellipse.minor_axis == 8.0


# ---------- synthetic + overlay (skip without cv2) ----------


def test_make_synthetic_eye():
    cv2 = pytest.importorskip("cv2")
    image, truth = make_synthetic_eye(width=424, height=187)
    assert image.shape == (187, 424)
    assert image.dtype == np.uint8
    assert truth["center_x"] == 212.0
    assert truth["center_y"] == 93.5
    assert 0 <= truth["major_axis"] <= 424
    # pupil should be darker than surrounding iris
    pupil_region = image[90:97, 208:217].mean()
    iris_region = image[60:70, 40:60].mean()
    assert pupil_region < iris_region


def test_draw_detection_overlay():
    cv2 = pytest.importorskip("cv2")
    image = np.full((187, 424, 3), 120, dtype=np.uint8)
    row = {
        "center_x": 212.0, "center_y": 93.0,
        "major_axis": 14.0, "minor_axis": 8.0, "angle_deg": 12.0,
    }
    out = draw_detection(image, row)
    assert out.shape == (187, 424, 3)
    # invalid geometry -> unchanged
    bad = draw_detection(image, {"center_x": math.nan, "major_axis": 14.0})
    assert np.array_equal(bad, image)


def test_write_smoke_manifest_requires_cv2():
    pytest.importorskip("cv2")
    from attention_pipeline.nir_pupil_benchmark.synthetic import write_smoke_manifest

    rows = write_smoke_manifest(".")
    assert rows  # would raise earlier if cv2 missing
