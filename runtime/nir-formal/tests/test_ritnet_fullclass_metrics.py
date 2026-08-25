from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ritnet_fullclass_contract import subject_output_paths
from ritnet_fullclass_metrics import summarize_fullclass, summarize_fullclass_from_source


def synthetic_labels() -> np.ndarray:
    labels = np.zeros((160, 320), dtype=np.uint8)
    cv2.ellipse(labels, (160, 80), (120, 48), 0, 0, 360, 1, -1)
    cv2.ellipse(labels, (160, 80), (42, 38), 0, 0, 360, 2, -1)
    cv2.ellipse(labels, (160, 80), (18, 16), 0, 0, 360, 3, -1)
    return labels


def source_pupil_from_reference(reference: dict) -> dict[str, str]:
    return {
        "ritnet_found": "True",
        "pupil_center_x": str(reference["pupil_center_x"]),
        "pupil_center_y": str(reference["pupil_center_y"]),
        "pupil_axis_a": str(reference["pupil_axis_a"]),
        "pupil_axis_b": str(reference["pupil_axis_b"]),
        "pupil_angle_deg": str(reference["pupil_angle_deg"]),
        "pupil_mask_area": str(reference["pupil_contour_area"]),
        "pupil_equiv_diameter": str(reference["pupil_equiv_diameter"]),
        "pupil_confidence": str(reference["pupil_confidence"]),
    }


def test_fullclass_counts_geometry_and_normalization():
    labels = synthetic_labels()
    probs = np.full(labels.shape, 0.9, dtype=np.float32)
    result = summarize_fullclass(labels, probs, analysis_size=(320, 160))

    assert (
        result["background_pixels"]
        + result["sclera_pixels"]
        + result["iris_pixels"]
        + result["pupil_pixels"]
        == 320 * 160
    )
    assert result["pupil_fit_valid"] is True
    assert result["iris_outer_fit_valid"] is True
    assert result["normalization_valid"] is True
    assert 0 < result["pupil_to_iris_diameter_ratio"] < 1
    assert 0 < result["pupil_to_iris_ellipse_area_ratio"] < 1
    assert result["pupil_center_offset_norm"] < 0.05
    assert result["ocular_component_count"] == 1
    assert result["ocular_largest_component_fraction"] == 1.0
    assert result["pupil_confidence"] > 0.89


def test_fast_path_reuses_source_pupil_without_changing_normalized_geometry():
    labels = synthetic_labels()
    probs = np.full(labels.shape, 0.9, dtype=np.float32)
    reference = summarize_fullclass(labels, probs, analysis_size=(320, 160))
    source = source_pupil_from_reference(reference)
    fast = summarize_fullclass_from_source(labels, source, analysis_size=(320, 160))

    assert fast["pupil_fit_valid"] is True
    assert fast["iris_outer_fit_valid"] is True
    assert fast["normalization_valid"] is True
    assert np.isclose(
        fast["pupil_to_iris_diameter_ratio"],
        reference["pupil_to_iris_diameter_ratio"],
        rtol=0,
        atol=1e-6,
    )
    assert np.isclose(
        fast["pupil_to_iris_ellipse_area_ratio"],
        reference["pupil_to_iris_ellipse_area_ratio"],
        rtol=0,
        atol=1e-6,
    )
    assert np.isclose(fast["pupil_confidence"], 0.9, atol=1e-6)


def test_fullclass_empty_pupil_is_not_normalizable():
    labels = synthetic_labels()
    labels[labels == 3] = 2
    probs = np.zeros(labels.shape, dtype=np.float32)
    result = summarize_fullclass(labels, probs, analysis_size=(320, 160))

    assert result["pupil_fit_valid"] is False
    assert result["iris_outer_fit_valid"] is True
    assert result["normalization_valid"] is False
    assert result["pupil_to_iris_diameter_ratio"] is None


def test_fast_path_missing_source_pupil_is_not_normalizable():
    labels = synthetic_labels()
    source = {
        "ritnet_found": "False",
        "pupil_center_x": "",
        "pupil_center_y": "",
        "pupil_axis_a": "",
        "pupil_axis_b": "",
        "pupil_angle_deg": "",
        "pupil_mask_area": "",
        "pupil_equiv_diameter": "",
        "pupil_confidence": "0",
    }
    result = summarize_fullclass_from_source(labels, source, analysis_size=(320, 160))
    assert result["pupil_fit_valid"] is False
    assert result["normalization_valid"] is False
    assert result["pupil_to_iris_diameter_ratio"] is None


def test_subject_number_is_present_in_every_subject_artifact_filename(tmp_path):
    paths = subject_output_paths(tmp_path, "sub-31")
    assert set(paths) == {"csv", "summary", "manifest", "completion"}
    for path in paths.values():
        assert path.name.startswith("sub-031_")
