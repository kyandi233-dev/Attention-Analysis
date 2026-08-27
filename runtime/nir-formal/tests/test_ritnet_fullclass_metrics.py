from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ritnet_fullclass_contract import (
    QC_STRIDE_FRAMES,
    subject_output_paths,
)
from ritnet_fullclass_metrics import summarize_fullclass, summarize_fullclass_from_source
from ritnet_fullclass_qc import build_qc_anchor_frames, qc_image_paths, render_qc_images


# These 320x160 helper tests are retained as regression coverage for historical
# metric/QC utility functions. They do not define the current production
# full-class geometry, which is covered by test_ritnet_native_metrics.py.
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


def test_historical_helper_counts_geometry_and_normalization():
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


def test_historical_helper_can_reuse_source_pupil_for_regression_only():
    labels = synthetic_labels()
    probs = np.full(labels.shape, 0.9, dtype=np.float32)
    reference = summarize_fullclass(labels, probs, analysis_size=(320, 160))
    source = source_pupil_from_reference(reference)
    replay = summarize_fullclass_from_source(labels, source, analysis_size=(320, 160))

    assert replay["pupil_fit_valid"] is True
    assert replay["iris_outer_fit_valid"] is True
    assert replay["normalization_valid"] is True
    assert np.isclose(
        replay["pupil_to_iris_diameter_ratio"],
        reference["pupil_to_iris_diameter_ratio"],
        rtol=0,
        atol=1e-6,
    )
    assert np.isclose(
        replay["pupil_to_iris_ellipse_area_ratio"],
        reference["pupil_to_iris_ellipse_area_ratio"],
        rtol=0,
        atol=1e-6,
    )
    assert np.isclose(replay["pupil_confidence"], 0.9, atol=1e-6)


def test_historical_helper_empty_pupil_is_not_normalizable():
    labels = synthetic_labels()
    labels[labels == 3] = 2
    probs = np.zeros(labels.shape, dtype=np.float32)
    result = summarize_fullclass(labels, probs, analysis_size=(320, 160))

    assert result["pupil_fit_valid"] is False
    assert result["iris_outer_fit_valid"] is True
    assert result["normalization_valid"] is False
    assert result["pupil_to_iris_diameter_ratio"] is None


def test_historical_helper_missing_source_pupil_is_not_normalizable():
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


def test_subject_number_is_present_in_every_canonical_subject_artifact_filename(tmp_path):
    paths = subject_output_paths(tmp_path, "sub-31")
    expected_files = {"csv", "summary", "manifest", "completion", "qc_index", "labels_dir"}
    assert expected_files.issubset(paths)
    for key in {"csv", "summary", "manifest", "completion", "qc_index"}:
        assert paths[key].name.startswith("sub-031_")
        assert "v2-native640" in paths[key].name
    assert paths["qc_dir"].name.startswith("sub-031_")
    assert "v2-native640" in paths["qc_dir"].name
    assert paths["labels_dir"].name.startswith("sub-031_")
    assert "v2-native640" in paths["labels_dir"].name


def test_qc_anchor_sampling_keeps_phase_boundaries_and_sparse_stride():
    rows = []
    for frame in range(100, 10101, 100):
        rows.append(
            {
                "phase": "block1",
                "phase_segment": "1",
                "frame_idx": str(frame),
            }
        )
    anchors = build_qc_anchor_frames(rows, stride_frames=QC_STRIDE_FRAMES)
    assert 100 in anchors
    assert 5100 in anchors or 5000 in anchors
    assert 10100 in anchors
    assert len(anchors) < 10


def test_qc_render_and_subject_numbered_names(tmp_path):
    labels = synthetic_labels()
    roi = np.full((80, 160), 120, dtype=np.uint8)
    labels_color, overlay = render_qc_images(roi, labels)
    assert labels_color.shape == (160, 320, 3)
    assert overlay.shape == (160, 320, 3)
    assert np.any(labels_color[labels == 1] != 0)
    assert np.any(labels_color[labels == 2] != 0)
    assert np.any(labels_color[labels == 3] != 0)

    row = {
        "phase": "block1",
        "phase_segment": "1",
        "frame_idx": "1234",
        "eye": "frame_left",
    }
    labels_path, overlay_path = qc_image_paths(tmp_path, "sub-31", row)
    assert labels_path.name.startswith("sub-031_")
    assert overlay_path.name.startswith("sub-031_")
    assert labels_path.name.endswith("_labels.png")
    assert overlay_path.name.endswith("_overlay.png")
