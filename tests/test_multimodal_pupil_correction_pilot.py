import numpy as np
import pandas as pd

from scripts.multimodal_pupil_correction_pilot import (
    METHOD_SPECS,
    _baseline_residualize,
    add_nir_bbox_features,
)


def test_add_nir_bbox_features_keeps_eye_specific_geometry():
    paired = pd.DataFrame(
        {
            "eye": ["left", "right"],
            "bbox_x1": [10, 20],
            "bbox_y1": [5, 8],
            "bbox_x2": [30, 50],
            "bbox_y2": [25, 48],
            "rgb_eye_outer_corner_distance_px": [40.0, 50.0],
            "rgb_eye_inner_canthus_distance_px": [20.0, 25.0],
        }
    )

    result = add_nir_bbox_features(paired)

    assert result["nir_bbox_width_px"].tolist() == [20, 30]
    assert result["nir_bbox_height_px"].tolist() == [20, 40]
    assert np.allclose(result["nir_bbox_geom_scale_px"], [20.0, np.sqrt(1200.0)])


def test_baseline_residualization_does_not_fit_task_rows():
    x = np.log(np.arange(1.0, 13.0))
    frame = pd.DataFrame(
        {
            "eye": ["left"] * 12,
            "nir_phase": ["baseline"] * 10 + ["task"] * 2,
            "log_pupil_diameter": list(2.0 + 0.5 * x[:10]) + [4.0, 4.2],
            "log_nir_bbox_geom": x,
        }
    )

    corrected, details = _baseline_residualize(frame, ("log_nir_bbox_geom",))

    assert details["by_eye"]["left"]["fit_rows"] == 10
    assert np.allclose(corrected.iloc[:10], corrected.iloc[0])
    assert corrected.iloc[10] != corrected.iloc[11]


def test_method_spec_keeps_rgb_iris_out_of_correction_predictors():
    assert "rgb_iris_diameter_px" not in METHOD_SPECS["M1"]["predictors"]
    assert "rgb_iris_diameter_px" not in METHOD_SPECS["M2a"]["predictors"]
    assert "rgb_iris_diameter_px" not in METHOD_SPECS["M2b"]["predictors"]
