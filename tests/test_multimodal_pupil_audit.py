import numpy as np
import pandas as pd
import pytest

from scripts.multimodal_pupil_audit import (
    derive_rgb_features,
    read_repeat_registry_csv,
    select_sample_times,
    timeline_stats,
)


def _mesh_frame(rows=3):
    data = {
        "FaceRectWidth": [100.0] * rows,
        "FaceRectHeight": [80.0] * rows,
        "face_rank": [0] * rows,
        "unix_ms": [1000, 1065, 1130][:rows],
    }
    for axis in ("x", "y"):
        for index in range(478):
            data[f"mesh_{axis}_{index}"] = [0.0] * rows
    data["mesh_x_470"] = [3.0] * rows
    data["mesh_y_471"] = [4.0] * rows
    data["mesh_x_475"] = [6.0] * rows
    data["mesh_y_476"] = [8.0] * rows
    return pd.DataFrame(data)


def test_derive_rgb_features_uses_declared_mesh_indices():
    derived = derive_rgb_features(_mesh_frame())
    assert np.allclose(derived["rgb_face_bbox_scale_px"], np.sqrt(8000.0))
    assert np.allclose(derived["rgb_right_iris_diameter_px"], 5.0)
    assert np.allclose(derived["rgb_left_iris_diameter_px"], 10.0)
    assert np.allclose(derived["rgb_iris_diameter_px"], 7.5)


def test_select_sample_times_is_deterministic_and_unique():
    paired = pd.DataFrame(
        {
            "unix_ms_nir": [1000, 1000, 1030, 1060, 1090],
            "unix_ms_rgb": [1001, 1001, 1031, 1061, 1091],
            "pupil_valid": [True, True, True, True, False],
        }
    )
    result = select_sample_times(paired, sample_size=2)
    assert result.tolist() == [1000, 1060]


def test_timeline_stats_reports_unique_timestamps_and_gaps():
    result = timeline_stats(pd.Series([1000, 1030, 1065, 1300]))
    assert result["unique_timestamps"] == 4
    assert result["span_ms"] == 300
    assert result["gaps_gt_200ms"] == 1


def test_repeat_registry_maps_all_sessions_to_one_local_group(tmp_path):
    registry_path = tmp_path / "repeat_registry.csv"
    pd.DataFrame(
        [
            {
                "local_repeat_participant_id": "beijing_xlsx_repeat_001",
                "experiment_ids": "031|059|068",
                "session_count": 3,
                "global_repeat_participant_id": "",
            }
        ]
    ).to_csv(registry_path, index=False, encoding="utf-8-sig")

    registry = read_repeat_registry_csv(registry_path)

    assert registry["by_experiment_id"]["031"]["local_repeat_participant_id"] == "beijing_xlsx_repeat_001"
    assert registry["by_experiment_id"]["059"]["local_repeat_participant_id"] == registry["by_experiment_id"]["068"]["local_repeat_participant_id"]
    assert registry["contains_raw_identity_values"] is False


def test_repeat_registry_rejects_raw_identity_columns(tmp_path):
    registry_path = tmp_path / "repeat_registry_with_pii.csv"
    pd.DataFrame(
        [
            {
                "local_repeat_participant_id": "beijing_xlsx_repeat_001",
                "experiment_ids": "031|059|068",
                "session_count": 3,
                "phone": "not serialized",
            }
        ]
    ).to_csv(registry_path, index=False, encoding="utf-8-sig")

    with pytest.raises(ValueError, match="raw identity columns"):
        read_repeat_registry_csv(registry_path)


def test_repeat_registry_uses_exact_experiment_id_tokens(tmp_path):
    registry_path = tmp_path / "repeat_registry_exact_tokens.csv"
    pd.DataFrame(
        [
            {
                "local_repeat_participant_id": "group_031",
                "experiment_ids": "031",
                "session_count": 1,
            },
            {
                "local_repeat_participant_id": "group_1031",
                "experiment_ids": "1031",
                "session_count": 1,
            },
        ]
    ).to_csv(registry_path, index=False, encoding="utf-8-sig")

    registry = read_repeat_registry_csv(registry_path)

    assert registry["by_experiment_id"]["031"]["local_repeat_participant_id"] == "group_031"
    assert registry["by_experiment_id"]["1031"]["local_repeat_participant_id"] == "group_1031"
