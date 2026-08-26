from __future__ import annotations

import numpy as np
import pandas as pd

from attention_pipeline.nir_formal_analysis.tables import (
    TrackSpec,
    _source_mode_features,
    _window_features,
    build_track_indices,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": ["sub-031"] * 4,
            "block": [1, 1, 1, 1],
            "phase": ["block1"] * 4,
            "phase_segment": [0, 0, 0, 0],
            "frame_idx": [1, 2, 3, 4],
            "unix_ms": [1000.0, 1033.0, 1066.0, 1099.0],
            "video_time_ms": [0.0, 33.0, 66.0, 99.0],
            "phase_time_ms": [0.0, 33.0, 66.0, 99.0],
            "binocular_PIR": [0.1, 0.2, np.nan, 0.4],
            "binocular_source_mode": [
                "binocular",
                "left_only",
                "missing",
                "right_only",
            ],
            "left_centered_PIR": [0.1, 0.2, 0.3, 0.4],
            "left_valid_primary": [True, True, False, True],
        }
    )


def test_build_track_indices_respects_validity_and_finite_values():
    tracks = [
        TrackSpec(
            name="binocular_primary",
            value_column="binocular_PIR",
            valid_column=None,
            source_mode_column="binocular_source_mode",
            family="primary",
        ),
        TrackSpec(
            name="left_primary",
            value_column="left_centered_PIR",
            valid_column="left_valid_primary",
            source_mode_column=None,
            family="eye_preserved",
        ),
    ]
    indices = build_track_indices(_frame(), "sub-031", tracks)
    binocular = indices[(1, "binocular_primary")]
    left = indices[(1, "left_primary")]
    assert binocular.valid.tolist() == [True, True, False, True]
    assert left.valid.tolist() == [True, True, False, True]


def test_window_features_keeps_missing_and_source_mode_information():
    track = TrackSpec(
        name="binocular_primary",
        value_column="binocular_PIR",
        valid_column=None,
        source_mode_column="binocular_source_mode",
        family="primary",
    )
    index = build_track_indices(_frame(), "sub-031", [track])[(1, track.name)]
    features = _window_features(index, 1000.0, 1100.0)
    assert features["n_nir_rows"] == 4
    assert features["n_pir_valid"] == 3
    assert np.isclose(features["pir_valid_fraction"], 0.75)
    assert np.isclose(features["source_mode_binocular_fraction"], 0.25)
    assert np.isclose(features["source_mode_left_only_fraction"], 0.25)
    assert np.isclose(features["source_mode_right_only_fraction"], 0.25)
    assert np.isclose(features["source_mode_missing_fraction"], 0.25)


def test_source_mode_features_none_does_not_invent_binocular_quality():
    values = _source_mode_features(None, 4)
    assert all(value is None for value in values.values())
