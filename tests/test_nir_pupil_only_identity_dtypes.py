from __future__ import annotations

import pandas as pd

from attention_pipeline.nir_analysis_ready.pupil_only import (
    apply_session_eye_standardization,
    build_wide_timepoints,
    compute_session_eye_baselines,
)
from attention_pipeline.nir_pupil_only import adapt_session_rows


def _source_rows() -> pd.DataFrame:
    # frame 2 is intentionally left-only. A plain int repeat_group_size would
    # be upcast on the right side of the outer merge and historically caused
    # the matched frame 1 identity comparison to see "1" versus "1.0".
    return pd.DataFrame(
        {
            "eye_metrics_schema_version": [7, 7, 7],
            "phase": ["block1", "block1", "block1"],
            "phase_segment": [0, 0, 0],
            "frame_idx": [1, 2, 1],
            "eye": ["frame_left", "frame_left", "frame_right"],
            "unix_ms": [1000.0, 1033.0, 1000.0],
            "video_time_ms": [0.0, 33.0, 0.0],
            "phase_time_ms": [0.0, 33.0, 0.0],
            "source_eye_status": ["observed", "observed", "observed"],
            "ritnet_status": ["success", "success", "success"],
            "pupil_found": [True, True, True],
            "pupil_fit_valid": [True, True, True],
            "pupil_center_x": [10.0, 10.2, 11.0],
            "pupil_center_y": [8.0, 8.1, 8.5],
            "pupil_geom_mean_diameter": [20.0, 21.0, 20.5],
        }
    )


def test_asymmetric_eye_merge_preserves_repeat_identity_without_false_mismatch() -> None:
    adapted = adapt_session_rows(
        _source_rows(),
        {
            "session_id": "sub-001",
            "analysis_group_token": "participant-001",
            "source_schema_version": 7,
            "repeat_group_size": 1,
        },
    )
    assert str(adapted["repeat_group_size"].dtype) == "Int64"
    assert str(adapted["is_repeat_session"].dtype) == "boolean"

    baselines = compute_session_eye_baselines(adapted)
    standardized = apply_session_eye_standardization(adapted, baselines)
    wide = build_wide_timepoints(standardized)

    assert len(wide) == 2
    assert wide["repeat_group_size"].astype("Int64").tolist() == [1, 1]
    assert wide["is_repeat_session"].astype("boolean").tolist() == [False, False]
    assert wide.loc[wide["frame_idx"].eq(1), "binocular_source_mode"].item() == "binocular"
    assert wide.loc[wide["frame_idx"].eq(2), "binocular_source_mode"].item() == "left_only"
