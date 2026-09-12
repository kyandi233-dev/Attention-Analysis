import numpy as np
import pandas as pd
import pytest

from attention_pipeline.nir_formal_analysis.pupil_blink_binocular import (
    audit_binocular_source_modes,
    build_binocular_measurement_timepoints,
)


def test_vectorized_binocular_fusion_keeps_geometry_and_rseg_independent():
    frame = pd.DataFrame({
        "session_id": ["s1"] * 4,
        "phase": ["block1"] * 4,
        "frame_idx": [1, 1, 2, 2],
        "eye": ["left", "right", "left", "right"],
        "unix_ms": [1000, 1000, 2000, 2000],
        "source_observed": [True] * 4,
        "ritnet_missing": [False] * 4,
        "interpolation_only": [False] * 4,
        "temporal_flagged": [False] * 4,
        "pupil_geom_mean_diameter": [10, 14, np.nan, 20],
        "hard_pupil_fraction": [0.2, 0.3, 0.4, 0.5],
        "hard_iris_fraction": [0.3, 0.2, 0.1, 0.5],
    })
    out = build_binocular_measurement_timepoints(frame)
    assert out.loc[0, "pupil_geom_mean_diameter"] == pytest.approx(12)
    assert out.loc[1, "pupil_geom_mean_diameter"] == pytest.approx(20)
    assert out.loc[1, "seg_pupil_fraction_within_pupil_iris_hard"] == pytest.approx(0.65)
    assert out.loc[1, "seg_pupil_fraction_within_pupil_iris_hard__source_mode"] == "binocular"


def test_source_mode_audit_reports_binocular_and_monocular_composition():
    frame = pd.DataFrame({
        "session_id": ["s1"] * 4,
        "phase": ["block1"] * 4,
        "frame_idx": [1, 1, 2, 2],
        "eye": ["left", "right", "left", "right"],
        "unix_ms": [1000, 1000, 2000, 2000],
        "source_observed": [True] * 4,
        "ritnet_missing": [False] * 4,
        "interpolation_only": [False] * 4,
        "temporal_flagged": [False] * 4,
        "pupil_geom_mean_diameter": [10, 14, np.nan, 20],
        "hard_pupil_fraction": [0.2, 0.3, 0.4, 0.5],
        "hard_iris_fraction": [0.3, 0.2, 0.1, 0.5],
    })
    timepoints = build_binocular_measurement_timepoints(frame)
    audit = audit_binocular_source_modes(timepoints)
    geometry_qc = audit[
        audit["signal"].eq("pupil_geom_mean_diameter")
        & audit["measurement_state"].eq("nir_qc_valid")
    ].set_index("source_mode")
    assert geometry_qc.loc["binocular", "n_timepoints"] == 1
    assert geometry_qc.loc["right_only", "n_timepoints"] == 1
    assert geometry_qc.loc["binocular", "fraction"] == pytest.approx(0.5)
    assert geometry_qc.loc["right_only", "fraction"] == pytest.approx(0.5)
