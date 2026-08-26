from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attention_pipeline.nir_analysis_ready.materialize import (
    CENTER_IN_COLUMN,
    IRIS_DIAMETER_COLUMN,
    IRIS_FIT_COLUMN,
    PIR_COLUMN,
    PIR_VALID_COLUMN,
    PUPIL_DIAMETER_COLUMN,
    PUPIL_FIT_COLUMN,
    apply_subject_eye_standardization,
    build_wide_timepoints,
    compute_subject_eye_baselines,
    derive_frame_validity,
)


def _row(
    *,
    eye: str,
    frame_idx: int,
    pir: float,
    production_valid: bool,
    center_in: bool = True,
    iris_diameter: float = 1.0,
) -> dict[str, object]:
    return {
        "subject": "sub-031",
        "phase": "block1" if frame_idx == 1 else "block2",
        "phase_segment": 0,
        "frame_idx": frame_idx,
        "video_time_ms": float(frame_idx * 1000),
        "unix_ms": float(1_800_000_000_000 + frame_idx * 1000),
        "phase_time_ms": float(frame_idx * 1000),
        "eye": eye,
        "source_row": frame_idx + (0 if eye == "left" else 100),
        PIR_COLUMN: pir,
        PIR_VALID_COLUMN: production_valid,
        PUPIL_FIT_COLUMN: True,
        IRIS_FIT_COLUMN: True,
        CENTER_IN_COLUMN: center_in,
        PUPIL_DIAMETER_COLUMN: 0.5,
        IRIS_DIAMETER_COLUMN: iris_diameter,
    }


def test_primary_recovers_edge_rejected_production_frame() -> None:
    frame = pd.DataFrame(
        [
            _row(
                eye="left",
                frame_idx=1,
                pir=0.30,
                production_valid=False,
            )
        ]
    )
    result = derive_frame_validity(frame)
    assert bool(result.loc[0, "pir_valid_primary"])
    assert not bool(result.loc[0, "pir_valid_strict"])


def test_primary_still_rejects_core_geometry_failures() -> None:
    frame = pd.DataFrame(
        [
            _row(
                eye="left",
                frame_idx=1,
                pir=0.30,
                production_valid=False,
                center_in=False,
            ),
            _row(
                eye="right",
                frame_idx=1,
                pir=0.80,
                production_valid=False,
                iris_diameter=0.4,
            ),
        ]
    )
    result = derive_frame_validity(frame)
    assert result["pir_valid_primary"].tolist() == [False, False]


def test_strict_must_be_subset_of_primary() -> None:
    frame = pd.DataFrame(
        [
            _row(
                eye="left",
                frame_idx=1,
                pir=0.30,
                production_valid=True,
                center_in=False,
            )
        ]
    )
    with pytest.raises(ValueError, match="subset"):
        derive_frame_validity(frame)


def test_subject_eye_centering_spans_blocks_and_binocular_fallback() -> None:
    raw = pd.DataFrame(
        [
            _row(eye="left", frame_idx=1, pir=0.20, production_valid=True),
            _row(eye="right", frame_idx=1, pir=0.30, production_valid=True),
            _row(eye="left", frame_idx=2, pir=0.40, production_valid=False),
            _row(
                eye="right",
                frame_idx=2,
                pir=0.35,
                production_valid=False,
                center_in=False,
            ),
        ]
    )
    valid = derive_frame_validity(raw)
    baselines = compute_subject_eye_baselines(valid)

    left = baselines[baselines["eye"] == "left"].iloc[0]
    right = baselines[baselines["eye"] == "right"].iloc[0]
    assert left["primary_median_PIR"] == pytest.approx(0.30)
    assert right["primary_median_PIR"] == pytest.approx(0.30)
    assert left["block1_n_primary_valid"] == 1
    assert left["block2_n_primary_valid"] == 1

    standardized = apply_subject_eye_standardization(valid, baselines)
    wide = build_wide_timepoints(standardized, time_tolerance_ms=1.0)

    first = wide[wide["frame_idx"] == 1].iloc[0]
    second = wide[wide["frame_idx"] == 2].iloc[0]
    assert first["binocular_source_mode"] == "binocular"
    assert first["binocular_PIR"] == pytest.approx(-0.05)
    assert second["binocular_source_mode"] == "left_only"
    assert second["binocular_PIR"] == pytest.approx(0.10)
    assert np.isnan(second["right_centered_PIR"])
    assert second["left_raw_PIR"] == pytest.approx(0.40)
    assert second["right_raw_PIR"] == pytest.approx(0.35)


def test_left_right_time_mismatch_is_contract_error() -> None:
    raw = pd.DataFrame(
        [
            _row(eye="left", frame_idx=1, pir=0.20, production_valid=True),
            _row(eye="right", frame_idx=1, pir=0.30, production_valid=True),
        ]
    )
    raw.loc[raw["eye"] == "right", "unix_ms"] += 5.0
    valid = derive_frame_validity(raw)
    baselines = compute_subject_eye_baselines(valid)
    standardized = apply_subject_eye_standardization(valid, baselines)
    with pytest.raises(ValueError, match="unix_ms mismatch"):
        build_wide_timepoints(standardized, time_tolerance_ms=1.0)
