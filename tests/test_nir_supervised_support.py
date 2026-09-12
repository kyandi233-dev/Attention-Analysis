import numpy as np
import pandas as pd
import pytest

from attention_pipeline.nir_formal_analysis.supervised_features import (
    build_raw_binocular_timepoints,
    select_preprobe_window,
)
from attention_pipeline.nir_formal_analysis.supervised_support import (
    audit_supervised_window_support,
)


def _multi_context_frame():
    return pd.DataFrame(
        {
            "session_id": ["sub-001"] * 6 + ["sub-002"] * 2,
            "block": [1] * 4 + [2] * 2 + [1] * 2,
            "frame_idx": list(range(1, 9)),
            "unix_ms": [70000.0, 80000.0, 90000.0, 99999.0, 90000.0, 95000.0, 90000.0, 95000.0],
            "left_raw_pupil_diameter": [10.0, 11.0, 12.0, 13.0, 1e5, 1e5, 2e5, 2e5],
            "right_raw_pupil_diameter": [14.0, 15.0, 16.0, 17.0, 1e5, 1e5, 2e5, 2e5],
            "left_pupil_valid_primary": [True] * 8,
            "right_pupil_valid_primary": [True] * 8,
        }
    )


def test_support_audit_reports_requested_available_validity_gap_and_source_mode():
    frame = pd.DataFrame(
        {
            "unix_ms": [1000.0, 2000.0, 4000.0],
            "raw_binocular_pupil": [10.0, np.nan, 14.0],
            "raw_binocular_source_mode": ["binocular", "missing", "left_only"],
        }
    )
    result = audit_supervised_window_support(
        frame,
        requested_start_ms=0.0,
        requested_end_ms=5000.0,
        available_start_ms=1000.0,
        available_end_ms=5000.0,
    )

    assert result["requested_duration_sec"] == pytest.approx(5.0)
    assert result["available_duration_sec"] == pytest.approx(4.0)
    assert result["available_duration_fraction"] == pytest.approx(0.8)
    assert result["window_truncated_by_available_start"] is True
    assert result["n_nir_rows"] == 3
    assert result["n_pupil_valid"] == 2
    assert result["pupil_valid_fraction"] == pytest.approx(2 / 3)
    assert result["sampling_rate_hz_estimate"] == pytest.approx(2 / 3)
    assert result["max_temporal_gap_sec"] == pytest.approx(2.0)
    assert result["source_mode_summary"] == "mixed"
    assert result["automatic_coverage_drop_allowed"] is False
    assert result["window_support_status"] == "available"


def test_no_rows_and_no_valid_samples_have_distinct_support_states():
    no_rows = pd.DataFrame(columns=["unix_ms", "raw_binocular_pupil", "raw_binocular_source_mode"])
    result = audit_supervised_window_support(
        no_rows,
        requested_start_ms=0.0,
        requested_end_ms=10000.0,
    )
    assert result["window_support_status"] == "not_available_no_nir_rows"
    assert result["pupil_level_mean_status"] == "not_estimable_no_valid_samples"

    invalid = pd.DataFrame(
        {
            "unix_ms": [1000.0, 2000.0],
            "raw_binocular_pupil": [np.nan, np.nan],
            "raw_binocular_source_mode": ["missing", "missing"],
        }
    )
    result = audit_supervised_window_support(
        invalid,
        requested_start_ms=0.0,
        requested_end_ms=10000.0,
    )
    assert result["window_support_status"] == "available_no_valid_pupil_samples"
    assert result["n_nir_rows"] == 2
    assert result["n_pupil_valid"] == 0


def test_other_block_and_other_session_cannot_change_current_probe_features():
    before = build_raw_binocular_timepoints(_multi_context_frame())
    changed = _multi_context_frame()
    changed.loc[
        ~(
            changed["session_id"].eq("sub-001")
            & changed["block"].eq(1)
        ),
        ["left_raw_pupil_diameter", "right_raw_pupil_diameter"],
    ] = 9e9
    after = build_raw_binocular_timepoints(changed)

    kwargs = dict(
        session_id="sub-001",
        block_num=1,
        probe_onset_ms=100000.0,
        window_sec=30,
    )
    a = select_preprobe_window(before, **kwargs)
    b = select_preprobe_window(after, **kwargs)
    audit_a = audit_supervised_window_support(
        a,
        requested_start_ms=70000.0,
        requested_end_ms=100000.0,
    )
    audit_b = audit_supervised_window_support(
        b,
        requested_start_ms=70000.0,
        requested_end_ms=100000.0,
    )

    for key in (
        "pupil_level_mean",
        "pupil_level_median",
        "pupil_variability_sd",
        "pupil_variability_mad",
        "pupil_variability_iqr",
        "pupil_trend_robust_binned_slope_per_sec",
    ):
        assert audit_a[key] == pytest.approx(audit_b[key])
