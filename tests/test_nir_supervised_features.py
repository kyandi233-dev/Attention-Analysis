import numpy as np
import pandas as pd
import pytest

from attention_pipeline.nir_formal_analysis.supervised_features import (
    BASE_SIGNAL,
    TREND_ALGORITHM,
    build_raw_binocular_timepoints,
    select_preprobe_window,
    summarize_supervised_window,
)


def _frame():
    return pd.DataFrame(
        {
            "session_id": ["sub-001"] * 5,
            "block": [1] * 5,
            "frame_idx": [1, 2, 3, 4, 5],
            "unix_ms": [96000.0, 98000.0, 99000.0, 100000.0, 101000.0],
            "left_raw_pupil_diameter": [10.0, 20.0, np.nan, 99.0, 999.0],
            "right_raw_pupil_diameter": [14.0, np.nan, 30.0, 101.0, 1001.0],
            "left_pupil_valid_primary": [True, True, False, True, True],
            "right_pupil_valid_primary": [True, False, True, True, True],
            # Deliberately impossible centered values: supervised construction
            # must ignore them.
            "binocular_pupil": [-500.0] * 5,
            "left_centered_pupil": [-700.0] * 5,
            "right_centered_pupil": [-900.0] * 5,
        }
    )


def test_raw_binocular_uses_canonical_eye_fusion_and_ignores_centered_values():
    result = build_raw_binocular_timepoints(_frame())

    assert result["base_signal"].eq(BASE_SIGNAL).all()
    assert result["raw_binocular_pupil"].tolist() == pytest.approx(
        [12.0, 20.0, 30.0, 100.0, 1000.0]
    )
    assert result["raw_binocular_source_mode"].tolist() == [
        "binocular",
        "left_only",
        "right_only",
        "binocular",
        "binocular",
    ]
    assert "binocular_pupil" not in result
    assert "left_centered_pupil" not in result


def test_preprobe_window_is_end_exclusive_and_block_local():
    points = build_raw_binocular_timepoints(_frame())
    window = select_preprobe_window(
        points,
        session_id="sub-001",
        block_num=1,
        probe_onset_ms=100000.0,
        window_sec=10,
    )

    assert window["unix_ms"].tolist() == [96000.0, 98000.0, 99000.0]
    assert window["raw_binocular_pupil"].tolist() == pytest.approx([12.0, 20.0, 30.0])
    assert window["window_end_exclusive"].all()


def test_post_probe_perturbation_cannot_change_selected_preprobe_values():
    before = build_raw_binocular_timepoints(_frame())
    changed = _frame()
    changed.loc[changed["unix_ms"] >= 100000.0, "left_raw_pupil_diameter"] = 1e9
    changed.loc[changed["unix_ms"] >= 100000.0, "right_raw_pupil_diameter"] = 1e9
    after = build_raw_binocular_timepoints(changed)

    kwargs = dict(
        session_id="sub-001",
        block_num=1,
        probe_onset_ms=100000.0,
        window_sec=10,
    )
    a = select_preprobe_window(before, **kwargs)
    b = select_preprobe_window(after, **kwargs)
    pd.testing.assert_series_equal(
        a["raw_binocular_pupil"],
        b["raw_binocular_pupil"],
        check_names=False,
    )


def test_only_frozen_window_lengths_are_accepted():
    points = build_raw_binocular_timepoints(_frame())
    with pytest.raises(ValueError, match="window_sec"):
        select_preprobe_window(
            points,
            session_id="sub-001",
            block_num=1,
            probe_onset_ms=100000.0,
            window_sec=60,
        )


def test_supervised_summary_reuses_frozen_compact_candidate_family():
    window = pd.DataFrame(
        {
            "unix_ms": [0.0, 1000.0, 2000.0, 3000.0],
            "raw_binocular_pupil": [10.0, 12.0, 14.0, 16.0],
        }
    )
    result = summarize_supervised_window(window)

    assert result["base_signal"] == BASE_SIGNAL
    assert result["trend_algorithm"] == TREND_ALGORITHM
    assert result["n_valid_pupil_samples"] == 4
    assert result["pupil_level_mean"] == pytest.approx(13.0)
    assert result["pupil_level_median"] == pytest.approx(13.0)
    assert result["pupil_variability_sd"] == pytest.approx(np.std([10, 12, 14, 16], ddof=1))
    assert result["pupil_variability_mad"] == pytest.approx(2.0)
    assert result["pupil_variability_iqr"] == pytest.approx(3.0)
    assert result["pupil_trend_robust_binned_slope_per_sec"] == pytest.approx(2.0)
    assert result["pupil_level_mean_status"] == "computable"
    assert result["pupil_variability_sd_status"] == "computable"
    assert result["pupil_trend_robust_binned_slope_per_sec_status"] == "computable"


def test_single_valid_sample_is_not_misreported_as_zero_variability():
    window = pd.DataFrame(
        {
            "unix_ms": [1000.0, 2000.0],
            "raw_binocular_pupil": [15.0, np.nan],
        }
    )
    result = summarize_supervised_window(window)

    assert result["pupil_level_mean"] == pytest.approx(15.0)
    assert result["pupil_level_mean_status"] == "computable"
    assert result["pupil_variability_sd"] is None
    assert result["pupil_variability_mad"] is None
    assert result["pupil_variability_iqr"] is None
    assert result["pupil_variability_sd_status"] == "not_estimable_low_valid_samples"
    assert result["pupil_trend_robust_binned_slope_per_sec"] is None
    assert result["pupil_trend_robust_binned_slope_per_sec_status"] == "not_estimable_low_valid_samples"


def test_trend_requires_actual_time_support_not_just_three_values():
    window = pd.DataFrame(
        {
            "unix_ms": [1000.0, 1000.0, 1000.0],
            "raw_binocular_pupil": [10.0, 11.0, 12.0],
        }
    )
    result = summarize_supervised_window(window)

    assert result["pupil_level_mean_status"] == "computable"
    assert result["pupil_variability_sd_status"] == "computable"
    assert result["pupil_trend_robust_binned_slope_per_sec"] is None
    assert result["pupil_trend_robust_binned_slope_per_sec_status"] == "not_estimable_time_support"
