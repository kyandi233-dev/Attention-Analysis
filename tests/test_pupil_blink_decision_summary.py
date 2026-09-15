import pandas as pd
import pytest

from attention_pipeline.nir_formal_analysis.pupil_blink_decision_summary import (
    summarize_bin_width_stability,
    summarize_binocular_sources,
    summarize_blink_recovery,
    summarize_probe_bin_support,
    validate_probe_candidate_bin_counts,
)


def _candidate_rows() -> pd.DataFrame:
    rows = []
    for signal, offset in (("geom", 0.0), ("hard", 1.0), ("soft", 2.0)):
        for probe in range(3):
            for width, n_bins in ((1.0, 30), (2.0, 15), (5.0, 6)):
                rows.append(
                    {
                        "session_id": "s1",
                        "probe_event_id": f"p{probe}",
                        "signal": signal,
                        "cleaning_track": "nir_qc_only",
                        "buffer_id": "none",
                        "bin_width_sec": width,
                        "window_sec": 30.0,
                        "n_valid_bins": n_bins,
                        "valid_fraction": 0.9,
                        "linear_status": "computable",
                        "quadratic_status": "computable",
                        "has_early_support": True,
                        "has_late_support": True,
                        "level_mean": offset + probe + 0.1,
                        "level_median": offset + probe,
                        "variability_sd": 1.0 + probe,
                        "variability_mad": 0.8 + probe,
                        "variability_iqr": 1.2 + probe,
                        "linear_slope_per_sec": (probe + 1.0) / width,
                        "quadratic_curvature_per_sec2": (probe + 0.5) / width,
                    }
                )
    return pd.DataFrame(rows)


def test_probe_support_is_signal_specific_not_three_signal_sum():
    summary = summarize_probe_bin_support(_candidate_rows())
    one_second = summary[summary["bin_width_sec"].eq(1.0)]
    assert len(one_second) == 3
    assert set(one_second["defined_bin_n"]) == {30}
    assert set(one_second["n_valid_bins_max"]) == {30.0}
    assert set(one_second["full_bin_support_fraction"]) == {1.0}


def test_probe_bin_guard_rejects_cross_signal_aggregation_artifact():
    bad = _candidate_rows().iloc[[0]].copy()
    bad["n_valid_bins"] = 90
    with pytest.raises(ValueError, match="aggregation likely mixed signals"):
        validate_probe_candidate_bin_counts(bad)


def test_bin_width_stability_pairs_only_within_signal_track_buffer():
    stability = summarize_bin_width_stability(_candidate_rows())
    # 3 signals x 3 width-pairs x 2 dynamic metrics.
    assert len(stability) == 18
    assert set(stability["n_pair"]) == {3}
    assert stability["spearman_rho"].eq(1.0).all()


def test_binocular_summary_keeps_measurement_states_separate():
    audit = pd.DataFrame(
        [
            {"session_id": "s1", "signal": "geom", "measurement_state": "raw_computable", "source_mode": "binocular", "n_timepoints": 80},
            {"session_id": "s1", "signal": "geom", "measurement_state": "raw_computable", "source_mode": "left_only", "n_timepoints": 20},
            {"session_id": "s1", "signal": "geom", "measurement_state": "nir_qc_valid", "source_mode": "binocular", "n_timepoints": 60},
            {"session_id": "s1", "signal": "geom", "measurement_state": "nir_qc_valid", "source_mode": "left_only", "n_timepoints": 40},
        ]
    )
    summary = summarize_binocular_sources(audit)
    totals = summary.groupby(["signal", "measurement_state"])["fraction"].sum()
    assert totals.eq(1.0).all()
    raw = summary[
        summary["measurement_state"].eq("raw_computable")
        & summary["source_mode"].eq("binocular")
    ].iloc[0]
    qc = summary[
        summary["measurement_state"].eq("nir_qc_valid")
        & summary["source_mode"].eq("binocular")
    ].iloc[0]
    assert raw["fraction"] == pytest.approx(0.8)
    assert qc["fraction"] == pytest.approx(0.6)


def test_blink_recovery_summary_keeps_start_and_end_anchors_separate():
    recovery = pd.DataFrame(
        [
            {"session_id": "s1", "signal": "geom", "anchor": "start", "relative_bin_center_ms": 25.0, "raw_computable_fraction": 0.2, "nir_qc_valid_fraction": 0.1, "raw_value_median": 3.0, "nir_qc_value_median": 3.1},
            {"session_id": "s1", "signal": "geom", "anchor": "end", "relative_bin_center_ms": 25.0, "raw_computable_fraction": 0.8, "nir_qc_valid_fraction": 0.7, "raw_value_median": 3.2, "nir_qc_value_median": 3.3},
        ]
    )
    summary = summarize_blink_recovery(recovery)
    assert set(summary["anchor"]) == {"start", "end"}
    assert len(summary) == 2
