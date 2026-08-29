from __future__ import annotations

import numpy as np

from attention_pipeline.nir_formal_analysis.event_response import (
    EventResponseConfig,
    event_response_features,
)


def test_event_response_reports_peak_latency_and_recovery() -> None:
    onset = 1000.0
    times = np.array([800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300], dtype=float)
    values = np.array([10.0, 10.1, 9.9, 10.0, 10.0, 11.0, 12.0, 11.0, 10.4, 10.1, 10.0], dtype=float)
    cfg = EventResponseConfig(
        min_baseline_valid_n=3,
        min_response_valid_n=5,
        min_late_recovery_valid_n=2,
        max_gap_sec=0.1,
        late_recovery_window_ms=150.0,
    )
    out = event_response_features(times, values, onset_ms=onset, config=cfg)
    assert out["event_response_status"] == "estimable"
    assert np.isclose(out["baseline_median"], 10.0)
    assert np.isclose(out["dilation_peak_amplitude"], 2.0)
    assert np.isclose(out["dilation_peak_latency_ms"], 100.0)
    assert out["dominant_peak_direction"] == "dilation"
    assert out["recovery_status"] == "recovered_within_response_window"
    assert out["recovery_time_after_onset_ms"] >= out["dominant_peak_latency_ms"]
    assert np.isfinite(out["late_recovery_abs_residual"])


def test_event_response_fails_closed_on_low_baseline_samples() -> None:
    onset = 1000.0
    times = np.array([950, 1000, 1050, 1100, 1150, 1200], dtype=float)
    values = np.array([10.0, 10.0, 11.0, 12.0, 11.0, 10.0], dtype=float)
    out = event_response_features(times, values, onset_ms=onset)
    assert out["event_response_status"] == "not_estimable"
    assert "low_baseline_valid_n" in out["event_response_reasons"]
    assert np.isnan(out["dominant_peak_latency_ms"])


def test_event_response_fails_closed_on_large_temporal_gap() -> None:
    onset = 1000.0
    times = np.array([800, 850, 900, 950, 1000, 1050, 1400, 1450, 1500, 1550, 1600, 1650], dtype=float)
    values = np.linspace(10.0, 11.0, len(times))
    cfg = EventResponseConfig(min_response_valid_n=5, max_gap_sec=0.25)
    out = event_response_features(times, values, onset_ms=onset, config=cfg)
    assert out["event_response_status"] == "not_estimable"
    assert "response_gap_exceeds_gate" in out["event_response_reasons"]


def test_event_response_is_capped_at_next_trial_onset() -> None:
    onset = 1000.0
    times = np.arange(800.0, 2200.0, 50.0)
    values = np.ones_like(times) * 10.0
    values[times >= 1600.0] = 99.0  # would be a future-trial contaminant
    cfg = EventResponseConfig(min_response_valid_n=5, max_gap_sec=0.1)
    out = event_response_features(
        times,
        values,
        onset_ms=onset,
        response_end_ms=1500.0,
        config=cfg,
    )
    assert out["event_response_status"] == "estimable"
    assert out["response_end_ms"] == 1500.0
    assert out["dilation_peak_amplitude"] < 50.0
