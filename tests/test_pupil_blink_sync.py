import pandas as pd
import pytest

from attention_pipeline.nir_formal_analysis.pupil_blink_sync import (
    audit_rgb_nir_sync_with_frames,
    rgb_blink_source_availability,
)


def test_frame_axis_sync_adds_range_and_residual_diagnostics():
    nir = pd.DataFrame({"session_id": ["s1"] * 4, "unix_ms": [1000, 1100, 1200, 1300]})
    events = pd.DataFrame({"session_id": ["s1"], "start_unix_ms": [1090], "end_unix_ms": [1210]})
    rgb = pd.DataFrame({"unix_ms": [1010, 1110, 1210, 1310]})
    row = audit_rgb_nir_sync_with_frames(nir, events, rgb).iloc[0]
    assert row["sync_evidence_level"] == "rgb_frame_axis_plus_blink_events"
    assert row["rgb_minus_nir_start_ms"] == pytest.approx(10)
    assert row["rgb_minus_nir_end_ms"] == pytest.approx(10)
    assert row["frame_nearest_residual_abs_median_ms"] == pytest.approx(10)


def test_event_only_sync_is_explicitly_limited():
    nir = pd.DataFrame({"session_id": ["s1"] * 2, "unix_ms": [1000, 1100]})
    events = pd.DataFrame({"session_id": ["s1"], "start_unix_ms": [1000], "end_unix_ms": [1000]})
    row = audit_rgb_nir_sync_with_frames(nir, events, None).iloc[0]
    assert row["sync_evidence_level"] == "blink_event_boundaries_only"
    assert not row["rgb_frame_axis_available"]


def test_rgb_frame_axis_with_zero_events_is_observed_zero_not_missing():
    events = pd.DataFrame(columns=["session_id", "start_unix_ms", "end_unix_ms"])
    rgb = pd.DataFrame({"unix_ms": [1000, 1100]})
    available, basis = rgb_blink_source_availability(events, rgb)
    assert available
    assert basis == "rgb_frame_axis_zero_detected_events"


def test_event_boundaries_without_frame_axis_remain_usable_with_limited_evidence():
    events = pd.DataFrame({"session_id": ["s1"], "start_unix_ms": [1000], "end_unix_ms": [1050]})
    available, basis = rgb_blink_source_availability(events, None)
    assert available
    assert basis == "blink_event_boundaries_only"


def test_no_rgb_frame_axis_and_no_events_is_unavailable_not_zero_blinks():
    events = pd.DataFrame(columns=["session_id", "start_unix_ms", "end_unix_ms"])
    available, basis = rgb_blink_source_availability(events, None)
    assert not available
    assert basis == "rgb_blink_evidence_unavailable_or_ambiguous"
