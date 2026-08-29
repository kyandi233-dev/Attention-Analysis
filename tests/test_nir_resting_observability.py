from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attention_pipeline.nir_formal_analysis.resting_observability import (
    RestingObservabilityConfig,
    baseline_interval_from_timeline,
    summarize_resting_rows,
)


def _adapted() -> pd.DataFrame:
    rows = []
    for frame_idx, ts in enumerate(np.arange(1000.0, 2000.0, 100.0)):
        for eye in ("left", "right"):
            primary = not (frame_idx in {3, 4})
            rows.append({
                "frame_idx": frame_idx,
                "unix_ms": ts,
                "eye": eye,
                "source_observed": True,
                "source_missing": False,
                "ritnet_missing": False,
                "roi_clipped": False,
                "geometry_invalid": not primary,
                "temporal_flagged": False,
                "interpolation_only": False,
                "pupil_valid_primary": primary,
                "pupil_valid_strict": primary,
                "pupil_geom_mean_diameter": 20.0 + frame_idx * 0.1 if primary else np.nan,
            })
    return pd.DataFrame(rows)


def test_baseline_interval_uses_explicit_timeline_events() -> None:
    timeline = pd.DataFrame({
        "event": ["all_modalities_started", "baseline_start", "baseline_stop", "cover"],
        "detail": ["", "", "duration=180.0s", ""],
        "unix_ms": [900.0, 1000.0, 181000.0, 181100.0],
    })
    interval = baseline_interval_from_timeline(timeline)
    assert interval["baseline_start_ms"] == 1000.0
    assert interval["baseline_stop_ms"] == 181000.0
    assert interval["baseline_duration_sec"] == 180.0


def test_baseline_interval_fails_closed_on_duplicate_or_missing_markers() -> None:
    timeline = pd.DataFrame({
        "event": ["baseline_start", "baseline_start", "baseline_stop"],
        "unix_ms": [1000.0, 1100.0, 181000.0],
    })
    with pytest.raises(ValueError, match="expected exactly one"):
        baseline_interval_from_timeline(timeline)


def test_resting_observability_is_audit_only_until_thresholds_are_frozen() -> None:
    out = summarize_resting_rows(
        _adapted(),
        start_ms=1000.0,
        stop_ms=2000.0,
        cfg=RestingObservabilityConfig(),
    )
    assert out["resting_observability_status"] == "audit_only_thresholds_not_frozen"
    assert out["resting_reference_status"] == "not_authorized_thresholds_not_frozen"
    assert out["resting_timepoint_n"] == 10
    assert np.isclose(out["primary_valid_any_fraction"], 0.8)
    assert "eyes-closed" in out["observability_semantics"]


def test_frozen_observability_gate_can_authorize_exploratory_reference_without_calling_missing_eyes_closed() -> None:
    cfg = RestingObservabilityConfig(
        max_contiguous_gap_sec=0.15,
        min_primary_valid_fraction=0.75,
        min_longest_primary_valid_sec=0.3,
    )
    out = summarize_resting_rows(_adapted(), start_ms=1000.0, stop_ms=2000.0, cfg=cfg)
    assert out["resting_observability_status"] == "observable_for_exploratory_reference"
    assert out["resting_reference_status"] == "exploratory_candidate_estimable"
    assert np.isfinite(out["resting_pupil_median_candidate"])
    assert "eyes-closed" in out["observability_semantics"]
