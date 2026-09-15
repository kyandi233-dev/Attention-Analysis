from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from attention_pipeline.nir_formal_analysis.ocular_g1_freeze_support import (
    build_ocular_g1_freeze_support,
    summarize_cross_signal_representation,
    summarize_source_mode_limitation,
    summarize_sync_semantics,
    summarize_temporal_support,
)
from attention_pipeline.nir_formal_analysis.pupil_blink_measurement import (
    GEOMETRY_SIGNAL,
    RSEG_HARD_SIGNAL,
)


def _candidate_rows() -> pd.DataFrame:
    rows = []
    for probe_i in range(4):
        for signal, scale in ((GEOMETRY_SIGNAL, 10.0), (RSEG_HARD_SIGNAL, 1.0)):
            rows.append(
                {
                    "session_id": "sub-001",
                    "probe_event_id": f"p{probe_i}",
                    "signal": signal,
                    "cleaning_track": "rgb_plus_nir_qc",
                    "buffer_id": "pre200_post200",
                    "bin_width_sec": 2.0,
                    "n_valid_bins": 15,
                    "t_min_sec": -29.0,
                    "t_max_sec": -1.0,
                    "temporal_span_sec": 28.0 if probe_i < 3 else 18.0,
                    "has_early_support": True,
                    "has_late_support": True,
                    "linear_status": "computable",
                    "quadratic_status": "computable",
                    "level_mean": scale * (probe_i + 1),
                    "level_median": scale * (probe_i + 1.1),
                    "variability_sd": scale * (probe_i + 0.2),
                    "variability_mad": scale * (probe_i + 0.1),
                    "variability_iqr": scale * (probe_i + 0.15),
                    "linear_slope_per_sec": scale * (-1) ** probe_i * (probe_i + 1),
                    "quadratic_curvature_per_sec2": scale * (-1) ** probe_i * (probe_i + 0.5),
                }
            )
    return pd.DataFrame(rows)


def test_cross_signal_summary_matches_probes_without_same_scale_assumption() -> None:
    out = summarize_cross_signal_representation(_candidate_rows())
    level = out[out["metric"].eq("level_mean")].iloc[0]
    assert level["paired_available_n"] == 4
    assert level["spearman_rho"] == 1.0
    assert level["agreement_interpretation"] == "cross_representation_rank_agreement_not_same_scale_identity"
    slope = out[out["metric"].eq("linear_slope_per_sec")].iloc[0]
    assert slope["dynamic_sign_agreement_fraction"] == 1.0


def test_temporal_support_grid_is_descriptive_and_requires_both_halves() -> None:
    frame = _candidate_rows()
    frame.loc[frame["probe_event_id"].eq("p3"), "has_late_support"] = False
    out = summarize_temporal_support(frame)
    row = out[out["signal"].eq(GEOMETRY_SIGNAL)].iloc[0]
    assert row["both_halves_support_fraction"] == 0.75
    assert row["support_span_ge_20s_fraction"] == 0.75
    assert row["support_span_ge_25s_fraction"] == 0.75


def test_sync_semantics_does_not_convert_gap_residual_into_sync_failure() -> None:
    frame = pd.DataFrame(
        [
            {
                "session_id": "sub-083",
                "sync_status": "audited",
                "sync_evidence_level": "rgb_frame_axis_plus_blink_events",
                "rgb_blink_source_available": True,
                "rgb_minus_nir_start_ms": 47.0,
                "rgb_minus_nir_end_ms": 13.0,
                "nir_frame_large_gap_n": 2,
                "rgb_frame_large_gap_n": 0,
                "nearest_residual_abs_p95_ms": 1200.0,
                "frame_nearest_residual_abs_p95_ms": 182000.0,
                "frame_nearest_residual_abs_max_ms": 190000.0,
            }
        ]
    )
    out = summarize_sync_semantics(frame).iloc[0]
    assert out["clock_boundary_evidence_status"] == "available_descriptive_only"
    assert out["nir_coverage_gap_status"] == "large_gap_present"
    assert out["legacy_sync_status"] == "audited"
    assert "no_auto_pass_fail" in out["interpretation"]


def test_source_mode_summary_marks_numeric_bias_as_not_estimable() -> None:
    frame = pd.DataFrame(
        [
            {"session_id": "s1", "signal": GEOMETRY_SIGNAL, "measurement_state": "nir_qc_valid", "source_mode": "binocular", "n_timepoints": 4},
            {"session_id": "s1", "signal": GEOMETRY_SIGNAL, "measurement_state": "nir_qc_valid", "source_mode": "left_only", "n_timepoints": 3},
            {"session_id": "s1", "signal": GEOMETRY_SIGNAL, "measurement_state": "nir_qc_valid", "source_mode": "right_only", "n_timepoints": 1},
            {"session_id": "s1", "signal": GEOMETRY_SIGNAL, "measurement_state": "nir_qc_valid", "source_mode": "missing", "n_timepoints": 2},
        ]
    )
    out = summarize_source_mode_limitation(frame).iloc[0]
    assert out["binocular_fraction"] == 0.4
    assert out["single_eye_fraction"] == 0.4
    assert out["missing_fraction"] == 0.2
    assert not bool(out["numeric_source_mode_bias_estimable_from_existing_g1_tables"])


def test_builder_reads_only_existing_audit_csvs(tmp_path: Path) -> None:
    _candidate_rows().to_csv(tmp_path / "probe_measurement_candidates.csv", index=False)
    pd.DataFrame(
        [{"session_id": "s1", "sync_status": "no_blink_events", "rgb_blink_source_available": True}]
    ).to_csv(tmp_path / "rgb_nir_sync_audit.csv", index=False)
    pd.DataFrame(
        [{"session_id": "s1", "signal": GEOMETRY_SIGNAL, "measurement_state": "nir_qc_valid", "source_mode": "binocular", "n_timepoints": 10}]
    ).to_csv(tmp_path / "binocular_source_mode_audit.csv", index=False)
    tables = build_ocular_g1_freeze_support(tmp_path)
    assert set(tables) == {
        "g1_cross_signal_representation_summary.csv",
        "g1_temporal_support_freeze_grid.csv",
        "g1_sync_semantics_split.csv",
        "g1_source_mode_limit_summary.csv",
    }
