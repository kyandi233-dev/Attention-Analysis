from pathlib import Path

import pandas as pd
import pytest

from attention_pipeline.nir_formal_analysis.ocular_science_freeze import build_frozen_ocular_science_output
from attention_pipeline.nir_formal_analysis.ocular_science_output import GEOMETRY, RSEG_HARD


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    rows = []
    for signal in (GEOMETRY, RSEG_HARD):
        for track, buffer_id in (("nir_qc_only", "none"), ("rgb_plus_nir_qc", "pre200_post200")):
            for probe in (1, 2):
                rows.append({
                    "session_id": "sub-001", "participant_group_id": "p01", "block_num": 1,
                    "probe_index_in_block": probe, "signal": signal, "cleaning_track": track,
                    "buffer_id": buffer_id, "bin_width_sec": 2.0, "level_mean": 0.4,
                    "level_median": 0.4, "variability_sd": 0.02, "variability_mad": 0.01,
                    "linear_slope_per_sec": 0.001, "quadratic_curvature_per_sec2": 0.0001,
                    "linear_status": "computable", "quadratic_status": "computable",
                    "temporal_span_sec": 18.0 if signal == RSEG_HARD and track == "rgb_plus_nir_qc" and probe == 1 else 28.0,
                    "has_early_support": True, "has_late_support": True,
                })
    g1 = tmp_path / "g1.csv"
    pd.DataFrame(rows).to_csv(g1, index=False)
    rgb = tmp_path / "rgb.csv"
    pd.DataFrame([
        {"session_id": "sub-001", "participant_group_id": "p01", "block_id": "B1", "probe_index_in_block": probe, "blink_event_rate_per_min": 4.0}
        for probe in (1, 2)
    ]).to_csv(rgb, index=False)
    temporal = tmp_path / "temporal.csv"
    pd.DataFrame([{
        "signal": RSEG_HARD, "cleaning_track": "rgb_plus_nir_qc", "buffer_id": "pre200_post200",
        "bin_width_sec": 2.0, "linear_span_ge_20s_fraction": 0.86, "quadratic_span_ge_20s_fraction": 0.86,
    }]).to_csv(temporal, index=False)
    return g1, rgb, temporal


def test_frozen_ocular_materializer_masks_short_dynamics_and_freezes_roles(tmp_path: Path) -> None:
    g1, rgb, temporal = _inputs(tmp_path)
    manifest = build_frozen_ocular_science_output(
        g1, tmp_path / "FormalScience", rgb_probe_features_path=rgb, temporal_support_summary_path=temporal
    )
    root = tmp_path / "FormalScience/Ocular/tables"
    wide = pd.read_csv(root / "ocular_probe_features_wide.csv")
    handoff = pd.read_csv(root / "ocular_feature_handoff.csv")
    row = wide[wide["probe_index_in_block"].eq(1)].iloc[0]
    assert pd.notna(row["ocular__rseg_hard__rgb_nir_qc__level_mean__2s"])
    assert pd.isna(row["ocular__rseg_hard__rgb_nir_qc__linear_slope_per_sec__2s"])
    assert pd.isna(row["ocular__rseg_hard__rgb_nir_qc__quadratic_curvature_per_sec2__2s"])
    assert handoff["registry_ready"].astype(bool).all()
    assert not handoff["researcher_freeze_required"].astype(bool).any()
    assert set(handoff.loc[handoff["predictor_column"].str.contains("__geometry__"), "report_role"]) == {"cross_representation_sensitivity"}
    selected = handoff.loc[handoff["report_role"].eq("first_round_selected"), "predictor_column"].tolist()
    for token in ("__level_mean__", "__variability_mad__", "__linear_slope_per_sec__", "__quadratic_curvature_per_sec2__"):
        assert any(token in value for value in selected)
    assert manifest["technical_freeze"]["minimum_temporal_span_sec"] == 20.0
    assert manifest["first_round_pupil_representation"] == RSEG_HARD
    assert manifest["first_round_variability_metric"] == "variability_mad"
    assert manifest["final_feature_registry_mutated"] is False
    assert manifest["supervised_model_run"] is False


def test_frozen_ocular_materializer_requires_freeze_evidence(tmp_path: Path) -> None:
    g1, _rgb, _temporal = _inputs(tmp_path)
    with pytest.raises(ValueError, match="requires RGB blink input and temporal freeze evidence"):
        build_frozen_ocular_science_output(g1, tmp_path / "FormalScience")
