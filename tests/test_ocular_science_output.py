from __future__ import annotations

from pathlib import Path

import pandas as pd

from attention_pipeline.nir_formal_analysis.ocular_science_output import (
    GEOMETRY,
    HANDOFF_COLUMNS,
    RSEG_HARD,
    build_ocular_science_output,
)


def _g1_candidates() -> pd.DataFrame:
    rows = []
    for participant_i in range(4):
        for probe_i in range(3):
            for signal, scale in ((GEOMETRY, 10.0), (RSEG_HARD, 1.0)):
                for track, buffer_id in (("nir_qc_only", "none"), ("rgb_plus_nir_qc", "pre200_post200")):
                    x = participant_i * 3 + probe_i + 1
                    rows.append(
                        {
                            "session_id": f"sub-{participant_i + 1:03d}",
                            "participant_group_id": f"p{participant_i + 1:02d}",
                            "block_num": 1,
                            "probe_index_in_block": probe_i + 1,
                            "signal": signal,
                            "cleaning_track": track,
                            "buffer_id": buffer_id,
                            "bin_width_sec": 2.0,
                            "level_mean": scale + x / 100.0,
                            "level_median": scale + x / 110.0,
                            "variability_sd": x / 50.0,
                            "variability_mad": x / 60.0,
                            "linear_slope_per_sec": x / 1000.0,
                            "quadratic_curvature_per_sec2": x / 10000.0,
                            "linear_status": "computable",
                            "quadratic_status": "computable",
                        }
                    )
    return pd.DataFrame(rows)


def _rgb_probes() -> pd.DataFrame:
    rows = []
    for participant_i in range(4):
        for probe_i in range(3):
            rows.append(
                {
                    "session_id": f"sub-{participant_i + 1:03d}",
                    "participant_group_id": f"p{participant_i + 1:02d}",
                    "block_id": "B1",
                    "probe_index_in_block": probe_i + 1,
                    "blink_event_rate_per_min": 4.0 + probe_i,
                }
            )
    return pd.DataFrame(rows)


def test_ocular_handoff_keeps_researcher_freeze_open(tmp_path: Path) -> None:
    g1 = tmp_path / "probe_measurement_candidates.csv"
    rgb = tmp_path / "rgb_probe_pre30s_strict_features.csv"
    _g1_candidates().to_csv(g1, index=False)
    _rgb_probes().to_csv(rgb, index=False)

    manifest = build_ocular_science_output(g1, tmp_path / "FormalScience", rgb_probe_features_path=rgb)
    out = tmp_path / "FormalScience/Ocular"
    handoff = pd.read_csv(out / "tables/ocular_feature_handoff.csv")
    wide = pd.read_csv(out / "tables/ocular_probe_features_wide.csv")

    assert handoff.columns.tolist() == HANDOFF_COLUMNS
    assert set(handoff["scientific_modality"]) == {"ocular"}
    assert {"nir", "rgb"}.issubset(set(handoff["source_namespace"]))
    assert not handoff["registry_ready"].astype(bool).any()
    assert handoff["researcher_freeze_required"].astype(bool).all()
    assert handoff["selection_policy"].str.contains("Q1/Q2 significance", regex=False).all()
    assert set(handoff.loc[handoff["scientific_feature_id"].eq("ocular.pupil_level"), "report_role"]) == {
        "primary_candidate",
        "sensitivity_alternative",
    }
    dynamics = handoff[handoff["scientific_feature_id"].isin({"ocular.pupil_linear_trend", "ocular.pupil_quadratic_curvature"})]
    assert set(dynamics["estimability_status"]) == {"pending_temporal_span_freeze"}
    assert "ocular__blink_event_rate_per_min__pre30s" in wide.columns
    assert manifest["technical_freeze"] == {"blink_buffer": "pre200_post200", "bin_width_sec": 2.0, "bin_n": 15}
    assert manifest["final_feature_registry_mutated"] is False
    assert manifest["supervised_model_run"] is False
    assert manifest["multimodal_model_run"] is False


def test_ocular_builder_excludes_soft_signal_from_handoff(tmp_path: Path) -> None:
    frame = _g1_candidates()
    soft = frame.iloc[[0]].copy()
    soft["signal"] = "seg_pupil_fraction_within_pupil_iris_soft"
    frame = pd.concat([frame, soft], ignore_index=True)
    path = tmp_path / "probe_measurement_candidates.csv"
    frame.to_csv(path, index=False)

    build_ocular_science_output(path, tmp_path / "FormalScience")
    handoff = pd.read_csv(tmp_path / "FormalScience/Ocular/tables/ocular_feature_handoff.csv")
    assert not handoff["candidate_representation_id"].str.contains("soft", case=False, regex=False).any()
