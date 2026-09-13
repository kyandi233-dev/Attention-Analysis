from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from attention_pipeline.rgb_formal.movement_science_output import (
    HANDOFF_COLUMNS,
    build_movement_science_output,
    build_ocular_movement_artifact_audit,
)


def _probe_rows(n: int = 24) -> pd.DataFrame:
    rows = []
    for i in range(n):
        participant = f"p{i % 4 + 1:02d}"
        session = f"sub-{i % 4 + 1:03d}"
        probe = i // 4 + 1
        rows.append(
            {
                "session_id": session,
                "participant_group_id": participant,
                "block_id": "b1" if probe <= 3 else "b2",
                "probe_index_in_block": probe if probe <= 3 else probe - 3,
                "body_motion_energy_median": 0.1 + 0.01 * i,
                "pose_lateral_right_per_sec_median": -0.2 + 0.02 * i,
                "pose_vertical_up_per_sec_median": 0.3 - 0.01 * i,
                "pose_radial_proximity_direction_score_median": -0.1 + 0.01 * i,
                "exposure_change_abs_median": 0.05 + 0.001 * i,
                "blink_event_rate_per_min": 5 + i % 3,
            }
        )
    return pd.DataFrame(rows)


def test_movement_builder_reclassifies_rgb_without_registry_mutation(tmp_path: Path) -> None:
    rgb_root = tmp_path / "rgb55"
    (rgb_root / "tables").mkdir(parents=True)
    (rgb_root / "models").mkdir(parents=True)
    probes = _probe_rows()
    probes.to_csv(rgb_root / "tables/rgb_probe_pre30s_strict_features.csv", index=False)

    # Existing participant-clustered 5.5 model output is curated, not refit here.
    pd.DataFrame(
        [
            {"feature": "body_motion_energy_median", "estimate": 0.1},
            {"feature": "blink_event_rate_per_min", "estimate": 0.2},
        ]
    ).to_csv(rgb_root / "models/rgb_q1_mnlogit.csv", index=False)

    manifest = build_movement_science_output(rgb_root, tmp_path / "FormalScience")
    out = tmp_path / "FormalScience/Movement"
    handoff = pd.read_csv(out / "tables/movement_feature_handoff.csv")

    assert handoff.columns.tolist() == HANDOFF_COLUMNS
    assert set(handoff["scientific_modality"]) == {"movement"}
    assert set(handoff["source_namespace"]) == {"rgb"}
    assert "body_motion_energy_median" in set(handoff["predictor_column"])
    assert "blink_event_rate_per_min" not in set(handoff["predictor_column"])
    assert not handoff["registry_ready"].astype(bool).any()
    assert handoff["researcher_freeze_required"].astype(bool).all()
    assert handoff["selection_policy"].str.contains("Q1/Q2 significance", regex=False).all()

    radial = handoff[handoff["predictor_column"].eq("pose_radial_proximity_direction_score_median")].iloc[0]
    assert "dimensionless" in str(radial["unit"])
    assert "nonphysical" in str(radial["redundancy_relation"])

    q1 = pd.read_csv(out / "tables/movement_q1_models.csv")
    assert q1["feature"].tolist() == ["body_motion_energy_median"]
    assert manifest["final_feature_registry_mutated"] is False
    assert manifest["supervised_model_run"] is False
    assert manifest["multimodal_model_run"] is False
    assert not list(tmp_path.rglob("feature_registry.yaml"))


def test_ocular_cross_artifact_audit_keeps_representation_selection_open() -> None:
    rgb = _probe_rows(24)
    g1_rows = []
    for _, row in rgb.iterrows():
        for signal, offset in (
            ("pupil_geom_mean_diameter", 2.0),
            ("seg_pupil_fraction_within_pupil_iris_hard", 0.3),
        ):
            g1_rows.append(
                {
                    "session_id": row["session_id"],
                    "participant_group_id": row["participant_group_id"],
                    "block_num": 1 if row["block_id"] == "b1" else 2,
                    "probe_index_in_block": row["probe_index_in_block"],
                    "signal": signal,
                    "cleaning_track": "nir_qc_only",
                    "buffer_id": "none",
                    "bin_width_sec": 2.0,
                    "window_sec": 30.0,
                    "median": offset + 0.5 * row["body_motion_energy_median"] + 0.01 * (_ % 3),
                }
            )
    audit = build_ocular_movement_artifact_audit(pd.DataFrame(g1_rows), rgb)

    assert not audit.empty
    assert set(audit["signal"]) == {
        "pupil_geom_mean_diameter",
        "seg_pupil_fraction_within_pupil_iris_hard",
    }
    assert set(audit["interpretation_scope"]) == {"measurement_artifact_sensitivity_only"}
    assert not audit["representation_selection_allowed"].astype(bool).any()
    assert not audit["q1_q2_used"].astype(bool).any()
    assert "body_motion_energy_median" in set(audit["artifact_predictor"])
