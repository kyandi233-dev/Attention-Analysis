from __future__ import annotations

from pathlib import Path

import pandas as pd

from attention_pipeline.rgb_formal.movement_science_figures import build_movement_science_figures


def test_movement_figures_use_real_rgb55_estimate_schema_and_audit_missingness(tmp_path: Path) -> None:
    root = tmp_path / "Movement"
    tables = root / "tables"
    tables.mkdir(parents=True)

    probe_rows = []
    cycle_rows = []
    session_rows = []
    for participant_idx in range(1, 5):
        participant = f"p{participant_idx:02d}"
        session = f"sub-{participant_idx:03d}"
        session_rows.append(
            {
                "session_id": session,
                "body_motion_observable_ratio": 0.90 + 0.01 * participant_idx,
                "exposure_change_observable_ratio": 0.88 + 0.01 * participant_idx,
                "pose_shoulders_observable_ratio": 0.85 + 0.01 * participant_idx,
            }
        )
        for block_idx, block in enumerate(("B1", "B2"), start=1):
            for probe_idx in range(3):
                base = participant_idx * 0.1 + block_idx * 0.03 + probe_idx * 0.005
                probe_rows.append(
                    {
                        "participant_group_id": participant,
                        "session_id": session,
                        "block_id": block,
                        "body_motion_energy_median": base,
                        "exposure_change_abs_median": 0.05 + 0.1 * base,
                        "pose_lateral_right_per_sec_median": -0.02 + 0.01 * probe_idx,
                        "pose_vertical_up_per_sec_median": 0.03 - 0.005 * probe_idx,
                        "pose_radial_proximity_direction_score_median": 0.01 * (probe_idx - 1),
                    }
                )
            for cycle_bin in (1, 2, 3):
                cycle_rows.append(
                    {
                        "participant_group_id": participant,
                        "block_id": block,
                        "cycle_bin": cycle_bin,
                        "body_motion_energy_median": participant_idx * 0.1 + block_idx * 0.02 + cycle_bin * 0.01,
                        "exposure_change_abs_median": 0.05 + cycle_bin * 0.003,
                    }
                )

    pd.DataFrame(probe_rows).to_csv(tables / "movement_probe_descriptive_source.csv", index=False)
    pd.DataFrame(cycle_rows).to_csv(tables / "rgb_block_cycle_source.csv", index=False)
    pd.DataFrame(session_rows).to_csv(tables / "rgb_session_coverage_source.csv", index=False)

    q1_rows = []
    for category, estimate in ((2, 0.12), (3, 0.05), (4, -0.08)):
        q1_rows.append(
            {
                "predictor": "body_motion_energy_median",
                "outcome": "q1_nominal_4class",
                "contrast_category": category,
                "reference_category": 1,
                "estimate_per_predictor_sd": estimate,
                "ci_low": estimate - 0.04,
                "ci_high": estimate + 0.04,
            }
        )
    pd.DataFrame(q1_rows).to_csv(tables / "movement_q1_models.csv", index=False)
    pd.DataFrame(
        [
            {
                "predictor": "body_motion_energy_median",
                "outcome": "q2_ordinal_4level",
                "estimate_per_predictor_sd": 0.11,
                "ci_low": 0.03,
                "ci_high": 0.19,
            }
        ]
    ).to_csv(tables / "movement_q2_models.csv", index=False)
    pd.DataFrame(
        [
            {
                "metric": "body_motion_energy_median",
                "term": "block2",
                "estimate": 0.02,
                "ci_low": -0.01,
                "ci_high": 0.05,
            },
            {
                "metric": "body_motion_energy_median",
                "term": "cycle_bin",
                "estimate": 0.03,
                "ci_low": 0.01,
                "ci_high": 0.05,
            },
        ]
    ).to_csv(tables / "movement_task_progression.csv", index=False)
    pd.DataFrame(
        [
            {
                "outcome": "body_motion_energy_median",
                "predictor": "go_correct_rt_cv",
                "estimate_per_predictor_sd": -0.07,
                "ci_low": -0.11,
                "ci_high": -0.03,
            }
        ]
    ).to_csv(tables / "movement_behavior_links.csv", index=False)

    generated = build_movement_science_figures(root)

    assert "figures/main/movement_q1_relationships.png" in generated
    assert "figures/main/movement_q2_relationships.png" in generated
    assert "figures/main/movement_block_pair_task_progression.png" in generated
    assert (root / "figures/main/movement_q1_relationships.svg").is_file()
    assert (root / "figures/main/movement_q2_relationships.svg").is_file()
    assert (root / "figures/main/movement_block_pair_task_progression.svg").is_file()

    manifest = pd.read_csv(root / "manifests/figure_manifest.csv")
    audit = pd.read_csv(root / "manifests/figure_audit.csv")
    assert {"movement_q1_relationships", "movement_q2_relationships"}.issubset(set(manifest["figure_id"]))
    assert set(manifest["status"]) == {"candidate"}
    assert not audit["internal_title_present"].astype(bool).any()

    # Pose-sensitivity model rows were intentionally not supplied: absence must
    # be auditable rather than silently disappearing.
    pose_audit = audit[audit["figure_id"].str.endswith("_pose_sensitivity")]
    assert not pose_audit.empty
    assert set(pose_audit["generation_status"]) == {"not_estimable"}
