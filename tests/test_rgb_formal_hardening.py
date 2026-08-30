from __future__ import annotations

from pathlib import Path

import pandas as pd

from attention_pipeline.rgb_formal import blink_candidates as blink
from attention_pipeline.rgb_formal.motion_qc import derive_motion_qc
from attention_pipeline.rgb_formal.pose_direction import derive_pose_direction


def _eye_points(indices: tuple[int, ...], openness: float, row: dict[str, object]) -> None:
    p0, p1, p2, p3, p4, p5 = indices
    coords = {
        p0: (0.0, 0.0), p3: (1.0, 0.0),
        p1: (0.25, openness / 2), p5: (0.25, -openness / 2),
        p2: (0.75, openness / 2), p4: (0.75, -openness / 2),
    }
    for idx, (x, y) in coords.items():
        row[f"mesh_x_{idx}"] = x
        row[f"mesh_y_{idx}"] = y


def _face_row(t: int, openness: float) -> dict[str, object]:
    row: dict[str, object] = {
        "unix_ms": t,
        "phase": "baseline",
        "primary_face": True,
        "face_track_id": "A",
        "face_count": 1,
        "face_rank": 0,
        "temporal_gap": False,
        "capture_gap_before": False,
    }
    _eye_points(blink.RIGHT_EYE, openness, row)
    _eye_points(blink.LEFT_EYE, openness, row)
    return row


def _pose_frame(t: int, *, world_z: float = 0.0, bbox_scale: float = 1.0) -> list[dict[str, object]]:
    rows = []
    for name, dx in (("left_shoulder", -0.1), ("right_shoulder", 0.1)):
        rows.append({
            "video_frame_position": t,
            "unix_ms": t,
            "pose_valid": True,
            "pose_count": 1,
            "pose_index": 0,
            "landmark_name": name,
            "x": 0.5 + dx,
            "y": 0.5,
            "visibility": 0.9,
            "presence": 0.9,
            "world_z": world_z,
            "pose_bbox_xmin": 0.2 - 0.05 * bbox_scale,
            "pose_bbox_ymin": 0.2 - 0.05 * bbox_scale,
            "pose_bbox_xmax": 0.8 + 0.05 * bbox_scale,
            "pose_bbox_ymax": 0.8 + 0.05 * bbox_scale,
        })
    return rows


def test_component_row_uses_blink_status_as_authoritative_component_status() -> None:
    from attention_pipeline.rgb_formal.runner import _component_row

    row = _component_row(
        "sub-001",
        "blink_candidates",
        {
            "primary_face_status": "generated",
            "blink_status": "not_estimable",
            "blink_reason": "bilateral_eye_or_time_interval_not_observable",
        },
        pd.Series({
            "participant_group_id": "participant:A",
            "participant_identity_source": "questionnaire_repeat_registry",
        }),
    )
    assert row["status"] == "not_estimable"
    assert row["reason"] == "bilateral_eye_or_time_interval_not_observable"


def test_motion_observability_honors_producer_and_temporal_qc() -> None:
    motion = pd.DataFrame({
        "unix_ms": [0, 100, 200],
        "global_motion_energy_per_sec": [0.0, 5.0, 7.0],
        "gray_mean_delta": [0.0, 10.0, 20.0],
        "motion_valid": [True, False, True],
        "gap_before": [False, False, True],
        "irregular_dt": [False, False, False],
    })
    out, status = derive_motion_qc(motion)
    assert bool(out.loc[0, "body_motion_observable"])
    assert not bool(out.loc[1, "body_motion_observable"])
    assert pd.isna(out.loc[1, "body_motion_energy"])
    assert bool(out.loc[1, "exposure_change_observable"])
    assert not bool(out.loc[2, "motion_temporal_valid"])
    assert pd.isna(out.loc[2, "body_motion_energy"])
    assert pd.isna(out.loc[2, "exposure_change_abs"])
    assert status["producer_invalid_rows"] == 1
    assert status["temporal_invalid_rows"] == 1


def test_bilateral_inconsistency_blocks_blink_candidate_event() -> None:
    rows = [
        _face_row(0, 0.30),
        _face_row(67, 0.30),
        _face_row(134, 0.03),
        _face_row(201, 0.30),
    ]
    _eye_points(blink.LEFT_EYE, 0.055, rows[2])
    frames, events, status = blink.derive_blink_candidates(
        pd.DataFrame(rows), minimum_valid_frames=2, minimum_closed_duration_ms=0
    )
    row = frames[pd.to_numeric(frames["unix_ms"]).eq(134)].iloc[0]
    assert bool(row["blink_bilateral_observable"])
    assert not bool(row["bilateral_eye_consistent"])
    assert not bool(row["blink_bilateral_valid_for_event"])
    assert not bool(row["blink_closed_bilateral_candidate"])
    assert events.empty
    assert status["bilateral_inconsistent_rows"] >= 1


def test_pose_radial_output_is_dimensionless_direction_score() -> None:
    raw = []
    raw += _pose_frame(0)
    raw += _pose_frame(100, world_z=-0.02, bbox_scale=1.1)
    out, status = derive_pose_direction(pd.DataFrame(raw), gap_reset_ms=300)
    assert "pose_radial_proximity_direction_score" in out.columns
    assert "pose_radial_proximity_candidate_per_sec" not in out.columns
    score = out.loc[1, "pose_radial_proximity_direction_score"]
    assert -1.0 <= float(score) <= 1.0
    assert status["radial_interpretation"] == "dimensionless_direction_agreement_score_not_physical_displacement"


def test_rgb_identity_reconciles_before_participant_group_gate() -> None:
    source = (Path(__file__).resolve().parents[1] / "src/attention_pipeline/rgb_formal/runner.py").read_text(encoding="utf-8")
    governed = source.split("def _governed_identity", 1)[1].split("def participant_inference_gate", 1)[0]
    assert "included_cohort(cohort, require_groups=False)" in governed
    assert "reconcile_formal_identity(" in governed
    assert "reconcile_cohort_identity(" not in governed


def test_rgb_identity_order_keeps_key_without_legacy_session() -> None:
    """Behavioral lock for the identity-order fix (audit 3.1).

    A session with a verified participant_key but an empty legacy
    repeat_participant_id must survive cohort inclusion and reconcile as
    resolved; a legacy-only session without an allow-listed governance status
    must survive as unresolved instead of failing before reconciliation.
    """
    from attention_pipeline.formal_analysis.cohort import included_cohort
    from attention_pipeline.formal_analysis.identity_contract import reconcile_formal_identity

    cohort = pd.DataFrame({
        "session_id": ["sub-001", "sub-002", "sub-003", "sub-004"],
        "include": [True, True, True, True],
        "repeat_participant_id": ["P-A", None, "P-C", "P-D"],
        "identity_status": ["confirmed", None, "confirmed", None],
    })
    registry = pd.DataFrame({
        "session_id": ["sub-001", "sub-002"],
        "participant_key": ["participant:A", "participant:B"],
    })
    # require_groups=True would raise on sub-002 before reconciliation (old bug).
    included = included_cohort(cohort, require_groups=False)
    identity = reconcile_formal_identity(
        included, registry, legacy_status_column="identity_status",
    )
    by = identity.set_index("session_id")
    assert len(identity) == 4
    assert by.loc["sub-002", "participant_group_id"] == "participant:B"
    assert by.loc["sub-003", "participant_group_id"] == "legacy:P-C"
    assert pd.isna(by.loc["sub-004", "participant_group_id"])
    assert by.loc["sub-004", "participant_identity_source"] == "unresolved"


def test_rgb_config_uses_shared_identity_contract() -> None:
    import yaml

    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/rgb_formal.yaml").read_text(encoding="utf-8"))
    identity = config["identity"]
    assert identity["formal_cluster_key"] == "participant_group_id"
    assert identity["identity_reconciliation_function"] == "reconcile_formal_identity"
    assert identity["legacy_identity_status_column"] == "identity_status"
    assert config["pose_confirmation"]["radial_direction_output"] == "pose_radial_proximity_direction_score"
