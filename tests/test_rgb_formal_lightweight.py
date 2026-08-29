from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from attention_pipeline.rgb_formal import blink_candidates as blink
from attention_pipeline.rgb_formal.motion_qc import derive_motion_qc
from attention_pipeline.rgb_formal.pose_direction import derive_pose_direction


def _eye_points(prefix: str, indices: tuple[int, ...], openness: float, row: dict[str, object]) -> None:
    p0, p1, p2, p3, p4, p5 = indices
    coords = {
        p0: (0.0, 0.0), p3: (1.0, 0.0),
        p1: (0.25, openness / 2), p5: (0.25, -openness / 2),
        p2: (0.75, openness / 2), p4: (0.75, -openness / 2),
    }
    for idx, (x, y) in coords.items():
        row[f"mesh_x_{idx}"] = x
        row[f"mesh_y_{idx}"] = y


def _face_row(t: int, openness: float, *, track: str = "A", primary: bool = True, face_count: int = 1, rank: int = 0) -> dict[str, object]:
    row: dict[str, object] = {
        "unix_ms": t, "phase": "baseline", "primary_face": primary,
        "face_track_id": track, "face_count": face_count, "face_rank": rank,
        "temporal_gap": False, "capture_gap_before": False,
    }
    _eye_points("right", blink.RIGHT_EYE, openness, row)
    _eye_points("left", blink.LEFT_EYE, openness, row)
    return row


def test_single_frame_blink_has_nominal_frame_duration() -> None:
    face = pd.DataFrame([
        _face_row(0, 0.30), _face_row(67, 0.30), _face_row(134, 0.03), _face_row(201, 0.30),
    ])
    _, events, status = blink.derive_blink_candidates(face, minimum_valid_frames=2)
    assert status["blink_status"] == "generated"
    assert len(events) == 1
    assert events.iloc[0]["frame_n"] == 1
    assert events.iloc[0]["duration_ms"] == pytest.approx(67.0)
    assert events.iloc[0]["duration_ms"] > 0


def test_blink_event_never_bridges_timestamp_gap() -> None:
    face = pd.DataFrame([
        _face_row(0, 0.30), _face_row(67, 0.03), _face_row(567, 0.03), _face_row(634, 0.30),
    ])
    _, events, status = blink.derive_blink_candidates(
        face, minimum_valid_frames=1, minimum_closed_duration_ms=0, gap_reset_ms=250
    )
    assert status["gap_break_rows"] >= 1
    assert len(events) == 2
    assert events["frame_n"].tolist() == [1, 1]


def test_multiface_rank0_is_not_reliable_primary_and_track_switch_splits() -> None:
    ambiguous = pd.DataFrame([
        _face_row(0, 0.30),
        _face_row(67, 0.03, primary=False, face_count=2, rank=0),
        _face_row(67, 0.03, primary=False, face_count=2, rank=1),
        _face_row(134, 0.30),
    ])
    frames, events, status = blink.derive_blink_candidates(
        ambiguous, minimum_valid_frames=1, minimum_closed_duration_ms=0
    )
    middle = frames[pd.to_numeric(frames["unix_ms"]).eq(67)].iloc[0]
    assert not bool(middle["primary_face_reliable"])
    assert status["rank0_only_rejected_frames"] == 1
    assert events.empty

    switched = pd.DataFrame([
        _face_row(0, 0.30, track="A"), _face_row(67, 0.03, track="A"),
        _face_row(134, 0.03, track="B"), _face_row(201, 0.30, track="B"),
    ])
    _, events2, status2 = blink.derive_blink_candidates(
        switched, minimum_valid_frames=1, minimum_closed_duration_ms=0
    )
    assert status2["track_reset_rows"] == 1
    assert len(events2) == 2


def test_face_parquet_is_read_with_column_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    available = sorted(blink.FACE_PROJECTION_COLUMNS | {f"AU{i:02d}" for i in range(1, 100)} | {f"junk_{i}" for i in range(3000)})
    monkeypatch.setattr(blink, "_parquet_columns", lambda _: available)
    seen: dict[str, object] = {}

    def fake_read(path: Path, *, columns: list[str]) -> pd.DataFrame:
        seen["columns"] = columns
        return pd.DataFrame({c: [0] for c in columns})

    monkeypatch.setattr(pd, "read_parquet", fake_read)
    blink.read_face_projection(Path("face.parquet"))
    columns = seen["columns"]
    assert isinstance(columns, list)
    assert len(columns) == len(blink.FACE_PROJECTION_COLUMNS)
    assert len(columns) < 100
    assert not any(str(c).startswith("AU") for c in columns)


def test_exposure_change_is_not_relabelled_as_body_motion() -> None:
    motion = pd.DataFrame({
        "unix_ms": [0, 100], "global_motion_energy_per_sec": [0.0, 0.0], "gray_mean_delta": [0.0, 40.0],
    })
    out, status = derive_motion_qc(motion)
    assert status["combined_risk_score_generated"] is False
    assert out.loc[1, "body_motion_energy"] == 0.0
    assert out.loc[1, "exposure_change_abs"] == 40.0
    assert out.loc[1, "motion_exposure_separation_contract"] == "separate_tracks_no_combined_risk_score"


def _pose_frame(t: int, x: float, y: float, *, visibility: float = 0.9, world_z: float = 0.0, bbox_scale: float = 1.0) -> list[dict[str, object]]:
    rows = []
    for name, dx in (("left_shoulder", -0.1), ("right_shoulder", 0.1)):
        rows.append({
            "video_frame_position": t, "unix_ms": t, "pose_valid": True, "pose_count": 1,
            "pose_index": 0, "landmark_name": name, "x": x + dx, "y": y,
            "visibility": visibility, "presence": 0.9, "world_z": world_z,
            "pose_bbox_xmin": 0.2 - 0.05 * bbox_scale, "pose_bbox_ymin": 0.2 - 0.05 * bbox_scale,
            "pose_bbox_xmax": 0.8 + 0.05 * bbox_scale, "pose_bbox_ymax": 0.8 + 0.05 * bbox_scale,
        })
    return rows


def test_pose_signed_differences_reset_at_gap_and_low_visibility() -> None:
    raw = []
    raw += _pose_frame(0, 0.50, 0.50)
    raw += _pose_frame(100, 0.55, 0.45, world_z=-0.02, bbox_scale=1.1)
    raw += _pose_frame(600, 0.60, 0.40, world_z=-0.04, bbox_scale=1.2)
    raw += _pose_frame(700, 0.65, 0.35, visibility=0.1)
    raw += _pose_frame(800, 0.70, 0.30, world_z=-0.06, bbox_scale=1.3)
    out, status = derive_pose_direction(pd.DataFrame(raw), gap_reset_ms=300)
    assert out.loc[1, "pose_lateral_right_per_sec"] > 0
    assert out.loc[1, "pose_vertical_up_per_sec"] > 0
    assert out.loc[1, "pose_radial_component_n"] == 3
    assert bool(out.loc[2, "pose_diff_reset"])
    assert pd.isna(out.loc[2, "pose_lateral_right_per_sec"])
    assert bool(out.loc[3, "pose_diff_reset"])
    assert bool(out.loc[4, "pose_diff_reset"])
    assert pd.isna(out.loc[4, "pose_lateral_right_per_sec"])
    assert status["gap_or_quality_reset_rows"] >= 4


def test_unresolved_identity_never_falls_back_to_session_id() -> None:
    from attention_pipeline.rgb_formal.runner import participant_inference_gate

    identity = pd.DataFrame({
        "session_id": ["sub-001", "sub-002"],
        "participant_group_id": ["participant:A", pd.NA],
        "participant_identity_source": ["participant_key", "unresolved"],
    })
    gate = participant_inference_gate(identity)
    assert gate["status"] == "not_estimable"
    assert gate["reason"] == "participant_identity_unresolved_no_session_id_fallback"
    assert gate["unresolved_session_n"] == 1
    assert "sub-002" not in identity["participant_group_id"].astype(str).tolist()


def test_mmwave_contract_forbids_overwrite_and_combined_risk_score() -> None:
    from attention_pipeline.rgb_formal.runner import mmwave_protection_contract

    contract = mmwave_protection_contract()
    assert contract["rgb_writes_mmwave_results"] is False
    assert contract["rgb_creates_mmwave_truth_table"] is False
    assert contract["blink_combined_with_motion_pose_risk_score"] is False
    assert contract["required_future_mmwave_tracks"][0] == "original_mmwave_result"


def test_lightweight_runner_does_not_import_heavy_science_or_figures() -> None:
    source = (Path(__file__).resolve().parents[1] / "src/attention_pipeline/rgb_formal/runner.py").read_text(encoding="utf-8")
    assert ".science import" not in source
    assert ".figures import" not in source
    assert "participant_cluster_bootstrap(" not in source
    assert "participant_exclusive_folds(" not in source
    assert "generate_rgb_figure_pack(" not in source


def test_perclos_is_retained_but_disabled_deferred() -> None:
    import yaml

    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/rgb_formal.yaml").read_text(encoding="utf-8"))
    perclos = config["ocular"]["perclos_candidate"]
    assert perclos["active"] is False
    assert perclos["status"] == "disabled_deferred"
    assert perclos["reason"] == "no validated closure-event contract"


def test_legacy_pipeline_entry_is_only_a_compatibility_alias() -> None:
    source = (Path(__file__).resolve().parents[1] / "src/attention_pipeline/rgb_formal/pipeline.py").read_text(encoding="utf-8")
    assert "from .runner import run_rgb_formal_v2" in source
    assert "return run_rgb_formal_v2(config_path, subjects=subjects)" in source
    assert "candidate_validation(summary)" not in source.split("def run_rgb_formal_pipeline", 1)[1]


def test_detected_or_face_valid_false_breaks_blink_observability() -> None:
    rows = [_face_row(0, 0.30), _face_row(67, 0.03), _face_row(134, 0.30)]
    rows[1]["detected"] = False
    face = pd.DataFrame(rows)
    frames, events, _ = blink.derive_blink_candidates(face, minimum_valid_frames=1, minimum_closed_duration_ms=0)
    middle = frames[pd.to_numeric(frames["unix_ms"]).eq(67)].iloc[0]
    assert not bool(middle["primary_face_reliable"])
    assert middle["primary_face_selection_source"] == "face_detection_or_validity_false"
    assert not bool(middle["blink_bilateral_observable"])
    assert events.empty


def test_blink_status_does_not_conflict_with_primary_face_status() -> None:
    face = pd.DataFrame([_face_row(0, 0.30)])
    _, _, status = blink.derive_blink_candidates(face, minimum_valid_frames=5)
    assert "status" not in status
    assert status["primary_face_status"] == "generated"
    assert status["blink_status"] == "not_estimable"
