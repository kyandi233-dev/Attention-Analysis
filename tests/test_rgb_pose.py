from __future__ import annotations

from types import SimpleNamespace

from attention_pipeline.rgb.pose import pose_result_rows


def _lm(x, y, z, visibility=0.9, presence=0.8):
    return SimpleNamespace(x=x, y=y, z=z, visibility=visibility, presence=presence)


def test_pose_result_rows_preserves_normalized_world_and_multiple_poses():
    pose1 = [_lm(i / 100, i / 100 + 0.1, -i / 1000) for i in range(33)]
    pose2 = [_lm(i / 100 + 0.2, i / 100 + 0.2, -i / 900) for i in range(33)]
    world1 = [_lm(i / 10, i / 20, i / 30, 0.7, 0.6) for i in range(33)]
    world2 = [_lm(i / 11, i / 21, i / 31, 0.5, 0.4) for i in range(33)]
    result = SimpleNamespace(
        pose_landmarks=[pose1, pose2],
        pose_world_landmarks=[world1, world2],
    )
    rows = pose_result_rows(result, base={"subject": "sub-031", "unix_ms": 123})
    assert len(rows) == 66
    assert rows[0]["pose_count"] == 2
    assert rows[0]["pose_index"] == 0
    assert rows[0]["landmark_name"] == "nose"
    assert rows[0]["visibility"] == 0.9
    assert rows[0]["world_visibility"] == 0.7
    assert rows[33]["pose_index"] == 1


def test_pose_result_rows_keeps_missing_frame_identity():
    result = SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[])
    rows = pose_result_rows(result, base={"subject": "sub-031", "video_frame_position": 10})
    assert len(rows) == 1
    assert rows[0]["pose_valid"] is False
    assert rows[0]["video_frame_position"] == 10
    assert rows[0]["landmark_index"] is None
