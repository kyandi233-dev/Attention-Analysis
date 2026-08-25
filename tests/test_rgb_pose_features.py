import pandas as pd

from attention_pipeline.rgb.pose_features import derive_pose_features


def _frame_rows(frame_pos, unix_ms, offset):
    coords = {
        "left_shoulder": (0.4 + offset, 0.4),
        "right_shoulder": (0.6 + offset, 0.4),
        "left_elbow": (0.35 + offset, 0.55),
        "right_elbow": (0.65 + offset, 0.55),
        "left_wrist": (0.3 + offset, 0.7),
        "right_wrist": (0.7 + offset, 0.7),
        "left_hip": (0.45 + offset, 0.75),
        "right_hip": (0.55 + offset, 0.75),
    }
    rows = []
    for name, (x, y) in coords.items():
        rows.append({
            "video_frame_position": frame_pos,
            "capture_frame_idx": frame_pos,
            "unix_ms": unix_ms,
            "phase": "block1",
            "block": 1,
            "trial_num": 1,
            "behavior_state": "trial",
            "pose_valid": True,
            "pose_count": 1,
            "pose_index": 0,
            "landmark_name": name,
            "x": x,
            "y": y,
            "visibility": 0.95,
            "presence": 0.95,
        })
    return rows


def test_pose_features_are_shoulder_width_normalized():
    table = pd.DataFrame(_frame_rows(1, 1000, 0.0) + _frame_rows(2, 1100, 0.02))
    features = derive_pose_features(table, subject="sub-test")
    assert len(features) == 2
    second = features.iloc[1]
    assert abs(second["shoulder_width_norm"] - 0.2) < 1e-9
    assert second["upper_body_motion_swidth_per_sec"] > 0
    assert second["primary_pose_ambiguous"] == False
