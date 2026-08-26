import pandas as pd

from attention_pipeline.rgb.pose_qc import summarize_pose_table


def test_pose_qc_reports_upper_body_separately():
    rows = []
    for frame in (10, 20):
        for name, vis, pres in [
            ("left_shoulder", 0.95, 0.98),
            ("right_shoulder", 0.90, 0.96),
            ("left_wrist", 0.80, 0.85),
            ("right_wrist", 0.75, 0.82),
            ("left_hip", 0.20, 0.10),
            ("right_hip", 0.15, 0.08),
        ]:
            rows.append({
                "video_frame_position": frame,
                "phase": "block1",
                "pose_valid": True,
                "pose_count": 1,
                "pose_index": 0,
                "landmark_name": name,
                "x": 0.5,
                "y": 0.5,
                "visibility": vis,
                "presence": pres,
                "world_x": 0.0,
            })
    table = pd.DataFrame(rows)
    summary = summarize_pose_table(
        table,
        subject="sub-test",
        upper_body_landmarks=["left_shoulder", "right_shoulder", "left_wrist", "right_wrist"],
        optional_trunk_landmarks=["left_hip", "right_hip"],
    )
    assert summary["pose_valid_fraction"] == 1.0
    assert summary["upper_body_group"]["mean_visibility"] > summary["optional_trunk_group"]["mean_visibility"]
    assert summary["landmarks"]["left_shoulder"]["frame_coverage"] == 1.0
