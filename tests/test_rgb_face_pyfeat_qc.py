import pandas as pd

from attention_pipeline.rgb.face_pyfeat_qc import summarize_pyfeat_benchmark


def test_pyfeat_qc_detects_multi_face_missing_inputs_and_v2_groups():
    sample = pd.DataFrame([
        {"benchmark_index": 0, "image_path": "C:/frames/a.jpg", "phase": "baseline"},
        {"benchmark_index": 1, "image_path": "C:/frames/b.jpg", "phase": "block1"},
        {"benchmark_index": 2, "image_path": "C:/frames/c.jpg", "phase": "block1"},
    ])
    raw = pd.DataFrame([
        {
            "input": "C:/frames/a.jpg", "FaceScore": 0.9, "AU01": 0.1,
            "Neutral": 0.7, "Happy": 0.1, "valence": 0.2,
            "gaze_pitch": 0.0, "Pitch": 0.1,
            "x_0": 10.0, "y_0": 11.0,
            "mesh_x_0": 10.0, "mesh_y_0": 11.0, "mesh_z_0": 0.1,
            "eyeBlinkLeft": 0.2, "Identity_1": 0.01,
        },
        {
            "input": "C:/frames/b.jpg", "FaceScore": 0.8, "AU01": 0.2,
            "Neutral": 0.6, "Happy": 0.2, "valence": 0.1,
            "gaze_pitch": 0.1, "Pitch": 0.2,
            "x_0": 12.0, "y_0": 13.0,
            "mesh_x_0": 12.0, "mesh_y_0": 13.0, "mesh_z_0": 0.2,
            "eyeBlinkLeft": 0.3, "Identity_1": 0.02,
        },
        {
            "input": "C:/frames/b.jpg", "FaceScore": 0.7, "AU01": 0.3,
            "Neutral": 0.5, "Happy": 0.3, "valence": 0.0,
            "gaze_pitch": 0.2, "Pitch": 0.3,
            "x_0": 14.0, "y_0": 15.0,
            "mesh_x_0": 14.0, "mesh_y_0": 15.0, "mesh_z_0": 0.3,
            "eyeBlinkLeft": 0.4, "Identity_1": 0.03,
        },
    ])
    summary, per_image = summarize_pyfeat_benchmark(sample, raw)
    assert summary["images_with_face"] == 2
    assert summary["images_without_face"] == 1
    assert summary["images_with_multiple_faces"] == 1
    assert summary["extra_face_rows_above_one_per_input"] == 1
    assert summary["field_groups"]["emotion_v2"]["column_count"] == 2
    assert summary["field_groups"]["landmark68_xy"]["column_count"] == 2
    assert summary["field_groups"]["mesh478_xyz"]["column_count"] == 3
    assert summary["field_groups"]["blendshapes"]["column_count"] == 1
    assert summary["field_groups"]["identity"]["column_count"] == 1
    assert summary["output_columns"] == len(raw.columns)
    assert per_image.loc[per_image["benchmark_index"] == 1, "face_count"].iloc[0] == 2
