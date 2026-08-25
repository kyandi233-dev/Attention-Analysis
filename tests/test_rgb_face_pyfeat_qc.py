import pandas as pd

from attention_pipeline.rgb.face_pyfeat_qc import summarize_pyfeat_benchmark


def test_pyfeat_qc_detects_multi_face_and_missing_inputs():
    sample = pd.DataFrame([
        {"benchmark_index": 0, "image_path": "C:/frames/a.jpg", "phase": "baseline"},
        {"benchmark_index": 1, "image_path": "C:/frames/b.jpg", "phase": "block1"},
        {"benchmark_index": 2, "image_path": "C:/frames/c.jpg", "phase": "block1"},
    ])
    raw = pd.DataFrame([
        {"input": "C:/frames/a.jpg", "FaceScore": 0.9, "AU01": 0.1, "anger": 0.2, "gaze_pitch": 0.0, "Pitch": 0.1},
        {"input": "C:/frames/b.jpg", "FaceScore": 0.8, "AU01": 0.2, "anger": 0.3, "gaze_pitch": 0.1, "Pitch": 0.2},
        {"input": "C:/frames/b.jpg", "FaceScore": 0.7, "AU01": 0.3, "anger": 0.4, "gaze_pitch": 0.2, "Pitch": 0.3},
    ])
    summary, per_image = summarize_pyfeat_benchmark(sample, raw)
    assert summary["images_with_face"] == 2
    assert summary["images_without_face"] == 1
    assert summary["images_with_multiple_faces"] == 1
    assert summary["extra_face_rows_above_one_per_input"] == 1
    assert per_image.loc[per_image["benchmark_index"] == 1, "face_count"].iloc[0] == 2
