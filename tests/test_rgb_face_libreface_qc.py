import pandas as pd

from attention_pipeline.rgb.face_libreface_qc import summarize_libreface_benchmark


def test_libreface_qc_reports_alignment_failure_and_component_coverage():
    sample = pd.DataFrame([
        {"benchmark_index": 0, "image_path": "C:/frames/a.jpg", "phase": "baseline"},
        {"benchmark_index": 1, "image_path": "C:/frames/b.jpg", "phase": "block1"},
        {"benchmark_index": 2, "image_path": "C:/frames/c.jpg", "phase": "block1"},
    ])
    alignment = pd.DataFrame([
        {"benchmark_index": 0, "alignment_success": True, "alignment_error": None, "headpose_json": "{}", "landmarks_json": "{}"},
        {"benchmark_index": 1, "alignment_success": False, "alignment_error": "no face", "headpose_json": None, "landmarks_json": None},
        {"benchmark_index": 2, "alignment_success": True, "alignment_error": None, "headpose_json": "{}", "landmarks_json": "{}"},
    ])
    components = {
        "au_detection": pd.DataFrame([
            {"benchmark_index": 0, "AU1": 1},
            {"benchmark_index": 2, "AU1": 0},
        ]),
        "au_intensity": pd.DataFrame([
            {"benchmark_index": 0, "AU1": 2.0},
            {"benchmark_index": 2, "AU1": 1.0},
        ]),
        "expression": pd.DataFrame([
            {"benchmark_index": 0, "expression": "Neutral"},
            {"benchmark_index": 2, "expression": "Happy"},
        ]),
        "gaze": pd.DataFrame([
            {"benchmark_index": 0, "gaze_yaw": 1.0, "gaze_pitch": 2.0},
            {"benchmark_index": 2, "gaze_yaw": 3.0, "gaze_pitch": 4.0},
        ]),
    }
    summary, per_image = summarize_libreface_benchmark(sample, alignment, components)
    assert summary["aligned_faces"] == 2
    assert summary["alignment_failures"] == 1
    assert summary["phase"]["block1"]["alignment_failures"] == 1
    assert summary["components"]["au_detection"]["covers_all_aligned_indices"] is True
    assert summary["components"]["gaze"]["column_count"] == 2
    assert summary["failed_inputs"][0]["benchmark_index"] == 1
    assert len(per_image) == 3
