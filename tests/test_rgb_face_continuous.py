import pandas as pd

from attention_pipeline.rgb.face_continuous import _nearest_position
from attention_pipeline.rgb.face_continuous_qc import _parse_headpose, _temporal_metrics


def test_nearest_position_uses_real_timestamps():
    times = [1000, 1033, 1067, 1101, 1134]
    assert _nearest_position(times, 1098, 0, 4) == 3
    assert _nearest_position(times, 1048, 0, 4) == 1


def test_parse_libreface_headpose_string_json():
    pitch, yaw, roll = _parse_headpose('"pitch:1.5, yaw:-2.0, roll:3.5"')
    assert pitch == 1.5
    assert yaw == -2.0
    assert roll == 3.5


def test_temporal_metrics_skip_marked_gap():
    table = pd.DataFrame({
        "dt_ms": [None, 100, 100, 500],
        "temporal_gap": [False, False, False, True],
        "AU01": [0.0, 0.1, 0.2, 5.0],
    })
    result = _temporal_metrics(table, ["AU01"])["AU01"]
    assert result["valid_step_pairs"] == 2
    assert result["max_abs_step"] < 1.0
