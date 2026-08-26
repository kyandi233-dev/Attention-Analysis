from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from scripts.rgb_formal_full_runner_v1 import _canonicalize_dataframe, _table_flush


def test_streaming_numeric_null_chunk_keeps_float_schema(tmp_path: Path):
    output = tmp_path / "motion.parquet"
    state = {"path": str(output), "writer": None, "schema": None, "rows": 0}
    first = [{"subject": "sub-test", "gray_mean_delta": None, "global_motion_energy": None}]
    second = [{"subject": "sub-test", "gray_mean_delta": 0.25, "global_motion_energy": 0.5}]

    _table_flush(first, state, "motion")
    _table_flush(second, state, "motion")
    state["writer"].close()

    table = pq.read_table(output)
    assert table.schema.field("gray_mean_delta").type == pq.read_schema(output).field("gray_mean_delta").type
    assert str(table.schema.field("gray_mean_delta").type) == "double"
    assert str(table.schema.field("global_motion_energy").type) == "double"
    values = pd.read_parquet(output)
    assert pd.isna(values.loc[0, "gray_mean_delta"])
    assert values.loc[1, "gray_mean_delta"] == 0.25


def test_pose_and_face_nullable_numeric_columns_are_stable():
    pose = _canonicalize_dataframe(pd.DataFrame([{"x": None, "landmark_index": None}]), "pose")
    face = _canonicalize_dataframe(pd.DataFrame([{"AU20": None, "frame": None}]), "face")
    assert str(pose.dtypes["x"]) == "Float64"
    assert str(pose.dtypes["landmark_index"]) == "Int64"
    assert str(face.dtypes["AU20"]) == "Float64"
    assert str(face.dtypes["frame"]) == "Int64"
