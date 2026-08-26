from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from scripts.face_formal_cuda import _append_parquet_chunk, _canonicalize_dataframe


def test_streaming_numeric_null_chunk_keeps_float_schema(tmp_path: Path):
    output = tmp_path / "face.parquet"
    state = {
        "path": str(output),
        "writer": None,
        "schema": None,
        "columns": None,
        "rows": 0,
    }
    first = pd.DataFrame([
        {
            "subject": "sub-test",
            "benchmark_index": 0,
            "AU20": None,
            "FaceScore": None,
            "detected": False,
        }
    ])
    second = pd.DataFrame([
        {
            "subject": "sub-test",
            "benchmark_index": 1,
            "AU20": 0.25,
            "FaceScore": 0.9,
            "detected": True,
        }
    ])

    _append_parquet_chunk(first, state)
    _append_parquet_chunk(second, state)
    state["writer"].close()

    schema = pq.read_schema(output)
    assert str(schema.field("AU20").type) == "double"
    assert str(schema.field("FaceScore").type) == "double"
    values = pd.read_parquet(output)
    assert pd.isna(values.loc[0, "AU20"])
    assert values.loc[1, "AU20"] == 0.25


def test_face_nullable_identity_and_science_columns_are_stable():
    face = _canonicalize_dataframe(
        pd.DataFrame(
            [
                {
                    "subject": "sub-test",
                    "benchmark_index": None,
                    "AU20": None,
                    "detected": None,
                    "phase": "baseline",
                }
            ]
        )
    )
    assert str(face.dtypes["benchmark_index"]) == "Int64"
    assert str(face.dtypes["AU20"]) == "Float64"
    assert str(face.dtypes["detected"]) == "boolean"
    assert str(face.dtypes["phase"]) == "string"
