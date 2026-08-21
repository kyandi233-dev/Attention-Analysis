from pathlib import Path

import pandas as pd
import pytest

from attention_pipeline.nir.formal import (
    block_window,
    formal_subject_paths,
    load_nir_timestamps,
    locate_continuous_window,
    normalize_subject,
)


def test_subject_paths_accept_trailing_underscore(tmp_path):
    assert normalize_subject("sub-011") == ("sub-011_", "sub-011")
    assert normalize_subject("sub-011_") == ("sub-011_", "sub-011")
    paths = formal_subject_paths(tmp_path, "sub-011_")
    assert paths["nir_video"] == tmp_path / "sub-011_" / "nir" / "sub-011_nir.avi"


def test_block_window_and_timestamp_alignment(tmp_path):
    master = tmp_path / "master.csv"
    pd.DataFrame([
        {"event": "block_start", "detail": "Block1_B", "unix_ms": 1000},
        {"event": "block_stop", "detail": "Block1_B", "unix_ms": 5000},
    ]).to_csv(master, index=False)
    assert block_window(master, 1) == (1000, 5000)
    timestamps = pd.DataFrame({
        "frame_idx": range(6),
        "unix_ms": [950, 1010, 1040, 1070, 1100, 1130],
        "status": [None] * 6,
    })
    result = locate_continuous_window(timestamps, 1000, 100)
    assert result == {
        "start_frame_idx": 1, "end_frame_idx": 4,
        "start_unix_ms": 1010, "end_unix_ms": 1100,
        "n_frames": 4, "start_offset_ms": 10,
    }


def test_timestamp_reader_rejects_real_gap(tmp_path):
    path = tmp_path / "ts.csv"
    path.write_text("0,1000,\n2,1033,\n", encoding="utf-8")
    timestamps = load_nir_timestamps(path)
    with pytest.raises(ValueError, match="frame-index gap"):
        locate_continuous_window(timestamps, 1000, 100)
