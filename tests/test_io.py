from pathlib import Path

import pandas as pd

from attention_pipeline.io import block_windows, load_timestamps, nearest_written_frame


def test_dropped_rows_do_not_consume_avi_position(tmp_path: Path):
    path = tmp_path / "timestamps.csv"
    path.write_text("10,1000,\n11,1033,dropped\n12,1066,\n", encoding="utf-8")
    rows = load_timestamps(path)
    assert rows.loc[0, "avi_frame_idx"] == 0
    assert pd.isna(rows.loc[1, "avi_frame_idx"])
    assert rows.loc[2, "avi_frame_idx"] == 1
    mapped = nearest_written_frame(rows, 1050)
    assert mapped["capture_frame_idx"] == 12
    assert mapped["avi_frame_idx"] == 1


def test_formal_timeline_has_six_unbuffered_blocks(config):
    timeline = config.path_value("raw_root") / "sub-000_" / "beh" / "master_timeline.csv"
    windows = block_windows(timeline)
    assert [x["condition"] for x in windows] == ["A", "B", "C", "C", "B", "A"]
    assert all(x["start_ms"] < x["end_ms"] for x in windows)

