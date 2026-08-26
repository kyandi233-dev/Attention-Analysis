from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from timestamp_mapping import read_timestamp_map


def _write_rows(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def test_capture_gaps_do_not_change_sequential_avi_mapping(tmp_path):
    path = tmp_path / "sub-100_nir_timestamps.csv"
    _write_rows(
        path,
        [
            [0, 1000, ""],
            [1, 1033, ""],
            [4, 1133, ""],
            [5, 1166, ""],
        ],
    )

    mapping = read_timestamp_map(path)

    assert mapping.avi_frame_count == 4
    assert mapping.unix_by_avi_frame == {0: 1000, 1: 1033, 2: 1133, 3: 1166}
    assert mapping.capture_by_avi_frame == {0: 0, 1: 1, 2: 4, 3: 5}
    assert mapping.capture_frame_gap_count == 1
    assert mapping.capture_frame_gap_frames == 2


def test_explicit_dropped_rows_are_excluded_from_avi_mapping(tmp_path):
    path = tmp_path / "sub-dropped_nir_timestamps.csv"
    _write_rows(
        path,
        [
            [0, 1000, ""],
            [1, 1033, "dropped"],
            [2, 1066, ""],
        ],
    )

    mapping = read_timestamp_map(path)

    assert mapping.avi_frame_count == 2
    assert mapping.capture_by_avi_frame == {0: 0, 1: 2}
    assert mapping.n_dropped_rows == 1
