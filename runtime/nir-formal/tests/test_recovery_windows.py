from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recovery_windows import map_timestamp_rows, read_reconstructed_blocks


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_reconstructed_blocks_map_each_modality_by_absolute_time(tmp_path):
    timeline = tmp_path / "reconstructed.csv"
    _write(
        timeline,
        [
            {"event": "reconstructed_block_start", "detail": "Block1_B", "unix_ms": 1000},
            {"event": "reconstructed_block_stop", "detail": "Block1_B", "unix_ms": 1300},
            {"event": "reconstructed_block_start", "detail": "Block2_B", "unix_ms": 2000},
            {"event": "reconstructed_block_stop", "detail": "Block2_B", "unix_ms": 2300},
        ],
    )
    blocks = read_reconstructed_blocks(timeline)
    nir = map_timestamp_rows("nir", [(10, 900), (11, 1000), (12, 1100), (13, 1200), (14, 2000)], blocks, source="nir.csv")
    mmwave = map_timestamp_rows("mmwave", [(100, 1000), (101, 1100), (103, 1200), (104, 2000)], blocks, source="mmwave.csv")
    assert nir[0].start_frame_idx == 11
    assert nir[0].end_frame_idx == 13
    assert nir[0].contiguous
    assert mmwave[0].frame_index_gap_count == 1
    assert mmwave[0].frame_index_gap_frames == 1
    assert mmwave[1].n_timestamp_rows == 1
