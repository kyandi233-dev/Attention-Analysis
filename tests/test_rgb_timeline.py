from __future__ import annotations

import csv

from attention_pipeline.rgb.timeline import formal_analysis_span, formal_rgb_intervals


def test_formal_rgb_timeline_uses_true_baseline_and_two_blocks(tmp_path):
    path = tmp_path / "master_timeline.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event", "detail", "unix_ms"])
        writer.writeheader()
        writer.writerows(
            [
                {"event": "baseline_start", "detail": "", "unix_ms": 900000},
                {"event": "baseline_stop", "detail": "duration=180s", "unix_ms": 1180000},
                {"event": "block_start", "detail": "Block1_B", "unix_ms": 1300000},
                {"event": "block_stop", "detail": "Block1_B", "unix_ms": 1400000},
                {"event": "block_start", "detail": "Block2_B", "unix_ms": 1600000},
                {"event": "block_stop", "detail": "Block2_B", "unix_ms": 1700000},
            ]
        )

    intervals = formal_rgb_intervals(path, baseline_duration_sec=180, expected_blocks=2)
    assert intervals[0].phase == "baseline"
    assert intervals[0].start_unix_ms == 1000000
    assert [item.phase for item in intervals if item.phase.startswith("block")] == ["block1", "block2"]
    assert formal_analysis_span(path) == (1000000, 1700000)
