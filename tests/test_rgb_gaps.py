from __future__ import annotations

import csv
from pathlib import Path

from attention_pipeline.config import Config
from attention_pipeline.rgb.discover import RGBSubjectFiles
from attention_pipeline.rgb.gaps import subject_timestamp_gaps
from attention_pipeline.rgb.timeline import detailed_rgb_intervals


def _write_timeline(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event", "detail", "unix_ms"])
        writer.writeheader()
        writer.writerows(
            [
                {"event": "baseline_stop", "detail": "duration=180s", "unix_ms": 1180000},
                {"event": "instructions", "detail": "", "unix_ms": 1181000},
                {"event": "practice_start", "detail": "", "unix_ms": 1190000},
                {"event": "practice_end", "detail": "", "unix_ms": 1200000},
                {"event": "block_start", "detail": "Block1_B", "unix_ms": 1300000},
                {"event": "block_stop", "detail": "Block1_B", "unix_ms": 1400000},
                {"event": "block_start", "detail": "Block2_B", "unix_ms": 1600000},
                {"event": "block_stop", "detail": "Block2_B", "unix_ms": 1700000},
            ]
        )


def test_detailed_timeline_keeps_instructions_practice_and_interblock(tmp_path):
    timeline = tmp_path / "master_timeline.csv"
    _write_timeline(timeline)
    phases = [item.phase for item in detailed_rgb_intervals(timeline)]
    assert "baseline" in phases
    assert "instructions" in phases
    assert "practice" in phases
    assert "block1" in phases
    assert "interblock_transition" in phases
    assert "block2" in phases


def test_timestamp_gap_preserves_frame_identity_and_phase(tmp_path):
    timeline = tmp_path / "master_timeline.csv"
    timestamps = tmp_path / "sub-031_rgb_timestamps.csv"
    _write_timeline(timeline)
    timestamps.write_text(
        "0,1300000,\n1,1300032,\n2,1300064,\n3,1301000,\n4,1301032,\n",
        encoding="utf-8",
    )
    config = Config(
        path=tmp_path / "rgb_analysis.yaml",
        data={
            "data": {"exclude": {}},
            "focuswave": {"baseline_duration_sec": 180, "expected_blocks": 2},
            "qc": {"timestamp_gap_warning_ms": 100},
        },
        digest="test",
    )
    files = RGBSubjectFiles(
        subject="sub-031",
        root=tmp_path,
        subject_dir=tmp_path,
        video=tmp_path / "sub-031_rgb.avi",
        timestamps=timestamps,
        behavior_dir=tmp_path,
        master_timeline=timeline,
        block1_behavior=None,
        block2_behavior=None,
    )

    gaps = subject_timestamp_gaps(files, config)
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap["previous_video_frame_position"] == 2
    assert gap["current_video_frame_position"] == 3
    assert gap["previous_capture_frame_idx"] == 2
    assert gap["current_capture_frame_idx"] == 3
    assert gap["missing_capture_frame_indices"] == 0
    assert gap["gap_duration_ms"] == 936
    assert gap["median_interval_ms_subject"] == 32.0
    assert gap["phase_midpoint"] == "block1"
    assert gap["inside_formal_block"] is True
