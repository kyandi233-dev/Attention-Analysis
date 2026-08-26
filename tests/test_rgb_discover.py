from __future__ import annotations

from attention_pipeline.config import load_config
from attention_pipeline.rgb.discover import discover_rgb_subjects


def test_rgb_discovery_finds_unique_subject_and_flags_duplicate(tmp_path):
    root1 = tmp_path / "root1"
    root2 = tmp_path / "root2"
    for root, subject in [(root1, "sub-031_"), (root1, "sub-032_"), (root2, "sub-032_")]:
        rgb = root / subject / "rgb"
        beh = root / subject / "beh"
        rgb.mkdir(parents=True, exist_ok=True)
        beh.mkdir(parents=True, exist_ok=True)
        stem = subject.rstrip("_")
        (rgb / f"{stem}_rgb.avi").write_bytes(b"")
        (rgb / f"{stem}_rgb_timestamps.csv").write_text("0,1000,ok\n", encoding="utf-8")
        (beh / "master_timeline.csv").write_text("event,detail,unix_ms\n", encoding="utf-8")

    config_path = tmp_path / "rgb.yaml"
    config_path.write_text(
        "data:\n"
        f"  roots:\n    - '{root1.as_posix()}'\n    - '{root2.as_posix()}'\n"
        "  min_subject_number: 31\n"
        "  rgb_video_pattern: 'sub-*_/rgb/*_rgb.avi'\n"
        "  rgb_timestamp_suffix: '_rgb_timestamps.csv'\n"
        "  behavior_dir: 'beh'\n"
        "  master_timeline: 'master_timeline.csv'\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    records, duplicates = discover_rgb_subjects(config)
    assert [item.subject for item in records] == ["sub-031"]
    assert "sub-032" in duplicates
