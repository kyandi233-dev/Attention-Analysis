from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase_windows import resolve_phase_windows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_focuswave_v313_windows_exclude_baseline_confirmation_and_practice_countdown(tmp_path):
    subject = tmp_path / "sub-031_"
    video = subject / "nir" / "sub-031_nir.avi"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"")

    _write_csv(
        subject / "beh" / "master_timeline.csv",
        [
            {"event": "baseline_start", "detail": "", "unix_ms": 1000},
            {"event": "baseline_stop", "detail": "duration=2.0s", "unix_ms": 4000},
            {"event": "cover", "detail": "", "unix_ms": 4100},
            {"event": "instructions", "detail": "", "unix_ms": 4200},
            {"event": "practice_start", "detail": "", "unix_ms": 5000},
            {"event": "practice_end", "detail": "", "unix_ms": 8000},
            {"event": "block_start", "detail": "Block1_B", "unix_ms": 9000},
            {"event": "block_stop", "detail": "Block1_B", "unix_ms": 12000},
            {"event": "rest_stop", "detail": "rest=120.0s", "unix_ms": 15000},
            {"event": "block_start", "detail": "Block2_B", "unix_ms": 16000},
            {"event": "block_stop", "detail": "Block2_B", "unix_ms": 19000},
        ],
    )
    _write_csv(
        subject / "beh" / "SART_031_Practice_run1.csv",
        [
            {"trial_num": 1, "absolute_onset_time": 5500},
            {"trial_num": 2, "absolute_onset_time": 6500},
        ],
    )

    # One NIR frame every 100 ms, frame 0 == unix 0.
    unix_by_frame = {idx: idx * 100 for idx in range(220)}
    windows = resolve_phase_windows(
        video,
        unix_by_frame,
        ["baseline", "instructions", "practice", "block1", "block2"],
        baseline_duration_sec=180,
        practice_trial_duration_ms=500,
    )
    by_phase = {window.phase: window for window in windows}

    # baseline_start is at 1000, but true baseline is the final 2 s before stop.
    assert by_phase["baseline"].start_unix_ms == 2000
    assert by_phase["baseline"].end_unix_ms == 4000

    assert by_phase["instructions"].start_unix_ms == 4200
    assert by_phase["instructions"].end_unix_ms == 5000

    # practice_start is 5000, but actual first trial onset is 5500; result page
    # after the last trial is also excluded by last-onset + trial duration.
    assert by_phase["practice"].start_unix_ms == 5500
    assert by_phase["practice"].end_unix_ms == 7000
    assert by_phase["practice"].source == "SART_031_Practice_run1.csv"

    assert by_phase["block1"].start_unix_ms == 9000
    assert by_phase["block1"].end_unix_ms == 12000
    assert by_phase["block2"].start_unix_ms == 16000
    assert by_phase["block2"].end_unix_ms == 19000


def test_missing_requested_block_is_rejected(tmp_path):
    subject = tmp_path / "sub-031_"
    video = subject / "nir" / "sub-031_nir.avi"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"")
    _write_csv(
        subject / "beh" / "master_timeline.csv",
        [
            {"event": "baseline_start", "detail": "", "unix_ms": 0},
            {"event": "baseline_stop", "detail": "duration=1.0s", "unix_ms": 1000},
        ],
    )
    unix_by_frame = {idx: idx * 100 for idx in range(20)}

    try:
        resolve_phase_windows(video, unix_by_frame, ["block2"])
    except ValueError as exc:
        assert "block2" in str(exc).lower()
    else:
        raise AssertionError("missing block2 should fail")
