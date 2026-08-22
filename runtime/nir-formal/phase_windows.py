"""Resolve FocusWave v3.1.3 analysis phases onto NIR frame indices.

The formal NIR pipeline intentionally analyses only explicit experimental phases.
For subjects collected with the current BB protocol (sub-031 and later), the
supported phases are:

- baseline: the true 180 s resting interval ending at ``baseline_stop``;
- instructions: ``instructions`` marker until ``practice_start``;
- practice: actual practice trials from the Practice CSV (countdown/results excluded);
- block1 / block2: ``block_start`` .. ``block_stop`` from master_timeline.csv.

Initial camera alignment, cover, inter-block rest/re-alignment, settlement pages,
and trailing recording are not included unless a future phase definition adds them.
"""
from __future__ import annotations

import csv
import re
from bisect import bisect_left
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


_BLOCK_RE = re.compile(r"^Block(?P<num>\d+)_")
_DURATION_RE = re.compile(r"duration=(?P<sec>[0-9]+(?:\.[0-9]+)?)s")


@dataclass(frozen=True)
class PhaseWindow:
    phase: str
    segment: int
    start_unix_ms: int
    end_unix_ms: int
    start_frame_idx: int
    end_frame_idx: int
    n_frames: int
    source: str

    def to_dict(self) -> dict:
        return asdict(self)


def _read_master_timeline(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            event = str(row.get("event", "")).strip()
            detail = str(row.get("detail", "") or "").strip()
            raw = str(row.get("unix_ms", "") or "").strip()
            try:
                unix_ms = int(float(raw))
            except ValueError:
                # Version/info rows at the end of master_timeline are not events.
                continue
            rows.append({"event": event, "detail": detail, "unix_ms": unix_ms})
    return rows


def _one_event(rows: list[dict[str, object]], event: str) -> dict[str, object]:
    found = [row for row in rows if row["event"] == event]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one {event!r}; found {len(found)}")
    return found[0]


def _event_time(rows: list[dict[str, object]], event: str) -> int:
    return int(_one_event(rows, event)["unix_ms"])


def _map_interval_to_frames(
    phase: str,
    segment: int,
    start_ms: int,
    end_ms: int,
    source: str,
    unix_by_frame: dict[int, int],
) -> PhaseWindow:
    if end_ms <= start_ms:
        raise ValueError(f"Invalid {phase} interval: {start_ms}..{end_ms}")
    if not unix_by_frame:
        raise ValueError("Formal phase analysis requires *_nir_timestamps.csv")

    pairs = sorted((int(ts), int(frame_idx)) for frame_idx, ts in unix_by_frame.items())
    times = [item[0] for item in pairs]
    start_pos = bisect_left(times, int(start_ms))
    end_pos = bisect_left(times, int(end_ms)) - 1  # half-open [start, end)
    if start_pos >= len(pairs) or end_pos < start_pos:
        raise ValueError(f"No NIR frames inside {phase} interval {start_ms}..{end_ms}")

    selected = pairs[start_pos : end_pos + 1]
    frame_ids = [frame_idx for _, frame_idx in selected]
    if any(b - a != 1 for a, b in zip(frame_ids, frame_ids[1:])):
        raise ValueError(f"NIR frame-index gap inside {phase} interval")

    return PhaseWindow(
        phase=phase,
        segment=int(segment),
        start_unix_ms=int(start_ms),
        end_unix_ms=int(end_ms),
        start_frame_idx=frame_ids[0],
        end_frame_idx=frame_ids[-1],
        n_frames=len(frame_ids),
        source=source,
    )


def _baseline_interval(rows: list[dict[str, object]], default_duration_sec: float) -> tuple[int, int, str]:
    stop = _one_event(rows, "baseline_stop")
    stop_ms = int(stop["unix_ms"])
    duration_sec = float(default_duration_sec)
    match = _DURATION_RE.search(str(stop.get("detail", "")))
    if match:
        duration_sec = float(match.group("sec"))
    if duration_sec <= 0:
        raise ValueError("Baseline duration must be positive")
    start_ms = stop_ms - int(round(duration_sec * 1000.0))

    # baseline_start is logged before the participant confirms the resting page.
    # It is only a sanity bound; the true resting interval is stop-duration..stop.
    marker_start = _event_time(rows, "baseline_start")
    if start_ms < marker_start:
        raise ValueError(
            f"Derived resting baseline starts before baseline_start marker: {start_ms} < {marker_start}"
        )
    return start_ms, stop_ms, f"baseline_stop-duration({duration_sec:g}s)"


def _instructions_interval(rows: list[dict[str, object]]) -> tuple[int, int, str]:
    return _event_time(rows, "instructions"), _event_time(rows, "practice_start"), "master_timeline"


def _practice_intervals(
    beh_dir: Path,
    rows: list[dict[str, object]],
    trial_duration_ms: int,
) -> list[tuple[int, int, str]]:
    if trial_duration_ms <= 0:
        raise ValueError("practice trial duration must be positive")
    practice_end = _event_time(rows, "practice_end")
    files = sorted(beh_dir.glob("*Practice_run*.csv"))
    intervals: list[tuple[int, int, str]] = []
    for path in files:
        onsets: list[int] = []
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if "absolute_onset_time" not in (reader.fieldnames or []):
                continue
            for row in reader:
                raw = str(row.get("absolute_onset_time", "") or "").strip()
                try:
                    onsets.append(int(float(raw)))
                except ValueError:
                    continue
        if not onsets:
            continue
        start_ms = min(onsets)
        end_ms = min(practice_end, max(onsets) + int(trial_duration_ms))
        if end_ms > start_ms:
            intervals.append((start_ms, end_ms, path.name))
    if intervals:
        return intervals

    # Explicit fallback for atypical/missing practice CSVs. The source is recorded
    # so this is never indistinguishable from trial-timestamp alignment.
    return [
        (
            _event_time(rows, "practice_start"),
            practice_end,
            "master_timeline_fallback_includes_countdown_or_result_page",
        )
    ]


def _block_intervals(rows: list[dict[str, object]]) -> dict[int, tuple[int, int, str]]:
    starts: dict[int, int] = {}
    stops: dict[int, int] = {}
    for row in rows:
        match = _BLOCK_RE.match(str(row.get("detail", "")))
        if not match:
            continue
        number = int(match.group("num"))
        if row["event"] == "block_start":
            starts[number] = int(row["unix_ms"])
        elif row["event"] == "block_stop":
            stops[number] = int(row["unix_ms"])
    result: dict[int, tuple[int, int, str]] = {}
    for number in sorted(set(starts) | set(stops)):
        if number not in starts or number not in stops:
            raise ValueError(f"Missing start/stop marker for Block{number}")
        result[number] = (starts[number], stops[number], "master_timeline")
    return result


def resolve_phase_windows(
    video: Path,
    unix_by_frame: dict[int, int],
    phases: Iterable[str],
    *,
    baseline_duration_sec: float = 180.0,
    practice_trial_duration_ms: int = 1150,
) -> list[PhaseWindow]:
    """Resolve requested FocusWave v3.1.3 phases to inclusive frame windows."""
    subject_dir = video.parent.parent
    beh_dir = subject_dir / "beh"
    timeline_path = beh_dir / "master_timeline.csv"
    rows = _read_master_timeline(timeline_path)
    blocks = _block_intervals(rows)

    requested = [str(value).strip().lower() for value in phases if str(value).strip()]
    windows: list[PhaseWindow] = []
    for phase in requested:
        if phase == "baseline":
            start_ms, end_ms, source = _baseline_interval(rows, baseline_duration_sec)
            windows.append(_map_interval_to_frames(phase, 1, start_ms, end_ms, source, unix_by_frame))
        elif phase == "instructions":
            start_ms, end_ms, source = _instructions_interval(rows)
            windows.append(_map_interval_to_frames(phase, 1, start_ms, end_ms, source, unix_by_frame))
        elif phase == "practice":
            for segment, (start_ms, end_ms, source) in enumerate(
                _practice_intervals(beh_dir, rows, int(practice_trial_duration_ms)), start=1
            ):
                windows.append(_map_interval_to_frames(phase, segment, start_ms, end_ms, source, unix_by_frame))
        elif phase.startswith("block") and phase[5:].isdigit():
            number = int(phase[5:])
            if number not in blocks:
                raise ValueError(f"Requested {phase}, but no matching block markers were found")
            start_ms, end_ms, source = blocks[number]
            windows.append(_map_interval_to_frames(phase, 1, start_ms, end_ms, source, unix_by_frame))
        else:
            raise ValueError(f"Unsupported formal phase: {phase}")

    windows.sort(key=lambda item: (item.start_unix_ms, item.phase, item.segment))
    for previous, current in zip(windows, windows[1:]):
        if current.start_unix_ms < previous.end_unix_ms:
            raise ValueError(
                f"Overlapping formal windows: {previous.phase}/{previous.segment} and "
                f"{current.phase}/{current.segment}"
            )
    return windows
