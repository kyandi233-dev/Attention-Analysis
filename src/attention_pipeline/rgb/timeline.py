from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


_BLOCK_RE = re.compile(r"Block(?P<num>\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class PhaseInterval:
    phase: str
    start_unix_ms: int
    end_unix_ms: int
    source: str


def read_master_timeline(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            raw = str(row.get("unix_ms", "") or "").strip()
            try:
                unix_ms = int(float(raw))
            except ValueError:
                continue
            rows.append(
                {
                    "event": str(row.get("event", "") or "").strip(),
                    "detail": str(row.get("detail", "") or "").strip(),
                    "unix_ms": unix_ms,
                }
            )
    return rows


def _event_times(rows: list[dict[str, object]], event: str) -> list[int]:
    return [int(row["unix_ms"]) for row in rows if row["event"] == event]


def _one_event_time(rows: list[dict[str, object]], event: str) -> int:
    found = _event_times(rows, event)
    if len(found) != 1:
        raise ValueError(f"Expected exactly one {event!r}; found {len(found)}")
    return found[0]


def block_intervals(rows: list[dict[str, object]]) -> dict[int, tuple[int, int]]:
    starts: dict[int, int] = {}
    stops: dict[int, int] = {}
    for row in rows:
        detail = str(row.get("detail", ""))
        match = _BLOCK_RE.search(detail)
        if not match:
            continue
        number = int(match.group("num"))
        event = str(row.get("event", ""))
        if event == "block_start":
            starts[number] = int(row["unix_ms"])
        elif event == "block_stop":
            stops[number] = int(row["unix_ms"])

    result: dict[int, tuple[int, int]] = {}
    for number in sorted(set(starts) | set(stops)):
        if number not in starts or number not in stops:
            raise ValueError(f"Missing block_start/block_stop for Block{number}")
        if stops[number] <= starts[number]:
            raise ValueError(f"Invalid Block{number} interval")
        result[number] = (starts[number], stops[number])
    return result


def formal_rgb_intervals(
    timeline_path: Path,
    *,
    baseline_duration_sec: float = 180.0,
    expected_blocks: int = 2,
) -> list[PhaseInterval]:
    """Resolve the continuous formal RGB analysis span into named intervals.

    The true resting baseline is defined as baseline_stop minus the configured
    duration. Formal blocks come from block_start/block_stop markers in the
    FocusWave master timeline. Gaps between named phases remain available as
    transition intervals instead of being silently assigned to neighboring blocks.
    """
    rows = read_master_timeline(timeline_path)
    baseline_stop = _one_event_time(rows, "baseline_stop")
    baseline_start = baseline_stop - int(round(float(baseline_duration_sec) * 1000.0))
    blocks = block_intervals(rows)
    if len(blocks) < expected_blocks:
        raise ValueError(f"Expected at least {expected_blocks} formal blocks; found {sorted(blocks)}")

    selected_blocks = {number: blocks[number] for number in sorted(blocks)[:expected_blocks]}
    intervals = [
        PhaseInterval("baseline", baseline_start, baseline_stop, "baseline_stop-duration")
    ]

    cursor = baseline_stop
    for number, (start_ms, stop_ms) in selected_blocks.items():
        if start_ms > cursor:
            intervals.append(
                PhaseInterval(
                    f"transition_before_block{number}", cursor, start_ms, "master_timeline_gap"
                )
            )
        intervals.append(
            PhaseInterval(f"block{number}", start_ms, stop_ms, "master_timeline")
        )
        cursor = stop_ms
    return intervals


def formal_analysis_span(
    timeline_path: Path,
    *,
    baseline_duration_sec: float = 180.0,
    expected_blocks: int = 2,
) -> tuple[int, int]:
    intervals = formal_rgb_intervals(
        timeline_path,
        baseline_duration_sec=baseline_duration_sec,
        expected_blocks=expected_blocks,
    )
    return intervals[0].start_unix_ms, intervals[-1].end_unix_ms
