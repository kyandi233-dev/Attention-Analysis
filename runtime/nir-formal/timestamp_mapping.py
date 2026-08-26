"""Map NIR capture timestamps to sequential frames stored in the AVI."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TimestampMap:
    path: Path
    unix_by_avi_frame: dict[int, int]
    capture_by_avi_frame: dict[int, int]
    status_by_avi_frame: dict[int, str]
    n_source_rows: int
    n_dropped_rows: int
    capture_frame_gap_count: int
    capture_frame_gap_frames: int

    @property
    def avi_frame_count(self) -> int:
        return len(self.unix_by_avi_frame)

    def get(self, avi_frame_idx: int, default: int | None = None) -> int | None:
        """Compatibility accessor for the legacy diagnostic path."""
        return self.unix_by_avi_frame.get(avi_frame_idx, default)

    def metadata(self) -> dict[str, int | str]:
        return {
            "path": str(self.path),
            "frame_index_semantics": "frame_idx=avi_frame_idx; capture_frame_idx=source_counter",
            "avi_frame_count": self.avi_frame_count,
            "timestamp_source_row_count": self.n_source_rows,
            "explicit_dropped_row_count": self.n_dropped_rows,
            "capture_frame_gap_count": self.capture_frame_gap_count,
            "capture_frame_gap_frames": self.capture_frame_gap_frames,
        }


def read_timestamp_map(path: Path) -> TimestampMap:
    """Read source counters and map usable timestamp rows to AVI positions."""
    source_rows: list[tuple[int, int, str]] = []
    seen_capture: set[int] = set()
    previous_capture: int | None = None
    previous_unix: int | None = None

    with path.open(newline="", encoding="utf-8-sig") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row or len(row) < 2:
                continue
            try:
                capture_frame_idx = int(float(row[0]))
                unix_ms = int(float(row[1]))
            except ValueError as exc:
                raise ValueError(f"Invalid timestamp row {line_number}: {path}") from exc

            if capture_frame_idx in seen_capture:
                raise ValueError(f"Duplicate timestamp capture frame {capture_frame_idx}: {path}")
            if previous_capture is not None and capture_frame_idx <= previous_capture:
                raise ValueError(f"NIR capture frame index must increase: {path}")
            if previous_unix is not None and unix_ms <= previous_unix:
                raise ValueError(f"NIR unix_ms must increase: {path}")

            status = row[2].strip().lower() if len(row) >= 3 else ""
            source_rows.append((capture_frame_idx, unix_ms, status))
            seen_capture.add(capture_frame_idx)
            previous_capture = capture_frame_idx
            previous_unix = unix_ms

    if not source_rows:
        raise ValueError(f"No timestamp rows found: {path}")

    gaps = [
        current - previous - 1
        for (previous, _, _), (current, _, _) in zip(source_rows, source_rows[1:])
        if current - previous > 1
    ]
    usable = [row for row in source_rows if row[2] != "dropped"]
    unix_by_avi_frame = {index: row[1] for index, row in enumerate(usable)}
    capture_by_avi_frame = {index: row[0] for index, row in enumerate(usable)}
    status_by_avi_frame = {index: row[2] for index, row in enumerate(usable)}

    return TimestampMap(
        path=path,
        unix_by_avi_frame=unix_by_avi_frame,
        capture_by_avi_frame=capture_by_avi_frame,
        status_by_avi_frame=status_by_avi_frame,
        n_source_rows=len(source_rows),
        n_dropped_rows=len(source_rows) - len(usable),
        capture_frame_gap_count=len(gaps),
        capture_frame_gap_frames=sum(gaps),
    )
