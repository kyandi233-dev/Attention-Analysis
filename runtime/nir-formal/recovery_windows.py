"""Build auditable task-window mappings for reconstructed timelines.

This module is deliberately modality-neutral. A reconstructed behavior timeline
defines absolute Unix-millisecond block boundaries; each modality maps those
boundaries through its own timestamp file. It never crops or rewrites raw video,
NIR timestamps, or mmWave NPZ files.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RecoveryWindow:
    modality: str
    phase: str
    segment: int
    start_unix_ms: int
    end_unix_ms: int
    start_frame_idx: int | None
    end_frame_idx: int | None
    n_timestamp_rows: int
    frame_index_gap_count: int
    frame_index_gap_frames: int
    contiguous: bool
    source: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def read_reconstructed_blocks(path: Path) -> list[tuple[str, int, int, str]]:
    """Read block_start/block_stop pairs from an external timeline CSV."""
    starts: dict[int, tuple[int, str]] = {}
    stops: dict[int, tuple[int, str]] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            event = str(row.get("event", "")).strip()
            detail = str(row.get("detail", "") or "").strip()
            try:
                timestamp = int(float(str(row.get("unix_ms", "")).strip()))
            except (TypeError, ValueError):
                continue
            number = _block_number(detail)
            if number is None:
                continue
            if event == "reconstructed_block_start" or event == "block_start":
                starts[number] = (timestamp, detail)
            elif event == "reconstructed_block_stop" or event == "block_stop":
                stops[number] = (timestamp, detail)
    numbers = sorted(set(starts) | set(stops))
    if not numbers:
        raise ValueError(f"No block start/stop rows found in {path}")
    blocks: list[tuple[str, int, int, str]] = []
    for number in numbers:
        if number not in starts or number not in stops:
            raise ValueError(f"Missing reconstructed start/stop for Block{number}")
        start, detail = starts[number]
        end, _ = stops[number]
        if end <= start:
            raise ValueError(f"Invalid Block{number} interval: {start}..{end}")
        blocks.append((f"block{number}", start, end, detail))
    return blocks


def _block_number(detail: str) -> int | None:
    text = detail.lower()
    if "block" not in text:
        return None
    digits = ""
    for char in text[text.index("block") + 5 :]:
        if char.isdigit():
            digits += char
        elif digits:
            break
    return int(digits) if digits else None


def read_timestamp_rows(path: Path, *, timestamp_column: int = 1) -> list[tuple[int, int]]:
    """Read ``frame_idx, absolute_unix_ms`` pairs from a modality timestamp CSV."""
    rows: list[tuple[int, int]] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if len(row) <= timestamp_column:
                continue
            try:
                frame_idx = int(float(row[0].strip()))
                unix_ms = int(float(row[timestamp_column].strip()))
            except ValueError:
                if line_number == 1:
                    continue
                raise ValueError(f"Invalid timestamp row {line_number}: {path}")
            rows.append((frame_idx, unix_ms))
    if not rows:
        raise ValueError(f"No timestamp rows found in {path}")
    return rows


def map_timestamp_rows(
    modality: str,
    rows: Iterable[tuple[int, int]],
    blocks: Iterable[tuple[str, int, int, str]],
    *,
    source: str,
) -> list[RecoveryWindow]:
    indexed = sorted((int(frame), int(timestamp)) for frame, timestamp in rows)
    result: list[RecoveryWindow] = []
    for phase, start_ms, end_ms, _detail in blocks:
        selected = [(frame, timestamp) for frame, timestamp in indexed if start_ms <= timestamp < end_ms]
        frame_ids = [frame for frame, _ in selected]
        gaps = [b - a - 1 for a, b in zip(frame_ids, frame_ids[1:]) if b - a > 1]
        result.append(
            RecoveryWindow(
                modality=modality,
                phase=phase,
                segment=1,
                start_unix_ms=start_ms,
                end_unix_ms=end_ms,
                start_frame_idx=frame_ids[0] if frame_ids else None,
                end_frame_idx=frame_ids[-1] if frame_ids else None,
                n_timestamp_rows=len(frame_ids),
                frame_index_gap_count=len(gaps),
                frame_index_gap_frames=sum(gaps),
                contiguous=not gaps,
                source=source,
            )
        )
    return result


def write_recovery_manifest(
    output: Path,
    *,
    subject: str,
    timeline: Path,
    windows: Iterable[RecoveryWindow],
    limitations: str,
) -> Path:
    payload = {
        "schema_version": 1,
        "subject": subject,
        "mode": "reconstructed_task_window_recovery",
        "source_timeline": str(Path(timeline).resolve()),
        "raw_inputs_modified": False,
        "windows": [window.to_dict() for window in windows],
        "limitations": limitations,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output
