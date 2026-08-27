"""Atomic deterministic artifact I/O for final full-class outputs."""
from __future__ import annotations

import csv
import gzip
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def atomic_write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: tuple[str, ...] | list[str],
) -> int:
    """Write a UTF-8 CSV and atomically replace the destination.

    Final eye/frame tables are intentionally plain CSV. Gzip previously added
    avoidable CPU work and made manual inspection/recovery harder; the lean
    scalar schema is already bounded by the per-subject output-size contract.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    count = 0
    try:
        with temp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        return count
    finally:
        if temp.exists():
            temp.unlink()


def iter_csv(path: Path) -> Iterator[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield dict(row)


def csv_fieldnames(path: Path) -> tuple[str, ...]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ())


# Legacy read-only helpers remain so older .csv.gz outputs stay inspectable and
# so stale callers fail safely during the transition. When handed a current
# plain .csv path they delegate to the plain reader and perform no decompression.
def iter_csv_gz(path: Path) -> Iterator[dict[str, str]]:
    path = Path(path)
    if path.suffix.lower() != ".gz":
        yield from iter_csv(path)
        return
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield dict(row)


def csv_gz_fieldnames(path: Path) -> tuple[str, ...]:
    path = Path(path)
    if path.suffix.lower() != ".gz":
        return csv_fieldnames(path)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ())


def artifact_size(path: Path) -> int:
    return int(Path(path).stat().st_size)


def directory_size(path: Path, *, ignore_names: set[str] | None = None) -> int:
    root = Path(path)
    ignored = ignore_names or set()
    total = 0
    if not root.exists():
        return 0
    for item in root.rglob("*"):
        if item.is_file() and item.name not in ignored:
            total += int(item.stat().st_size)
    return total
