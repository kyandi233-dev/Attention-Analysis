"""Atomic deterministic small-artifact I/O for final full-class outputs."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


GZIP_COMPRESSLEVEL = 6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def atomic_write_csv_gz(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: tuple[str, ...] | list[str],
) -> int:
    """Write deterministic gzip CSV and atomically replace destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    count = 0
    raw = None
    gzip_file = None
    text = None
    try:
        raw = temp.open("wb")
        gzip_file = gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            compresslevel=GZIP_COMPRESSLEVEL,
            mtime=0,
        )
        text = io.TextIOWrapper(gzip_file, encoding="utf-8", newline="")
        writer = csv.DictWriter(text, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
            count += 1
        text.flush()
        gzip_file.flush()
        text.detach()
        text = None
        gzip_file.close()
        gzip_file = None
        raw.flush()
        os.fsync(raw.fileno())
        raw.close()
        raw = None
        os.replace(temp, path)
        return count
    finally:
        if text is not None:
            try:
                text.close()
            except Exception:
                pass
        if gzip_file is not None:
            try:
                gzip_file.close()
            except Exception:
                pass
        if raw is not None:
            try:
                raw.close()
            except Exception:
                pass
        if temp.exists():
            temp.unlink()


def iter_csv_gz(path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(Path(path), "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield dict(row)


def csv_gz_fieldnames(path: Path) -> tuple[str, ...]:
    with gzip.open(Path(path), "rt", encoding="utf-8", newline="") as handle:
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
