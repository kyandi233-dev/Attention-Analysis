"""Transactional temporary checkpoint store for final full-class inference.

This SQLite database is a work artifact, not a final scientific output. It
prevents long RITnet runs from losing completed numeric rows after interruption
without requiring full per-eye segmentation masks on disk.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


WORKSTORE_SCHEMA_VERSION = 1


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def identity_digest(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(identity)).encode("utf-8")).hexdigest()


def _key_from_row(row: Mapping[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(row.get("phase") or ""),
        int(float(row["phase_segment"])),
        int(float(row["frame_idx"])),
        str(row.get("eye") or ""),
    )


class FullClassWorkStore:
    def __init__(self, path: Path, *, identity: Mapping[str, Any]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.identity = dict(identity)
        self.identity_digest = identity_digest(self.identity)
        self.connection = sqlite3.connect(str(self.path))
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._initialize()

    def _initialize(self) -> None:
        with self.connection:
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS eye_rows (
                    row_ordinal INTEGER PRIMARY KEY,
                    phase TEXT NOT NULL,
                    phase_segment INTEGER NOT NULL,
                    frame_idx INTEGER NOT NULL,
                    eye TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    UNIQUE(phase, phase_segment, frame_idx, eye)
                )
                """
            )
        existing = dict(self.connection.execute("SELECT key, value FROM meta"))
        if not existing:
            with self.connection:
                self.connection.executemany(
                    "INSERT INTO meta(key, value) VALUES (?, ?)",
                    [
                        ("schema_version", str(WORKSTORE_SCHEMA_VERSION)),
                        ("identity_json", canonical_json(self.identity)),
                        ("identity_digest", self.identity_digest),
                    ],
                )
            return
        if int(existing.get("schema_version", -1)) != WORKSTORE_SCHEMA_VERSION:
            raise RuntimeError("workstore schema version mismatch")
        if existing.get("identity_digest") != self.identity_digest:
            raise RuntimeError("workstore identity digest differs from current run")
        try:
            stored_identity = json.loads(existing.get("identity_json", ""))
        except Exception as exc:
            raise RuntimeError("workstore identity_json is unreadable") from exc
        if stored_identity != self.identity:
            raise RuntimeError("workstore identity differs from current run")

    @property
    def stored_rows(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM eye_rows").fetchone()[0])

    def validate_prefix(self, source_rows: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...]) -> int:
        """Require stored rows to be exactly source ordinals 0..N-1 with matching keys."""
        stored = list(
            self.connection.execute(
                "SELECT row_ordinal, phase, phase_segment, frame_idx, eye FROM eye_rows ORDER BY row_ordinal"
            )
        )
        if len(stored) > len(source_rows):
            raise RuntimeError("workstore contains more rows than current source eyes")
        for expected_ordinal, record in enumerate(stored):
            ordinal, phase, segment, frame_idx, eye = record
            if int(ordinal) != expected_ordinal:
                raise RuntimeError(
                    f"workstore is not a contiguous prefix at position {expected_ordinal}: ordinal={ordinal}"
                )
            expected_key = _key_from_row(source_rows[expected_ordinal])
            actual_key = (str(phase), int(segment), int(frame_idx), str(eye))
            if actual_key != expected_key:
                raise RuntimeError(
                    f"workstore/source key mismatch at ordinal {expected_ordinal}: "
                    f"stored={actual_key}, source={expected_key}"
                )
        return len(stored)

    def append_rows(self, items: Iterable[tuple[int, Mapping[str, Any]]]) -> None:
        records = []
        for ordinal, row in items:
            phase, segment, frame_idx, eye = _key_from_row(row)
            records.append(
                (
                    int(ordinal),
                    phase,
                    int(segment),
                    int(frame_idx),
                    eye,
                    canonical_json(dict(row)),
                )
            )
        if not records:
            return
        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO eye_rows(
                    row_ordinal, phase, phase_segment, frame_idx, eye, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                records,
            )

    def iter_rows(self) -> Iterator[dict[str, Any]]:
        cursor = self.connection.execute(
            "SELECT payload_json FROM eye_rows ORDER BY row_ordinal"
        )
        for (payload,) in cursor:
            value = json.loads(payload)
            if not isinstance(value, dict):
                raise RuntimeError("workstore payload is not a JSON object")
            yield value

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "FullClassWorkStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
