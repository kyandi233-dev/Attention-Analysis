"""Transactional, resumable evidence store for 400x640 RITnet hard labels.

The committed NPZ chunks are the source of truth. CSV metadata can be rebuilt
from those chunks after an interrupted metadata commit. Reopening an already
finalized, unchanged store is read-only: it must not rewrite the store manifest
or change any completion hash.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ritnet_native_metrics import NATIVE_LABEL_SHAPE, validate_native_labels

LABEL_STORE_SCHEMA_VERSION = 2
DEFAULT_CHUNK_ROWS = 128
PROBABILITY_STAT_NAMES = ("mean", "median", "p05", "p95", "min", "max")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _read_json(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class LabelStoreVerification:
    valid: bool
    stored_rows: int
    chunk_count: int
    label_shape_verified: bool
    label_dtype_verified: bool
    label_value_domain_verified: bool
    label_index_unique_verified: bool
    chunk_hashes_verified: bool
    index_chunk_match_verified: bool
    errors: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "stored_rows": self.stored_rows,
            "chunk_count": self.chunk_count,
            "label_shape_verified": self.label_shape_verified,
            "label_dtype_verified": self.label_dtype_verified,
            "label_value_domain_verified": self.label_value_domain_verified,
            "label_index_unique_verified": self.label_index_unique_verified,
            "chunk_hashes_verified": self.chunk_hashes_verified,
            "index_chunk_match_verified": self.index_chunk_match_verified,
            "errors": list(self.errors),
        }


class RitnetLabelStore:
    INDEX_FIELDS = [
        "row_ordinal",
        "frame_idx",
        "eye",
        "eye_code",
        "chunk_id",
        "chunk_offset",
        "chunk_file",
    ]
    CHUNK_MANIFEST_FIELDS = [
        "chunk_id",
        "chunk_file",
        "sha256",
        "size_bytes",
        "row_count",
        "first_row_ordinal",
        "last_row_ordinal",
    ]
    _INDEX_INT_FIELDS = {
        "row_ordinal", "frame_idx", "eye_code", "chunk_id", "chunk_offset"
    }
    _CHUNK_INT_FIELDS = {
        "chunk_id", "size_bytes", "row_count", "first_row_ordinal", "last_row_ordinal"
    }

    def __init__(
        self,
        root: Path,
        *,
        identity: dict[str, Any],
        eye_mapping: dict[str, int],
        chunk_rows: int = DEFAULT_CHUNK_ROWS,
        compression: str = "npz_compressed",
    ) -> None:
        self.root = Path(root)
        self.chunks_dir = self.root / "chunks"
        self.index_path = self.root / "label_index.csv"
        self.chunk_manifest_path = self.root / "chunk_manifest.csv"
        self.store_manifest_path = self.root / "store_manifest.json"
        self.identity = identity
        self.identity_digest = canonical_digest(identity)
        self.eye_mapping = {str(k): int(v) for k, v in eye_mapping.items()}
        if not self.eye_mapping or len(set(self.eye_mapping.values())) != len(self.eye_mapping):
            raise ValueError("eye_mapping must be non-empty with unique codes")
        if any(code < 0 or code > 255 for code in self.eye_mapping.values()):
            raise ValueError("eye_mapping codes must fit uint8")
        self.chunk_rows = int(chunk_rows)
        if self.chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        if compression not in {"npz_compressed", "npz_stored"}:
            raise ValueError("compression must be npz_compressed or npz_stored")
        self.compression = compression
        self.index_rows: list[dict[str, Any]] = []
        self.chunk_rows_meta: list[dict[str, Any]] = []
        self._open_or_create()

    @property
    def stored_rows(self) -> int:
        return len(self.index_rows)

    @property
    def next_row_ordinal(self) -> int:
        return self.stored_rows

    def _open_or_create(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        if self.store_manifest_path.exists():
            manifest = _read_json(self.store_manifest_path)
            expected = {
                "schema_version": LABEL_STORE_SCHEMA_VERSION,
                "identity_digest": self.identity_digest,
                "chunk_rows": self.chunk_rows,
                "compression": self.compression,
                "shape": list(NATIVE_LABEL_SHAPE),
                "dtype": "uint8",
                "class_mapping": {
                    "0": "background", "1": "sclera", "2": "iris", "3": "pupil"
                },
                "eye_mapping": self.eye_mapping,
                "probability_stat_names": list(PROBABILITY_STAT_NAMES),
            }
            for key, value in expected.items():
                if manifest.get(key) != value:
                    raise RuntimeError(
                        f"Label-store resume identity mismatch for {key}: "
                        f"stored={manifest.get(key)!r}, current={value!r}"
                    )
            self.index_rows = self._read_index_csv(self.index_path)
            self.chunk_rows_meta = self._read_chunk_manifest_csv(self.chunk_manifest_path)
            repaired = self._repair_metadata_from_committed_chunks()
            report = self.verify()
            if not report.valid:
                raise RuntimeError(
                    "Existing label store failed verification: " + "; ".join(report.errors)
                )
            if repaired:
                # A committed chunk existed beyond the last metadata commit. The old
                # top-level completion is no longer valid until the caller finalizes again.
                self._write_store_manifest(status="running")
            return

        unexpected = [p for p in self.root.iterdir() if p.name != "chunks"]
        unexpected += list(self.chunks_dir.iterdir())
        if unexpected:
            raise RuntimeError(
                "Label-store directory contains artifacts but no store_manifest.json: "
                + ", ".join(str(p) for p in unexpected[:5])
            )
        _atomic_csv(self.index_path, [], self.INDEX_FIELDS)
        _atomic_csv(self.chunk_manifest_path, [], self.CHUNK_MANIFEST_FIELDS)
        self.index_rows = []
        self.chunk_rows_meta = []
        self._write_store_manifest(status="running")

    @classmethod
    def _read_typed_csv(
        cls, path: Path, *, fieldnames: list[str], int_fields: set[str]
    ) -> list[dict[str, Any]]:
        if not path.is_file():
            raise RuntimeError(f"Missing label-store artifact: {path}")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if list(reader.fieldnames or []) != fieldnames:
                raise RuntimeError(
                    f"CSV schema mismatch for {path}: expected={fieldnames}, got={reader.fieldnames}"
                )
            rows: list[dict[str, Any]] = []
            for row_number, raw in enumerate(reader, start=2):
                converted: dict[str, Any] = {}
                for key in fieldnames:
                    value = raw.get(key)
                    if value is None:
                        raise RuntimeError(f"Missing {key} at {path}:{row_number}")
                    if key in int_fields:
                        try:
                            converted[key] = int(value)
                        except ValueError as exc:
                            raise RuntimeError(
                                f"Invalid integer {key}={value!r} at {path}:{row_number}"
                            ) from exc
                    else:
                        converted[key] = str(value)
                rows.append(converted)
            return rows

    @classmethod
    def _read_index_csv(cls, path: Path) -> list[dict[str, Any]]:
        return cls._read_typed_csv(
            path, fieldnames=cls.INDEX_FIELDS, int_fields=cls._INDEX_INT_FIELDS
        )

    @classmethod
    def _read_chunk_manifest_csv(cls, path: Path) -> list[dict[str, Any]]:
        return cls._read_typed_csv(
            path,
            fieldnames=cls.CHUNK_MANIFEST_FIELDS,
            int_fields=cls._CHUNK_INT_FIELDS,
        )

    def _repair_metadata_from_committed_chunks(self) -> bool:
        """Rebuild CSV metadata from committed chunks; return True only if changed."""
        known = {int(row["chunk_id"]): row for row in self.chunk_rows_meta}
        files = sorted(self.chunks_dir.glob("chunk-*.npz"))
        canonical_meta: list[dict[str, Any]] = []
        canonical_index: list[dict[str, Any]] = []
        reverse_eye = {v: k for k, v in self.eye_mapping.items()}
        next_ordinal = 0

        for expected_chunk_id, path in enumerate(files):
            try:
                file_chunk_id = int(path.stem.split("-")[-1])
            except ValueError as exc:
                raise RuntimeError(f"Invalid committed chunk filename: {path}") from exc
            if file_chunk_id != expected_chunk_id:
                raise RuntimeError(
                    f"Committed chunk sequence has a gap: expected {expected_chunk_id}, got {file_chunk_id}"
                )
            with np.load(path, allow_pickle=False) as payload:
                n = int(payload["labels"].shape[0])
                self._validate_chunk_file(
                    path,
                    expected_chunk_rows=n,
                    expected_first_ordinal=next_ordinal,
                )
                ordinals = payload["row_ordinal"]
                frames = payload["frame_idx"]
                eye_codes = payload["eye_code"]
                for offset in range(n):
                    code = int(eye_codes[offset])
                    if code not in reverse_eye:
                        raise RuntimeError(f"Unknown eye_code={code} in {path}")
                    canonical_index.append(
                        {
                            "row_ordinal": int(ordinals[offset]),
                            "frame_idx": int(frames[offset]),
                            "eye": reverse_eye[code],
                            "eye_code": code,
                            "chunk_id": expected_chunk_id,
                            "chunk_offset": offset,
                            "chunk_file": str(path.relative_to(self.root)).replace("\\", "/"),
                        }
                    )
            actual_sha = sha256_file(path)
            old = known.get(expected_chunk_id)
            if old is not None and old["sha256"] != actual_sha:
                raise RuntimeError(f"Committed chunk hash mismatch: {path}")
            canonical_meta.append(
                {
                    "chunk_id": expected_chunk_id,
                    "chunk_file": str(path.relative_to(self.root)).replace("\\", "/"),
                    "sha256": actual_sha,
                    "size_bytes": int(path.stat().st_size),
                    "row_count": n,
                    "first_row_ordinal": next_ordinal,
                    "last_row_ordinal": next_ordinal + n - 1,
                }
            )
            next_ordinal += n

        if len(known) > len(files):
            raise RuntimeError("chunk_manifest references a missing committed chunk")

        changed = canonical_meta != self.chunk_rows_meta or canonical_index != self.index_rows
        if changed:
            _atomic_csv(self.index_path, canonical_index, self.INDEX_FIELDS)
            _atomic_csv(
                self.chunk_manifest_path, canonical_meta, self.CHUNK_MANIFEST_FIELDS
            )
            self.index_rows = canonical_index
            self.chunk_rows_meta = canonical_meta
        return changed

    def _write_store_manifest(self, *, status: str, expected_rows: int | None = None) -> None:
        payload = {
            "schema_version": LABEL_STORE_SCHEMA_VERSION,
            "status": status,
            "identity": self.identity,
            "identity_digest": self.identity_digest,
            "format": "chunked_npz",
            "compression": self.compression,
            "chunk_rows": self.chunk_rows,
            "shape": list(NATIVE_LABEL_SHAPE),
            "dtype": "uint8",
            "class_mapping": {
                "0": "background", "1": "sclera", "2": "iris", "3": "pupil"
            },
            "eye_mapping": self.eye_mapping,
            "probability_stat_names": list(PROBABILITY_STAT_NAMES),
            "probability_summary_note": (
                "Per-row class-3 probability summaries are checkpointed with each chunk; "
                "full probability maps are not stored."
            ),
            "stored_rows": self.stored_rows,
            "chunk_count": len(self.chunk_rows_meta),
            "expected_rows": expected_rows,
            "index_file": self.index_path.name,
            "chunk_manifest_file": self.chunk_manifest_path.name,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "label_index_sha256": sha256_file(self.index_path),
            "chunk_manifest_sha256": sha256_file(self.chunk_manifest_path),
        }
        _atomic_json(self.store_manifest_path, payload)

    def append_chunk(
        self,
        *,
        labels: np.ndarray,
        row_ordinal: np.ndarray,
        frame_idx: np.ndarray,
        eye: list[str],
        pupil_probability_available: np.ndarray,
        pupil_probability_stats: np.ndarray,
    ) -> dict[str, Any]:
        labels = np.asarray(labels)
        if labels.ndim != 3 or labels.shape[1:] != NATIVE_LABEL_SHAPE:
            raise ValueError(f"chunk labels must have shape [N,400,640]; got {labels.shape}")
        if labels.dtype != np.uint8:
            raise TypeError(f"chunk labels must be uint8; got {labels.dtype}")
        n = int(labels.shape[0])
        if n <= 0 or n > self.chunk_rows:
            raise ValueError(f"chunk row count must be 1..{self.chunk_rows}; got {n}")
        for i in range(n):
            validate_native_labels(labels[i])

        ordinals = np.asarray(row_ordinal, dtype=np.int64)
        frames = np.asarray(frame_idx, dtype=np.int64)
        available = np.asarray(pupil_probability_available, dtype=np.uint8)
        stats = np.asarray(pupil_probability_stats, dtype=np.float32)
        if ordinals.shape != (n,) or frames.shape != (n,) or available.shape != (n,):
            raise ValueError("row_ordinal, frame_idx and probability availability must have shape [N]")
        if stats.shape != (n, len(PROBABILITY_STAT_NAMES)):
            raise ValueError(f"pupil_probability_stats must have shape [N,6]; got {stats.shape}")
        if len(eye) != n:
            raise ValueError("eye must contain N values")
        if not np.isin(available, (0, 1)).all():
            raise ValueError("pupil_probability_available values must be 0 or 1")
        if np.isinf(stats).any():
            raise ValueError("pupil_probability_stats must not contain +/-inf")
        for i in range(n):
            finite = np.isfinite(stats[i])
            if not bool(available[i]) and finite.any():
                raise ValueError("unavailable probability row must contain only NaN stats")
            if bool(available[i]) and finite.any() and not finite.all():
                raise ValueError("available probability stats must be either all finite or all NaN")

        expected_ordinals = np.arange(
            self.next_row_ordinal, self.next_row_ordinal + n, dtype=np.int64
        )
        if not np.array_equal(ordinals, expected_ordinals):
            raise ValueError(
                f"chunk ordinals must be contiguous from {self.next_row_ordinal}; got "
                f"{ordinals[:3].tolist()}..."
            )
        try:
            eye_codes = np.asarray(
                [self.eye_mapping[str(value)] for value in eye], dtype=np.uint8
            )
        except KeyError as exc:
            raise ValueError(
                f"eye value not present in frozen eye_mapping: {exc.args[0]!r}"
            ) from exc

        chunk_id = len(self.chunk_rows_meta)
        final_path = self.chunks_dir / f"chunk-{chunk_id:06d}.npz"
        if final_path.exists():
            raise FileExistsError(final_path)
        temp_path = self.chunks_dir / f".{final_path.name}.{uuid.uuid4().hex}.tmp"
        try:
            with temp_path.open("wb") as handle:
                writer = np.savez_compressed if self.compression == "npz_compressed" else np.savez
                writer(
                    handle,
                    labels=np.ascontiguousarray(labels),
                    row_ordinal=ordinals,
                    frame_idx=frames,
                    eye_code=eye_codes,
                    pupil_probability_available=available,
                    pupil_probability_stats=stats,
                )
                handle.flush()
                os.fsync(handle.fileno())
            self._validate_chunk_file(
                temp_path,
                expected_chunk_rows=n,
                expected_first_ordinal=int(ordinals[0]),
            )
            chunk_sha = sha256_file(temp_path)
            os.replace(temp_path, final_path)
            if sha256_file(final_path) != chunk_sha:
                raise RuntimeError("chunk SHA256 changed across atomic rename")
        finally:
            if temp_path.exists():
                temp_path.unlink()

        chunk_rel = str(final_path.relative_to(self.root)).replace("\\", "/")
        chunk_meta = {
            "chunk_id": chunk_id,
            "chunk_file": chunk_rel,
            "sha256": chunk_sha,
            "size_bytes": int(final_path.stat().st_size),
            "row_count": n,
            "first_row_ordinal": int(ordinals[0]),
            "last_row_ordinal": int(ordinals[-1]),
        }
        new_index_rows = list(self.index_rows)
        for offset in range(n):
            new_index_rows.append(
                {
                    "row_ordinal": int(ordinals[offset]),
                    "frame_idx": int(frames[offset]),
                    "eye": str(eye[offset]),
                    "eye_code": int(eye_codes[offset]),
                    "chunk_id": chunk_id,
                    "chunk_offset": offset,
                    "chunk_file": chunk_rel,
                }
            )
        new_chunk_meta = list(self.chunk_rows_meta) + [chunk_meta]
        _atomic_csv(self.index_path, new_index_rows, self.INDEX_FIELDS)
        _atomic_csv(
            self.chunk_manifest_path, new_chunk_meta, self.CHUNK_MANIFEST_FIELDS
        )
        self.index_rows = new_index_rows
        self.chunk_rows_meta = new_chunk_meta
        self._write_store_manifest(status="running")
        return chunk_meta

    def _validate_chunk_file(
        self,
        path: Path,
        *,
        expected_chunk_rows: int | None = None,
        expected_first_ordinal: int | None = None,
    ) -> int:
        with np.load(path, allow_pickle=False) as payload:
            required = {
                "labels", "row_ordinal", "frame_idx", "eye_code",
                "pupil_probability_available", "pupil_probability_stats",
            }
            if set(payload.files) != required:
                raise ValueError(f"chunk keys mismatch: {sorted(payload.files)}")
            labels = payload["labels"]
            ordinals = payload["row_ordinal"]
            frames = payload["frame_idx"]
            eye_codes = payload["eye_code"]
            available = payload["pupil_probability_available"]
            stats = payload["pupil_probability_stats"]
            if labels.ndim != 3 or labels.shape[1:] != NATIVE_LABEL_SHAPE or labels.dtype != np.uint8:
                raise ValueError(f"invalid labels array in {path}: {labels.shape} {labels.dtype}")
            n = int(labels.shape[0])
            if expected_chunk_rows is not None and n != expected_chunk_rows:
                raise ValueError(f"chunk row count mismatch: expected {expected_chunk_rows}, got {n}")
            if ordinals.shape != (n,) or ordinals.dtype != np.int64:
                raise ValueError("invalid row_ordinal array")
            if frames.shape != (n,) or frames.dtype != np.int64:
                raise ValueError("invalid frame_idx array")
            if eye_codes.shape != (n,) or eye_codes.dtype != np.uint8:
                raise ValueError("invalid eye_code array")
            if available.shape != (n,) or available.dtype != np.uint8 or not np.isin(available, (0, 1)).all():
                raise ValueError("invalid pupil_probability_available array")
            if stats.shape != (n, len(PROBABILITY_STAT_NAMES)) or stats.dtype != np.float32:
                raise ValueError("invalid pupil_probability_stats array")
            if np.isinf(stats).any():
                raise ValueError("pupil_probability_stats contains infinity")
            if expected_first_ordinal is not None:
                expected = np.arange(
                    expected_first_ordinal, expected_first_ordinal + n, dtype=np.int64
                )
                if not np.array_equal(ordinals, expected):
                    raise ValueError("chunk row ordinals are not contiguous")
            for i in range(n):
                validate_native_labels(labels[i])
                finite = np.isfinite(stats[i])
                if not bool(available[i]) and finite.any():
                    raise ValueError("unavailable probability row has finite stats")
                if bool(available[i]) and finite.any() and not finite.all():
                    raise ValueError("probability stats are partially missing")
            return n

    def verify(self, expected_rows: int | None = None) -> LabelStoreVerification:
        errors: list[str] = []
        shape_ok = dtype_ok = domain_ok = hashes_ok = index_unique_ok = index_match_ok = True
        stored = 0
        try:
            index_rows = self._read_index_csv(self.index_path)
            chunk_meta = self._read_chunk_manifest_csv(self.chunk_manifest_path)
        except Exception as exc:
            return LabelStoreVerification(
                False, 0, 0, False, False, False, False, False, False, (str(exc),)
            )

        ordinals_seen: list[int] = []
        keys_seen: set[tuple[int, str]] = set()
        expected_index_rows: list[tuple[int, int, str, int, int, str]] = []
        reverse_eye = {v: k for k, v in self.eye_mapping.items()}

        for expected_chunk_id, meta in enumerate(chunk_meta):
            try:
                chunk_id = int(meta["chunk_id"])
                if chunk_id != expected_chunk_id:
                    raise ValueError(f"chunk id sequence break: {chunk_id} != {expected_chunk_id}")
                expected_rel = f"chunks/chunk-{chunk_id:06d}.npz"
                if meta["chunk_file"] != expected_rel:
                    raise ValueError(f"unexpected chunk path: {meta['chunk_file']}")
                path = self.root / meta["chunk_file"]
                if not path.is_file():
                    raise FileNotFoundError(path)
                if int(meta["size_bytes"]) != int(path.stat().st_size):
                    raise ValueError(f"chunk size mismatch: {path}")
                if sha256_file(path) != meta["sha256"]:
                    hashes_ok = False
                    raise ValueError(f"chunk hash mismatch: {path}")
                first = int(meta["first_row_ordinal"])
                rows = int(meta["row_count"])
                if int(meta["last_row_ordinal"]) != first + rows - 1:
                    raise ValueError(f"chunk ordinal range mismatch: {path}")
                self._validate_chunk_file(
                    path, expected_chunk_rows=rows, expected_first_ordinal=first
                )
                with np.load(path, allow_pickle=False) as payload:
                    labels = payload["labels"]
                    if labels.shape[1:] != NATIVE_LABEL_SHAPE:
                        shape_ok = False
                    if labels.dtype != np.uint8:
                        dtype_ok = False
                    if not np.isin(np.unique(labels), (0, 1, 2, 3)).all():
                        domain_ok = False
                    ordinals = payload["row_ordinal"]
                    frames = payload["frame_idx"]
                    codes = payload["eye_code"]
                    for offset in range(rows):
                        eye_text = reverse_eye.get(int(codes[offset]))
                        if eye_text is None:
                            raise ValueError(f"unknown eye_code={int(codes[offset])}")
                        expected_index_rows.append(
                            (
                                int(ordinals[offset]), int(frames[offset]), eye_text,
                                chunk_id, offset, expected_rel,
                            )
                        )
                stored += rows
            except Exception as exc:
                errors.append(str(exc))

        for row in index_rows:
            try:
                ordinal = int(row["row_ordinal"])
                frame = int(row["frame_idx"])
                eye_text = str(row["eye"])
                chunk_id = int(row["chunk_id"])
                offset = int(row["chunk_offset"])
                chunk_file = str(row["chunk_file"])
                ordinals_seen.append(ordinal)
                key = (frame, eye_text)
                if key in keys_seen:
                    index_unique_ok = False
                    errors.append(f"duplicate label index key frame/eye={key}")
                keys_seen.add(key)
                if self.eye_mapping.get(eye_text) != int(row["eye_code"]):
                    index_match_ok = False
                    errors.append(f"eye mapping mismatch at row {ordinal}")
                expected = (
                    ordinal, frame, eye_text, chunk_id, offset, chunk_file
                )
                if ordinal >= len(expected_index_rows) or expected_index_rows[ordinal] != expected:
                    index_match_ok = False
                    errors.append(f"index/chunk mismatch at row {ordinal}")
            except Exception as exc:
                index_match_ok = False
                errors.append(f"invalid index row: {exc}")

        if ordinals_seen != list(range(len(index_rows))):
            index_unique_ok = False
            errors.append("label index row_ordinal is not unique contiguous 0..N-1")
        if len(index_rows) != stored:
            index_match_ok = False
            errors.append(f"index row count {len(index_rows)} != chunk row count {stored}")
        if expected_rows is not None and stored != int(expected_rows):
            errors.append(f"stored rows {stored} != expected rows {expected_rows}")

        try:
            manifest = _read_json(self.store_manifest_path)
            if manifest.get("label_index_sha256") != sha256_file(self.index_path):
                hashes_ok = False
                errors.append("store_manifest label_index_sha256 mismatch")
            if manifest.get("chunk_manifest_sha256") != sha256_file(self.chunk_manifest_path):
                hashes_ok = False
                errors.append("store_manifest chunk_manifest_sha256 mismatch")
        except Exception as exc:
            hashes_ok = False
            errors.append(f"store_manifest verification failed: {exc}")

        valid = bool(
            not errors and shape_ok and dtype_ok and domain_ok and hashes_ok
            and index_unique_ok and index_match_ok
        )
        return LabelStoreVerification(
            valid, stored, len(chunk_meta), shape_ok, dtype_ok, domain_ok,
            index_unique_ok, hashes_ok, index_match_ok, tuple(errors)
        )

    def iter_rows(self) -> Iterator[dict[str, Any]]:
        reverse_eye = {v: k for k, v in self.eye_mapping.items()}
        for meta in self.chunk_rows_meta:
            chunk_id = int(meta["chunk_id"])
            path = self.root / str(meta["chunk_file"])
            with np.load(path, allow_pickle=False) as payload:
                labels = payload["labels"]
                ordinals = payload["row_ordinal"]
                frames = payload["frame_idx"]
                codes = payload["eye_code"]
                available = payload["pupil_probability_available"]
                stats = payload["pupil_probability_stats"]
                for offset in range(labels.shape[0]):
                    yield {
                        "row_ordinal": int(ordinals[offset]),
                        "frame_idx": int(frames[offset]),
                        "eye": reverse_eye[int(codes[offset])],
                        "eye_code": int(codes[offset]),
                        "chunk_id": chunk_id,
                        "chunk_offset": offset,
                        "chunk_file": str(meta["chunk_file"]),
                        "labels": np.ascontiguousarray(labels[offset]),
                        "pupil_probability_available": bool(available[offset]),
                        "pupil_probability_stats": np.asarray(stats[offset], dtype=np.float32),
                    }

    def finalize(self, expected_rows: int) -> LabelStoreVerification:
        report = self.verify(expected_rows=expected_rows)
        if not report.valid:
            raise RuntimeError(
                "Cannot finalize invalid label store: " + "; ".join(report.errors)
            )
        self._write_store_manifest(status="complete", expected_rows=int(expected_rows))
        manifest = _read_json(self.store_manifest_path)
        if manifest.get("status") != "complete":
            raise RuntimeError("store manifest did not finalize as complete")
        if int(manifest.get("expected_rows", -1)) != int(expected_rows):
            raise RuntimeError("store manifest expected_rows mismatch after finalization")
        if manifest.get("label_index_sha256") != sha256_file(self.index_path):
            raise RuntimeError("label index hash mismatch after finalization")
        if manifest.get("chunk_manifest_sha256") != sha256_file(self.chunk_manifest_path):
            raise RuntimeError("chunk manifest hash mismatch after finalization")
        return report
