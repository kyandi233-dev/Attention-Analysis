from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ritnet_label_store import RitnetLabelStore, sha256_file


def labels(n: int, value: int = 0) -> np.ndarray:
    x = np.full((n, 400, 640), value, dtype=np.uint8)
    x[:, 150:250, 200:440] = 1
    x[:, 175:225, 280:360] = 2
    x[:, 190:210, 310:330] = 3
    return x


def make_store(tmp_path: Path, identity=None) -> RitnetLabelStore:
    return RitnetLabelStore(
        tmp_path / "labels",
        identity=identity or {"subject": "sub-031", "source": "abc"},
        eye_mapping={"frame_left": 0, "frame_right": 1},
        chunk_rows=2,
    )


def append(store: RitnetLabelStore, start: int, n: int) -> None:
    stats = np.full((n, 6), 0.9 + start / 100.0, dtype=np.float32)
    store.append_chunk(
        labels=labels(n),
        row_ordinal=np.arange(start, start + n, dtype=np.int64),
        frame_idx=np.arange(100 + start, 100 + start + n, dtype=np.int64),
        eye=["frame_left" if i % 2 == 0 else "frame_right" for i in range(n)],
        pupil_probability_available=np.ones(n, dtype=np.uint8),
        pupil_probability_stats=stats,
    )


def test_store_resume_and_probability_stats_are_checkpointed(tmp_path):
    store = make_store(tmp_path)
    append(store, 0, 2)
    append(store, 2, 1)
    report = store.finalize(3)
    assert report.valid
    assert store.stored_rows == 3

    reopened = make_store(tmp_path)
    assert reopened.next_row_ordinal == 3
    rows = list(reopened.iter_rows())
    assert len(rows) == 3
    assert rows[0]["labels"].shape == (400, 640)
    assert np.isclose(rows[2]["pupil_probability_stats"][0], 0.92)


def test_reopening_finalized_store_is_byte_stable(tmp_path):
    store = make_store(tmp_path)
    append(store, 0, 2)
    store.finalize(2)
    manifest_path = store.store_manifest_path
    before_hash = sha256_file(manifest_path)
    before_text = manifest_path.read_text(encoding="utf-8")
    before_manifest = json.loads(before_text)
    assert before_manifest["status"] == "complete"

    reopened = make_store(tmp_path)
    after_hash = sha256_file(manifest_path)
    after_text = manifest_path.read_text(encoding="utf-8")
    after_manifest = json.loads(after_text)

    assert reopened.verify(expected_rows=2).valid
    assert after_manifest["status"] == "complete"
    assert after_hash == before_hash
    assert after_text == before_text


def test_resume_rejects_identity_mismatch(tmp_path):
    store = make_store(tmp_path)
    append(store, 0, 1)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        make_store(tmp_path, identity={"subject": "sub-031", "source": "different"})


def test_corrupted_chunk_is_detected(tmp_path):
    store = make_store(tmp_path)
    append(store, 0, 1)
    chunk = tmp_path / "labels" / "chunks" / "chunk-000000.npz"
    with chunk.open("ab") as handle:
        handle.write(b"corruption")
    report = store.verify()
    assert report.valid is False
    assert report.chunk_hashes_verified is False


def test_duplicate_frame_eye_is_rejected_by_verification(tmp_path):
    store = make_store(tmp_path)
    store.append_chunk(
        labels=labels(2),
        row_ordinal=np.array([0, 1], dtype=np.int64),
        frame_idx=np.array([100, 100], dtype=np.int64),
        eye=["frame_left", "frame_left"],
        pupil_probability_available=np.ones(2, dtype=np.uint8),
        pupil_probability_stats=np.ones((2, 6), dtype=np.float32),
    )
    report = store.verify()
    assert report.valid is False
    assert report.label_index_unique_verified is False


def test_orphan_committed_chunk_is_adopted_after_metadata_tear(tmp_path):
    store = make_store(tmp_path)
    append(store, 0, 1)
    orphan = tmp_path / "labels" / "chunks" / "chunk-000001.npz"
    with orphan.open("wb") as handle:
        np.savez_compressed(
            handle,
            labels=labels(1),
            row_ordinal=np.array([1], dtype=np.int64),
            frame_idx=np.array([101], dtype=np.int64),
            eye_code=np.array([1], dtype=np.uint8),
            pupil_probability_available=np.ones(1, dtype=np.uint8),
            pupil_probability_stats=np.ones((1, 6), dtype=np.float32),
        )
    reopened = make_store(tmp_path)
    assert reopened.stored_rows == 2
    assert reopened.verify().valid
    manifest = json.loads(reopened.store_manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "running"
