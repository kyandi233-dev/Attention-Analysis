from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ritnet_label_store import RitnetLabelStore, canonical_digest, sha256_file
from ritnet_native_completion import verify_fullclass_completion


def atomicish_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def build_artifacts(tmp_path: Path) -> tuple[Path, dict]:
    identity = {"subject": "sub-031", "source_eyes_sha256": "abc", "git_commit": "deadbeef"}
    store_root = tmp_path / "sub-031_labels"
    store = RitnetLabelStore(
        store_root,
        identity={"resume_identity_digest": canonical_digest(identity)},
        eye_mapping={"frame_left": 0, "frame_right": 1},
        chunk_rows=2,
    )
    labels = np.zeros((2, 400, 640), dtype=np.uint8)
    labels[:, 100:300, 100:500] = 1
    labels[:, 160:240, 250:390] = 2
    labels[:, 185:215, 300:340] = 3
    store.append_chunk(
        labels=labels,
        row_ordinal=np.array([0, 1], dtype=np.int64),
        frame_idx=np.array([100, 101], dtype=np.int64),
        eye=["frame_left", "frame_right"],
        pupil_probability_available=np.ones(2, dtype=np.uint8),
        pupil_probability_stats=np.ones((2, 6), dtype=np.float32),
    )
    report = store.finalize(2)
    assert report.valid

    output_csv = tmp_path / "out.csv"
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["native_label_row_ordinal", "frame_idx", "eye"])
        writer.writeheader()
        writer.writerow({"native_label_row_ordinal": 0, "frame_idx": 100, "eye": "frame_left"})
        writer.writerow({"native_label_row_ordinal": 1, "frame_idx": 101, "eye": "frame_right"})

    qc_index = tmp_path / "qc_index.csv"
    with qc_index.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_idx", "eye", "reason"])
        writer.writeheader()
        writer.writerow({"frame_idx": 100, "eye": "frame_left", "reason": "anchor"})

    summary = tmp_path / "summary.json"
    manifest = tmp_path / "manifest.json"
    atomicish_json(summary, {"rows": 2})
    atomicish_json(manifest, {"identity": identity})

    completion = tmp_path / "completion.json"
    marker = {
        "schema_version": 2,
        "extension_version": "ritnet-fullclass-v2-native640",
        "status": "complete",
        "resume_identity": identity,
        "resume_identity_digest": canonical_digest(identity),
        "label_store_identity_digest": store.identity_digest,
        "expected_rows": 2,
        "processed_rows": 2,
        "stored_label_rows": 2,
        "label_store_verified": True,
        "label_value_domain_verified": True,
        "label_shape_verified": True,
        "label_index_unique_verified": True,
        "label_csv_key_match_verified": True,
        "output_csv": str(output_csv),
        "label_index": str(store.index_path),
        "chunk_manifest": str(store.chunk_manifest_path),
        "store_manifest": str(store.store_manifest_path),
        "summary": str(summary),
        "manifest": str(manifest),
        "qc_index": str(qc_index),
        "label_store_root": str(store_root),
        "output_csv_sha256": sha256_file(output_csv),
        "label_index_sha256": sha256_file(store.index_path),
        "chunk_manifest_sha256": sha256_file(store.chunk_manifest_path),
        "store_manifest_sha256": sha256_file(store.store_manifest_path),
        "summary_sha256": sha256_file(summary),
        "manifest_sha256": sha256_file(manifest),
        "qc_index_sha256": sha256_file(qc_index),
        "artifact_hashes_verified_at_utc": "2026-08-27T00:00:00+00:00",
    }
    atomicish_json(completion, marker)
    return completion, identity


def test_completion_verifies_full_artifact_chain(tmp_path):
    completion, identity = build_artifacts(tmp_path)
    result = verify_fullclass_completion(completion, expected_identity=identity)
    assert result.valid, result.errors


def test_completion_detects_csv_mutation(tmp_path):
    completion, identity = build_artifacts(tmp_path)
    marker = json.loads(completion.read_text(encoding="utf-8"))
    output = Path(marker["output_csv"])
    output.write_text(output.read_text(encoding="utf-8-sig") + "2,102,frame_left\n", encoding="utf-8")
    result = verify_fullclass_completion(completion, expected_identity=identity)
    assert result.valid is False
    assert any("hash mismatch" in error or "CSV" in error for error in result.errors)


def test_completion_detects_qc_index_mutation(tmp_path):
    completion, identity = build_artifacts(tmp_path)
    marker = json.loads(completion.read_text(encoding="utf-8"))
    qc_index = Path(marker["qc_index"])
    qc_index.write_text(qc_index.read_text(encoding="utf-8-sig") + "101,frame_right,manual\n", encoding="utf-8")
    result = verify_fullclass_completion(completion, expected_identity=identity)
    assert result.valid is False
    assert any("qc_index" in error for error in result.errors)


def test_completion_detects_chunk_corruption(tmp_path):
    completion, identity = build_artifacts(tmp_path)
    marker = json.loads(completion.read_text(encoding="utf-8"))
    chunk = Path(marker["label_store_root"]) / "chunks" / "chunk-000000.npz"
    with chunk.open("ab") as handle:
        handle.write(b"oops")
    result = verify_fullclass_completion(completion, expected_identity=identity)
    assert result.valid is False
    assert any("label-store" in error or "hash mismatch" in error for error in result.errors)
