from __future__ import annotations

import sqlite3

import pytest

from ritnet_fullclass_workstore import (
    V7_CORE_VERSION,
    V8_CORE_VERSION,
    FullClassWorkStore,
)


def source_row(frame, eye):
    return {"phase": "block1", "phase_segment": 1, "frame_idx": frame, "eye": eye}


def payload(frame, eye, value):
    return {**source_row(frame, eye), "hard_pupil_fraction": value}


def scientific_identity(core_version, *, subject="sub-031", model_hash="m" * 64):
    return {
        "core_version": core_version,
        "subject": subject,
        "source_identity": {"source_video_sha256": "s" * 64, "eyes_sha256": "e" * 64},
        "git_commit": "a" * 40,
        "git_branch": "amd-DirectML",
        "config_sha256": "c" * 64,
        "ritnet_model_sha256": model_hash,
        "ritnet_external_data_sha256": "d" * 64,
        "ritnet_input": [640, 400],
        "ritnet_batch_size": 16,
        "ritnet_precision": "fp32",
        "class_mapping": {"0": "background", "1": "sclera", "2": "iris", "3": "pupil"},
        "roi_algorithm_version": "roi-v1",
        "valid_source_mask_version": "mask-v1",
        "roi_contract": {
            "target_width": 640,
            "target_height": 400,
            "aspect_ratio": 1.6,
            "expand_horizontal_each_side": 0.30,
            "expand_vertical_each_side": 0.45,
            "padding_mode": "replicate",
        },
        "uncertainty_algorithm_version": "unc-v1",
        "uncertainty_domain_version": "domain-v1",
        "soft_class_fraction_domain_version": "soft-v1",
        "temporal_qc_version": "temporal-v1",
        "eye_metrics_schema_version": 6,
        "frame_coverage_schema_version": 2,
    }


def test_workstore_commits_and_reopens_exact_prefix(tmp_path):
    path = tmp_path / "work.sqlite"
    identity = {"subject": "sub-031", "source": "abc"}
    source = [source_row(10, "frame_left"), source_row(10, "frame_right"), source_row(11, "frame_left")]

    with FullClassWorkStore(path, identity=identity) as store:
        store.append_rows([(0, payload(10, "frame_left", 0.1)), (1, payload(10, "frame_right", 0.2))])
        assert store.validate_prefix(source) == 2

    with FullClassWorkStore(path, identity=identity) as reopened:
        assert reopened.validate_prefix(source) == 2
        rows = list(reopened.iter_rows())
        assert [row["hard_pupil_fraction"] for row in rows] == [0.1, 0.2]


def test_workstore_rejects_different_resume_identity(tmp_path):
    path = tmp_path / "work.sqlite"
    with FullClassWorkStore(path, identity={"subject": "sub-031"}):
        pass
    with pytest.raises(RuntimeError, match="identity digest"):
        FullClassWorkStore(path, identity={"subject": "sub-032"})


def test_workstore_prefix_rejects_source_key_change(tmp_path):
    path = tmp_path / "work.sqlite"
    identity = {"subject": "sub-031"}
    with FullClassWorkStore(path, identity=identity) as store:
        store.append_rows([(0, payload(10, "frame_left", 0.1))])
        with pytest.raises(RuntimeError, match="key mismatch"):
            store.validate_prefix([source_row(10, "frame_right")])


def test_workstore_unique_eye_key_prevents_duplicate_frame_eye(tmp_path):
    path = tmp_path / "work.sqlite"
    with FullClassWorkStore(path, identity={"subject": "sub-031"}) as store:
        store.append_rows([(0, payload(10, "frame_left", 0.1))])
        with pytest.raises(Exception):
            store.append_rows([(1, payload(10, "frame_left", 0.2))])


def test_v7_checkpoint_migrates_to_v8_when_scientific_identity_is_identical(tmp_path):
    path = tmp_path / "work.sqlite"
    stored = scientific_identity(V7_CORE_VERSION)
    source = [source_row(10, "frame_left")]
    with FullClassWorkStore(path, identity=stored) as store:
        store.append_rows([(0, payload(10, "frame_left", 0.1))])

    current = scientific_identity(V8_CORE_VERSION)
    current["git_commit"] = "b" * 40
    current["config_sha256"] = "z" * 64
    current["summary_workers"] = 4

    with FullClassWorkStore(path, identity=current) as migrated:
        assert migrated.validate_prefix(source) == 1
        assert list(migrated.iter_rows())[0]["hard_pupil_fraction"] == 0.1

    con = sqlite3.connect(path)
    meta = dict(con.execute("SELECT key, value FROM meta"))
    con.close()
    assert "resume_migrated_from_identity_digest" in meta
    assert V8_CORE_VERSION in meta["identity_json"]


def test_v7_checkpoint_rejects_v8_migration_when_model_hash_differs(tmp_path):
    path = tmp_path / "work.sqlite"
    stored = scientific_identity(V7_CORE_VERSION)
    with FullClassWorkStore(path, identity=stored):
        pass

    current = scientific_identity(V8_CORE_VERSION, model_hash="x" * 64)
    with pytest.raises(RuntimeError, match="scientific run"):
        FullClassWorkStore(path, identity=current)


def test_v7_checkpoint_rejects_v8_migration_when_source_differs(tmp_path):
    path = tmp_path / "work.sqlite"
    stored = scientific_identity(V7_CORE_VERSION)
    with FullClassWorkStore(path, identity=stored):
        pass

    current = scientific_identity(V8_CORE_VERSION, subject="sub-032")
    with pytest.raises(RuntimeError, match="scientific run"):
        FullClassWorkStore(path, identity=current)
