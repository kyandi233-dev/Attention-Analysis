from __future__ import annotations

import json
import sqlite3

import pytest

from ritnet_fullclass_workstore import V8_CORE_VERSION, FullClassWorkStore


ANALYSIS_VERSION = "source-backed-output-mask-v2-pupil-geometry-only"
UNCERTAINTY_VERSION = "cohort-ocular-mean-only-v1"
UNCERTAINTY_DOMAIN = "source-valid-ocular-mean-only-v1"
SOFT_DOMAIN = "source-valid-class-probability-mean-v1"


def identity(*, model_hash="m" * 64):
    return {
        "core_version": V8_CORE_VERSION,
        "subject": "sub-031",
        "source_identity": {"source_video_sha256": "s" * 64, "source_eyes_sha256": "e" * 64},
        "git_commit": "a" * 40,
        "git_branch": "nvidia-cuda-v8",
        "config_sha256": "c" * 64,
        "execution_backend": "onnxruntime-cuda",
        "execution_provider": "CUDAExecutionProvider",
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
        "analysis_domain_version": ANALYSIS_VERSION,
        "uncertainty_algorithm_version": UNCERTAINTY_VERSION,
        "uncertainty_domain_version": UNCERTAINTY_DOMAIN,
        "soft_class_fraction_domain_version": SOFT_DOMAIN,
        "temporal_qc_version": "temporal-v1",
        "eye_metrics_schema_version": 6,
        "frame_coverage_schema_version": 2,
    }


def source_row(frame=10, eye="frame_left"):
    return {"phase": "block1", "phase_segment": 1, "frame_idx": frame, "eye": eye}


def payload(frame=10, eye="frame_left"):
    return {
        **source_row(frame, eye),
        "eye_metrics_schema_version": 6,
        "ritnet_status": "success",
        "analysis_domain_version": ANALYSIS_VERSION,
        "uncertainty_algorithm_version": UNCERTAINTY_VERSION,
        "uncertainty_domain_version": UNCERTAINTY_DOMAIN,
        "soft_class_fraction_domain_version": SOFT_DOMAIN,
        "hard_pupil_fraction": 0.1,
    }


def meta(path):
    con = sqlite3.connect(path)
    try:
        return dict(con.execute("SELECT key, value FROM meta"))
    finally:
        con.close()


def test_v8_checkpoint_rebinds_only_after_prefix_and_payload_validation(tmp_path):
    path = tmp_path / "work.sqlite"
    stored = identity()
    with FullClassWorkStore(path, identity=stored) as store:
        store.append_rows([(0, payload())])

    before = meta(path)
    current = identity()
    current["git_commit"] = "b" * 40
    current["git_branch"] = "analysis/repaired-cuda-runner"
    current["config_sha256"] = "z" * 64
    current["summary_workers"] = 4
    current["qc_image_max_count"] = 40

    with FullClassWorkStore(path, identity=current) as resumed:
        # Opening alone must never rewrite checkpoint provenance.
        assert meta(path)["identity_json"] == before["identity_json"]
        assert resumed.validate_prefix([source_row()]) == 1
        assert list(resumed.iter_rows())[0]["hard_pupil_fraction"] == 0.1

    after = meta(path)
    assert json.loads(after["identity_json"]) == current
    assert after["resume_migration_kind"] == "v8_to_v8_scientific_identity"
    assert after["resume_migrated_from_identity_digest"] == before["identity_digest"]


def test_v8_checkpoint_bad_prefix_never_mutates_identity(tmp_path):
    path = tmp_path / "work.sqlite"
    stored = identity()
    with FullClassWorkStore(path, identity=stored) as store:
        store.append_rows([(0, payload())])
    before = meta(path)

    current = identity()
    current["git_commit"] = "b" * 40
    with FullClassWorkStore(path, identity=current) as resumed:
        with pytest.raises(RuntimeError, match="key mismatch"):
            resumed.validate_prefix([source_row(frame=11)])

    after = meta(path)
    assert after["identity_json"] == before["identity_json"]
    assert after["identity_digest"] == before["identity_digest"]
    assert "resume_migration_kind" not in after


def test_v8_checkpoint_payload_mismatch_never_mutates_identity(tmp_path):
    path = tmp_path / "work.sqlite"
    stored = identity()
    bad = payload()
    bad["uncertainty_domain_version"] = "wrong"
    with FullClassWorkStore(path, identity=stored) as store:
        store.append_rows([(0, bad)])
    before = meta(path)

    current = identity()
    current["config_sha256"] = "z" * 64
    with FullClassWorkStore(path, identity=current) as resumed:
        with pytest.raises(RuntimeError, match="payload uncertainty_domain_version mismatch"):
            resumed.validate_prefix([source_row()])

    after = meta(path)
    assert after["identity_json"] == before["identity_json"]
    assert after["identity_digest"] == before["identity_digest"]


def test_v8_checkpoint_rejects_changed_numeric_science(tmp_path):
    path = tmp_path / "work.sqlite"
    with FullClassWorkStore(path, identity=identity()) as store:
        store.append_rows([(0, payload())])

    with pytest.raises(RuntimeError, match="scientific run"):
        FullClassWorkStore(path, identity=identity(model_hash="x" * 64))


def test_v8_checkpoint_rejects_directml_execution_identity(tmp_path):
    path = tmp_path / "work.sqlite"
    with FullClassWorkStore(path, identity=identity()) as store:
        store.append_rows([(0, payload())])

    current = identity()
    current["execution_backend"] = "onnxruntime-directml"
    current["execution_provider"] = "DmlExecutionProvider"
    with pytest.raises(RuntimeError, match="scientific run"):
        FullClassWorkStore(path, identity=current)
