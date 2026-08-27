from __future__ import annotations

from ritnet_fullclass_final_completion import completion_work_identity_compatible


def identity():
    return {
        "core_version": "fullclass-final-core-v8-interface-safe-plain-csv",
        "subject": "sub-031",
        "git_commit": "a" * 40,
        "git_branch": "amd-DirectML",
        "config_sha256": "c" * 64,
        "ritnet_model_sha256": "m" * 64,
        "ritnet_external_data_sha256": "d" * 64,
        "source_identity": {"source_video_sha256": "v" * 64},
        "eye_metrics_schema_version": 6,
        "frame_coverage_schema_version": 2,
    }


def test_completion_skip_allows_only_git_provenance_drift():
    stored = identity()
    current = identity()
    current["git_commit"] = "b" * 40
    current["git_branch"] = "renamed-working-branch"
    assert completion_work_identity_compatible(stored, current)


def test_completion_skip_rejects_config_drift():
    stored = identity()
    current = identity()
    current["git_commit"] = "b" * 40
    current["config_sha256"] = "x" * 64
    assert not completion_work_identity_compatible(stored, current)


def test_completion_skip_rejects_core_or_source_drift():
    stored = identity()
    current = identity()
    current["core_version"] = "fullclass-final-core-v9"
    assert not completion_work_identity_compatible(stored, current)

    current = identity()
    current["source_identity"] = {"source_video_sha256": "z" * 64}
    assert not completion_work_identity_compatible(stored, current)
