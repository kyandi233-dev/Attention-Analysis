from __future__ import annotations

import sys
from pathlib import Path


RUNTIME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME))

from ritnet_fullclass_final_completion import _scientific_identity


def test_same_result_inputs_with_detached_head_and_branch_are_scientifically_equivalent():
    """A temporary acceptance worktree must not invalidate the same result."""
    recorded = {
        "git_commit": "a00dd0802740662980080c28577dfbb72991deef",
        "git_branch": "HEAD",
        "config_sha256": "config",
        "ritnet_model_sha256": "model",
        "ritnet_external_data_sha256": "external-data",
        "source_identity": {"source_video_sha256": "source"},
    }
    expected = {**recorded, "git_branch": "nvidia-cuda"}

    code = {"runtime/nir-formal/ritnet_fullclass_final_engine.py": "engine"}
    assert _scientific_identity(recorded, result_code_hashes=code) == _scientific_identity(
        expected, result_code_hashes=code
    )


def test_scientific_identity_ignores_git_refs_but_keeps_result_inputs_strict():
    recorded = {
        "git_commit": "a00dd0802740662980080c28577dfbb72991deef",
        "git_branch": "HEAD",
        "config_sha256": "config",
        "ritnet_model_sha256": "model",
        "source_identity": {"source_video_sha256": "source"},
    }
    current = {**recorded, "git_commit": "4d555149c749b18292e4589824a7600ec9a5477f", "git_branch": "nvidia-cuda"}
    code = {"runtime/nir-formal/ritnet_fullclass_final_engine.py": "engine"}

    assert _scientific_identity(recorded, result_code_hashes=code) == _scientific_identity(
        current, result_code_hashes=code
    )
    assert _scientific_identity(recorded, result_code_hashes=code) != _scientific_identity(
        {**current, "config_sha256": "other-config"}, result_code_hashes=code
    )
