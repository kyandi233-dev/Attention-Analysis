from ritnet_fullclass_final_engine import (
    CORE_VERSION,
    VALIDATION_GEOMETRY_VERSION,
    VALIDATION_WORK_DIRNAME,
)
from ritnet_fullclass_workstore import (
    SCIENTIFIC_IDENTITY_KEYS,
    V8_CORE_VERSION,
    _v8_to_v8_identity_compatible,
)


def _identity(*, validation_version: str | None, git_commit: str) -> dict[str, object]:
    identity: dict[str, object] = {
        key: f"value::{key}" for key in SCIENTIFIC_IDENTITY_KEYS
    }
    identity.update(
        {
            "core_version": V8_CORE_VERSION,
            "git_commit": git_commit,
            "git_branch": "nvidia-cuda-v8-geometry-validation",
            "config_sha256": "config",
        }
    )
    if validation_version is not None:
        identity["validation_geometry_version"] = validation_version
    return identity


def test_geometry_validation_checkpoint_namespace_is_separate_from_production():
    assert CORE_VERSION == V8_CORE_VERSION
    assert VALIDATION_WORK_DIRNAME == ".ritnet-fullclass-geometry-validation-work"
    assert VALIDATION_WORK_DIRNAME != ".ritnet-fullclass-work"


def test_same_validation_geometry_can_resume_across_provenance_only_git_change():
    stored = _identity(validation_version=VALIDATION_GEOMETRY_VERSION, git_commit="a" * 40)
    current = _identity(validation_version=VALIDATION_GEOMETRY_VERSION, git_commit="b" * 40)
    assert _v8_to_v8_identity_compatible(stored, current)


def test_changed_validation_geometry_invalidates_checkpoint():
    stored = _identity(validation_version="shadow-three-path-v0", git_commit="a" * 40)
    current = _identity(validation_version=VALIDATION_GEOMETRY_VERSION, git_commit="b" * 40)
    assert not _v8_to_v8_identity_compatible(stored, current)


def test_production_checkpoint_cannot_be_rebound_as_geometry_validation():
    stored = _identity(validation_version=None, git_commit="a" * 40)
    current = _identity(validation_version=VALIDATION_GEOMETRY_VERSION, git_commit="b" * 40)
    assert not _v8_to_v8_identity_compatible(stored, current)
