from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from attention_pipeline.config import Config
from attention_pipeline.formal_analysis import behavior_adapter
from attention_pipeline.nir_formal_analysis import identity_audit as nir_identity
from attention_pipeline.nir_pupil_only.contract import adapt_session_rows
from attention_pipeline.path_registry import PathRegistry, load_path_registry
from attention_pipeline.rgb_formal import runner as rgb_runner


def _cohort() -> pd.DataFrame:
    return pd.DataFrame([
        {"session_id": "sub-001", "include": True, "repeat_participant_id": "old-A", "identity_status": "confirmed"},
        {"session_id": "sub-002", "include": True, "repeat_participant_id": "old-A", "identity_status": "confirmed"},
        {"session_id": "sub-003", "include": True, "repeat_participant_id": "old-B", "identity_status": "confirmed"},
        {"session_id": "sub-004", "include": True, "repeat_participant_id": "old-C", "identity_status": "unreviewed"},
        {"session_id": "sub-005", "include": True, "repeat_participant_id": pd.NA, "identity_status": pd.NA},
    ])


def _registry() -> pd.DataFrame:
    return pd.DataFrame([
        {"session_id": "sub-001", "participant_key": "P001", "visit_order": 1, "prior_visit_count": 0, "total_visit_count": 2},
        {"session_id": "sub-005", "participant_key": "P005", "visit_order": 1, "prior_visit_count": 0, "total_visit_count": 1},
    ])


def _config(tmp_path: Path) -> Config:
    registry = PathRegistry(
        path=tmp_path / "paths.local.yaml",
        data={"paths": {"formal_raw_roots": [str(tmp_path)]}},
        digest="test-paths",
    )
    data = {
        "data": {"roots_path_key": "formal_raw_roots"},
        "cohort": {
            "manifest_path_key": "cohort_manifest",
            "session_column": "session_id",
            "include_column": "include",
            "repeat_group_column": "repeat_participant_id",
            "identity_registry_path_key": "repeat_registry",
            "legacy_identity_status_column": "identity_status",
            "allowed_legacy_identity_statuses": ["confirmed"],
        },
        "identity": {
            "cohort_manifest_path_key": "cohort_manifest",
            "repeat_registry_path_key": "repeat_registry",
            "questionnaire_path_key": "questionnaire_derived_data",
            "cohort_session_column": "session_id",
            "cohort_include_column": "include",
            "legacy_repeat_group_column": "repeat_participant_id",
            "legacy_identity_status_column": "identity_status",
            "allowed_legacy_identity_statuses": ["confirmed"],
        },
    }
    return Config(path=tmp_path / "science.yaml", data=data, digest="science", path_registry=registry)


def _group_map(frame: pd.DataFrame) -> dict[str, object]:
    return frame.set_index("session_id")["participant_group_id"].to_dict()


def test_behavior_nir_rgb_share_identical_participant_resolution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cohort = _cohort()
    registry = _registry()
    config = _config(tmp_path)

    monkeypatch.setattr(behavior_adapter, "load_cohort_manifest", lambda *args, **kwargs: cohort.copy())
    monkeypatch.setattr(behavior_adapter, "load_repeat_registry", lambda *args, **kwargs: registry.copy())
    _, behavior_identity = behavior_adapter.prepare_behavior_runtime_config(config)

    monkeypatch.setattr(nir_identity, "load_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(nir_identity, "load_cohort_manifest", lambda *args, **kwargs: cohort.copy())
    monkeypatch.setattr(nir_identity, "load_repeat_registry", lambda *args, **kwargs: registry.copy())
    nir_frame = nir_identity.load_reconciled_identity("unused.yaml")

    monkeypatch.setattr(rgb_runner, "load_cohort_manifest", lambda *args, **kwargs: cohort.copy())
    monkeypatch.setattr(rgb_runner, "load_repeat_registry", lambda *args, **kwargs: registry.copy())
    monkeypatch.setattr(rgb_runner, "load_questionnaire_data", lambda *args, **kwargs: registry.copy())
    monkeypatch.setattr(rgb_runner, "validate_questionnaire_registry_consistency", lambda *args, **kwargs: pd.DataFrame())
    _, rgb_frame, _, _ = rgb_runner._governed_identity(config)

    assert _group_map(behavior_identity) == _group_map(nir_frame) == _group_map(rgb_frame)
    groups = _group_map(behavior_identity)
    assert groups["sub-001"] == "P001"
    assert groups["sub-002"] == "P001"  # unambiguous crosswalk through old-A
    assert groups["sub-003"] == "legacy:old-B"  # governance-approved legacy-only group
    assert pd.isna(groups["sub-004"])  # legacy group exists but governance status is not approved
    assert groups["sub-005"] == "P005"  # verified participant_key works without a legacy group
    assert not any(value == session for session, value in groups.items() if pd.notna(value))


def _eye_metrics_with_oar() -> pd.DataFrame:
    return pd.DataFrame({
        "eye_metrics_schema_version": [7],
        "phase": ["block1"],
        "phase_segment": ["block1"],
        "frame_idx": [1],
        "eye": ["frame_left"],
        "unix_ms": [1000.0],
        "video_time_ms": [1000.0],
        "phase_time_ms": [1000.0],
        "source_eye_status": ["observed"],
        "ritnet_status": ["success"],
        "pupil_found": [True],
        "pupil_fit_valid": [True],
        "pupil_center_x": [20.0],
        "pupil_center_y": [10.0],
        "pupil_geom_mean_diameter": [8.0],
        "fullclass_ocular_aperture_ratio_median": [0.42],
        "fullclass_ocular_aperture_ratio_p90": [0.61],
    })


def test_staged_nir_preserves_ocular_aperture_only_as_qc_candidate() -> None:
    out = adapt_session_rows(
        _eye_metrics_with_oar(),
        {"session_id": "sub-001", "analysis_group_token": "P001", "source_schema_version": 7},
    )
    assert out.loc[0, "fullclass_ocular_aperture_ratio_median"] == pytest.approx(0.42)
    assert out.loc[0, "fullclass_ocular_aperture_ratio_p90"] == pytest.approx(0.61)
    assert bool(out.loc[0, "ocular_aperture_available"])
    assert out.loc[0, "ocular_aperture_role"] == "nir_eye_opening_candidate_qc"
    assert out.loc[0, "ocular_aperture_interpretation"] == "not_ear_not_blink_not_perclos"


def test_path_registry_accepts_current_and_previous_schema_versions(tmp_path: Path) -> None:
    for version in (1, 2, 3):
        path = tmp_path / f"paths-v{version}.yaml"
        path.write_text(f"version: {version}\npaths:\n  x: derived\n", encoding="utf-8")
        registry = load_path_registry(path)
        assert registry.path_value("x") == (tmp_path / "derived").resolve()

    invalid = tmp_path / "paths-v4.yaml"
    invalid.write_text("version: 4\npaths:\n  x: derived\n", encoding="utf-8")
    with pytest.raises(ValueError, match="不支持的路径注册表版本"):
        load_path_registry(invalid)
