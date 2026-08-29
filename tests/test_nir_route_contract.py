from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _yaml(name: str):
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_staged_nir_and_legacy_adapter_use_different_manifest_path_keys() -> None:
    ready = _yaml("nir_analysis_ready.yaml")
    paths = _yaml("paths.example.yaml")
    assert ready["paths"]["source_manifest"] == "@path:nir_analysis_ready_source_manifest_json"
    assert "nir_analysis_ready_source_manifest_json" in paths["paths"]
    assert "nir_source_manifest" in paths["paths"]
    assert paths["paths"]["nir_analysis_ready_source_manifest_json"] != paths["paths"]["nir_source_manifest"]
    assert ready["source_contract"]["manifest_format"] == "json_object_with_sessions"


def test_nir_formal_uses_behavior_v2_and_canonical_participant_group() -> None:
    config = _yaml("nir_formal_analysis.yaml")
    assert config["paths"]["behavior_config"] == "configs/behavior_formal_v2.yaml"
    assert config["identity"]["formal_cluster_key"] == "participant_group_id"
    assert config["identity"]["analysis_group_token_role"] == "nir_compatibility_alias_only"
    assert config["analysis_policy"]["behavior_runtime_must_use_shared_portable_path_registry"] is True


def test_nir_behavior_loader_reuses_behavior_runtime_and_parent_path_registry() -> None:
    source = (ROOT / "src/attention_pipeline/nir_behavior/discovery.py").read_text(encoding="utf-8")
    assert "prepare_behavior_runtime_config" in source
    assert "replace(loaded, path_registry=config.path_registry)" in source
    assert "runtime, cohort = prepare_behavior_runtime_config(bconfig)" in source
    assert "extract_formal_trials(runtime, session)" in source
    assert "extract_formal_trials(bconfig" not in source
