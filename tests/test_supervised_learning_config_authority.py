from __future__ import annotations

from pathlib import Path

import yaml


def test_supervised_config_points_to_current_1_16_10_authority() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "supervised_learning_v1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    pipeline = config["pipeline"]
    assert pipeline["evidence_branch"] == "codex/code-fix-ledger"
    assert "1.16.10-监督学习模态与设备定义修订及代码迁移计划" in pipeline["evidence_method_entry"]
    assert "1.16.1-监督学习心理意义、训练权重与多层评价修订" in pipeline["inherited_training_evaluation_entry"]
    assert pipeline["implementation_plan"].endswith("/issues/65")


def test_supervised_config_freezes_reporting_provenance_and_comparability_flags() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "supervised_learning_v1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    outputs = config["outputs"]

    for key in (
        "preserve_feature_comparison_plan",
        "preserve_scientific_modality",
        "preserve_source_namespace",
        "preserve_required_devices",
        "preserve_preprocessing_dependencies",
        "preserve_comparison_role",
        "preserve_includes_behavior_reference",
        "paired_comparisons_require_same_analysis_set",
    ):
        assert outputs[key] is True
    assert outputs["cross_analysis_set_increment_ranking_allowed"] is False

    note = config["feature_registry"]["note"]
    assert "scientific modality" in note
    assert "source_namespace" in note
    assert "required_devices" in note
