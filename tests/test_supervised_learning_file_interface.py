from attention_pipeline.supervised_learning.file_interface import (
    FILE_INTERFACE_SCHEMA_VERSION,
    PAIRED_PROVENANCE_FILENAME,
    PROBE_FEATURE_STATUS_REQUIRED_COLUMNS,
    TASK_A_INPUT_REQUIRED_COLUMNS,
    TASK_A_OUTPUT_FILES,
    TASK_B_ANALYSIS_SET_FILES,
    TASK_B_QUALITY_FILES,
    supervised_file_interface_manifest,
)


def test_machine_readable_file_interface_uses_current_task_b_and_task_a_names():
    manifest = supervised_file_interface_manifest()

    assert manifest["schema_version"] == FILE_INTERFACE_SCHEMA_VERSION
    assert manifest["canonical_probe_key"] == [
        "session_id",
        "block_id",
        "probe_index_in_block",
    ]
    assert manifest["participant_group_column"] == "participant_group_id"
    assert manifest["primary_window_name"] == "pre_30s"

    assert TASK_B_QUALITY_FILES["probe_feature_status"] == "probe_feature_status.csv"
    assert TASK_B_ANALYSIS_SET_FILES["analysis_sets"] == "analysis_sets.csv"
    assert TASK_A_OUTPUT_FILES["probe_predictions"] == "probe_predictions.csv"
    assert TASK_A_OUTPUT_FILES["run_manifest"] == "run_manifest.json"
    assert TASK_A_OUTPUT_FILES["paired_comparison_provenance"] == PAIRED_PROVENANCE_FILENAME


def test_probe_feature_status_contract_keeps_source_namespace_compatibility_field_explicit():
    assert "modality" in PROBE_FEATURE_STATUS_REQUIRED_COLUMNS
    assert "feature" in PROBE_FEATURE_STATUS_REQUIRED_COLUMNS
    assert "feature_computable" in PROBE_FEATURE_STATUS_REQUIRED_COLUMNS
    assert "eligible_for_missing_strategy" in PROBE_FEATURE_STATUS_REQUIRED_COLUMNS
    assert "missing_kind" in PROBE_FEATURE_STATUS_REQUIRED_COLUMNS

    manifest = supervised_file_interface_manifest()
    semantics = manifest["task_b_probe_feature_status_modality_field_semantics"]
    assert "producer/source namespace" in semantics
    assert "scientific modality" in semantics


def test_task_a_input_contract_requires_comparison_specific_identity_fields():
    required = set(TASK_A_INPUT_REQUIRED_COLUMNS)
    assert {
        "analysis_set_id",
        "membership_type",
        "comparison_models",
        "required_features",
        "required_feature_records",
        "required_outcomes",
    } <= required

    manifest = supervised_file_interface_manifest()
    assert manifest["comparison_specific_analysis_set_required"] is True
    assert manifest["cross_analysis_set_raw_increment_ranking_allowed"] is False
