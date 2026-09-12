"""Machine-readable file/schema interface for the 1.16.10 supervised pipeline.

The human-readable companion is
``docs/supervised_learning/1.16.10-feature-and-file-interface.md``.  This module
keeps stable filenames and minimum column contracts discoverable by tests and future
orchestration code so documentation cannot silently drift from implementation.
"""
from __future__ import annotations

from attention_pipeline.multimodal_formal.alignment import KEY_COLUMNS, PRIMARY_WINDOW
from attention_pipeline.supervised_learning.comparison_provenance import (
    PROVENANCE_SCHEMA_VERSION,
)
from attention_pipeline.supervised_learning.reporting import (
    BOOTSTRAP_FILENAME,
    FAILURES_FILENAME,
    FOLD_AUDITS_FILENAME,
    MANIFEST_FILENAME,
    MODEL_SCORES_FILENAME,
    PAIRED_BOOTSTRAP_FILENAME,
    PAIRED_FOLD_ESTIMABILITY_FILENAME,
    PAIRED_MODEL_INCREMENTS_FILENAME,
    PAIRED_PARTICIPANT_INCREMENTS_FILENAME,
    PARTICIPANT_SCORES_FILENAME,
    PREDICTIONS_FILENAME,
    PROBE_TRAJECTORY_FILENAME,
    SESSION_DISCRIMINATION_FILENAME,
)


FILE_INTERFACE_SCHEMA_VERSION = "1.16.10-file-interface-v1"
PARTICIPANT_GROUP_COLUMN = "participant_group_id"

TASK_B_QUALITY_FILES = {
    "formal_probe_identity": "formal_probe_identity.csv",
    "modality_probe_status": "modality_probe_status.csv",
    "probe_feature_status": "probe_feature_status.csv",
    "feature_coverage": "feature_coverage.csv",
    "modality_availability": "modality_availability.csv",
}
TASK_B_ANALYSIS_SET_FILES = {
    "analysis_sets": "analysis_sets.csv",
    "analysis_set_summary": "analysis_set_summary.csv",
}
PAIRED_PROVENANCE_FILENAME = "paired_comparison_provenance.json"

PROBE_FEATURE_STATUS_REQUIRED_COLUMNS = (
    *KEY_COLUMNS,
    PARTICIPANT_GROUP_COLUMN,
    "modality",  # compatibility field: producer/source namespace, not scientific modality
    "feature",
    "value",
    "feature_computable",
    "eligible_for_missing_strategy",
    "missing_kind",
)

ANALYSIS_SET_REQUIRED_COLUMNS = (
    *KEY_COLUMNS,
    PARTICIPANT_GROUP_COLUMN,
    "analysis_set_id",
    "comparison_models",
    "required_features",
    "required_feature_records",
    "feature_identity_mode",
    "required_outcomes",
    "included_complete",
    "included_missing_aware",
)

TASK_A_INPUT_REQUIRED_COLUMNS = (
    *KEY_COLUMNS,
    PARTICIPANT_GROUP_COLUMN,
    "probe_event_id",
    "q1_nominal_4class",
    "analysis_set_id",
    "membership_type",
    "comparison_models",
    "required_features",
    "required_feature_records",
    "feature_identity_mode",
    "required_outcomes",
)

TASK_A_OUTPUT_FILES = {
    "probe_predictions": PREDICTIONS_FILENAME,
    "fold_audits": FOLD_AUDITS_FILENAME,
    "failures": FAILURES_FILENAME,
    "participant_log_loss": PARTICIPANT_SCORES_FILENAME,
    "model_evaluation": MODEL_SCORES_FILENAME,
    "participant_bootstrap": BOOTSTRAP_FILENAME,
    "paired_participant_increments": PAIRED_PARTICIPANT_INCREMENTS_FILENAME,
    "paired_model_increments": PAIRED_MODEL_INCREMENTS_FILENAME,
    "paired_fold_estimability": PAIRED_FOLD_ESTIMABILITY_FILENAME,
    "paired_increment_bootstrap": PAIRED_BOOTSTRAP_FILENAME,
    "probe_trajectory": PROBE_TRAJECTORY_FILENAME,
    "session_discrimination": SESSION_DISCRIMINATION_FILENAME,
    "paired_comparison_provenance": PAIRED_PROVENANCE_FILENAME,
    "run_manifest": MANIFEST_FILENAME,
}


def supervised_file_interface_manifest() -> dict[str, object]:
    return {
        "schema_version": FILE_INTERFACE_SCHEMA_VERSION,
        "canonical_probe_key": list(KEY_COLUMNS),
        "participant_group_column": PARTICIPANT_GROUP_COLUMN,
        "primary_window_name": PRIMARY_WINDOW,
        "task_b_quality_files": dict(TASK_B_QUALITY_FILES),
        "task_b_probe_feature_status_required_columns": list(
            PROBE_FEATURE_STATUS_REQUIRED_COLUMNS
        ),
        "task_b_probe_feature_status_modality_field_semantics": (
            "compatibility producer/source namespace; scientific modality is supplied by "
            "registry required_feature_records"
        ),
        "task_b_analysis_set_files": dict(TASK_B_ANALYSIS_SET_FILES),
        "task_b_analysis_set_required_columns": list(ANALYSIS_SET_REQUIRED_COLUMNS),
        "task_a_input_required_columns_before_predictors": list(
            TASK_A_INPUT_REQUIRED_COLUMNS
        ),
        "task_a_output_files": dict(TASK_A_OUTPUT_FILES),
        "paired_comparison_provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "comparison_specific_analysis_set_required": True,
        "cross_analysis_set_raw_increment_ranking_allowed": False,
    }
