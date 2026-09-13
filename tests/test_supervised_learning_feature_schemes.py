from __future__ import annotations

import pandas as pd
import pytest

from attention_pipeline.supervised_learning.feature_schemes import (
    FeatureScheme,
    feature_scheme_from_mapping,
    load_feature_schemes,
    require_scheme_columns,
    validate_mainline_feature_scheme,
)
from attention_pipeline.supervised_learning.task import SupervisedLearningContractError


def test_valid_predeclared_behavior_scheme_uses_raw_omission() -> None:
    scheme = feature_scheme_from_mapping(
        {
            "feature_set_id": "behavior_candidate_a",
            "columns": [
                "go_correct_rt_median_ms",
                "go_correct_rt_cv",
                "go_correct_rt_theilsen_slope_ms_per_s",
                "raw_go_omission_rate",
                "commission_rate",
                "dprime_loglinear",
            ],
            "modality_blocks": ["behavior"],
        }
    )
    assert scheme.feature_set_id == "behavior_candidate_a"
    assert "raw_go_omission_rate" in scheme.columns


def test_structural_leakage_and_audit_columns_fail_closed() -> None:
    forbidden = [
        "q1_nominal_4class",
        "q1_binary",
        "p_q1_equals_1",
        "predicted_q1_binary",
        "participant_group_id",
        "session_id",
        "probe_event_id",
        "probe_id",
        "probe_order_in_block",
        "probe_index_in_block",
        "probe_index_global",
        "probe_time_ms",
        "probe_onset_unix_ms",
        "window_start_unix_ms",
        "window_effective_start_unix_ms",
        "window_end_unix_ms",
        "analysis_set_id",
        "comparison_models",
        "run_id",
        "model_id",
        "outer_fold_group",
        "pupil_geom_mean_diameter_within",
        "pupil_geom_mean_diameter_between",
    ]
    for column in forbidden:
        with pytest.raises(SupervisedLearningContractError):
            validate_mainline_feature_scheme(FeatureScheme("bad", (column,)))


def test_duplicate_features_duplicate_modality_blocks_and_duplicate_scheme_ids_are_rejected() -> None:
    with pytest.raises(SupervisedLearningContractError, match="duplicate columns"):
        validate_mainline_feature_scheme(FeatureScheme("dup", ("x", "x")))

    with pytest.raises(SupervisedLearningContractError, match="duplicate modality_blocks"):
        validate_mainline_feature_scheme(
            FeatureScheme("dup-block", ("x",), modality_blocks=("behavior", "behavior"))
        )

    with pytest.raises(SupervisedLearningContractError, match="feature_set_id values must be unique"):
        load_feature_schemes(
            {
                "candidates": [
                    {"feature_set_id": "same", "columns": ["x"]},
                    {"feature_set_id": "same", "columns": ["y"]},
                ]
            }
        )


def test_upstream_column_contract_is_not_silently_reduced() -> None:
    frame = pd.DataFrame({"participant_group_id": ["P1"], "x": [1.0]})
    scheme = FeatureScheme("needs_xy", ("x", "y"))
    validate_mainline_feature_scheme(scheme)
    with pytest.raises(SupervisedLearningContractError, match="missing upstream columns"):
        require_scheme_columns(frame, scheme)


def test_pending_interface_can_have_no_candidates_until_task_c_d_supply_them() -> None:
    assert load_feature_schemes({"status": "interface_only_pending_task_c_d", "candidates": []}) == []
