import numpy as np
import pandas as pd
import pytest

from attention_pipeline.nir_formal_analysis.rejected_rework import (
    NIRContractError,
    assert_primary_probe_unit,
    audit_brightness_direction,
    build_analysis_ready_timepoints,
    derive_behavior_endpoints,
    figure_denominator_contract,
    incremental_comparison,
    majority_baseline,
    participant_exclusive_folds,
    repeat_estimability,
    safe_model_fit,
    stimulus_history_features,
    validate_window_registry,
)
from attention_pipeline.nir_pipeline_validation.report_admission import (
    FIGURE01_10,
    chinese_figure_contract_ok,
    report_admission,
)


def _pupil_rows():
    rows = []
    for frame_idx, unix_ms in enumerate([1000, 1100, 1200, 1300]):
        for eye, offset in (("left", 0), ("right", 1)):
            rows.append(
                {
                    "subject": "synthetic-subject",
                    "participant_id": "synthetic-participant",
                    "session_id": "synthetic-session",
                    "block_id": "B1",
                    "phase": "block1",
                    "phase_segment": "block1",
                    "frame_idx": frame_idx,
                    "eye": eye,
                    "unix_ms": unix_ms,
                    "pupil_geom_mean_diameter": 10 + frame_idx + offset,
                    "quality_track": "observed",
                }
            )
    return pd.DataFrame(rows)


def test_pupil_bridge_has_no_pir_and_preserves_hierarchy():
    result = build_analysis_ready_timepoints(_pupil_rows())
    assert len(result) == 4
    assert "binocular_pupil_diameter_px" in result
    assert not any("pir" in column.lower() for column in result.columns)
    assert result["participant_id"].eq("synthetic-participant").all()


def test_pir_semantics_fail_closed():
    bad = _pupil_rows().assign(fullclass_pupil_to_iris_diameter_ratio=0.4)
    with pytest.raises(NIRContractError, match="legacy iris/PIR"):
        build_analysis_ready_timepoints(bad)


def test_behavior_endpoints_are_not_merged_correct():
    trials = pd.DataFrame(
        {
            "trial_id": [1, 2, 3, 4],
            "is_go": [True, True, False, False],
            "responded": [True, False, False, True],
            "correct": [1, 0, 1, 0],
        }
    )
    result = derive_behavior_endpoints(trials)
    assert result["endpoint_name"].tolist() == [
        "go_omission",
        "go_omission",
        "nogo_commission",
        "nogo_commission",
    ]
    assert result["endpoint_value"].tolist() == [0, 1, 0, 1]


def test_primary_probe_unit_rejects_duplicate_counting():
    probes = pd.DataFrame(
        {
            "participant_id": ["p1", "p1"],
            "session_id": ["s1", "s1"],
            "block_id": ["B1", "B1"],
            "probe_id": ["q1", "q1"],
        }
    )
    with pytest.raises(NIRContractError, match="one row per probe"):
        assert_primary_probe_unit(probes)


def test_window_registry_has_one_primary_probe_window():
    registry = pd.DataFrame(
        [
            {"window_name": "pre20", "unit": "probe", "role": "primary", "start_offset_ms": -20000, "end_offset_ms": 0},
            {"window_name": "pre10", "unit": "probe", "role": "sensitivity", "start_offset_ms": -10000, "end_offset_ms": 0},
            {"window_name": "pre10trial", "unit": "trial", "role": "sensitivity", "start_offset_ms": -10000, "end_offset_ms": 0},
        ]
    )
    validate_window_registry(registry)
    with pytest.raises(NIRContractError):
        validate_window_registry(registry.assign(role="primary"))


def test_brightness_direction_and_multi_stimulus_history():
    audit_brightness_direction(
        ["history_luminance_mean", "pupil_tonic"], analysis_kind="pre_event_tonic"
    )
    with pytest.raises(NIRContractError, match="current-stimulus"):
        audit_brightness_direction(
            ["history_luminance_mean", "current_relative_luminance"],
            analysis_kind="pre_event_tonic",
        )
    audit_brightness_direction(
        ["current_relative_luminance", "local_baseline_pupil"],
        analysis_kind="post_event_phasic",
    )
    events = pd.DataFrame(
        {
            "onset_ms": [100, 300, 700, 1200],
            "relative_luminance": [0.1, 0.2, 0.3, 0.9],
        }
    )
    features = stimulus_history_features(events, target_ms=1000, start_ms=0, end_ms=1000)
    assert features["history_event_count"] == 3
    assert features["history_sequence_onset_ms"] == (100, 300, 700)
    with pytest.raises(NIRContractError):
        stimulus_history_features(events, target_ms=1000, start_ms=0, end_ms=1200)


def test_participant_exclusive_outer_folds_keep_repeat_sessions_together():
    frame = pd.DataFrame(
        {
            "participant_id": np.repeat([f"p{i}" for i in range(6)], 4),
            "session_id": ["s1", "s1", "s2", "s2"] * 6,
        }
    )
    groups = frame["participant_id"].to_numpy()
    for train, test in participant_exclusive_folds(frame, n_splits=3):
        assert not (set(groups[train]) & set(groups[test]))


def test_imbalance_metrics_and_incremental_design():
    outcome = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    baseline = majority_baseline(outcome)
    assert {"pr_auc", "balanced_accuracy", "brier_score"}.issubset(baseline)
    comparison = incremental_comparison(
        outcome,
        [0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.2, 0.7, 0.8, 0.9],
    )
    assert comparison["model_role"].tolist() == [
        "behavior_design_baseline",
        "behavior_plus_nir",
    ]
    assert "delta_pr_auc" in comparison


def test_model_failure_is_materialized_not_empty_success():
    frame = pd.DataFrame({"participant_id": ["p1", "p2", "p3"]})

    def fail():
        raise np.linalg.LinAlgError("Singular matrix")

    result, failures = safe_model_fit(
        fail,
        model_name="mixed",
        endpoint="go_omission",
        stage="hierarchical_inference",
        input_unit="probe",
        frame=frame,
    )
    assert result is None
    assert failures.loc[0, "error_type"] == "LinAlgError"
    assert failures.loc[0, "admission_status"] == "blocked"


def test_two_block_repeat_is_not_stability_evidence():
    rows = []
    for participant in ("p1", "p2"):
        rows.extend(
            [
                {"participant_id": participant, "session_order": 1, "value": 1.0},
                {"participant_id": participant, "session_order": 2, "value": 1.1},
            ]
        )
    result = repeat_estimability(pd.DataFrame(rows), min_pairs=3)
    assert result["estimable"] is False
    assert "two block values" in result["reason"]


def test_figure_count_units_cannot_share_axis():
    bad = pd.DataFrame(
        [
            {"figure_id": "Figure01", "panel_id": "A", "count_unit": "frame", "denominator": 100, "n": 90},
            {"figure_id": "Figure01", "panel_id": "A", "count_unit": "session", "denominator": 44, "n": 2},
        ]
    )
    with pytest.raises(NIRContractError, match="must not mix"):
        figure_denominator_contract(bad)


def test_figure_registry_01_10_and_chinese_metadata():
    assert [figure.figure_id for figure in FIGURE01_10] == [
        f"Figure{index:02d}" for index in range(1, 11)
    ]
    assert chinese_figure_contract_ok(
        {
            "title_zh": "行为基线与NIR增量",
            "x_label_zh": "预测概率",
            "y_label_zh": "观察比例",
            "n_unit": "participant",
            "effect_label": "效应量",
            "ci_label": "95%置信区间",
        }
    )


def test_report_gate_blocks_failures_and_missing_behavior_v3():
    failures = pd.DataFrame(
        [
            {
                "stage": "model",
                "error_type": "LinAlgError",
                "error_message": "Singular matrix",
                "input_unit": "probe",
                "admission_status": "blocked",
            }
        ]
    )
    models = pd.DataFrame([{"endpoint": "go_omission", "status": "failed"}])
    gate = report_admission(
        failures=failures,
        model_results=models,
        brightness_audit_passed=True,
        probe_unit_audit_passed=True,
        behavior_v3_contract_frozen=False,
    )
    assert gate["admitted"] is False
    assert "blocking_failures_present" in gate["reasons"]


def test_report_gate_can_admit_only_after_all_contracts_pass():
    failures = pd.DataFrame(
        columns=["stage", "error_type", "error_message", "input_unit", "admission_status"]
    )
    models = pd.DataFrame(
        [
            {"endpoint": "go_omission", "status": "success"},
            {"endpoint": "nogo_commission", "status": "success"},
        ]
    )
    gate = report_admission(
        failures=failures,
        model_results=models,
        brightness_audit_passed=True,
        probe_unit_audit_passed=True,
        behavior_v3_contract_frozen=True,
    )
    assert gate["admitted"] is True
