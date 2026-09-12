from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from attention_pipeline.supervised_learning.entrypoint import run_supervised_from_config
from attention_pipeline.supervised_learning.task import SupervisedLearningContractError


def _probe_table() -> pd.DataFrame:
    rows = []
    for group_index in range(6):
        group = f"P{group_index:02d}"
        session = f"S{group_index:02d}"
        for probe, q1 in enumerate((1, 2, 1, 3), start=1):
            rows.append(
                {
                    "participant_group_id": group,
                    "session_id": session,
                    "block_id": "B1",
                    "probe_event_id": f"{session}|B1|P{probe}",
                    "q1_nominal_4class": q1,
                    "analysis_set_id": "synthetic-common-set",
                    "membership_type": "included_missing_aware",
                    "required_outcomes": '["q1_nominal_4class"]',
                    "behavior_signal": 1.0 if q1 == 1 else -1.0,
                }
            )
    return pd.DataFrame(rows)


def _config(input_path: Path, output_root: Path) -> tuple[dict, dict]:
    config = {
        "pipeline": {"name": "focuswave-supervised-learning-v1", "version": "test", "random_seed": 7},
        "paths": {
            "input_table": {"path_key": "supervised_learning_input_probe_table"},
            "output_root": {"path_key": "supervised_learning_output_root"},
        },
        "task": {
            "name": "q1_equals_1_vs_2_3_4",
            "analysis_unit": "probe_preceding_window",
            "primary_window_seconds": 30,
            "source_column": "q1_nominal_4class",
            "positive_values": [1],
            "negative_values": [2, 3, 4],
            "positive_label": 1,
            "negative_label": 0,
            "positive_probability_name": "p_q1_equals_1",
        },
        "validation": {
            "outer": {"method": "leave_one_participant_out", "group_column": "participant_group_id", "participant_disjoint": True},
            "inner": {"method": "grouped_k_fold", "n_splits": 5, "group_column": "participant_group_id", "refit_preprocessing_per_split": True},
            "zero_individual_calibration": True,
            "forbid_test_participant_sequence_statistics": True,
            "forbid_test_participant_future_information": True,
            "require_analysis_set_id": True,
        },
        "preprocessing": {
            "participant_equal_weighted_median_imputation_fit_on_training_only": True,
            "participant_equal_standardization_fit_on_training_only": True,
            "data_dependent_column_handling_fit_on_training_only": True,
            "participant_equal_training_weights_normalized_to_mean_one": True,
            "unified_global_coverage_cutoff": None,
            "participant_specific_within_between_mainline": False,
        },
        "feature_schemes": {
            "model_families": {
                "behavior": {
                    "candidates": [
                        {"feature_set_id": "behavior_signal_v1", "columns": ["behavior_signal"], "modality_blocks": ["behavior"]}
                    ]
                }
            }
        },
        "models": {
            "primary": {"kind": "logistic_l2", "C_candidates": [0.1, 1.0], "max_iter": 500},
            "selection_metric": "participant_macro_log_loss",
        },
        "uncertainty": {
            "participant_cluster_bootstrap": {
                "method": "fixed_oof_participant_cluster_percentile",
                "replicates": 1000,
                "seed": 20260830,
                "confidence_level": 0.95,
                "paired_model_resampling": True,
                "retrain_within_bootstrap": False,
            }
        },
        "outputs": {"preserve_failures": True},
    }
    paths = {
        "version": 3,
        "paths": {
            "supervised_learning_input_probe_table": str(input_path),
            "supervised_learning_output_root": str(output_root),
        },
    }
    return config, paths


def _write_configs(tmp_path: Path, config_data: dict, paths_data: dict) -> tuple[Path, Path]:
    config_path = tmp_path / "supervised.yaml"
    paths_path = tmp_path / "paths.local.yaml"
    config_path.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")
    paths_path.write_text(yaml.safe_dump(paths_data, sort_keys=False), encoding="utf-8")
    return config_path, paths_path


def _registry_config() -> dict:
    return {
        "features": [
            {
                "feature_id": "behavior_signal",
                "scientific_feature_id": "behavior_signal",
                "columns": ["behavior_signal"],
                "role": "behavior",
                "modality": "behavior",
                "raw_source": "synthetic behavior",
                "source_namespace": "behavior",
                "required_devices": [],
                "behavior_reference_eligible": True,
                "behavior_increment_eligible": False,
                "allowed_device_packages": [],
            },
            {
                "feature_id": "rgb_signal",
                "scientific_feature_id": "rgb_signal",
                "columns": ["rgb_signal"],
                "role": "sensor",
                "modality": "ocular",
                "raw_source": "synthetic RGB",
                "source_namespace": "rgb",
                "required_devices": ["rgb"],
                "behavior_increment_eligible": True,
                "modality_model_eligible": True,
                "allowed_device_packages": ["M3"],
            },
        ]
    }


def test_run_supervised_from_config_end_to_end(tmp_path) -> None:
    input_path = tmp_path / "probe_table.csv"
    output_root = tmp_path / "outputs"
    _probe_table().to_csv(input_path, index=False)
    config_data, paths_data = _config(input_path, output_root)
    config_path, paths_path = _write_configs(tmp_path, config_data, paths_data)

    manifest = run_supervised_from_config(config_path, paths_config=paths_path, run_id="synthetic-run")

    run_root = output_root / "synthetic-run"
    assert manifest["status"] == "complete"
    assert manifest["analysis_set_id"] == "synthetic-common-set"
    assert manifest["membership_type"] == "included_missing_aware"
    assert manifest["n_prediction_rows"] == len(_probe_table())
    assert manifest["outer_test_outcomes_passed_to_model"] is False
    assert manifest["outer_evaluation"]["bootstrap_replicates"] == 1000
    assert manifest["outer_evaluation"]["bootstrap_seed"] == 20260830
    assert manifest["analysis_set_required_outcomes"] == ["q1_nominal_4class"]
    assert manifest["analysis_set_outcome_scope_verified"] is True
    assert (run_root / "probe_predictions.csv").is_file()
    assert (run_root / "participant_log_loss.csv").is_file()
    assert (run_root / "model_evaluation.csv").is_file()
    assert (run_root / "participant_bootstrap.json").is_file()
    assert (run_root / "fold_audits.json").is_file()
    assert (run_root / "failures.csv").is_file()
    saved = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    assert saved["provenance"]["input_table"] == str(input_path)
    assert len(saved["provenance"]["input_sha256"]) == 64


def test_direct_task_a_api_rejects_extra_q2_sample_filter(tmp_path) -> None:
    input_path = tmp_path / "probe_table.csv"
    output_root = tmp_path / "outputs"
    frame = _probe_table()
    frame["required_outcomes"] = '["q1_nominal_4class", "q2_ordinal_4level"]'
    frame.to_csv(input_path, index=False)
    config_data, paths_data = _config(input_path, output_root)
    config_path, paths_path = _write_configs(tmp_path, config_data, paths_data)

    with pytest.raises(
        SupervisedLearningContractError,
        match="does not match the frozen Task-A target source",
    ):
        run_supervised_from_config(config_path, paths_config=paths_path, run_id="blocked-q2-filter")


def test_registry_run_consumes_only_models_declared_by_current_analysis_set(tmp_path) -> None:
    input_path = tmp_path / "probe_table.csv"
    output_root = tmp_path / "outputs"
    frame = _probe_table()
    frame["rgb_signal"] = frame["behavior_signal"] * 0.5
    declared = ["behavior_reference", "behavior_plus::rgb_signal"]
    frame["comparison_models"] = json.dumps(declared)
    frame["required_features"] = json.dumps({"behavior": ["behavior_signal"], "ocular": ["rgb_signal"]})
    frame.to_csv(input_path, index=False)

    config_data, paths_data = _config(input_path, output_root)
    config_data["feature_registry"] = _registry_config()
    config_path, paths_path = _write_configs(tmp_path, config_data, paths_data)

    manifest = run_supervised_from_config(config_path, paths_config=paths_path, run_id="registry-run")
    saved_predictions = pd.read_csv(output_root / "registry-run" / "probe_predictions.csv")

    assert manifest["status"] == "complete"
    assert set(saved_predictions["model_id"]) == set(declared)
    assert set(saved_predictions["membership_type"]) == {"included_missing_aware"}
    assert manifest["n_prediction_rows"] == len(frame) * len(declared)
    saved_manifest = json.loads((output_root / "registry-run" / "run_manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["analysis_set_declared_models"] == declared
    assert saved_manifest["analysis_set_feature_scope_verified"] is True
    assert saved_manifest["analysis_set_required_feature_columns"] == ["behavior_signal", "rgb_signal"]
    assert saved_manifest["declared_model_predictor_union"] == ["behavior_signal", "rgb_signal"]
    assert saved_manifest["analysis_set_required_outcomes"] == ["q1_nominal_4class"]
    assert saved_manifest["analysis_set_outcome_scope_verified"] is True


def test_registry_run_rejects_model_not_declared_in_frozen_registry(tmp_path) -> None:
    input_path = tmp_path / "probe_table.csv"
    output_root = tmp_path / "outputs"
    frame = _probe_table()
    frame["comparison_models"] = json.dumps(["behavior_reference", "not_in_registry"])
    frame["required_features"] = json.dumps({"behavior": ["behavior_signal"]})
    frame.to_csv(input_path, index=False)

    config_data, paths_data = _config(input_path, output_root)
    config_data["feature_registry"] = {
        "features": [
            {
                "feature_id": "behavior_signal",
                "scientific_feature_id": "behavior_signal",
                "columns": ["behavior_signal"],
                "role": "behavior",
                "modality": "behavior",
                "raw_source": "synthetic behavior",
                "source_namespace": "behavior",
                "required_devices": [],
                "behavior_reference_eligible": True,
                "allowed_device_packages": [],
            }
        ]
    }
    config_path, paths_path = _write_configs(tmp_path, config_data, paths_data)

    with pytest.raises(SupervisedLearningContractError, match="absent from frozen feature registry"):
        run_supervised_from_config(config_path, paths_config=paths_path, run_id="blocked-registry")


def test_registry_run_rejects_extra_sample_filter_feature(tmp_path) -> None:
    input_path = tmp_path / "probe_table.csv"
    output_root = tmp_path / "outputs"
    frame = _probe_table()
    frame["rgb_signal"] = frame["behavior_signal"] * 0.5
    frame["unused_filter"] = 1.0
    frame["comparison_models"] = json.dumps(["behavior_reference", "behavior_plus::rgb_signal"])
    frame["required_features"] = json.dumps(
        {"behavior": ["behavior_signal"], "ocular": ["rgb_signal", "unused_filter"]}
    )
    frame.to_csv(input_path, index=False)
    config_data, paths_data = _config(input_path, output_root)
    config_data["feature_registry"] = _registry_config()
    config_path, paths_path = _write_configs(tmp_path, config_data, paths_data)

    with pytest.raises(SupervisedLearningContractError, match="extra_sample_filters=.*unused_filter"):
        run_supervised_from_config(config_path, paths_config=paths_path, run_id="extra-filter")


def test_registry_run_rejects_predictor_missing_from_sample_contract(tmp_path) -> None:
    input_path = tmp_path / "probe_table.csv"
    output_root = tmp_path / "outputs"
    frame = _probe_table()
    frame["rgb_signal"] = frame["behavior_signal"] * 0.5
    frame["comparison_models"] = json.dumps(["behavior_reference", "behavior_plus::rgb_signal"])
    frame["required_features"] = json.dumps({"behavior": ["behavior_signal"]})
    frame.to_csv(input_path, index=False)
    config_data, paths_data = _config(input_path, output_root)
    config_data["feature_registry"] = _registry_config()
    config_path, paths_path = _write_configs(tmp_path, config_data, paths_data)

    with pytest.raises(SupervisedLearningContractError, match="predictors_missing_from_sample_contract=.*rgb_signal"):
        run_supervised_from_config(config_path, paths_config=paths_path, run_id="missing-filter")


def test_runtime_config_cannot_reenable_global_coverage_gate(tmp_path) -> None:
    input_path = tmp_path / "probe_table.csv"
    output_root = tmp_path / "outputs"
    _probe_table().to_csv(input_path, index=False)
    config_data, paths_data = _config(input_path, output_root)
    config_data["preprocessing"]["unified_global_coverage_cutoff"] = 0.8
    config_path, paths_path = _write_configs(tmp_path, config_data, paths_data)

    with pytest.raises(SupervisedLearningContractError, match="unified global coverage cutoff"):
        run_supervised_from_config(config_path, paths_config=paths_path, run_id="blocked")


def test_runtime_config_cannot_disable_participant_equal_preprocessing(tmp_path) -> None:
    input_path = tmp_path / "probe_table.csv"
    output_root = tmp_path / "outputs"
    _probe_table().to_csv(input_path, index=False)
    config_data, paths_data = _config(input_path, output_root)
    config_data["preprocessing"]["participant_equal_weighted_median_imputation_fit_on_training_only"] = False
    config_path, paths_path = _write_configs(tmp_path, config_data, paths_data)

    with pytest.raises(SupervisedLearningContractError, match="must remain true"):
        run_supervised_from_config(config_path, paths_config=paths_path, run_id="blocked")


def test_runtime_config_requires_five_inner_participant_folds(tmp_path) -> None:
    input_path = tmp_path / "probe_table.csv"
    output_root = tmp_path / "outputs"
    _probe_table().to_csv(input_path, index=False)
    config_data, paths_data = _config(input_path, output_root)
    config_data["validation"]["inner"]["n_splits"] = 2
    config_path, paths_path = _write_configs(tmp_path, config_data, paths_data)

    with pytest.raises(SupervisedLearningContractError, match="exactly 5 participant-grouped folds"):
        run_supervised_from_config(config_path, paths_config=paths_path, run_id="blocked")


def test_runtime_config_rejects_legacy_prediction_folds(tmp_path) -> None:
    input_path = tmp_path / "probe_table.csv"
    output_root = tmp_path / "outputs"
    _probe_table().to_csv(input_path, index=False)
    config_data, paths_data = _config(input_path, output_root)
    config_data["validation"]["prediction_folds"] = 5
    config_path, paths_path = _write_configs(tmp_path, config_data, paths_data)

    with pytest.raises(SupervisedLearningContractError, match="prediction_folds is deprecated"):
        run_supervised_from_config(config_path, paths_config=paths_path, run_id="blocked")


def test_runtime_config_rejects_old_probe_or_fold_mean_selection_metric(tmp_path) -> None:
    input_path = tmp_path / "probe_table.csv"
    output_root = tmp_path / "outputs"
    _probe_table().to_csv(input_path, index=False)
    config_data, paths_data = _config(input_path, output_root)
    config_data["models"]["selection_metric"] = "mean_inner_log_loss"
    config_path, paths_path = _write_configs(tmp_path, config_data, paths_data)

    with pytest.raises(SupervisedLearningContractError, match="participant_macro_log_loss"):
        run_supervised_from_config(config_path, paths_config=paths_path, run_id="blocked")


def test_runtime_config_rejects_bootstrap_contract_drift(tmp_path) -> None:
    input_path = tmp_path / "probe_table.csv"
    output_root = tmp_path / "outputs"
    _probe_table().to_csv(input_path, index=False)
    config_data, paths_data = _config(input_path, output_root)
    config_data["uncertainty"]["participant_cluster_bootstrap"]["replicates"] = 2000
    config_path, paths_path = _write_configs(tmp_path, config_data, paths_data)

    with pytest.raises(SupervisedLearningContractError, match="frozen D10 value 1000"):
        run_supervised_from_config(config_path, paths_config=paths_path, run_id="blocked")


def test_runtime_config_cannot_split_inner_and_outer_participant_keys(tmp_path) -> None:
    input_path = tmp_path / "probe_table.csv"
    output_root = tmp_path / "outputs"
    _probe_table().to_csv(input_path, index=False)
    config_data, paths_data = _config(input_path, output_root)
    config_data["validation"]["inner"]["group_column"] = "session_id"
    config_path, paths_path = _write_configs(tmp_path, config_data, paths_data)

    with pytest.raises(SupervisedLearningContractError, match="same participant grouping column"):
        run_supervised_from_config(config_path, paths_config=paths_path, run_id="blocked")


def test_runtime_config_cannot_replace_participant_grouping_with_session_grouping(tmp_path) -> None:
    input_path = tmp_path / "probe_table.csv"
    output_root = tmp_path / "outputs"
    _probe_table().to_csv(input_path, index=False)
    config_data, paths_data = _config(input_path, output_root)
    config_data["validation"]["outer"]["group_column"] = "session_id"
    config_data["validation"]["inner"]["group_column"] = "session_id"
    config_path, paths_path = _write_configs(tmp_path, config_data, paths_data)

    with pytest.raises(SupervisedLearningContractError, match="participant_group_id"):
        run_supervised_from_config(config_path, paths_config=paths_path, run_id="blocked")


def test_runtime_config_cannot_mix_modality_compositions_within_one_model_family(tmp_path) -> None:
    input_path = tmp_path / "probe_table.csv"
    output_root = tmp_path / "outputs"
    frame = _probe_table()
    frame["ocular_signal"] = 0.5
    frame.to_csv(input_path, index=False)
    config_data, paths_data = _config(input_path, output_root)
    config_data["feature_schemes"]["model_families"]["behavior"]["candidates"].append(
        {
            "feature_set_id": "behavior_plus_ocular_wrong_family",
            "columns": ["behavior_signal", "ocular_signal"],
            "modality_blocks": ["behavior", "ocular"],
        }
    )
    config_path, paths_path = _write_configs(tmp_path, config_data, paths_data)

    with pytest.raises(SupervisedLearningContractError, match="same modality_blocks"):
        run_supervised_from_config(config_path, paths_config=paths_path, run_id="blocked")


def test_formal_run_requires_one_nonblank_analysis_set_id(tmp_path) -> None:
    output_root = tmp_path / "outputs"
    config_data, paths_data = _config(tmp_path / "probe_table.csv", output_root)

    cases = []
    missing = _probe_table().drop(columns=["analysis_set_id"])
    cases.append((missing, "contain analysis_set_id"))
    blank = _probe_table()
    blank["analysis_set_id"] = " "
    cases.append((blank, "blank"))
    multiple = _probe_table()
    multiple.loc[multiple.index[-1], "analysis_set_id"] = "other-set"
    cases.append((multiple, "exactly one analysis_set_id"))

    for index, (frame, message) in enumerate(cases):
        input_path = tmp_path / f"probe_table_{index}.csv"
        frame.to_csv(input_path, index=False)
        config_data["paths"]["input_table"] = {"path_key": "supervised_learning_input_probe_table"}
        paths_data["paths"]["supervised_learning_input_probe_table"] = str(input_path)
        config_path, paths_path = _write_configs(tmp_path, config_data, paths_data)
        with pytest.raises(SupervisedLearningContractError, match=message):
            run_supervised_from_config(config_path, paths_config=paths_path, run_id=f"blocked-{index}")


def test_formal_run_requires_one_explicit_membership_type(tmp_path) -> None:
    output_root = tmp_path / "outputs"
    config_data, paths_data = _config(tmp_path / "probe_table.csv", output_root)

    cases = []
    missing = _probe_table().drop(columns=["membership_type"])
    cases.append((missing, "requires membership_type"))
    blank = _probe_table()
    blank["membership_type"] = " "
    cases.append((blank, "blank"))
    mixed = _probe_table()
    mixed.loc[mixed.index[-1], "membership_type"] = "included_complete"
    cases.append((mixed, "one membership_type per run"))

    for index, (frame, message) in enumerate(cases):
        input_path = tmp_path / f"membership_table_{index}.csv"
        frame.to_csv(input_path, index=False)
        paths_data["paths"]["supervised_learning_input_probe_table"] = str(input_path)
        config_path, paths_path = _write_configs(tmp_path, config_data, paths_data)
        with pytest.raises(SupervisedLearningContractError, match=message):
            run_supervised_from_config(config_path, paths_config=paths_path, run_id=f"membership-blocked-{index}")
