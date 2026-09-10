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
    for group_index in range(5):
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
            "q2_is_predictor": False,
        },
        "validation": {
            "outer": {"method": "leave_one_participant_out", "group_column": "participant_group_id", "participant_disjoint": True},
            "inner": {"method": "grouped_k_fold", "n_splits": 2, "group_column": "participant_group_id", "refit_preprocessing_per_split": True},
            "zero_individual_calibration": True,
            "forbid_test_participant_sequence_statistics": True,
            "forbid_test_participant_future_information": True,
        },
        "preprocessing": {
            "median_imputation_fit_on_training_only": True,
            "standardization_fit_on_training_only": True,
            "data_dependent_column_handling_fit_on_training_only": True,
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
            "selection_metric": "mean_inner_log_loss",
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
    assert manifest["n_prediction_rows"] == len(_probe_table())
    assert manifest["outer_test_outcomes_passed_to_model"] is False
    assert (run_root / "probe_predictions.csv").is_file()
    assert (run_root / "fold_audits.json").is_file()
    assert (run_root / "failures.csv").is_file()
    saved = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    assert saved["provenance"]["input_table"] == str(input_path)
    assert len(saved["provenance"]["input_sha256"]) == 64


def test_runtime_config_cannot_reenable_global_coverage_gate(tmp_path) -> None:
    input_path = tmp_path / "probe_table.csv"
    output_root = tmp_path / "outputs"
    _probe_table().to_csv(input_path, index=False)
    config_data, paths_data = _config(input_path, output_root)
    config_data["preprocessing"]["unified_global_coverage_cutoff"] = 0.8
    config_path, paths_path = _write_configs(tmp_path, config_data, paths_data)

    with pytest.raises(SupervisedLearningContractError, match="unified global coverage cutoff"):
        run_supervised_from_config(config_path, paths_config=paths_path, run_id="blocked")


def test_runtime_config_cannot_disable_training_only_preprocessing(tmp_path) -> None:
    input_path = tmp_path / "probe_table.csv"
    output_root = tmp_path / "outputs"
    _probe_table().to_csv(input_path, index=False)
    config_data, paths_data = _config(input_path, output_root)
    config_data["preprocessing"]["median_imputation_fit_on_training_only"] = False
    config_path, paths_path = _write_configs(tmp_path, config_data, paths_data)

    with pytest.raises(SupervisedLearningContractError, match="must remain true"):
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
