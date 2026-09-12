from __future__ import annotations

import json

import pandas as pd
import pytest

from attention_pipeline.supervised_learning.reporting import write_supervised_run
from attention_pipeline.supervised_learning.runner import SupervisedRunResult


def _complete_result() -> SupervisedRunResult:
    predictions = pd.DataFrame(
        {
            "run_id": ["run-001", "run-001"],
            "analysis_set_id": ["set-a", "set-a"],
            "membership_type": ["included_complete", "included_complete"],
            "participant_group_id": ["P01", "P02"],
            "model_id": ["behavior", "behavior"],
            "outer_fold_group": ["P01", "P02"],
            "session_id": ["S01", "S02"],
            "block_id": ["B1", "B1"],
            "probe_event_id": ["S01|B1|P1", "S02|B1|P1"],
            "q1_nominal_4class": [1, 2],
            "q1_binary": [1, 0],
            "feature_set_id": ["behavior-a", "behavior-a"],
            "selected_c": [1.0, 1.0],
            "p_q1_equals_1": [0.8, 0.2],
            "predicted_q1_binary": [1, 0],
            "model_failed": [False, False],
            "failure_reason": ["", ""],
        }
    )
    return SupervisedRunResult(
        predictions=predictions,
        fold_audits=[
            {
                "run_id": "run-001",
                "analysis_set_id": "set-a",
                "membership_type": "included_complete",
                "model_id": "behavior",
                "outer_fold_group": "P01",
                "failed": False,
                "reason": "",
            },
            {
                "run_id": "run-001",
                "analysis_set_id": "set-a",
                "membership_type": "included_complete",
                "model_id": "behavior",
                "outer_fold_group": "P02",
                "failed": False,
                "reason": "",
            },
        ],
        failures=pd.DataFrame(),
        metadata={
            "run_id": "run-001",
            "analysis_set_id": "set-a",
            "membership_type": "included_complete",
            "task": "q1_equals_1_vs_2_3_4",
            "n_input_rows": 2,
            "n_participant_groups": 2,
            "n_models": 1,
        },
    )


def _paired_result() -> SupervisedRunResult:
    rows = []
    audits = []
    for model_id, probabilities in {
        "behavior_reference": [0.6, 0.4],
        "behavior_plus::blink_rate": [0.9, 0.1],
    }.items():
        for participant, session, q1, probability in zip(
            ["P01", "P02"], ["S01", "S02"], [1, 2], probabilities, strict=True
        ):
            rows.append(
                {
                    "run_id": "run-paired",
                    "analysis_set_id": "set-a",
                    "membership_type": "included_complete",
                    "participant_group_id": participant,
                    "model_id": model_id,
                    "outer_fold_group": participant,
                    "session_id": session,
                    "block_id": "B1",
                    "probe_event_id": f"{session}|B1|P1",
                    "q1_nominal_4class": q1,
                    "q1_binary": 1 if q1 == 1 else 0,
                    "feature_set_id": model_id,
                    "selected_c": 1.0,
                    "p_q1_equals_1": probability,
                    "predicted_q1_binary": 1 if probability >= 0.5 else 0,
                    "model_failed": False,
                    "failure_reason": "",
                }
            )
            audits.append(
                {
                    "run_id": "run-paired",
                    "analysis_set_id": "set-a",
                    "membership_type": "included_complete",
                    "model_id": model_id,
                    "outer_fold_group": participant,
                    "failed": False,
                    "reason": "",
                }
            )
    return SupervisedRunResult(
        predictions=pd.DataFrame(rows),
        fold_audits=audits,
        failures=pd.DataFrame(),
        metadata={
            "run_id": "run-paired",
            "analysis_set_id": "set-a",
            "membership_type": "included_complete",
            "task": "q1_equals_1_vs_2_3_4",
            "n_input_rows": 2,
            "n_participant_groups": 2,
            "n_models": 2,
            "paired_comparisons": [
                {
                    "comparison_type": "behavior_increment",
                    "feature_id": "blink_rate",
                    "baseline_model_id": "behavior_reference",
                    "added_model_id": "behavior_plus::blink_rate",
                }
            ],
        },
    )


def test_write_supervised_run_is_immutable_and_auditable(tmp_path) -> None:
    result = _complete_result()
    manifest = write_supervised_run(
        result,
        output_root=tmp_path,
        provenance={"input_sha256": "abc", "code_sha": "def"},
    )

    run_root = tmp_path / "run-001"
    assert manifest["status"] == "complete"
    assert manifest["membership_type"] == "included_complete"
    assert manifest["n_prediction_rows"] == 2
    assert manifest["n_failed_folds"] == 0
    assert manifest["n_estimable_models"] == 1
    assert manifest["outer_evaluation"]["aggregation"] == "participant_equal_within_participant_probe_equal"
    assert manifest["outer_evaluation"]["bootstrap_replicates"] == 1000
    assert manifest["outer_evaluation"]["bootstrap_seed"] == 20260830
    assert manifest["outer_evaluation"]["bootstrap_retrain_models"] is False

    assert (run_root / "probe_predictions.csv").is_file()
    assert (run_root / "fold_audits.json").is_file()
    assert (run_root / "failures.csv").is_file()
    assert (run_root / "participant_log_loss.csv").is_file()
    assert (run_root / "model_evaluation.csv").is_file()
    assert (run_root / "participant_bootstrap.json").is_file()
    assert (run_root / "paired_participant_increments.csv").is_file()
    assert (run_root / "paired_model_increments.csv").is_file()
    assert (run_root / "paired_increment_bootstrap.json").is_file()
    assert (run_root / "run_manifest.json").is_file()

    participant = pd.read_csv(run_root / "participant_log_loss.csv")
    model = pd.read_csv(run_root / "model_evaluation.csv")
    predictions = pd.read_csv(run_root / "probe_predictions.csv")
    bootstrap = json.loads((run_root / "participant_bootstrap.json").read_text(encoding="utf-8"))
    assert set(participant["participant_group_id"]) == {"P01", "P02"}
    assert set(participant["membership_type"]) == {"included_complete"}
    assert set(predictions["membership_type"]) == {"included_complete"}
    assert model.loc[0, "status"] == "estimable"
    assert model.loc[0, "membership_type"] == "included_complete"
    assert model.loc[0, "participant_equal_log_loss"] == pytest.approx(participant["mean_log_loss"].mean())
    assert bootstrap[0]["method"] == "fixed_oof_participant_cluster_percentile"
    assert bootstrap[0]["replicates"] == 1000
    assert bootstrap[0]["seed"] == 20260830
    assert bootstrap[0]["membership_type"] == "included_complete"

    saved = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    assert saved["analysis_set_id"] == "set-a"
    assert saved["membership_type"] == "included_complete"
    assert saved["provenance"]["input_sha256"] == "abc"
    assert saved["n_declared_paired_comparisons"] == 0

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_supervised_run(result, output_root=tmp_path)


def test_declared_paired_increment_is_written_from_same_oof_archive(tmp_path) -> None:
    result = _paired_result()
    manifest = write_supervised_run(result, output_root=tmp_path)
    run_root = tmp_path / "run-paired"

    summary = pd.read_csv(run_root / "paired_model_increments.csv")
    participants = pd.read_csv(run_root / "paired_participant_increments.csv")
    bootstrap = json.loads((run_root / "paired_increment_bootstrap.json").read_text(encoding="utf-8"))

    assert manifest["n_declared_paired_comparisons"] == 1
    assert manifest["n_estimable_paired_comparisons"] == 1
    assert summary.loc[0, "status"] == "estimable"
    assert summary.loc[0, "feature_id"] == "blink_rate"
    assert summary.loc[0, "membership_type"] == "included_complete"
    assert summary.loc[0, "overall_log_loss_increment"] > 0
    assert set(participants["participant_group_id"]) == {"P01", "P02"}
    assert set(participants["membership_type"]) == {"included_complete"}
    assert bootstrap[0]["paired_model_resampling"] is True
    assert bootstrap[0]["membership_type"] == "included_complete"
    assert bootstrap[0]["point_estimate"] == pytest.approx(summary.loc[0, "overall_log_loss_increment"])


def test_write_supervised_run_rejects_prediction_row_loss(tmp_path) -> None:
    result = _complete_result()
    result.predictions = result.predictions.iloc[:1].copy()
    with pytest.raises(ValueError, match="prediction row count mismatch"):
        write_supervised_run(result, output_root=tmp_path)


def test_write_supervised_run_rejects_missing_or_duplicate_fold_audits(tmp_path) -> None:
    missing = _complete_result()
    missing.fold_audits = missing.fold_audits[:1]
    with pytest.raises(ValueError, match="fold audit count mismatch"):
        write_supervised_run(missing, output_root=tmp_path)

    duplicate = _complete_result()
    duplicate.fold_audits[1] = dict(duplicate.fold_audits[0])
    with pytest.raises(ValueError, match="duplicate fold audit rows"):
        write_supervised_run(duplicate, output_root=tmp_path)


def test_write_supervised_run_rejects_failure_key_disagreement(tmp_path) -> None:
    result = _complete_result()
    result.predictions.loc[0, "model_failed"] = True
    result.predictions.loc[0, "failure_reason"] = "expected synthetic failure"
    result.predictions.loc[0, "feature_set_id"] = None
    result.predictions.loc[0, "selected_c"] = float("nan")
    result.predictions.loc[0, "p_q1_equals_1"] = float("nan")
    result.predictions.loc[0, "predicted_q1_binary"] = pd.NA
    with pytest.raises(ValueError, match="failed-fold keys disagree"):
        write_supervised_run(result, output_root=tmp_path)
