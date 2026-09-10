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
        fold_audits=[{"outer_fold_group": "P01"}, {"outer_fold_group": "P02"}],
        failures=pd.DataFrame(),
        metadata={
            "run_id": "run-001",
            "analysis_set_id": "set-a",
            "task": "q1_equals_1_vs_2_3_4",
            "n_input_rows": 2,
            "n_participant_groups": 2,
            "n_models": 1,
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
    assert manifest["n_prediction_rows"] == 2
    assert manifest["n_failed_folds"] == 0
    assert (run_root / "probe_predictions.csv").is_file()
    assert (run_root / "fold_audits.json").is_file()
    assert (run_root / "failures.csv").is_file()
    assert (run_root / "run_manifest.json").is_file()

    saved = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    assert saved["analysis_set_id"] == "set-a"
    assert saved["provenance"]["input_sha256"] == "abc"

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_supervised_run(result, output_root=tmp_path)


def test_write_supervised_run_rejects_prediction_row_loss(tmp_path) -> None:
    result = _complete_result()
    result.predictions = result.predictions.iloc[:1].copy()
    with pytest.raises(ValueError, match="prediction row count mismatch"):
        write_supervised_run(result, output_root=tmp_path)
