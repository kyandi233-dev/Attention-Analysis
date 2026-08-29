from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from attention_pipeline.formal_analysis.execution_control import (
    ExecutionLedger,
    resolve_optional_steps,
)


def test_optional_step_selection_defaults_all_and_supports_only_skip() -> None:
    available = ("models", "figures", "sensitivity")
    assert resolve_optional_steps(available) == set(available)
    assert resolve_optional_steps(available, only="models,figures") == {"models", "figures"}
    assert resolve_optional_steps(available, skip="figures") == {"models", "sensitivity"}
    assert resolve_optional_steps(available, only="models,figures", skip="figures") == {"models"}
    with pytest.raises(ValueError, match="unknown optional formal step"):
        resolve_optional_steps(available, skip="rgb_magic")


def test_ledger_records_completed_and_skipped_and_marks_partial(tmp_path: Path) -> None:
    ledger = ExecutionLedger(pipeline="test")
    value = ledger.run("required", lambda: 7, required=True)
    assert value == 7
    skipped = ledger.run("figures", lambda: 9, required=False, requested=False, skip_reason="user skip")
    assert skipped is None
    assert ledger.run_status == "partial_run"
    ledger.write(tmp_path)

    rows = pd.read_csv(tmp_path / "execution_steps.csv")
    assert list(rows["status"]) == ["completed", "skipped"]
    assert rows.loc[0, "elapsed_seconds"] >= 0
    assert rows.loc[1, "skip_reason"] == "user skip"
    manifest = json.loads((tmp_path / "execution_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_status"] == "partial_run"


def test_required_step_cannot_be_skipped() -> None:
    ledger = ExecutionLedger(pipeline="test")
    with pytest.raises(ValueError, match="required formal step cannot be skipped"):
        ledger.run("identity", lambda: None, required=True, requested=False)


def test_failed_step_is_recorded_before_exception_propagates() -> None:
    ledger = ExecutionLedger(pipeline="test")

    def boom() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        ledger.run("model", boom, required=False)
    assert ledger.run_status == "failed"
    assert ledger.records[0].status == "failed"
    assert ledger.records[0].error_type == "RuntimeError"
