from __future__ import annotations

from attention_pipeline.nir_formal_analysis.baseline_contract import baseline_contract_rows


def test_three_baseline_semantics_are_explicit_and_not_conflated() -> None:
    rows = {row["reference_name"]: row for row in baseline_contract_rows()}
    centering = rows["session_eye_centering_reference"]
    event = rows["pre_event_local_baseline"]
    resting = rows["resting_or_task_start_baseline"]

    assert centering["resting_physiological_baseline"] is False
    assert "session × eye" in centering["scope"]
    assert event["window"] == "-200 ms <= time relative to trial onset < 0 ms"
    assert event["resting_physiological_baseline"] is False
    assert resting["status"] == "not_defined_without_protocol_evidence"
    assert resting["implementation"] == "none"
