from __future__ import annotations

from attention_pipeline.nir_formal_analysis.baseline_contract import baseline_contract_rows


def test_three_reference_semantics_are_explicit_and_not_conflated() -> None:
    rows = {row["reference_name"]: row for row in baseline_contract_rows()}
    centering = rows["session_eye_centering_reference"]
    event = rows["pre_event_local_baseline"]
    resting = rows["resting_task_start_interval"]

    assert centering["resting_physiological_baseline"] is False
    assert "session × eye" in centering["scope"]
    assert event["window"] == "-200 ms <= time relative to trial onset < 0 ms"
    assert event["resting_physiological_baseline"] is False

    # The formal program now supplies protocol evidence for the ~3-min interval,
    # but that is not the same as automatically admitting a resting pupil value.
    assert resting["status"] == "protocol_interval_verified_reference_pending_observability_gate"
    assert "baseline_start" in resting["window"]
    assert "baseline_stop" in resting["window"]
    assert resting["implementation"].endswith("run_resting_observability")
    assert resting["resting_physiological_baseline"] is True
    assert "阈值未预先冻结前不授权" in resting["interpretation_zh"]
