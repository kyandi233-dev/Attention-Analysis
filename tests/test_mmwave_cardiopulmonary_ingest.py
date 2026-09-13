from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from attention_pipeline.multimodal_formal.mmwave_cardiopulmonary_ingest import (
    BR_COLUMN,
    HR_COLUMN,
    PHYSIOLOGY_QUALIFICATION,
    TIME_LEGALITY_STATUS,
    MmwaveCardiopulmonaryIngestError,
    audit_mmwave_cardiopulmonary_snapshot,
)


def _behavior() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": ["sub-001"] * 4,
            "block_id": ["b1", "b1", "b2", "b2"],
            "probe_index_in_block": [1, 2, 1, 2],
            "participant_group_id": ["pg-001"] * 4,
            "probe_event_id": ["e1", "e2", "e3", "e4"],
            "probe_order_in_block": [1, 2, 1, 2],
            "window_name": ["pre_30s"] * 4,
        }
    )


def _snapshot() -> pd.DataFrame:
    probe = np.array([100_000.0, 140_000.0, 200_000.0, 240_000.0])
    nominal = probe - 30_000.0
    effective = nominal.copy()
    # Legal formal-Block truncation of the nominal 30-second interval.
    effective[0] = nominal[0] + 10_000.0
    state = ["AVAILABLE", "AVAILABLE", "SOURCE_UNAVAILABLE", "SOURCE_MALFORMED"]
    return pd.DataFrame(
        {
            "participant_group_id": ["pg-001"] * 4,
            "repeat_participant_id": ["legacy-001"] * 4,
            "session_id": ["sub-001"] * 4,
            "block_id": ["b1", "b1", "b2", "b2"],
            "probe_index_in_block": [1, 2, 1, 2],
            "probe_id": ["p1", "p2", "p3", "p4"],
            "window_name": ["pre_30s"] * 4,
            "window_nominal_start_unix_ms": nominal,
            "window_effective_start_unix_ms": effective,
            "window_end_unix_ms": probe,
            "probe_onset_unix_ms": probe,
            "alignment_clock_source": ["dll_host_receive_enqueue"] * 4,
            HR_COLUMN: [72.0, 74.0, np.nan, np.nan],
            BR_COLUMN: [15.0, 16.0, np.nan, np.nan],
            "source_availability_state": [
                "AVAILABLE",
                "AVAILABLE",
                "UNAVAILABLE",
                "AVAILABLE",
            ],
            "source_readability_state": [
                "READABLE",
                "READABLE",
                "NOT_READABLE",
                "NOT_READABLE",
            ],
            "estimability_state": [
                "ESTIMABLE",
                "ESTIMABLE",
                "NOT_ESTIMABLE",
                "NOT_ESTIMABLE",
            ],
            "measurement_qc_state": [
                "SUPPORTING_ONLY",
                "SUPPORTING_ONLY",
                "NOT_APPLICABLE",
                "FAILED",
            ],
            "malformed_state": ["OK", "OK", "OK", "MALFORMED"],
            "integration_state": state,
            "producer_state": state,
            "missing_reason": [
                "",
                "",
                "source_unavailable",
                "timestamp_count_mismatch",
            ],
            "snapshot_version": ["mmwave_integration_snapshot_v1"] * 4,
            "producer_commit": ["producer-sha"] * 4,
            "producer_adapter_version": ["adapter-v1"] * 4,
            "source_run_id": ["source-run"] * 4,
            "snapshot_run_id": ["snapshot-run"] * 4,
            # Producer diagnostics must never be promoted by this bridge.
            "mmwave_motion_proxy_median": [0.1, 0.2, np.nan, np.nan],
            "mmwave_rmssd_ms": [20.0, 21.0, np.nan, np.nan],
        }
    )


def _audit(
    snapshot: pd.DataFrame | None = None,
    behavior: pd.DataFrame | None = None,
):
    return audit_mmwave_cardiopulmonary_snapshot(
        _snapshot() if snapshot is None else snapshot,
        _behavior() if behavior is None else behavior,
        strict_canonical_counts=False,
    )


# Real FocusWave snapshots carry Unix-epoch milliseconds (~1.7e12).  The small
# values used by ``_snapshot()`` cannot expose a relative-tolerance defect, so the
# endpoint-guard tests deliberately shift the time columns to a realistic epoch.
REAL_EPOCH_MS = 1_756_000_000_000

_TIME_COLUMNS = (
    "window_nominal_start_unix_ms",
    "window_effective_start_unix_ms",
    "window_end_unix_ms",
    "probe_onset_unix_ms",
)


def _at_real_epoch(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Shift a fixture to realistic Unix-ms magnitude without changing deltas."""
    out = snapshot.copy()
    for column in _TIME_COLUMNS:
        out[column] = out[column] + REAL_EPOCH_MS
    return out


def test_ingest_preserves_unavailable_and_malformed_as_distinct_states() -> None:
    result = _audit()

    assert result.manifest["status"] == "SMOKE_ONLY"
    assert (
        result.manifest["engineering_integration_qualification"]
        == "PENDING_GOVERNED_REAL_RUN"
    )
    assert result.manifest["governed_real_snapshot_rerun_complete"] is False
    assert result.manifest["governed_probe_n"] == 4
    assert result.manifest["available_probe_n"] == 2
    assert result.manifest["source_unavailable_probe_n"] == 1
    assert result.manifest["source_malformed_probe_n"] == 1
    assert result.manifest["retained_missing_or_error_probe_n"] == 2
    assert result.manifest["hr_available_n"] == 2
    assert result.manifest["br_available_n"] == 2
    assert result.manifest["duplicate_key_n"] == 0
    assert result.manifest["missing_key_n"] == 0
    assert result.manifest["extra_key_n"] == 0
    assert result.manifest["participant_identity_inferred_from_folder"] is False
    assert result.manifest["source_unavailable_zero_imputed"] is False
    assert result.manifest["source_malformed_zero_imputed"] is False
    assert result.manifest["q1_q2_used_for_ingest_decision"] is False
    assert result.manifest["models_trained"] is False
    assert result.manifest["final_feature_registry_modified"] is False

    assert len(result.taskb_source) == 4
    retained = result.taskb_source["integration_state"].ne("AVAILABLE")
    assert result.taskb_source.loc[retained, [HR_COLUMN, BR_COLUMN]].isna().all().all()
    assert "mmwave_motion_proxy_median" not in result.taskb_source.columns
    assert "mmwave_rmssd_ms" not in result.taskb_source.columns

    audit = result.ingest_audit.set_index(["block_id", "probe_index_in_block"])
    assert bool(audit.loc[("b2", 1), "source_unavailable"])
    assert not bool(audit.loc[("b2", 1), "source_malformed"])
    assert bool(audit.loc[("b2", 2), "source_malformed"])
    assert not bool(audit.loc[("b2", 2), "source_unavailable"])

    status = result.quality_tables["probe_feature_status"]
    unavailable = status[
        status["feature"].eq(HR_COLUMN)
        & status["block_id"].eq("b2")
        & status["probe_index_in_block"].eq(1)
    ].iloc[0]
    malformed = status[
        status["feature"].eq(HR_COLUMN)
        & status["block_id"].eq("b2")
        & status["probe_index_in_block"].eq(2)
    ].iloc[0]
    assert unavailable["missing_kind"] == "structural_source_missing"
    assert malformed["missing_kind"] == "structural_source_unreadable"
    assert not bool(unavailable["eligible_for_missing_strategy"])
    assert not bool(malformed["eligible_for_missing_strategy"])

    summary = result.analysis_set_summary.set_index("membership")
    assert int(summary.loc["included_complete", "probe_n"]) == 2
    assert int(summary.loc["included_missing_aware", "probe_n"]) == 2


def test_handoff_uses_cardiopulmonary_science_and_mmwave_device_without_prediction_promotion() -> None:
    result = _audit()
    handoff = result.feature_handoff

    assert set(handoff["scientific_modality"]) == {"cardiopulmonary"}
    assert set(handoff["source_namespace"]) == {"mmwave"}
    assert all(json.loads(value) == ["mmwave"] for value in handoff["required_devices"])
    assert set(handoff["time_legality_status"]) == {TIME_LEGALITY_STATUS}
    assert set(handoff["physiology_qualification"]) == {PHYSIOLOGY_QUALIFICATION}
    for column in (
        "standalone_eligible",
        "behavior_increment_eligible",
        "modality_model_eligible",
        "full_model_eligible",
        "full_leave_one_out_eligible",
        "registry_ready",
    ):
        assert not handoff[column].astype(bool).any()
    assert handoff["researcher_freeze_required"].astype(bool).all()

    records = json.loads(
        result.analysis_sets["required_feature_records"].drop_duplicates().iloc[0]
    )
    assert {record["scientific_modality"] for record in records} == {
        "cardiopulmonary"
    }
    assert {record["source_namespace"] for record in records} == {"mmwave"}


def test_block_start_truncation_is_time_legal_and_reported() -> None:
    result = _audit()
    assert result.manifest["time_legality"]["time_legality_status"] == TIME_LEGALITY_STATUS
    assert result.manifest["time_legality"]["right_exclusive_end"] is True
    assert (
        result.manifest["time_legality"]["alignment_clock_source"]
        == "dll_host_receive_enqueue"
    )
    assert result.manifest["time_legality"]["truncated_window_n"] == 1
    assert int(result.ingest_audit["effective_start_truncated"].sum()) == 1


def test_wrong_clock_fails_closed() -> None:
    snapshot = _snapshot()
    snapshot.loc[0, "alignment_clock_source"] = "python_processing_timestamp"
    with pytest.raises(MmwaveCardiopulmonaryIngestError, match="verified_pre_probe_only"):
        _audit(snapshot=snapshot)


def test_default_relative_tolerance_would_admit_hour_scale_endpoint_drift() -> None:
    """Documents the defect this round fixes.

    NumPy's default ``rtol`` (~1e-5) applied to Unix-ms operands (~1e12) spans
    roughly 1e7 ms -- hours -- so an endpoint guard written without ``rtol=0.0``
    silently admits arbitrary drift instead of millisecond-level drift.
    """
    end = np.float64(REAL_EPOCH_MS)
    probe_one_hour_later = np.float64(REAL_EPOCH_MS + 3_600_000)

    # Default rtol: wrongly treated as close.
    assert np.isclose(end, probe_one_hour_later, atol=1.0)
    # Explicit rtol=0.0: correctly rejected.
    assert not np.isclose(end, probe_one_hour_later, atol=1.0, rtol=0.0)


def test_zero_endpoint_delta_is_legal_at_real_epoch() -> None:
    result = _audit(snapshot=_at_real_epoch(_snapshot()))

    audit = result.manifest["endpoint_delta_audit"]
    assert audit["n_total"] == 4
    assert audit["n_zero"] == 4
    assert audit["n_nonzero"] == 0
    assert audit["n_exceeding_provisional_atol"] == 0
    assert audit["nonzero_delta_frequencies_ms"] == {}


def test_hour_scale_endpoint_drift_fails_closed_at_real_epoch() -> None:
    """The guard must reject hour-scale drift that the old default rtol admitted."""
    snapshot = _at_real_epoch(_snapshot())
    snapshot.loc[0, "window_end_unix_ms"] = (
        snapshot.loc[0, "probe_onset_unix_ms"] + 3_600_000
    )

    with pytest.raises(
        MmwaveCardiopulmonaryIngestError, match="verified_pre_probe_only"
    ):
        _audit(snapshot=snapshot)


def test_endpoint_delta_within_provisional_atol_is_admitted_not_frozen() -> None:
    """+1 ms is admitted only because it sits inside the CURRENT PROVISIONAL atol.

    This is deliberately not asserted as a permanent scientific contract; whether
    the final rule is exact equality or a smaller constant is decided by the
    governed-cohort endpoint audit plus producer evidence.
    """
    snapshot = _at_real_epoch(_snapshot())
    snapshot.loc[0, "window_end_unix_ms"] = snapshot.loc[0, "probe_onset_unix_ms"] + 1

    result = _audit(snapshot=snapshot)

    audit = result.manifest["endpoint_delta_audit"]
    assert audit["n_nonzero"] == 1
    assert audit["n_exceeding_provisional_atol"] == 0
    assert audit["nonzero_delta_frequencies_ms"] == {"1": 1}
    assert audit["provisional_atol_ms"] == 1.0
    assert audit["provisional_atol_is_frozen_conclusion"] is False


def test_endpoint_delta_beyond_provisional_atol_fails_closed() -> None:
    snapshot = _at_real_epoch(_snapshot())
    snapshot.loc[0, "window_end_unix_ms"] = snapshot.loc[0, "probe_onset_unix_ms"] + 2

    with pytest.raises(
        MmwaveCardiopulmonaryIngestError, match="verified_pre_probe_only"
    ):
        _audit(snapshot=snapshot)


def test_endpoint_delta_audit_reports_nonzero_rows_and_scope_limit() -> None:
    snapshot = _at_real_epoch(_snapshot())
    snapshot.loc[1, "window_end_unix_ms"] = snapshot.loc[1, "probe_onset_unix_ms"] - 1

    result = _audit(snapshot=snapshot)

    audit = result.manifest["endpoint_delta_audit"]
    assert audit["n_total"] == 4
    assert audit["n_zero"] == 3
    assert audit["n_nonzero"] == 1
    assert audit["nonzero_delta_frequencies_ms"] == {"-1": 1}
    assert audit["nonzero_session_n"] == 1
    assert audit["nonzero_participant_group_n"] == 1
    assert audit["nonzero_sessions"] == ["sub-001"]
    assert audit["n_exceeding_provisional_atol"] == 0

    # Scope limit: this layer cannot verify frame membership.
    assert audit["audit_scope"] == "metadata_contract_window_end_vs_probe_onset"
    assert audit["frame_membership_available"] is False

    # Non-zero rows are listed explicitly, not only summarised.
    assert len(result.endpoint_delta_nonzero) == 1
    assert set(result.endpoint_delta_nonzero["endpoint_delta_ms"]) == {-1.0}
    for column in (
        "session_id",
        "block_id",
        "probe_index_in_block",
        "participant_group_id",
        "window_end_unix_ms",
        "probe_onset_unix_ms",
        "endpoint_delta_ms",
    ):
        assert column in result.endpoint_delta.columns


def test_endpoint_delta_outputs_are_written(tmp_path) -> None:
    from attention_pipeline.multimodal_formal.mmwave_cardiopulmonary_ingest import (
        write_mmwave_cardiopulmonary_ingest_audit,
    )

    snapshot = _at_real_epoch(_snapshot())
    snapshot.loc[0, "window_end_unix_ms"] = snapshot.loc[0, "probe_onset_unix_ms"] + 1
    result = _audit(snapshot=snapshot)

    paths = write_mmwave_cardiopulmonary_ingest_audit(tmp_path, result)

    delta_csv = tmp_path / "endpoint_delta_ms.csv"
    summary_json = tmp_path / "endpoint_delta_summary.json"
    assert delta_csv.exists()
    assert summary_json.exists()
    assert paths["endpoint_delta_ms"].endswith("endpoint_delta_ms.csv")

    written = pd.read_csv(delta_csv)
    assert len(written) == 4
    assert "endpoint_delta_ms" in written.columns

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["n_total"] == 4
    assert summary["n_nonzero"] == 1
    assert summary["frame_membership_available"] is False


def test_nominal_window_must_be_30_seconds() -> None:
    snapshot = _snapshot()
    snapshot.loc[0, "window_nominal_start_unix_ms"] += 1000
    with pytest.raises(MmwaveCardiopulmonaryIngestError, match="verified_pre_probe_only"):
        _audit(snapshot=snapshot)


def test_participant_identity_mismatch_fails_instead_of_inferring_from_session_name() -> None:
    snapshot = _snapshot()
    snapshot["participant_group_id"] = "wrong-participant"
    with pytest.raises(MmwaveCardiopulmonaryIngestError, match="identity disagrees"):
        _audit(snapshot=snapshot)


def test_duplicate_probe_key_fails() -> None:
    snapshot = pd.concat([_snapshot(), _snapshot().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate probe keys"):
        _audit(snapshot=snapshot)


def test_missing_probe_key_fails_key_conservation() -> None:
    snapshot = _snapshot().iloc[:-1].copy()
    with pytest.raises(MmwaveCardiopulmonaryIngestError, match="key conservation failed"):
        _audit(snapshot=snapshot)


def test_unavailable_or_malformed_rows_cannot_be_zero_filled() -> None:
    for state in ("SOURCE_UNAVAILABLE", "SOURCE_MALFORMED"):
        snapshot = _snapshot()
        row = snapshot["integration_state"].eq(state)
        snapshot.loc[row, HR_COLUMN] = 0.0
        snapshot.loc[row, BR_COLUMN] = 0.0
        with pytest.raises(MmwaveCardiopulmonaryIngestError, match="zero-fill"):
            _audit(snapshot=snapshot)
