from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from attention_pipeline.multimodal_formal.mmwave_cardiopulmonary_ingest import (
    BR_COLUMN,
    ENDPOINT_ATOL_MS,
    ENDPOINT_CONTRACT,
    ENDPOINT_RTOL,
    HR_COLUMN,
    PHYSIOLOGY_QUALIFICATION,
    TIME_LEGALITY_STATUS,
    MmwaveCardiopulmonaryIngestError,
    audit_mmwave_cardiopulmonary_snapshot,
    write_mmwave_cardiopulmonary_ingest_audit,
)
from scripts.mmwave_cardiopulmonary_ingest_audit import _resolve_probe_index


def _behavior() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": ["sub-001"] * 4,
            "block_id": ["b1", "b1", "b2", "b2"],
            "probe_index_in_block": [1, 2, 1, 2],
            "probe_order_in_block": [1, 2, 1, 2],
            "participant_group_id": ["pg-001"] * 4,
            "probe_event_id": ["e1", "e2", "e3", "e4"],
            "window_name": ["pre_30s"] * 4,
        }
    )


def _snapshot() -> pd.DataFrame:
    probe = np.array([100_000.0, 140_000.0, 200_000.0, 240_000.0])
    nominal = probe - 30_000.0
    effective = nominal.copy()
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


REAL_EPOCH_MS = 1_756_000_000_000
_TIME_COLUMNS = (
    "window_nominal_start_unix_ms",
    "window_effective_start_unix_ms",
    "window_end_unix_ms",
    "probe_onset_unix_ms",
)


def _at_real_epoch(snapshot: pd.DataFrame) -> pd.DataFrame:
    out = snapshot.copy()
    for column in _TIME_COLUMNS:
        out[column] = out[column] + REAL_EPOCH_MS
    return out


def test_ingest_preserves_unavailable_and_malformed_as_distinct_states() -> None:
    result = _audit()
    assert result.manifest["status"] == "SMOKE_ONLY"
    assert result.manifest["engineering_integration_qualification"] == "PENDING_GOVERNED_REAL_RUN"
    assert result.manifest["available_probe_n"] == 2
    assert result.manifest["source_unavailable_probe_n"] == 1
    assert result.manifest["source_malformed_probe_n"] == 1
    assert result.manifest["retained_missing_or_error_probe_n"] == 2
    assert result.manifest["source_unavailable_zero_imputed"] is False
    assert result.manifest["source_malformed_zero_imputed"] is False
    retained = result.taskb_source["integration_state"].ne("AVAILABLE")
    assert result.taskb_source.loc[retained, [HR_COLUMN, BR_COLUMN]].isna().all().all()
    assert "mmwave_motion_proxy_median" not in result.taskb_source.columns
    assert "mmwave_rmssd_ms" not in result.taskb_source.columns

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


def test_handoff_is_interface_ready_but_time_and_prediction_blocked() -> None:
    result = _audit()
    handoff = result.feature_handoff
    assert set(handoff["scientific_modality"]) == {"cardiopulmonary"}
    assert set(handoff["source_namespace"]) == {"mmwave"}
    assert all(json.loads(value) == ["mmwave"] for value in handoff["required_devices"])
    assert set(handoff["time_legality_status"]) == {"blocked_upstream_contract_mismatch"}
    assert TIME_LEGALITY_STATUS == "blocked_upstream_contract_mismatch"
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


def test_manifest_separates_metadata_contract_from_formal_time_legality() -> None:
    result = _audit()
    time = result.manifest["time_legality"]
    assert time["metadata_contract_verified"] is True
    assert time["declared_endpoint_contract"] == "exact_integer_equality"
    assert time["frame_membership_verified"] is False
    assert time["source_code_provenance_closed"] is False
    assert time["time_legality_status"] == "blocked_upstream_contract_mismatch"


def test_block_start_truncation_is_metadata_legal_and_reported() -> None:
    result = _audit()
    assert result.manifest["time_legality"]["truncated_window_n"] == 1
    assert int(result.ingest_audit["effective_start_truncated"].sum()) == 1
    assert result.ingest_audit["metadata_time_contract_valid"].astype(bool).all()


def test_wrong_clock_fails_closed() -> None:
    snapshot = _snapshot()
    snapshot.loc[0, "alignment_clock_source"] = "python_processing_timestamp"
    with pytest.raises(MmwaveCardiopulmonaryIngestError, match="metadata time-contract"):
        _audit(snapshot=snapshot)


def test_default_relative_tolerance_would_admit_hour_scale_endpoint_drift() -> None:
    end = np.float64(REAL_EPOCH_MS)
    probe_one_hour_later = np.float64(REAL_EPOCH_MS + 3_600_000)
    assert np.isclose(end, probe_one_hour_later, atol=1.0)
    assert not np.isclose(end, probe_one_hour_later, atol=0.0, rtol=0.0)


def test_exact_endpoint_contract_is_frozen() -> None:
    assert ENDPOINT_CONTRACT == "exact_integer_equality"
    assert ENDPOINT_ATOL_MS == 0.0
    assert ENDPOINT_RTOL == 0.0
    result = _audit(snapshot=_at_real_epoch(_snapshot()))
    summary = result.manifest["endpoint_delta_audit"]
    assert summary["endpoint_contract"] == "exact_integer_equality"
    assert summary["endpoint_contract_frozen"] is True
    assert summary["endpoint_atol_ms"] == 0.0
    assert summary["endpoint_rtol"] == 0.0
    assert summary["n_zero"] == 4
    assert summary["n_nonzero"] == 0
    assert summary["n_violating_exact_endpoint_contract"] == 0


@pytest.mark.parametrize("delta_ms", [1, -1, 2, 3_600_000])
def test_any_nonzero_endpoint_delta_fails_closed(delta_ms: int) -> None:
    snapshot = _at_real_epoch(_snapshot())
    snapshot.loc[0, "window_end_unix_ms"] = (
        snapshot.loc[0, "probe_onset_unix_ms"] + delta_ms
    )
    with pytest.raises(MmwaveCardiopulmonaryIngestError, match="metadata time-contract"):
        _audit(snapshot=snapshot)


def test_endpoint_delta_audit_states_frame_membership_is_unavailable() -> None:
    result = _audit(snapshot=_at_real_epoch(_snapshot()))
    audit = result.manifest["endpoint_delta_audit"]
    assert audit["audit_scope"] == "metadata_contract_window_end_vs_probe_onset"
    assert audit["frame_membership_available"] is False
    assert audit["source_code_provenance_closed"] is False
    assert audit["formal_time_legality_status"] == "blocked_upstream_contract_mismatch"


def test_endpoint_delta_outputs_are_written(tmp_path) -> None:
    result = _audit(snapshot=_at_real_epoch(_snapshot()))
    paths = write_mmwave_cardiopulmonary_ingest_audit(tmp_path, result)
    assert (tmp_path / "endpoint_delta_ms.csv").exists()
    assert (tmp_path / "endpoint_delta_nonzero_ms.csv").exists()
    assert (tmp_path / "endpoint_delta_summary.json").exists()
    assert paths["endpoint_delta_ms"].endswith("endpoint_delta_ms.csv")
    summary = json.loads(
        (tmp_path / "endpoint_delta_summary.json").read_text(encoding="utf-8")
    )
    assert summary["n_total"] == 4
    assert summary["n_nonzero"] == 0
    assert summary["frame_membership_available"] is False


def test_nominal_window_must_be_exactly_30_seconds() -> None:
    snapshot = _snapshot()
    snapshot.loc[0, "window_nominal_start_unix_ms"] += 1
    with pytest.raises(MmwaveCardiopulmonaryIngestError, match="metadata time-contract"):
        _audit(snapshot=snapshot)


def test_participant_identity_mismatch_fails() -> None:
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


def test_behavior_probe_order_alias_is_accepted_when_canonical_missing() -> None:
    behavior = _behavior().drop(columns=["probe_index_in_block"])
    resolved, alias = _resolve_probe_index(behavior)
    assert alias == "probe_order_in_block"
    assert resolved["probe_index_in_block"].tolist() == [1, 2, 1, 2]


def test_behavior_dual_probe_index_columns_must_agree() -> None:
    behavior = _behavior()
    behavior.loc[0, "probe_order_in_block"] = 9
    with pytest.raises(ValueError, match="aliases disagree"):
        _resolve_probe_index(behavior)


def test_behavior_dual_probe_index_columns_can_agree_without_alias_application() -> None:
    resolved, alias = _resolve_probe_index(_behavior())
    assert alias is None
    assert resolved["probe_index_in_block"].tolist() == [1, 2, 1, 2]
