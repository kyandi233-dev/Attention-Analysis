import pandas as pd
import pytest

from attention_pipeline.nir_formal_analysis.supervised_probe_table import (
    build_supervised_probe_table,
    supervised_nir_manifest,
)


def _analysis_ready():
    return pd.DataFrame(
        {
            "participant_group_id": ["P001"] * 6,
            "session_id": ["sub-001"] * 6,
            "block": [1] * 6,
            "frame_idx": list(range(1, 7)),
            "unix_ms": [70000.0, 76000.0, 82000.0, 88000.0, 94000.0, 99999.0],
            "left_raw_pupil_diameter": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "right_raw_pupil_diameter": [14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
            "left_pupil_valid_primary": [True] * 6,
            "right_pupil_valid_primary": [True] * 6,
        }
    )


def _probes():
    return pd.DataFrame(
        {
            "participant_group_id": ["P001"],
            "session_id": ["sub-001"],
            "block_num": [1],
            "probe_index_global": [7],
            "probe_index_in_block": [3],
            "probe_event_id": ["sub-001_b1_p3"],
            "probe_time_ms": [100000.0],
            "q1_nominal_4class": [1],
            "q2_ordinal_4level": [2],
        }
    )


def test_probe_table_primary_and_sensitivity_windows_share_unique_identity():
    result = build_supervised_probe_table(
        _analysis_ready(), _probes(), windows_sec=(30, 10, 20)
    )

    assert len(result) == 3
    assert result["participant_group_id"].eq("P001").all()
    assert result["probe_event_id"].eq("sub-001_b1_p3").all()
    assert result["probe_onset_ms"].eq(100000.0).all()
    assert set(result["window_sec"]) == {10, 20, 30}
    assert result.loc[result["window_sec"].eq(30), "window_role"].iloc[0] == "primary"
    assert result.loc[result["window_sec"].isin([10, 20]), "window_role"].eq("window_sensitivity").all()
    assert result["analysis_set_id"].isna().all() if "analysis_set_id" in result else True


def test_probe_table_rejects_trial_absolute_onset_as_probe_time():
    probes = _probes().drop(columns=["probe_time_ms"]).assign(absolute_onset_time=99500.0)
    with pytest.raises(ValueError, match="explicit probe time"):
        build_supervised_probe_table(_analysis_ready(), probes, windows_sec=(30,))


def test_probe_table_retains_explicit_window_end_compatibility():
    probes = _probes().drop(columns=["probe_time_ms"]).assign(window_end_ms=100000.0)
    result = build_supervised_probe_table(_analysis_ready(), probes, windows_sec=(30,))
    assert result["probe_onset_ms"].eq(100000.0).all()


def test_formal_probe_table_refuses_to_promote_analysis_group_token_to_participant_id():
    frame = _analysis_ready().drop(columns=["participant_group_id"])
    probes = _probes().drop(columns=["participant_group_id"]).assign(analysis_group_token="legacy")
    with pytest.raises(ValueError, match="participant_group_id"):
        build_supervised_probe_table(frame, probes)


def test_manifest_declares_zero_calibration_and_no_training_side_effects():
    manifest = supervised_nir_manifest()
    assert manifest["base_signal"] == "pupil_geom_mean_diameter"
    assert manifest["production_input_schema"] == "formal_candidate_sidecar_long"
    assert manifest["production_raw_column"] == "pupil_geom_mean_diameter__raw"
    assert manifest["production_validity_column"] == "pupil_geom_mean_diameter__valid_primary"
    assert manifest["zero_calibration"] is True
    assert manifest["session_level_baseline_used"] is False
    assert manifest["participant_within_between_used"] is False
    assert manifest["imputation_performed"] is False
    assert manifest["standardization_performed"] is False
    assert manifest["analysis_set_id_generated"] is False
    assert manifest["automatic_coverage_drop"] is False
    assert manifest["head_motion_sensitivity_status"] == "not_validated"
    assert manifest["scale_sensitivity_status"] == "not_validated"


def test_probe_table_respects_behavior_defined_available_bounds():
    probes = _probes().assign(
        block_analysis_start_ms=95000.0,
        block_analysis_end_ms=120000.0,
        available_start_ms=95000.0,
        available_end_ms=100000.0,
    )
    result = build_supervised_probe_table(
        _analysis_ready(), probes, windows_sec=(30,)
    )
    row = result.iloc[0]
    assert row["requested_duration_sec"] == pytest.approx(30.0)
    assert row["available_duration_sec"] == pytest.approx(5.0)
    assert row["available_duration_fraction"] == pytest.approx(1 / 6)
    assert bool(row["window_truncated_by_available_start"]) is True


def test_probe_table_rejects_participant_identity_mismatch_with_nir_session():
    frame = _analysis_ready().assign(participant_group_id="P002")
    with pytest.raises(ValueError, match="participant_group_id mismatch"):
        build_supervised_probe_table(frame, _probes())


def test_session_specific_participant_inference_works_across_multiple_sessions():
    frame = pd.concat(
        [
            _analysis_ready(),
            _analysis_ready().assign(session_id="sub-002", participant_group_id="P002"),
        ],
        ignore_index=True,
    )
    probes = pd.concat(
        [
            _probes().drop(columns=["participant_group_id"]),
            _probes()
            .drop(columns=["participant_group_id"])
            .assign(session_id="sub-002", probe_index_global=8),
        ],
        ignore_index=True,
    )
    result = build_supervised_probe_table(frame, probes)
    assert set(
        result[["session_id", "participant_group_id"]].itertuples(index=False, name=None)
    ) == {("sub-001", "P001"), ("sub-002", "P002")}
