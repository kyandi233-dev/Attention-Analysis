from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from attention_pipeline.behavior_formal.behavior_supervised_contract import (
    FIRST_ROUND_OMISSION_DESCRIPTIVE_QC_ONLY,
)
from attention_pipeline.behavior_formal.behavior_supervised_interface import (
    DESCRIPTIVE_QC_COLUMNS,
    ERROR_CONTROL_CANDIDATES,
    FIRST_ROUND_CANDIDATE_POOL,
    RT_LEVEL_CANDIDATES,
    RT_TREND_CANDIDATES,
    RT_VARIABILITY_CANDIDATES,
    BehaviorSupervisedInterfaceError,
    build_behavior_supervised_feature_audit,
    build_behavior_supervised_probe_table,
    materialize_behavior_supervised_interface,
)


ROOT = Path(__file__).resolve().parents[1]


def _primary_probe() -> pd.DataFrame:
    rows = []
    for i in range(3):
        n_rt = [12, 3, 10][i]
        participant = f"P-{i // 2}"
        raw_n = i
        clean_n = 0 if i == 0 else 1
        timing_n = raw_n - clean_n
        go_n = 16
        raw_rate = raw_n / go_n
        clean_rate = clean_n / go_n
        timing_rate = timing_n / go_n
        rows.append({
            "participant_group_id": participant,
            "repeat_participant_id": participant,
            "participant_identity_source": "questionnaire_repeat_registry",
            "participant_identity_resolved_for_clustering": True,
            "session_id": f"S-{i}",
            "block_id": "B1",
            "probe_event_id": f"S-{i}|B1|probe|1",
            "window_seconds_nominal": 30,
            "analysis_role": "primary_probe",
            "formal_independent_sample": True,
            "anchor_trial_excluded": True,
            "window_crosses_block": False,
            "probe_order_in_block": 1,
            "anchor_trial_num": 20,
            "probe_time_ms": 30000 + i * 1000,
            "q1_nominal_4class": [1, 2, 4][i],
            "q2_ordinal_4level": [2, 3, 1][i],
            "go_correct_rt_mean_ms": 400.0 + i,
            "go_correct_rt_median_ms": 395.0 + i,
            "go_correct_rt_cv": 0.12 + i * 0.01,
            "go_correct_rt_sd_ms": 45.0 + i,
            "go_correct_rt_mad_ms": 30.0 + i,
            "go_correct_rt_iqr_ms": 60.0 + i,
            "go_correct_rt_theilsen_slope_ms_per_s": 0.5 + i,
            "raw_go_omission_rate": raw_rate,
            "clean_go_omission_rate": clean_rate,
            "timing_ambiguous_go_omission_rate": timing_rate,
            "omission_prestimulus_only_ambiguity_rate": 0.0,
            "omission_carryover_only_ambiguity_rate": 0.0,
            "omission_prestimulus_and_carryover_ambiguity_rate": 0.0,
            "late_go_response_candidate_rate": 0.0,
            "anticipatory_go_response_candidate_rate": 0.0,
            "commission_rate": 0.10 + 0.01 * i,
            "dprime_loglinear": 2.0 - 0.1 * i,
            "omission_rate": raw_rate,
            "trial_opportunities": 20,
            "go_opportunities": go_n,
            "nogo_opportunities": 4,
            "correct_go_rt_opportunities": n_rt,
            "rt_variability_valid_n": n_rt,
            "rt_cv_min_n": 2,
            "rt_cv_status": "estimable",
            "rt_slope_min_n": 5,
            "rt_slope_status": "estimable" if n_rt >= 5 else "not_estimable_low_rt_n",
            "sdt_status": "estimable",
            "omission_numerator": raw_n,
            "omission_denominator": go_n,
            "commission_numerator": 0,
            "commission_denominator": 4,
            "raw_go_omission_n": raw_n,
            "clean_go_omission_n": clean_n,
            "timing_ambiguous_go_omission_n": timing_n,
            "omission_taxonomy_denominator": go_n,
        })
    return pd.DataFrame(rows)


def test_interface_preserves_all_probe_rows_and_feature_cells_without_imputation() -> None:
    source = _primary_probe()
    source.loc[1, "go_correct_rt_mad_ms"] = np.nan
    out = build_behavior_supervised_probe_table(source)
    assert len(out) == len(source)
    assert out["probe_event_id"].is_unique
    assert pd.isna(out.loc[1, "go_correct_rt_mad_ms"])
    assert "analysis_set_id" not in out.columns
    assert "omission_rate" not in out.columns
    assert set(FIRST_ROUND_CANDIDATE_POOL).issubset(out.columns)
    assert out["repeat_participant_id"].equals(out["participant_group_id"])


def test_clean_timing_and_finer_omission_fields_are_preserved_only_for_qc_audit() -> None:
    out = build_behavior_supervised_probe_table(_primary_probe())
    audit = build_behavior_supervised_feature_audit(out)
    qc = audit[audit["field"].isin(DESCRIPTIVE_QC_COLUMNS)]
    assert len(qc) == len(DESCRIPTIVE_QC_COLUMNS)
    assert qc["role"].eq("descriptive_qc_sensitivity_only").all()
    raw = audit[audit["field"].eq("raw_go_omission_rate")].iloc[0]
    assert raw["role"] == "first_round_supervised_omission_candidate"
    assert audit["automatic_drop_allowed"].eq(False).all()


def test_interface_rejects_duplicate_or_non_30s_probe_rows() -> None:
    duplicate = pd.concat([_primary_probe(), _primary_probe().iloc[[0]]], ignore_index=True)
    with pytest.raises(BehaviorSupervisedInterfaceError, match="probe_event_id must be unique"):
        build_behavior_supervised_probe_table(duplicate)

    wrong_window = _primary_probe()
    wrong_window.loc[0, "window_seconds_nominal"] = 20
    with pytest.raises(BehaviorSupervisedInterfaceError, match="30-second probes only"):
        build_behavior_supervised_probe_table(wrong_window)


def test_materialized_interface_writes_probe_table_audit_and_manifest_without_analysis_set(tmp_path) -> None:
    source_path = tmp_path / "probe_primary_30s.csv"
    output_root = tmp_path / "supervised_interface_v1"
    _primary_probe().to_csv(source_path, index=False)
    manifest = materialize_behavior_supervised_interface(source_path, output_root, config_digest="abc")
    assert manifest["status"] == "complete"
    assert manifest["n_rows"] == 3
    assert manifest["row_filter_applied"] is False
    assert manifest["imputation_applied"] is False
    assert manifest["analysis_set_id_generated"] is False
    assert manifest["participant_identity_contract_checked"] is True
    assert manifest["primary_probe_role_contract_checked"] is True
    assert manifest["rt_cv_handoff_contract_checked"] is True
    assert manifest["omission_partition_handoff_contract_checked"] is True
    assert (output_root / "behavior_supervised_probe_30s.csv").is_file()
    assert (output_root / "behavior_supervised_feature_audit.csv").is_file()
    saved = json.loads((output_root / "behavior_supervised_interface_manifest.json").read_text(encoding="utf-8"))
    assert saved["source_sha256"] == manifest["source_sha256"]
    with pytest.raises(FileExistsError):
        materialize_behavior_supervised_interface(source_path, output_root)


def test_candidate_yaml_exactly_matches_python_candidate_and_qc_contracts() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "behavior_supervised_candidates_v1.yaml").read_text(encoding="utf-8"))
    assert cfg["source"]["primary_window_seconds"] == 30
    assert tuple(cfg["candidate_dimensions"]["rt_level"]["columns"]) == RT_LEVEL_CANDIDATES
    variability = cfg["candidate_dimensions"]["rt_variability"]
    assert variability["preferred_current"] == RT_VARIABILITY_CANDIDATES[0]
    assert tuple(variability["limited_alternatives"]) == RT_VARIABILITY_CANDIDATES[1:]
    assert tuple(cfg["candidate_dimensions"]["rt_trend"]["columns"]) == RT_TREND_CANDIDATES
    assert tuple(cfg["candidate_dimensions"]["omission"]["columns"]) == ("raw_go_omission_rate",)
    assert tuple(cfg["candidate_dimensions"]["error_control"]["columns"]) == ERROR_CONTROL_CANDIDATES[1:]
    assert tuple(cfg["descriptive_qc_sensitivity_only"]) == FIRST_ROUND_OMISSION_DESCRIPTIVE_QC_ONLY
    assert cfg["outcomes"]["q2_ordinal_4level"]["first_round_predictor_allowed"] is False
    assert cfg["policies"]["no_imputation_in_task_c_interface"] is True
    assert cfg["policies"]["no_row_deletion_by_feature_coverage_in_task_c"] is True
    assert cfg["policies"]["analysis_set_id_generated_by_task_c"] is False
    assert cfg["policies"]["analysis_set_id_owned_by_task_b"] is True
    assert cfg["policies"]["all_nonraw_omission_rates_not_first_round_predictors"] is True
