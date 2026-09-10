from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from attention_pipeline.behavior_formal.behavior_supervised_interface import (
    FIRST_ROUND_CANDIDATE_POOL,
    BehaviorSupervisedInterfaceError,
    build_behavior_supervised_feature_audit,
    build_behavior_supervised_probe_table,
    materialize_behavior_supervised_interface,
)


ROOT = Path(__file__).resolve().parents[1]


def _primary_probe() -> pd.DataFrame:
    rows = []
    for i in range(3):
        rows.append({
            "participant_group_id": f"P-{i // 2}",
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
            "go_correct_rt_cv": np.nan if i == 1 else 0.12 + i * 0.01,
            "go_correct_rt_sd_ms": 45.0 + i,
            "go_correct_rt_mad_ms": 30.0 + i,
            "go_correct_rt_iqr_ms": 60.0 + i,
            "go_correct_rt_theilsen_slope_ms_per_s": 0.5 + i,
            "raw_go_omission_rate": 0.05 * i,
            "clean_go_omission_rate": 0.03 * i,
            "timing_ambiguous_go_omission_rate": 0.02 * i,
            "commission_rate": 0.10 + 0.01 * i,
            "dprime_loglinear": 2.0 - 0.1 * i,
            "omission_rate": 0.05 * i,
            "trial_opportunities": 20,
            "go_opportunities": 16,
            "nogo_opportunities": 4,
            "correct_go_rt_opportunities": [12, 3, 10][i],
            "omission_numerator": i,
            "omission_denominator": 16,
            "commission_numerator": 0,
            "commission_denominator": 4,
            "raw_go_omission_n": i,
            "clean_go_omission_n": i,
            "timing_ambiguous_go_omission_n": 0,
            "omission_taxonomy_denominator": 16,
        })
    return pd.DataFrame(rows)


def test_interface_preserves_all_probe_rows_and_missing_feature_cells_without_imputation() -> None:
    source = _primary_probe()
    out = build_behavior_supervised_probe_table(source)
    assert len(out) == len(source)
    assert out["probe_event_id"].is_unique
    assert out.loc[1, "go_correct_rt_cv"] != out.loc[1, "go_correct_rt_cv"]  # NaN remains NaN
    assert "analysis_set_id" not in out.columns
    assert "omission_rate" not in out.columns
    assert set(FIRST_ROUND_CANDIDATE_POOL).issubset(out.columns)


def test_clean_and_timing_omission_are_preserved_only_for_qc_audit() -> None:
    out = build_behavior_supervised_probe_table(_primary_probe())
    audit = build_behavior_supervised_feature_audit(out)
    qc = audit[audit["field"].isin(["clean_go_omission_rate", "timing_ambiguous_go_omission_rate"])]
    assert len(qc) == 2
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
    assert (output_root / "behavior_supervised_probe_30s.csv").is_file()
    assert (output_root / "behavior_supervised_feature_audit.csv").is_file()
    saved = json.loads((output_root / "behavior_supervised_interface_manifest.json").read_text(encoding="utf-8"))
    assert saved["source_sha256"] == manifest["source_sha256"]
    with pytest.raises(FileExistsError):
        materialize_behavior_supervised_interface(source_path, output_root)


def test_candidate_yaml_matches_c4_boundaries() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "behavior_supervised_candidates_v1.yaml").read_text(encoding="utf-8"))
    assert cfg["source"]["primary_window_seconds"] == 30
    assert cfg["candidate_dimensions"]["omission"]["columns"] == ["raw_go_omission_rate"]
    assert cfg["outcomes"]["q2_ordinal_4level"]["first_round_predictor_allowed"] is False
    assert cfg["policies"]["no_imputation_in_task_c_interface"] is True
    assert cfg["policies"]["no_row_deletion_by_feature_coverage_in_task_c"] is True
    assert cfg["policies"]["analysis_set_id_generated_by_task_c"] is False
    assert cfg["policies"]["analysis_set_id_owned_by_task_b"] is True
