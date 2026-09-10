from __future__ import annotations

from pathlib import Path
import runpy

import pandas as pd
import pytest
import yaml

from attention_pipeline.behavior_formal.behavior_supervised_interface import (
    BehaviorSupervisedInterfaceError,
    build_behavior_supervised_probe_table,
)


ROOT = Path(__file__).resolve().parents[1]


def _minimal_probe() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "participant_group_id": "P-01",
            "repeat_participant_id": "P-01",
            "session_id": "sub-031",
            "block_id": "B1",
            "probe_event_id": "sub-031|B1|probe|1",
            "window_seconds_nominal": 30,
            "analysis_role": "primary_probe",
            "formal_independent_sample": True,
            "anchor_trial_excluded": True,
            "window_crosses_block": False,
            "q1_nominal_4class": 1,
            "q2_ordinal_4level": 2,
            "go_correct_rt_mean_ms": 400.0,
            "go_correct_rt_median_ms": 395.0,
            "go_correct_rt_cv": 0.10,
            "go_correct_rt_sd_ms": 40.0,
            "go_correct_rt_mad_ms": 30.0,
            "go_correct_rt_iqr_ms": 55.0,
            "go_correct_rt_theilsen_slope_ms_per_s": 0.2,
            "raw_go_omission_rate": 0.05,
            "clean_go_omission_rate": 0.03,
            "timing_ambiguous_go_omission_rate": 0.02,
            "commission_rate": 0.1,
            "dprime_loglinear": 2.0,
            "correct_go_rt_opportunities": 8,
            "rt_cv_status": "estimable",
            "rt_slope_status": "estimable",
            "sdt_status": "estimable",
        }
    ])


def test_rt_cv_runtime_contract_rejects_reintroduced_empirical_gate() -> None:
    namespace = runpy.run_path(str(ROOT / "scripts" / "sart_formal_analysis.py"))
    validate = namespace["_validate_rt_cv_contract"]
    cfg = yaml.safe_load((ROOT / "configs" / "behavior_formal_v2.yaml").read_text(encoding="utf-8"))["behavior"]
    validate(cfg)
    altered = dict(cfg)
    altered["rt_cv_min_n"] = 20
    with pytest.raises(ValueError, match="RT-CV mathematical contract"):
        validate(altered)


def test_behavior_interface_rejects_session_id_as_participant_group() -> None:
    frame = _minimal_probe()
    frame["participant_group_id"] = frame["session_id"]
    with pytest.raises(BehaviorSupervisedInterfaceError, match="participant identity contract"):
        build_behavior_supervised_probe_table(frame)


def test_behavior_interface_requires_primary_independent_probe_role() -> None:
    frame = _minimal_probe()
    frame["analysis_role"] = "window_sensitivity_only"
    with pytest.raises(BehaviorSupervisedInterfaceError, match="analysis_role"):
        build_behavior_supervised_probe_table(frame)

    frame = _minimal_probe()
    frame["formal_independent_sample"] = False
    with pytest.raises(BehaviorSupervisedInterfaceError, match="formal_independent_sample"):
        build_behavior_supervised_probe_table(frame)


def test_behavior_interface_preserves_estimability_status_fields() -> None:
    out = build_behavior_supervised_probe_table(_minimal_probe())
    for column in ("rt_cv_status", "rt_slope_status", "sdt_status"):
        assert column in out.columns
