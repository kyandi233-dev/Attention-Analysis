from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import pytest
import yaml

from attention_pipeline.behavior_formal.science_v3 import aggregate_behavior_metrics
from attention_pipeline.config import load_config
from attention_pipeline.formal_analysis.behavior_adapter import assert_current_behavior_rt_cv_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "behavior_formal_v2.yaml"


def _trial_frame(n_valid_rt: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i in range(n_valid_rt):
        rows.append(
            {
                "is_no_go": 0,
                "correct": 1,
                "omission": 0,
                "commission": 0,
                "rt": 300.0 + i * 10.0,
                "time_in_block_sec": float(i),
            }
        )
    return pd.DataFrame(rows)


def _formal_rt_cv_contract() -> tuple[int, str]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    behavior = config["behavior"]
    return int(behavior["rt_cv_min_n"]), str(behavior["rt_cv_min_n_basis"])


def test_formal_config_uses_only_sample_sd_mathematical_requirement() -> None:
    minimum_n, basis = _formal_rt_cv_contract()
    assert minimum_n == 2
    assert basis == "sample_sd_mathematical_requirement_not_empirical_gate"


def test_current_formal_runtime_fails_closed_if_rt_cv_contract_drifts_or_disappears() -> None:
    config = load_config(CONFIG_PATH, use_env_paths=False)
    assert_current_behavior_rt_cv_contract(config)

    data = copy.deepcopy(config.data)
    data["behavior"]["rt_cv_min_n"] = 20
    with pytest.raises(ValueError, match="requires rt_cv_min_n=2"):
        assert_current_behavior_rt_cv_contract(replace(config, data=data))

    data = copy.deepcopy(config.data)
    del data["behavior"]["rt_cv_min_n"]
    with pytest.raises(ValueError, match="must explicitly set behavior.rt_cv_min_n=2"):
        assert_current_behavior_rt_cv_contract(replace(config, data=data))

    data = copy.deepcopy(config.data)
    data["behavior"]["rt_cv_min_n_basis"] = "empirical_gate"
    with pytest.raises(ValueError, match="rt_cv_min_n_basis"):
        assert_current_behavior_rt_cv_contract(replace(config, data=data))


def test_rt_cv_guard_does_not_redefine_generic_or_historical_adapter_configs() -> None:
    config = load_config(CONFIG_PATH, use_env_paths=False)

    data = copy.deepcopy(config.data)
    data.pop("pipeline", None)
    data["behavior"].pop("rt_cv_min_n", None)
    assert_current_behavior_rt_cv_contract(replace(config, data=data))

    data = copy.deepcopy(config.data)
    data["pipeline"]["name"] = "historical-behavior-reproduction"
    data["behavior"]["rt_cv_min_n"] = 20
    assert_current_behavior_rt_cv_contract(replace(config, data=data))


def test_two_to_nineteen_valid_rt_cv_is_preserved_by_formal_runner_annotation() -> None:
    minimum_n, _ = _formal_rt_cv_contract()
    namespace = runpy.run_path(str(ROOT / "scripts" / "sart_formal_analysis.py"))
    annotate_scale = namespace["_annotate_scale"]

    for n_valid in (2, 3, 5, 12, 19):
        metrics = aggregate_behavior_metrics(_trial_frame(n_valid))
        assert np.isfinite(metrics["go_correct_rt_cv"])
        frame = pd.DataFrame([metrics])
        annotated = annotate_scale(
            frame,
            unit="probe",
            rt_cv_min_n=minimum_n,
            rt_slope_min_n=5,
            is_probe=True,
        )
        assert np.isfinite(annotated.loc[0, "go_correct_rt_cv"])
        assert annotated.loc[0, "rt_cv_status"] == "estimable"


def test_one_valid_rt_remains_not_estimable_because_sample_sd_is_undefined() -> None:
    minimum_n, _ = _formal_rt_cv_contract()
    metrics = aggregate_behavior_metrics(_trial_frame(1))
    assert np.isnan(metrics["go_correct_rt_cv"])

    namespace = runpy.run_path(str(ROOT / "scripts" / "sart_formal_analysis.py"))
    annotate_scale = namespace["_annotate_scale"]
    annotated = annotate_scale(
        pd.DataFrame([metrics]),
        unit="probe",
        rt_cv_min_n=minimum_n,
        rt_slope_min_n=5,
        is_probe=True,
    )
    assert np.isnan(annotated.loc[0, "go_correct_rt_cv"])
    assert annotated.loc[0, "rt_cv_status"] == "not_estimable_low_rt_n"


def test_rt_cv_contract_does_not_change_probe_window_definition() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    behavior = config["behavior"]
    assert behavior["primary_probe_window_seconds"] == 30
    assert behavior["probe_windows_seconds"] == [10, 20, 30]
    assert behavior["rt_slope_min_n"] == 5
