from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from attention_pipeline.supervised_learning.predefined_reduced import (
    PredefinedReducedContractError,
    build_predefined_reduced_plan,
    validate_input_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "supervised_learning_v1.yaml"
DESIGN_CONFIG = REPO_ROOT / "configs" / "supervised_predefined_reduced_v1.yaml"


def _configs() -> tuple[dict, dict]:
    return (
        yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8")),
        yaml.safe_load(DESIGN_CONFIG.read_text(encoding="utf-8")),
    )


def test_predefined_plan_keeps_full_benchmark_and_exact_three_candidates() -> None:
    base, design = _configs()
    plan = build_predefined_reduced_plan(base, design)

    assert plan.benchmark_model_id == "frozen_full_benchmark"
    assert plan.selected_model_id == "predefined_reduced_nested"
    assert [scheme.feature_set_id for scheme in plan.candidate_schemes] == [
        "full_11",
        "behavior_core",
        "behavior_core_plus_ocular",
    ]
    assert len(plan.benchmark_scheme.columns) == 11
    assert plan.candidate_schemes[0].columns == plan.benchmark_scheme.columns
    assert plan.candidate_schemes[1].columns == (
        "commission_rate",
        "go_correct_rt_median_ms",
    )
    assert set(plan.candidate_schemes[2].effective_modalities) == {"behavior", "ocular"}
    assert plan.common_core_columns == ("commission_rate", "go_correct_rt_median_ms")


def test_predefined_plan_fails_if_full_candidate_is_posthoc_reduced() -> None:
    base, design = _configs()
    design["models"]["candidates"][0]["feature_ids"].pop()
    with pytest.raises(PredefinedReducedContractError, match="exactly reproduce"):
        build_predefined_reduced_plan(base, design)


def test_input_contract_rejects_same_shape_but_different_file(tmp_path: Path) -> None:
    base, design = _configs()
    plan = build_predefined_reduced_plan(base, design)
    frame = pd.DataFrame(
        {
            "analysis_set_id": ["AS.full"],
            "membership_type": ["included_missing_aware"],
            "participant_group_id": ["P01"],
            "comparison_models": ['["full"]'],
            "required_features": ['{"behavior": ["commission_rate"]}'],
            "required_outcomes": ['["q1_nominal_4class"]'],
            "q1_nominal_4class": [1],
        }
    )
    path = tmp_path / "not-the-frozen-input.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(PredefinedReducedContractError, match="SHA-256 drift"):
        validate_input_contract(frame, path, design, plan)

