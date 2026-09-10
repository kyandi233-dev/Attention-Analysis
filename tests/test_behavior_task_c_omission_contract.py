from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from attention_pipeline.behavior_formal.behavior_supervised_contract import (
    FIRST_ROUND_OMISSION_DESCRIPTIVE_QC_ONLY,
    FIRST_ROUND_SUPERVISED_OMISSION_PREDICTORS,
    OMISSION_COMPATIBILITY_ALIASES,
    BehaviorSupervisedContractError,
    omission_supervised_role,
    validate_first_round_omission_predictors,
)


ROOT = Path(__file__).resolve().parents[1]


def test_first_round_supervised_omission_contract_uses_raw_only() -> None:
    assert FIRST_ROUND_SUPERVISED_OMISSION_PREDICTORS == ("raw_go_omission_rate",)
    assert set(FIRST_ROUND_OMISSION_DESCRIPTIVE_QC_ONLY) == {
        "clean_go_omission_rate",
        "timing_ambiguous_go_omission_rate",
    }
    assert omission_supervised_role("raw_go_omission_rate") == "first_round_supervised_predictor"
    assert omission_supervised_role("clean_go_omission_rate") == "descriptive_qc_sensitivity_only"
    assert omission_supervised_role("timing_ambiguous_go_omission_rate") == "descriptive_qc_sensitivity_only"


def test_clean_timing_and_aliases_fail_closed_as_first_round_predictors() -> None:
    for forbidden in (
        "clean_go_omission_rate",
        "timing_ambiguous_go_omission_rate",
        "omission_rate",
    ):
        with pytest.raises(BehaviorSupervisedContractError, match="raw_go_omission_rate"):
            validate_first_round_omission_predictors([forbidden])

    assert validate_first_round_omission_predictors(["raw_go_omission_rate"]) == (
        "raw_go_omission_rate",
    )


def test_legacy_omission_rate_is_explicit_alias_not_independent_predictor() -> None:
    assert OMISSION_COMPATIBILITY_ALIASES["omission_rate"] == "raw_go_omission_rate"
    assert omission_supervised_role("omission_rate") == "compatibility_alias_not_independent_predictor"


def test_science_config_matches_machine_readable_supervised_omission_contract() -> None:
    config = yaml.safe_load((ROOT / "configs" / "behavior_formal_v2.yaml").read_text(encoding="utf-8"))
    omission = config["behavior"]["omission_endpoints"]
    assert tuple(omission["first_round_supervised_predictors"]) == FIRST_ROUND_SUPERVISED_OMISSION_PREDICTORS
    assert tuple(omission["first_round_supervised_descriptive_qc_only"]) == FIRST_ROUND_OMISSION_DESCRIPTIVE_QC_ONLY
    assert omission["compatibility_alias_not_independent_predictor"] == "omission_rate"
    assert config["analysis_policy"]["first_round_supervised_omission_only_raw"] is True
    assert config["analysis_policy"]["clean_timing_omission_remain_available_for_descriptive_qc"] is True
