from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from attention_pipeline.behavior_formal.behavior_error_taxonomy import OMISSION_QC_RATE_METRICS
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
    expected_descriptive = {
        "clean_go_omission_rate",
        "timing_ambiguous_go_omission_rate",
        *OMISSION_QC_RATE_METRICS,
    }
    assert set(FIRST_ROUND_OMISSION_DESCRIPTIVE_QC_ONLY) == expected_descriptive
    assert omission_supervised_role("raw_go_omission_rate") == "first_round_supervised_predictor"
    for metric in expected_descriptive:
        assert omission_supervised_role(metric) == "descriptive_qc_sensitivity_only"


def test_all_nonraw_omission_rates_and_aliases_fail_closed_as_first_round_predictors() -> None:
    forbidden_fields = (
        "clean_go_omission_rate",
        "timing_ambiguous_go_omission_rate",
        *OMISSION_QC_RATE_METRICS,
        *OMISSION_COMPATIBILITY_ALIASES.keys(),
    )
    for forbidden in forbidden_fields:
        with pytest.raises(BehaviorSupervisedContractError, match="raw_go_omission_rate"):
            validate_first_round_omission_predictors([forbidden])

    assert validate_first_round_omission_predictors(["raw_go_omission_rate"]) == (
        "raw_go_omission_rate",
    )


def test_legacy_omission_rate_is_explicit_alias_not_independent_predictor() -> None:
    assert OMISSION_COMPATIBILITY_ALIASES["omission_rate"] == "raw_go_omission_rate"
    assert omission_supervised_role("omission_rate") == "compatibility_alias_not_independent_predictor"


def test_science_config_has_one_current_primary_omission_role_and_historical_provenance() -> None:
    config = yaml.safe_load((ROOT / "configs" / "behavior_formal_v2.yaml").read_text(encoding="utf-8"))
    omission = config["behavior"]["omission_endpoints"]
    assert tuple(omission["first_round_supervised_predictors"]) == FIRST_ROUND_SUPERVISED_OMISSION_PREDICTORS
    assert tuple(omission["first_round_supervised_descriptive_qc_only"]) == (
        "clean_go_omission_rate",
        "timing_ambiguous_go_omission_rate",
    )
    assert omission["compatibility_alias_not_independent_predictor"] == "omission_rate"

    policy = config["analysis_policy"]
    assert policy["raw_go_omission_is_current_primary_omission_endpoint"] is True
    assert policy["clean_timing_omission_are_descriptive_qc_sensitivity"] is True
    assert policy["historical_three_omission_endpoint_outputs_retained_for_provenance"] is True
    assert "raw_clean_timing_ambiguous_omission_are_prespecified_formal_endpoints" not in policy
    assert policy["first_round_supervised_omission_only_raw"] is True
    assert policy["clean_timing_omission_remain_available_for_descriptive_qc"] is True
