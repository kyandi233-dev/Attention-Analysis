from __future__ import annotations

import pandas as pd
import pytest

from attention_pipeline.formal_analysis.identity_questionnaire import (
    left_join_questionnaire,
    load_repeat_registry,
    reconcile_cohort_identity,
    validate_questionnaire_registry_consistency,
)


def _registry_rows() -> pd.DataFrame:
    return pd.DataFrame([
        {"subid": "sub-031_", "participant_key": "P001", "experiment_id": 31, "visit_order": 1, "prior_visit_count": 0, "total_visit_count": 2, "is_first_visit": 1},
        {"subid": "sub-087_", "participant_key": "P001", "experiment_id": 87, "visit_order": 2, "prior_visit_count": 1, "total_visit_count": 2, "is_first_visit": 0},
        {"subid": "sub-041_", "participant_key": "P002", "experiment_id": 41, "visit_order": 1, "prior_visit_count": 0, "total_visit_count": 1, "is_first_visit": 1},
    ])


def test_repeat_registry_normalizes_trailing_underscore_and_validates_order(tmp_path) -> None:
    path = tmp_path / "registry.csv"
    _registry_rows().to_csv(path, index=False, encoding="utf-8-sig")
    out = load_repeat_registry(path)
    assert out["session_id"].tolist() == ["sub-031", "sub-087", "sub-041"]
    assert out.loc[out["session_id"].eq("sub-087"), "visit_order"].iloc[0] == 2


def test_repeat_registry_rejects_noncontinuous_visit_order(tmp_path) -> None:
    rows = _registry_rows()
    rows.loc[rows["subid"].eq("sub-087_"), "visit_order"] = 3
    path = tmp_path / "registry.csv"
    rows.to_csv(path, index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="visit_order"):
        load_repeat_registry(path)


def test_questionnaire_registry_core_mismatch_fails_closed() -> None:
    registry = _registry_rows().copy()
    questionnaire = registry.copy()
    registry["session_id"] = registry["subid"].str.rstrip("_")
    questionnaire["session_id"] = questionnaire["subid"].str.rstrip("_")
    questionnaire.loc[questionnaire["subid"].eq("sub-087_"), "participant_key"] = "P999"
    with pytest.raises(ValueError, match="核心身份字段不一致"):
        validate_questionnaire_registry_consistency(questionnaire, registry)


def test_cohort_membership_is_not_derived_from_questionnaire_and_missing_questionnaire_is_kept() -> None:
    cohort = pd.DataFrame([
        {"session_id": "sub-031", "include": True, "repeat_participant_id": "old-A"},
        {"session_id": "sub-087", "include": True, "repeat_participant_id": "old-A"},
        {"session_id": "sub-102", "include": True, "repeat_participant_id": "old-C"},
    ])
    registry = _registry_rows().copy()
    registry["session_id"] = registry["subid"].str.rstrip("_")
    reconciled = reconcile_cohort_identity(cohort, registry)
    assert set(reconciled["session_id"]) == {"sub-031", "sub-087", "sub-102"}
    assert reconciled.loc[reconciled["session_id"].eq("sub-031"), "participant_group_id"].iloc[0] == "P001"
    assert pd.isna(reconciled.loc[reconciled["session_id"].eq("sub-102"), "participant_key"].iloc[0])
    assert reconciled.loc[reconciled["session_id"].eq("sub-102"), "participant_group_id"].iloc[0] == "legacy:old-C"

    session = pd.DataFrame({"session_id": ["sub-031", "sub-087", "sub-102"], "value": [1, 2, 3]})
    questionnaire = registry.copy()
    joined = left_join_questionnaire(session, questionnaire)
    assert len(joined) == 3
    assert joined.loc[joined["session_id"].eq("sub-102"), "questionnaire_present"].iloc[0] == 0
    assert joined.loc[joined["session_id"].eq("sub-102"), "value"].iloc[0] == 3
