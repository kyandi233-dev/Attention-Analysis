from __future__ import annotations

import pandas as pd
import pytest

from attention_pipeline.formal_analysis.identity_contract import assert_participant_group_contract


def _base() -> pd.DataFrame:
    return pd.DataFrame({
        "session_id": ["sub-001", "sub-002"],
        "participant_key": ["P001", pd.NA],
        "participant_group_id": ["P001", "legacy:G2"],
        "repeat_participant_id": ["P001", "legacy:G2"],
        "participant_identity_source": ["questionnaire_repeat_registry", "governed_cohort_fallback_no_questionnaire_identity"],
        "participant_identity_resolved_for_clustering": [True, True],
    })


def test_identity_contract_accepts_exact_source_and_alias_alignment() -> None:
    assert_participant_group_contract(_base(), require_resolved=True)


def test_verified_participant_key_must_equal_canonical_group() -> None:
    frame = _base()
    frame.loc[0, "participant_group_id"] = "wrong"
    frame.loc[0, "repeat_participant_id"] = "wrong"
    with pytest.raises(ValueError, match="verified participant_key"):
        assert_participant_group_contract(frame, require_resolved=True)


def test_compatibility_alias_missingness_must_match_canonical_group() -> None:
    frame = _base()
    frame.loc[1, "repeat_participant_id"] = pd.NA
    with pytest.raises(ValueError, match="compatibility alias"):
        assert_participant_group_contract(frame, require_resolved=True)


def test_unresolved_source_cannot_carry_group() -> None:
    frame = _base()
    frame.loc[1, "participant_identity_source"] = "unresolved"
    with pytest.raises(ValueError, match="unresolved cannot carry"):
        assert_participant_group_contract(frame, require_resolved=True)


def test_resolved_flag_must_match_group_missingness() -> None:
    frame = _base()
    frame.loc[1, "participant_identity_resolved_for_clustering"] = False
    with pytest.raises(ValueError, match="disagrees with participant_group_id"):
        assert_participant_group_contract(frame, require_resolved=True)
