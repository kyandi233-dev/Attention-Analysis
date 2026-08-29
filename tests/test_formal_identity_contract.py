from __future__ import annotations

import pandas as pd
import pytest

from attention_pipeline.formal_analysis.identity_contract import (
    assert_participant_group_contract,
    reconcile_formal_identity,
)


def _registry(rows):
    return pd.DataFrame(rows)


def test_participant_key_is_preferred_and_never_rewritten_from_session() -> None:
    cohort = pd.DataFrame({
        "session_id": ["sub-031", "sub-032"],
        "include": [True, True],
        "repeat_participant_id": ["old-a", "old-b"],
        "identity_status": ["temporary_confirmed", "temporary_confirmed"],
    })
    registry = _registry([
        {"session_id": "sub-031", "participant_key": "participant:A"},
        {"session_id": "sub-032", "participant_key": "participant:B"},
    ])
    out = reconcile_formal_identity(cohort, registry)
    assert out["participant_group_id"].tolist() == ["participant:A", "participant:B"]
    assert out["participant_identity_source"].eq("questionnaire_repeat_registry").all()
    assert not out["participant_group_id"].isin(out["session_id"]).any()
    assert out["repeat_participant_id"].equals(out["participant_group_id"])


def test_questionnaire_missing_session_can_crosswalk_via_verified_overlap() -> None:
    cohort = pd.DataFrame({
        "session_id": ["sub-031", "sub-102"],
        "include": [True, True],
        "repeat_participant_id": ["legacy-p", "legacy-p"],
        "identity_status": ["temporary_confirmed", pd.NA],
    })
    registry = _registry([
        {"session_id": "sub-031", "participant_key": "participant:P"},
    ])
    out = reconcile_formal_identity(cohort, registry)
    row = out.set_index("session_id").loc["sub-102"]
    assert row["participant_group_id"] == "participant:P"
    assert row["participant_identity_source"] == "governed_cohort_crosswalk_for_missing_questionnaire"


def test_legacy_only_group_requires_governance_status() -> None:
    cohort = pd.DataFrame({
        "session_id": ["sub-040", "sub-041"],
        "include": [True, True],
        "repeat_participant_id": ["legacy-ok", "legacy-no"],
        "identity_status": ["temporary_confirmed", "unreviewed"],
    })
    registry = _registry({"session_id": pd.Series(dtype=str), "participant_key": pd.Series(dtype=str)})
    out = reconcile_formal_identity(cohort, registry)
    rows = out.set_index("session_id")
    assert rows.loc["sub-040", "participant_group_id"] == "legacy:legacy-ok"
    assert rows.loc["sub-040", "participant_identity_source"] == "governed_cohort_fallback_no_questionnaire_identity"
    assert pd.isna(rows.loc["sub-041", "participant_group_id"])
    assert rows.loc["sub-041", "participant_identity_source"] == "unresolved"


def test_new_session_with_participant_key_does_not_need_legacy_group_value() -> None:
    cohort = pd.DataFrame({
        "session_id": ["sub-200"],
        "include": [True],
        "repeat_participant_id": [pd.NA],
        "identity_status": [pd.NA],
    })
    registry = _registry([{"session_id": "sub-200", "participant_key": "participant:new"}])
    out = reconcile_formal_identity(cohort, registry)
    assert out.loc[0, "participant_group_id"] == "participant:new"


def test_unresolved_identity_is_retained_but_not_valid_for_inference() -> None:
    cohort = pd.DataFrame({
        "session_id": ["sub-201"],
        "include": [True],
        "repeat_participant_id": [pd.NA],
        "identity_status": [pd.NA],
    })
    out = reconcile_formal_identity(
        cohort,
        _registry({"session_id": pd.Series(dtype=str), "participant_key": pd.Series(dtype=str)}),
    )
    assert out.loc[0, "include"]
    assert pd.isna(out.loc[0, "participant_group_id"])
    assert out.loc[0, "participant_identity_source"] == "unresolved"
    with pytest.raises(ValueError, match="unresolved"):
        assert_participant_group_contract(out, require_resolved=True)
    assert_participant_group_contract(out, require_resolved=False)


def test_compatibility_alias_drift_fails_closed() -> None:
    frame = pd.DataFrame({
        "session_id": ["sub-031"],
        "participant_group_id": ["participant:A"],
        "repeat_participant_id": ["participant:B"],
    })
    with pytest.raises(ValueError, match="drifted"):
        assert_participant_group_contract(frame)


def test_session_id_as_participant_group_is_explicitly_rejected() -> None:
    frame = pd.DataFrame({
        "session_id": ["sub-031"],
        "participant_group_id": ["sub-031"],
        "repeat_participant_id": ["sub-031"],
    })
    with pytest.raises(ValueError, match="session_id"):
        assert_participant_group_contract(frame)
