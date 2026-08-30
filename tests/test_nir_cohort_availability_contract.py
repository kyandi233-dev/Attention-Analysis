from __future__ import annotations

import pandas as pd

import attention_pipeline.nir_analysis_ready as nir_ready
from attention_pipeline.nir_pupil_only import cohort_topology_summary, validate_cohort_topology


def _row(session, group, size):
    return {
        "session_id": session,
        "analysis_group_token": group,
        "source_schema_version": 6,
        "repeat_group_size": size,
    }


def test_nir_topology_allows_more_than_two_sessions_per_participant():
    rows = [
        _row("sub-001", "p1", 3),
        _row("sub-002", "p1", 3),
        _row("sub-003", "p1", 3),
        _row("sub-004", "p2", 2),
        _row("sub-005", "p2", 2),
        _row("sub-006", "p3", 1),
    ]
    observed = cohort_topology_summary(rows)
    assert observed == {
        "n_sessions": 6,
        "n_analysis_groups": 3,
        "n_double_session_repeat_groups": 1,
    }
    assert validate_cohort_topology(
        rows,
        expected_sessions=6,
        expected_analysis_groups=3,
        expected_double_session_repeat_groups=1,
    ) == observed


def test_full_contract_separates_governed_cohort_from_nir_availability(monkeypatch):
    cohort = pd.DataFrame(
        [
            {"session_id": "sub-001", "include": True, "repeat_participant_id": "p1"},
            {"session_id": "sub-002", "include": True, "repeat_participant_id": "p1"},
            {"session_id": "sub-003", "include": True, "repeat_participant_id": "p1"},
            {"session_id": "sub-004", "include": True, "repeat_participant_id": "p2"},
        ]
    )
    records = [
        _row("sub-001", "p1", 3),
        _row("sub-002", "p1", 3),
        _row("sub-004", "p2", 1),
    ]
    source_manifest = {
        "session_count": 3,
        "unavailable_session_count": 1,
        "sessions": records,
        "unavailable_sessions": [
            {
                "session_id": "sub-003",
                "status": "source_missing",
                "reason": "no_current_contract_source",
            }
        ],
    }

    class FakeConfig:
        def section(self, name):
            if name == "cohort_topology":
                return {
                    "sessions": 4,
                    "analysis_groups": 2,
                    "double_session_repeat_groups": 0,
                }
            raise KeyError(name)

    monkeypatch.setattr(nir_ready, "load_config", lambda *_args, **_kwargs: FakeConfig())
    monkeypatch.setattr(nir_ready, "load_cohort_manifest", lambda *_args, **_kwargs: cohort)
    monkeypatch.setattr(
        nir_ready,
        "load_source_manifest",
        lambda *_args, **_kwargs: (source_manifest, records),
    )

    contract = nir_ready._full_contract("unused.yaml")
    assert contract["canonical_cohort_topology"] == {
        "n_sessions": 4,
        "n_analysis_groups": 2,
        "n_double_session_repeat_groups": 0,
    }
    assert contract["nir_availability"] == {
        "n_available_sessions": 3,
        "n_unavailable_sessions": 1,
        "n_canonical_sessions": 4,
        "complete_accounting": True,
        "availability_does_not_redefine_cohort": True,
    }
    assert contract["source_available_subset_topology"] == {
        "n_sessions": 3,
        "n_analysis_groups": 2,
        "n_double_session_repeat_groups": 0,
    }
