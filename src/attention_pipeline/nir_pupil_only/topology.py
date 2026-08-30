"""Governed cohort topology helpers for the pupil-only NIR route.

A repeat participant may contribute any positive number of sessions.  The
number of exactly-two-session groups is a descriptive/audit count, not a
constraint that caps group size at two.
"""
from __future__ import annotations

from typing import Iterable, Mapping

from .contract import SourceIdentity


def cohort_topology_summary(records: Iterable[Mapping[str, object]]) -> dict[str, int]:
    rows = [SourceIdentity.from_mapping(value) for value in records]
    sessions = {row.session_id for row in rows}
    if len(sessions) != len(rows):
        raise ValueError("topology manifest contains duplicate session_id")
    groups: dict[str, list[str]] = {}
    for row in rows:
        groups.setdefault(row.analysis_group_token, []).append(row.session_id)
    if any(len(value) < 1 for value in groups.values()):
        raise ValueError("analysis groups must contain at least one session")
    return {
        "n_sessions": len(sessions),
        "n_analysis_groups": len(groups),
        "n_double_session_repeat_groups": int(sum(len(value) == 2 for value in groups.values())),
    }


def validate_cohort_topology(
    records: Iterable[Mapping[str, object]],
    *,
    expected_sessions: int = 116,
    expected_analysis_groups: int = 61,
    expected_double_session_repeat_groups: int = 10,
) -> dict[str, int]:
    summary = cohort_topology_summary(records)
    expected = {
        "n_sessions": int(expected_sessions),
        "n_analysis_groups": int(expected_analysis_groups),
        "n_double_session_repeat_groups": int(expected_double_session_repeat_groups),
    }
    if summary != expected:
        raise ValueError(f"cohort topology mismatch: observed={summary}, expected={expected}")
    return summary
