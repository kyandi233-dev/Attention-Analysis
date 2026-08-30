"""Governed cohort topology helpers for the pupil-only NIR route.

A repeat participant may contribute any positive number of sessions.  The
number of exactly-two-session groups is a descriptive/audit count, not a
constraint that caps group size at two.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from .contract import SourceIdentity


def cohort_topology_summary(records: Iterable[Mapping[str, object]]) -> dict[str, Any]:
    rows = [SourceIdentity.from_mapping(value) for value in records]
    sessions = {row.session_id for row in rows}
    if len(sessions) != len(rows):
        raise ValueError("topology manifest contains duplicate session_id")
    groups: dict[str, list[str]] = {}
    for row in rows:
        groups.setdefault(row.analysis_group_token, []).append(row.session_id)
    sizes = [len(value) for value in groups.values()]
    if any(size < 1 for size in sizes):
        raise ValueError("analysis groups must contain at least one session")
    distribution = {str(size): int(sizes.count(size)) for size in sorted(set(sizes))}
    return {
        "n_sessions": len(sessions),
        "n_analysis_groups": len(groups),
        "n_repeated_participant_groups": int(sum(size > 1 for size in sizes)),
        "max_sessions_per_participant": int(max(sizes, default=0)),
        "group_size_distribution": distribution,
    }


def validate_cohort_topology(
    records: Iterable[Mapping[str, object]],
    *,
    expected_sessions: int = 116,
    expected_analysis_groups: int = 61,
) -> dict[str, Any]:
    """Validate governed membership while allowing any positive visit count."""
    summary = cohort_topology_summary(records)
    expected = {
        "n_sessions": int(expected_sessions),
        "n_analysis_groups": int(expected_analysis_groups),
    }
    observed = {key: int(summary[key]) for key in expected}
    if observed != expected:
        raise ValueError(f"cohort topology mismatch: observed={observed}, expected={expected}")
    return summary
