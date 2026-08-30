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
