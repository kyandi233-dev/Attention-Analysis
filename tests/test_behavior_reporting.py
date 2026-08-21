from attention_pipeline.behavior.reporting import load_cohort, phase1_tables


def test_phase1_cohort_audit_preserves_all_formal_trials(config):
    cohort = load_cohort(config)
    tables = phase1_tables(config, cohort)
    assert len(cohort) == 11 * 6 * 216
    assert int(cohort["is_probe"].sum()) == 11 * 6 * 4
    assert tables["subject_block_audit"]["trials"].eq(216).all()
    assert int(tables["summary"].set_index("metric").loc["duplicate_subject_block_trial", "value"]) == 0


def test_probe_states_remain_separate_categories(config):
    cohort = load_cohort(config)
    counts = cohort.loc[cohort["is_probe"].eq(1), "probe_response"].value_counts()
    assert counts.to_dict() == {1: 214, 2: 29, 3: 14, 4: 7}

