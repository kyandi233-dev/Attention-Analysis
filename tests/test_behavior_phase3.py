from attention_pipeline.behavior.reporting import load_cohort, phase3_tables


def test_candidate_window_cartesian_product_and_block_boundary(config):
    tables = phase3_tables(config, load_cohort(config))
    rolling = tables["rolling_evidence"]
    assert set(rolling["time_window_sec"]) == {30, 60, 90, 120}
    assert set(rolling["nogo_window_target"]) == {6, 8, 12}
    assert (rolling["window_actual_start_ms"] >= rolling.groupby(["subject", "block_num"])["window_actual_start_ms"].transform("min")).all()


def test_probe_evidence_keeps_four_nominal_states_and_candidates(config):
    tables = phase3_tables(config, load_cohort(config))
    probes = tables["probe_evidence"]
    assert len(probes) == 264 * 3 * 3
    assert set(probes["probe_response"]) == {1, 2, 3, 4}
    assert set(probes["time_window_sec"]) == {30, 60, 90}
    assert set(probes["nogo_window_target"]) == {6, 8, 12}
    assert {"nogo_actual_span_sec", "nogo_evidence_age_sec", "window_actual_coverage_sec"}.issubset(probes)
