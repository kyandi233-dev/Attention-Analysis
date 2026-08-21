from attention_pipeline.behavior.reporting import load_cohort, phase2_tables


def test_fixed_cycle_positions_and_nogo_opportunities(config):
    tables = phase2_tables(load_cohort(config))
    position = tables["position_metrics"]
    actual = {
        condition: group.loc[group["is_no_go"].eq(1), "position_in_cycle"].astype(int).tolist()
        for condition, group in position.groupby("condition")
    }
    assert actual == {"A": [5, 9, 14, 18], "B": [5, 14], "C": [5]}


def test_block_and_cycle_tables_preserve_design(config):
    tables = phase2_tables(load_cohort(config))
    assert len(tables["block_metrics"]) == 11 * 6
    assert len(tables["cycle_metrics"]) == 11 * 6 * 12
    assert set(tables["same_condition_pairs"]["condition_occurrence"]) == {1, 2}
