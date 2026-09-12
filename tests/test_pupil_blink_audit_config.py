import pandas as pd

from attention_pipeline.nir_formal_analysis.pupil_blink_audit_config import (
    probe_block,
    probe_onset_ms,
    read_table,
)


def test_current_behavior_probe_time_field_is_accepted():
    row = pd.Series({"probe_time_ms": 12345.0})
    assert probe_onset_ms(row) == 12345.0


def test_current_behavior_block_id_is_accepted():
    assert probe_block(pd.Series({"block_id": "B1"})) == 1
    assert probe_block(pd.Series({"block_id": "B2"})) == 2


def test_empty_rgb_blink_event_csv_is_observed_zero_events(tmp_path):
    path = tmp_path / "rgb_blink_candidate_events.csv"
    pd.DataFrame().to_csv(path, index=False)
    assert read_table(path).empty
