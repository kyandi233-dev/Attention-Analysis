import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from attention_pipeline.behavior.evidence import rolling_evidence, summarize_window
from attention_pipeline.behavior.extract import block_metrics, extract_trials


def test_rt_is_annotated_never_silently_deleted(config):
    trials = extract_trials(config, "sub-000")
    raw_rows = sum(1 for _ in trials.itertuples())
    assert raw_rows == 6 * 216
    assert trials["rt"].notna().sum() > 0
    assert {"rt_qc_lt_100", "rt_qc_lt_150", "rt_qc_gt_1000", "rt_qc_gt_1150"}.issubset(trials)
    assert trials["condition_x_position"].notna().all()


def test_dprime_uses_correct_go_and_nogo_commission():
    frame = pd.DataFrame({
        "subject_id": ["x"] * 10,
        "block_num": [1] * 10,
        "condition": ["A"] * 10,
        "is_no_go": [0] * 8 + [1] * 2,
        "correct": [1] * 6 + [0] * 2 + [1] * 2,
        "commission": [0] * 8 + [1, 0],
        "rt": [200] * 6 + [np.nan] * 4,
        "omission": [0] * 6 + [1] * 2 + [0] * 2,
    })
    row = block_metrics(frame).iloc[0]
    expected = NormalDist().inv_cdf(6.5 / 9) - NormalDist().inv_cdf(1.5 / 3)
    assert row["correct_go_hits"] == 6
    assert row["nogo_commissions"] == 1
    assert math.isclose(row["dprime_loglinear"], expected)


def _synthetic_trials():
    rows = []
    for block_num, base in ((1, 0), (2, 1_000_000)):
        for i in range(20):
            rows.append({
                "subject_id": "x", "block_num": block_num, "condition": "A",
                "absolute_onset_time": base + i * 1000, "is_no_go": int(i % 4 == 0),
                "commission": int(i == 8), "rt": np.nan if i % 4 == 0 else 250 + i,
                "position_in_cycle": i % 18 + 1,
            })
    return pd.DataFrame(rows)


def test_window_does_not_cross_block(config):
    evidence = rolling_evidence(config, _synthetic_trials())
    assert set(evidence["block_num"]) == {1, 2}
    block2 = evidence.loc[evidence["block_num"].eq(2)]
    assert block2["window_end_ms"].min() >= 1_000_000


def test_nogo_jeffreys_uncertainty_and_status():
    block = _synthetic_trials().query("block_num == 1")
    result = summarize_window(block, end_ms=19_000, duration_sec=30, nogo_n=6)
    assert result["nogo_opportunities_actual"] == 5
    assert result["window_status"] == "insufficient_nogo"
    assert 0 < result["commission_jeffreys_ci95_low"] < result["commission_jeffreys_ci95_high"] < 1

