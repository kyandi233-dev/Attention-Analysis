from __future__ import annotations

import pandas as pd

from attention_pipeline.behavior_formal.science_v3 import (
    PROBE_MODEL_METRICS,
    BehaviorScienceConfig,
    fit_q1_nominal,
    fit_q2_ordinal,
)


FORMAL = {
    "raw_go_omission_rate",
    "clean_go_omission_rate",
    "timing_ambiguous_go_omission_rate",
}


def _small_probe() -> pd.DataFrame:
    rows = []
    for i in range(8):
        raw = 0.04 + i * 0.005
        clean = raw * 0.6
        ambiguous = raw - clean
        rows.append({
            "repeat_participant_id": f"P{i % 2}",
            "session_id": f"sub-{31 + i:03d}",
            "q1_nominal_4class": i % 4 + 1,
            "q2_ordinal_4level": i % 4 + 1,
            "raw_go_omission_rate": raw,
            "clean_go_omission_rate": clean,
            "timing_ambiguous_go_omission_rate": ambiguous,
        })
    return pd.DataFrame(rows)


def test_probe_model_inventory_uses_explicit_formal_omission_names_once() -> None:
    assert FORMAL.issubset(set(PROBE_MODEL_METRICS))
    assert "omission_rate" not in PROBE_MODEL_METRICS
    for metric in FORMAL:
        assert PROBE_MODEL_METRICS.count(metric) == 1


def test_q1_q2_fail_closed_rows_name_all_formal_omission_predictors() -> None:
    frame = _small_probe()
    cfg = BehaviorScienceConfig(min_model_rows=50, min_participant_groups=6)
    q1, q1_fail = fit_q1_nominal(frame, cfg)
    q2, q2_fail = fit_q2_ordinal(frame, cfg)
    assert q1.empty
    assert q2.empty
    assert FORMAL.issubset(set(q1_fail["predictor"]))
    assert FORMAL.issubset(set(q2_fail["predictor"]))
    assert q1_fail[q1_fail["predictor"].isin(FORMAL)]["status"].eq("not_estimable").all()
    assert q2_fail[q2_fail["predictor"].isin(FORMAL)]["status"].eq("not_estimable").all()
