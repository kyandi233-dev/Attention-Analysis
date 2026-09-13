from __future__ import annotations

import numpy as np
import pandas as pd

from attention_pipeline.nir_formal_analysis.ocular_postfreeze_analysis import (
    FROZEN_OCULAR_FEATURES,
    PRIMARY_BEHAVIOR_LINKS,
    SENSITIVITY_BEHAVIOR_LINKS,
    _bh_fdr,
    add_ocular_within_between,
    prepare_analysis_table,
)


def _tables():
    rows_o, rows_b = [], []
    for p in range(1, 4):
        pid = f"p{p:02d}"
        for s in range(1, 3):
            sid = f"sub-{p:02d}-{s}"
            for block in ("B1", "B2"):
                for probe in range(1, 11):
                    key = {
                        "participant_group_id": pid,
                        "session_id": sid,
                        "block_id": block,
                        "probe_index_in_block": probe,
                    }
                    o = dict(key)
                    for j, spec in enumerate(FROZEN_OCULAR_FEATURES.values(), start=1):
                        o[spec["column"]] = p + s / 10 + probe / 100 + j / 1000
                    rows_o.append(o)
                    rows_b.append({
                        "repeat_participant_id": pid,
                        "session_id": sid,
                        "block_id": block,
                        "probe_order_in_block": probe,
                        "q1_nominal_4class": (probe - 1) % 4 + 1,
                        "q2_ordinal_4level": (probe - 1) % 4 + 1,
                        "go_correct_rt_cv": 0.1 + probe / 100,
                        "go_correct_rt_theilsen_slope_ms_per_s": probe / 10,
                        "raw_go_omission_rate": (probe % 3) / 10,
                        "commission_rate": (probe % 2) / 10,
                        "omission_numerator": probe % 3,
                        "omission_denominator": 10,
                        "commission_numerator": probe % 2,
                        "commission_denominator": 4,
                    })
    return pd.DataFrame(rows_o), pd.DataFrame(rows_b)


def test_prepare_analysis_table_uses_exact_canonical_probe_identity():
    ocular, behavior = _tables()
    merged, audit = prepare_analysis_table(ocular, behavior)
    assert len(merged) == len(ocular) == len(behavior)
    assert audit["key_universe_exact_match"] is True
    assert set(merged["block_id"].unique()) == {"b1", "b2"}
    assert np.isclose(merged["progression_centered"].min(), -0.5)
    assert np.isclose(merged["progression_centered"].max(), 0.5)


def test_within_between_decomposition_centers_within_participant():
    ocular, behavior = _tables()
    merged, _ = prepare_analysis_table(ocular, behavior)
    col = FROZEN_OCULAR_FEATURES["level_mean"]["column"]
    d = add_ocular_within_between(merged, col)
    means = d.groupby("participant_group_id")["ocular_within"].mean()
    assert np.allclose(means.to_numpy(float), 0.0, atol=1e-12)
    assert d.groupby("participant_group_id")["ocular_between"].nunique().eq(1).all()


def test_behavior_main_contract_does_not_preempt_rt_level_freeze_or_promote_dprime():
    assert "go_correct_rt_mean_ms" not in PRIMARY_BEHAVIOR_LINKS
    assert "go_correct_rt_median_ms" not in PRIMARY_BEHAVIOR_LINKS
    assert "dprime_loglinear" not in PRIMARY_BEHAVIOR_LINKS
    assert "dprime_loglinear" in SENSITIVITY_BEHAVIOR_LINKS
    assert set(PRIMARY_BEHAVIOR_LINKS) == {
        "go_correct_rt_cv",
        "go_correct_rt_theilsen_slope_ms_per_s",
        "raw_go_omission_rate",
        "commission_rate",
    }


def test_first_round_ocular_contract_is_exactly_five_frozen_features():
    assert set(FROZEN_OCULAR_FEATURES) == {
        "level_mean", "variability_mad", "linear_slope_per_sec",
        "quadratic_curvature_per_sec2", "blink_event_rate_per_min",
    }
    assert all("geometry" not in spec["column"] for spec in FROZEN_OCULAR_FEATURES.values())
    pupil_columns = [
        spec["column"]
        for spec in FROZEN_OCULAR_FEATURES.values()
        if "blink" not in spec["column"]
    ]
    assert all("__rseg_hard__rgb_nir_qc__" in col for col in pupil_columns)
    assert all("__rseg_hard__nir_qc__" not in col for col in pupil_columns)


def test_bh_fdr_is_bounded_and_order_consistent():
    p = np.array([0.001, 0.01, 0.04, 0.5, np.nan])
    q = _bh_fdr(p)
    assert np.isnan(q[-1])
    assert np.all((q[:-1] >= p[:-1]) & (q[:-1] <= 1.0))
    assert np.all(np.diff(q[:-1]) >= -1e-12)
