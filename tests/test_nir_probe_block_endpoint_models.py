"""Contract tests for frozen-endpoint probe and block/session NIR models.

These tests use synthetic data only.  They pin the statistical contracts:

1. Q1 multinomial logistic uses reference category 1 with contrasts 2/3/4 and
   participant-cluster robust covariance;
2. Q2 ordinal models report the cumulative-logit fallback family and note the
   OrdinalGEE environment limitation;
3. within/between decomposition is participant-centered (within sums to zero
   inside each participant group);
4. insufficient samples write ``not_estimable`` failure rows instead of empty
   result tables;
5. block-level omission and commission remain two separate outcomes and are
   never merged into a combined accuracy outcome.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from attention_pipeline.nir_formal_analysis.block_session_models import (
    fit_block_session_models,
)
from attention_pipeline.nir_formal_analysis.probe_pupil_models import (
    Q2_FAMILY,
    _project_windows,
    fit_probe_endpoint_models,
)
from attention_pipeline.nir_formal_analysis.scientific_models import (
    add_pupil_within_between,
)


def _probe_table(rng: np.random.Generator) -> pd.DataFrame:
    """10 participant groups x 8 probes with all four Q1/Q2 levels present."""
    n_groups, probes_per_group = 10, 8
    rows = []
    for group_index in range(n_groups):
        token = f"p{group_index:02d}"
        session = f"s{group_index:02d}"
        base = 40.0 + 4.0 * group_index
        for probe_index in range(probes_per_group):
            rows.append({
                "session_id": session,
                "analysis_group_token": token,
                "block_num": int(probe_index < 4) + 1,
                "probe_index_global": group_index * probes_per_group + probe_index + 1,
                "probe_response": float((probe_index % 4) + 1),
                "probe_vigilance": float(np.repeat([1, 2, 3, 4], 2)[probe_index]),
                "pupil_geom_mean_diameter": base + rng.normal(0.0, 2.0),
                "hard_pupil_fraction": 0.80 + 0.02 * group_index + rng.normal(0.0, 0.02),
            })
    return pd.DataFrame(rows)


def test_probe_within_between_decomposition_is_participant_centered() -> None:
    rng = np.random.default_rng(0)
    table = _probe_table(rng)
    out = add_pupil_within_between(
        table, value_col="pupil_geom_mean_diameter", group_col="analysis_group_token"
    )
    assert out["pupil_between"].notna().all()
    assert out["pupil_within"].notna().all()
    sums = out.groupby("analysis_group_token")["pupil_within"].sum()
    assert np.isclose(sums.to_numpy(dtype=float), 0.0).all()


def test_probe_q1_reference_category_is_1_with_contrasts_2_3_4() -> None:
    rng = np.random.default_rng(1)
    table = _probe_table(rng)
    results, failures = fit_probe_endpoint_models(table)
    q1 = results[results["model_name"].str.startswith("Q1_")]
    assert not q1.empty, f"Q1 must be estimable; failures={failures.to_dict('records')}"
    assert q1["model_family"].eq("MNLogit_cluster_robust").all()
    assert q1["reference_category"].eq(1).all()
    assert set(q1["contrast_category"].astype(int)) == {2, 3, 4}
    assert set(q1["pupil_term"]) == {"pupil_within", "pupil_between"}
    assert q1["status"].eq("estimable").all()
    assert q1["estimate"].notna().all() and q1["se"].notna().all()
    assert (q1["ci_high"] >= q1["ci_low"]).all()


def test_probe_q2_uses_cumulative_logit_fallback_and_declares_it() -> None:
    rng = np.random.default_rng(2)
    table = _probe_table(rng)
    results, _ = fit_probe_endpoint_models(table)
    q2 = results[results["model_name"].str.startswith("Q2_")]
    assert not q2.empty, "Q2 must be estimable on the full synthetic table"
    assert q2["model_family"].eq(Q2_FAMILY).all()
    assert q2["family_note"].str.contains("OrdinalGEE").all()
    assert set(q2["pupil_term"]) == {"pupil_within", "pupil_between"}
    assert q2["status"].eq("estimable").all()


def test_probe_models_fail_closed_when_samples_insufficient() -> None:
    rng = np.random.default_rng(3)
    table = _probe_table(rng).iloc[:4]  # 2 groups x 2 probes, below all gates
    results, failures = fit_probe_endpoint_models(table)
    assert results.empty
    assert not failures.empty
    assert failures["status"].eq("not_estimable").all()
    labels = set(failures["model_name"])
    assert any(name.startswith("Q1_") for name in labels)
    assert any(name.startswith("Q2_") for name in labels)


def test_probe_q1_single_category_is_not_estimable() -> None:
    rng = np.random.default_rng(4)
    table = _probe_table(rng)
    table["probe_response"] = 1.0  # one category only
    results, failures = fit_probe_endpoint_models(table)
    q1_failures = failures[failures["outcome"].eq("probe_response")]
    assert not q1_failures.empty
    assert q1_failures["status"].eq("not_estimable").all()
    assert q1_failures["reason"].str.contains("categories").all()
    assert results[results["model_name"].str.startswith("Q1_")].empty


def test_project_windows_fuses_eyes_and_respects_boundaries() -> None:
    sidecar = pd.DataFrame({
        "eye": ["left"] * 3 + ["right"] * 3,
        "unix_ms": [0, 10, 20, 0, 10, 20],
        "pupil_geom_mean_diameter__raw": [2.0, 4.0, 6.0, 10.0, 20.0, 30.0],
        "pupil_geom_mean_diameter__valid_primary": [True] * 6,
    })
    windows = pd.DataFrame({
        "window_start_ms": [0.0, 15.0],
        "window_end_ms": [15.0, 25.0],
    })
    out = _project_windows(sidecar, windows, ("pupil_geom_mean_diameter",))
    assert out.loc[0, "pupil_geom_mean_diameter"] == 9.0  # (3 + 15) / 2 binocular
    assert out.loc[0, "pupil_geom_mean_diameter_eye_source"] == "binocular"
    assert out.loc[1, "pupil_geom_mean_diameter"] == 18.0  # (6 + 30) / 2
    assert out.loc[0, "pupil_geom_mean_diameter_n_valid"] == 4
    assert out.loc[0, "pupil_geom_mean_diameter_valid_fraction"] == 1.0


def test_project_windows_falls_back_to_single_eye() -> None:
    sidecar = pd.DataFrame({
        "eye": ["left", "left", "right", "right"],
        "unix_ms": [0, 10, 0, 10],
        "hard_pupil_fraction__raw": [0.5, 0.7, np.nan, np.nan],
        "hard_pupil_fraction__valid_primary": [True, True, False, False],
    })
    windows = pd.DataFrame({"window_start_ms": [0.0], "window_end_ms": [15.0]})
    out = _project_windows(sidecar, windows, ("hard_pupil_fraction",))
    assert out.loc[0, "hard_pupil_fraction"] == 0.6  # left-only median of 0.5/0.7
    assert out.loc[0, "hard_pupil_fraction_eye_source"] == "left_only"
    assert out.loc[0, "hard_pupil_fraction_n_valid"] == 2


def _block_table(rng: np.random.Generator) -> pd.DataFrame:
    """10 participant groups x 1 session x 4 blocks with behavior outcomes."""
    rows = []
    for group_index in range(10):
        token = f"p{group_index:02d}"
        session = f"s{group_index:02d}"
        base = 40.0 + 4.0 * group_index
        for block_num in range(1, 5):
            raw = base + rng.normal(0.0, 1.0)
            omission_n = int(rng.integers(0, 6))
            commission_n = int(rng.integers(0, 4))
            rows.append({
                "session_id": session,
                "analysis_group_token": token,
                "block_num": block_num,
                "metric": "pupil_geom_mean_diameter",
                "binocular_raw_median": raw,
                "binocular_centered_median": raw - base,
                "dprime_loglinear": 1.0 + 0.2 * (raw - base) + rng.normal(0.0, 0.3),
                "criterion_c": -0.1 + rng.normal(0.0, 0.2),
                "beta": 0.8 + rng.normal(0.0, 0.2),
                "go_correct_rt_mean_ms": 420.0 + rng.normal(0.0, 25.0),
                "go_correct_rt_median_ms": 400.0 + rng.normal(0.0, 25.0),
                "go_correct_rt_sd_ms": 80.0 + rng.normal(0.0, 10.0),
                "go_correct_rt_cv": 0.19 + rng.normal(0.0, 0.02),
                "go_correct_rt_theilsen_slope_ms_per_s": rng.normal(0.0, 0.5),
                "omission_rate": omission_n / 20.0,
                "omission_numerator": omission_n,
                "omission_denominator": 20,
                "commission_rate": commission_n / 10.0,
                "commission_numerator": commission_n,
                "commission_denominator": 10,
            })
    return pd.DataFrame(rows)


def test_block_continuous_outcomes_use_participant_lmm() -> None:
    rng = np.random.default_rng(5)
    table = _block_table(rng)
    results, failures = fit_block_session_models(table)
    assert not results.empty, f"expected estimable block models; failures={failures.to_dict('records')}"
    lmm = results[results["model_family"].eq("LMM")]
    assert not lmm.empty
    assert lmm["outcome"].eq("dprime_loglinear").any()
    assert set(lmm["pupil_term"]) == {"pupil_within", "pupil_between"}
    assert lmm["status"].eq("estimable").all()
    assert lmm["estimate"].notna().all() and lmm["se"].notna().all()


def test_block_rate_outcomes_use_binomial_gee() -> None:
    rng = np.random.default_rng(6)
    table = _block_table(rng)
    results, failures = fit_block_session_models(table)
    gee = results[results["model_family"].eq("GEE_binomial_exchangeable")]
    assert not gee.empty, f"expected estimable rate models; failures={failures.to_dict('records')}"
    assert set(gee["outcome"]) == {"omission_rate", "commission_rate"}
    assert gee["status"].eq("estimable").all()
    assert (gee["ci_high"] >= gee["ci_low"]).all()


def test_block_omission_and_commission_are_never_collapsed() -> None:
    rng = np.random.default_rng(7)
    table = _block_table(rng)
    _, failures = fit_block_session_models(table, min_participant_groups=99, min_rows=999)
    labels = set(failures["model_name"])
    assert any("omission_rate" in name for name in labels)
    assert any("commission_rate" in name for name in labels)
    assert not any("combined" in name or "correct_combined" in name for name in labels)
    assert failures["status"].eq("not_estimable").all()


def test_block_models_fail_closed_when_table_empty() -> None:
    results, failures = fit_block_session_models(pd.DataFrame())
    assert results.empty
    assert not failures.empty
    assert failures["status"].eq("not_estimable").all()
    assert failures["reason"].eq("empty_block_session_table").all()
