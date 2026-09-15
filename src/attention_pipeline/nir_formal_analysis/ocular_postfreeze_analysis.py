"""Frozen-post Ocular explanatory science layer for FocusWave.

This module starts from the already-frozen FormalScience/Ocular probe table.
It never reopens P3 representation selection, never re-runs NIR producers, and
never mutates the supervised feature registry.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

SCHEMA_VERSION = "ocular-postfreeze-science-v1"
Q1_REFERENCE_CATEGORY = 1
Q1_CATEGORIES = (1, 2, 3, 4)
Q2_LEVELS = (1, 2, 3, 4)
MIN_MODEL_ROWS = 24
MIN_PARTICIPANT_GROUPS = 6

FROZEN_OCULAR_FEATURES: dict[str, dict[str, str]] = {
    "level_mean": {
        "column": "ocular__rseg_hard__rgb_nir_qc__level_mean__2s",
        "label": "Pupil level (hard R_seg mean)",
        "unit": "proportion",
    },
    "variability_mad": {
        "column": "ocular__rseg_hard__rgb_nir_qc__variability_mad__2s",
        "label": "Pupil variability (hard R_seg MAD)",
        "unit": "proportion",
    },
    "linear_slope_per_sec": {
        "column": "ocular__rseg_hard__rgb_nir_qc__linear_slope_per_sec__2s",
        "label": "Pupil linear slope",
        "unit": "proportion/s",
    },
    "quadratic_curvature_per_sec2": {
        "column": "ocular__rseg_hard__rgb_nir_qc__quadratic_curvature_per_sec2__2s",
        "label": "Pupil quadratic curvature",
        "unit": "proportion/s^2",
    },
    "blink_event_rate_per_min": {
        "column": "ocular__blink_event_rate_per_min__pre30s",
        "label": "Blink event rate",
        "unit": "events/min",
    },
}

# RT mean/median are intentionally excluded until the Behavior researcher freeze.
PRIMARY_BEHAVIOR_LINKS: dict[str, dict[str, str]] = {
    "go_correct_rt_cv": {"family": "gaussian_gee", "unit": "ratio"},
    "go_correct_rt_theilsen_slope_ms_per_s": {"family": "gaussian_gee", "unit": "ms/s"},
    "raw_go_omission_rate": {
        "family": "binomial_gee", "unit": "proportion",
        "numerator": "omission_numerator", "denominator": "omission_denominator",
    },
    "commission_rate": {
        "family": "binomial_gee", "unit": "proportion",
        "numerator": "commission_numerator", "denominator": "commission_denominator",
    },
}
SENSITIVITY_BEHAVIOR_LINKS = (
    "dprime_loglinear", "go_correct_rt_sd_ms", "go_correct_rt_mad_ms", "go_correct_rt_iqr_ms",
)
CANONICAL_KEYS = (
    "participant_group_id", "session_id", "block_id", "probe_index_in_block",
)


@dataclass(frozen=True)
class AnalysisInputs:
    ocular_wide: Path
    behavior_probe: Path
    ocular_root: Path
    analysis_code_sha: str
    authoritative: bool = True


def _read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    return pd.read_csv(p, encoding="utf-8-sig", low_memory=False)


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_z(series: pd.Series) -> pd.Series:
    x = _numeric(series)
    sd = float(x.std(ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=x.index, dtype=float)
    return (x - float(x.mean())) / sd


def _normalize_block(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    return text.replace({"1": "b1", "2": "b2", "block1": "b1", "block2": "b2"})


def normalize_ocular_identity(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    missing = sorted(set(CANONICAL_KEYS) - set(out.columns))
    if missing:
        raise ValueError(f"Ocular frozen table missing canonical keys: {missing}")
    out["participant_group_id"] = out["participant_group_id"].astype("string").str.strip()
    out["session_id"] = out["session_id"].astype("string").str.strip()
    out["block_id"] = _normalize_block(out["block_id"])
    out["probe_index_in_block"] = _numeric(out["probe_index_in_block"]).astype("Int64")
    if out[list(CANONICAL_KEYS)].isna().any(axis=None):
        raise ValueError("Ocular frozen table contains missing canonical identity")
    if out.duplicated(list(CANONICAL_KEYS)).any():
        raise ValueError("Ocular frozen table is not unique on canonical probe key")
    return out


def normalize_behavior_identity(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "participant_group_id" not in out and "repeat_participant_id" in out:
        out["participant_group_id"] = out["repeat_participant_id"]
    if "probe_index_in_block" not in out and "probe_order_in_block" in out:
        out["probe_index_in_block"] = out["probe_order_in_block"]
    if "q1_nominal_4class" not in out and "probe_response" in out:
        out["q1_nominal_4class"] = out["probe_response"]
    if "q2_ordinal_4level" not in out and "probe_vigilance" in out:
        out["q2_ordinal_4level"] = out["probe_vigilance"]
    missing = sorted(set(CANONICAL_KEYS) - set(out.columns))
    if missing:
        raise ValueError(f"Behavior probe table missing canonical keys: {missing}")
    out["participant_group_id"] = out["participant_group_id"].astype("string").str.strip()
    out["session_id"] = out["session_id"].astype("string").str.strip()
    out["block_id"] = _normalize_block(out["block_id"])
    out["probe_index_in_block"] = _numeric(out["probe_index_in_block"]).astype("Int64")
    if out[list(CANONICAL_KEYS)].isna().any(axis=None):
        raise ValueError("Behavior probe table contains missing canonical identity")
    if out.duplicated(list(CANONICAL_KEYS)).any():
        raise ValueError("Behavior probe table is not unique on canonical probe key")
    return out


def _key_tuples(frame: pd.DataFrame) -> set[tuple[str, str, str, int]]:
    return set(zip(
        frame["participant_group_id"].astype(str), frame["session_id"].astype(str),
        frame["block_id"].astype(str), frame["probe_index_in_block"].astype(int),
    ))


def prepare_analysis_table(
    ocular: pd.DataFrame,
    behavior: pd.DataFrame,
    *,
    require_exact_key_universe: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join frozen Ocular and formal Behavior on the canonical probe key."""
    o = normalize_ocular_identity(ocular)
    b = normalize_behavior_identity(behavior)
    okeys, bkeys = _key_tuples(o), _key_tuples(b)
    only_o, only_b = okeys - bkeys, bkeys - okeys
    if require_exact_key_universe and (only_o or only_b):
        raise ValueError(
            f"canonical probe universes differ: ocular_only={len(only_o)}, behavior_only={len(only_b)}"
        )
    keep_behavior = list(CANONICAL_KEYS) + [
        c for c in (
            "q1_nominal_4class", "q2_ordinal_4level", *PRIMARY_BEHAVIOR_LINKS.keys(),
            "omission_numerator", "omission_denominator", "commission_numerator", "commission_denominator",
            *SENSITIVITY_BEHAVIOR_LINKS,
        ) if c in b.columns
    ]
    merged = o.merge(b[keep_behavior], on=list(CANONICAL_KEYS), how="inner", validate="one_to_one")
    merged["block_b2"] = merged["block_id"].astype(str).eq("b2").astype(int)
    probe_index = _numeric(merged["probe_index_in_block"])
    merged["progression_centered"] = (probe_index - 1.0) / 9.0 - 0.5
    audit = {
        "ocular_probe_n": int(len(o)), "behavior_probe_n": int(len(b)), "merged_probe_n": int(len(merged)),
        "ocular_only_key_n": int(len(only_o)), "behavior_only_key_n": int(len(only_b)),
        "participant_group_n": int(merged["participant_group_id"].nunique()),
        "session_n": int(merged["session_id"].nunique()),
        "key_universe_exact_match": not only_o and not only_b,
    }
    return merged, audit


def add_ocular_within_between(
    frame: pd.DataFrame, feature_col: str, *, participant_col: str = "participant_group_id",
) -> pd.DataFrame:
    out = frame.copy()
    value = _numeric(out[feature_col])
    between = value.groupby(out[participant_col].astype(str)).transform("mean")
    out["ocular_between"] = between
    out["ocular_within"] = value - between
    out["ocular_finite_n_within_participant"] = value.notna().groupby(
        out[participant_col].astype(str)
    ).transform("sum")
    return out


def _fit_gate(fit: Any) -> str | None:
    if getattr(fit, "converged", True) is False:
        return "model_did_not_converge"
    params = np.asarray(fit.params, dtype=float)
    bse = np.asarray(fit.bse, dtype=float)
    if params.size == 0 or not np.isfinite(params).all() or not np.isfinite(bse).all():
        return "nonfinite_parameter_or_se"
    return None


def _bh_fdr(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    vals = p[finite]
    order = np.argsort(vals)
    ranked = vals[order]
    n = len(ranked)
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    restored = np.empty_like(adj)
    restored[order] = adj
    out[np.flatnonzero(finite)] = restored
    return out


def add_family_fdr(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        out["q_value_bh_within_family"] = pd.Series(dtype=float)
    elif "p_value" not in out:
        out["q_value_bh_within_family"] = np.nan
    else:
        out["q_value_bh_within_family"] = _bh_fdr(_numeric(out["p_value"]))
    return out


def _failure(
    analysis_family: str, feature_id: str, outcome: str, model_family: str,
    data: pd.DataFrame, reason: str,
) -> dict[str, Any]:
    return {
        "analysis_family": analysis_family, "feature_id": feature_id, "outcome": outcome,
        "model_family": model_family, "status": "not_estimable", "reason": reason,
        "n_rows": int(len(data)),
        "participant_group_n": int(data.get("participant_group_id", pd.Series(dtype=str)).astype(str).nunique()),
        "session_n": int(data.get("session_id", pd.Series(dtype=str)).astype(str).nunique()),
    }


def _sample_gate(d: pd.DataFrame, min_rows: int, min_groups: int) -> str | None:
    if len(d) < min_rows:
        return "insufficient_rows"
    if d["participant_group_id"].astype(str).nunique() < min_groups:
        return "insufficient_participant_groups"
    return None


def _base_result(
    analysis_family: str, feature_id: str, feature_col: str, outcome: str,
    model_family: str, term: str, estimate: float, se: float, p_value: float,
    d: pd.DataFrame, *, effect_scale: str,
    reference_category: float | int | None = None,
    contrast_category: float | int | None = None,
) -> dict[str, Any]:
    return {
        "analysis_family": analysis_family, "feature_id": feature_id,
        "ocular_predictor": feature_col, "outcome": outcome, "model_family": model_family,
        "term": term, "estimate": estimate, "se": se,
        "ci_low": estimate - 1.96 * se, "ci_high": estimate + 1.96 * se,
        "p_value": p_value, "reference_category": reference_category,
        "contrast_category": contrast_category, "effect_scale": effect_scale,
        "n_rows": int(len(d)),
        "participant_group_n": int(d["participant_group_id"].astype(str).nunique()),
        "session_n": int(d["session_id"].astype(str).nunique()),
        "cluster_level": "participant_group_id", "status": "estimable",
        "selection_use": "never_for_feature_selection",
    }


def fit_task_progression_models(
    table: pd.DataFrame, *, min_rows: int = MIN_MODEL_ROWS, min_groups: int = MIN_PARTICIPANT_GROUPS,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    family = "GEE_Gaussian_exchangeable"
    for feature_id, spec in FROZEN_OCULAR_FEATURES.items():
        col = spec["column"]
        if col not in table:
            failures.append(_failure("task_progression", feature_id, col, family, table, "feature_column_missing"))
            continue
        d = table.dropna(subset=[col, "block_b2", "progression_centered", "participant_group_id", "session_id"]).copy()
        gate = _sample_gate(d, min_rows, min_groups)
        if gate or _numeric(d[col]).nunique() < 2:
            failures.append(_failure("task_progression", feature_id, col, family, d, gate or "outcome_has_single_level"))
            continue
        try:
            fit = sm.GEE.from_formula(
                f"{col} ~ block_b2 * progression_centered", groups="participant_group_id", data=d,
                family=sm.families.Gaussian(), cov_struct=sm.cov_struct.Exchangeable(),
            ).fit(maxiter=100)
            reason = _fit_gate(fit)
            if reason:
                failures.append(_failure("task_progression", feature_id, col, family, d, reason))
                continue
            for term in ("block_b2", "progression_centered", "block_b2:progression_centered"):
                if term in fit.params:
                    rows.append(_base_result(
                        "task_progression", feature_id, col, col, family, term,
                        float(fit.params[term]), float(fit.bse[term]), float(fit.pvalues[term]), d,
                        effect_scale="native ocular units; progression is centered full-block fraction",
                    ))
        except Exception as exc:
            failures.append(_failure("task_progression", feature_id, col, family, d, f"{type(exc).__name__}: {exc}"))
    return add_family_fdr(pd.DataFrame(rows)), failures


def _decomposed_model_frame(table: pd.DataFrame, feature_col: str, required: list[str]) -> pd.DataFrame:
    d = add_ocular_within_between(table, feature_col)
    d = d.dropna(subset=[*required, "ocular_within", "ocular_between", "participant_group_id", "session_id"]).copy()
    d["ocular_within_z"] = _safe_z(d["ocular_within"])
    d["ocular_between_z"] = _safe_z(d["ocular_between"])
    return d.dropna(subset=["ocular_within_z", "ocular_between_z"])


def fit_q1_models(
    table: pd.DataFrame, *, min_rows: int = MIN_MODEL_ROWS, min_groups: int = MIN_PARTICIPANT_GROUPS,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    outcome, family = "q1_nominal_4class", "MNLogit_cluster_robust"
    for feature_id, spec in FROZEN_OCULAR_FEATURES.items():
        col = spec["column"]
        if col not in table or outcome not in table:
            failures.append(_failure("q1", feature_id, outcome, family, table, "feature_or_outcome_missing"))
            continue
        d = _decomposed_model_frame(table, col, [outcome, "block_b2", "progression_centered"])
        levels = sorted(_numeric(d[outcome]).dropna().astype(int).unique())
        gate = _sample_gate(d, min_rows, min_groups)
        if set(levels) != set(Q1_CATEGORIES):
            gate = "all_Q1_categories_1_4_required"
        if gate:
            failures.append(_failure("q1", feature_id, outcome, family, d, gate))
            continue
        try:
            ordered = [Q1_REFERENCE_CATEGORY] + [x for x in levels if x != Q1_REFERENCE_CATEGORY]
            mapping = {level: idx for idx, level in enumerate(ordered)}
            y = _numeric(d[outcome]).astype(int).map(mapping).astype(int)
            x = pd.DataFrame({
                "ocular_within_z": d["ocular_within_z"], "ocular_between_z": d["ocular_between_z"],
                "block_b2": _numeric(d["block_b2"]), "progression_centered": _numeric(d["progression_centered"]),
            }, index=d.index)
            x = sm.add_constant(x, has_constant="add").astype(float)
            groups = pd.factorize(d["participant_group_id"].astype(str))[0]
            fit = sm.MNLogit(y, x).fit(
                method="newton", maxiter=300, disp=False,
                cov_type="cluster", cov_kwds={"groups": groups},
            )
            reason = _fit_gate(fit)
            if reason:
                failures.append(_failure("q1", feature_id, outcome, family, d, reason))
                continue
            params, bse, pvals = pd.DataFrame(fit.params), pd.DataFrame(fit.bse), pd.DataFrame(fit.pvalues)
            for eq_index, category in enumerate(ordered[1:]):
                if eq_index not in params.columns:
                    continue
                for term in ("ocular_within_z", "ocular_between_z"):
                    est, se = float(params.loc[term, eq_index]), float(bse.loc[term, eq_index])
                    row = _base_result(
                        "q1", feature_id, col, outcome, family, term, est, se,
                        float(pvals.loc[term, eq_index]), d,
                        effect_scale="log-odds per 1 SD component",
                        reference_category=Q1_REFERENCE_CATEGORY, contrast_category=int(category),
                    )
                    row.update({
                        "odds_ratio": math.exp(est), "or_ci_low": math.exp(est - 1.96 * se),
                        "or_ci_high": math.exp(est + 1.96 * se),
                    })
                    rows.append(row)
        except Exception as exc:
            failures.append(_failure("q1", feature_id, outcome, family, d, f"{type(exc).__name__}: {exc}"))
    return add_family_fdr(pd.DataFrame(rows)), failures


def fit_q2_models(
    table: pd.DataFrame, *, min_rows: int = MIN_MODEL_ROWS, min_groups: int = MIN_PARTICIPANT_GROUPS,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    outcome = "q2_ordinal_4level"
    family = "OrderedModel_cumulative_logit_cluster_robust"
    for feature_id, spec in FROZEN_OCULAR_FEATURES.items():
        col = spec["column"]
        if col not in table or outcome not in table:
            failures.append(_failure("q2", feature_id, outcome, family, table, "feature_or_outcome_missing"))
            continue
        d = _decomposed_model_frame(table, col, [outcome, "block_b2", "progression_centered"])
        levels = sorted(_numeric(d[outcome]).dropna().astype(int).unique())
        gate = _sample_gate(d, min_rows, min_groups)
        if not set(levels).issubset(set(Q2_LEVELS)) or len(levels) < 3:
            gate = "Q2_requires_at_least_three_levels_within_1_4"
        if gate:
            failures.append(_failure("q2", feature_id, outcome, family, d, gate))
            continue
        try:
            x = pd.DataFrame({
                "ocular_within_z": d["ocular_within_z"], "ocular_between_z": d["ocular_between_z"],
                "block_b2": _numeric(d["block_b2"]), "progression_centered": _numeric(d["progression_centered"]),
            }, index=d.index).astype(float)
            y = _numeric(d[outcome]).astype(int)
            groups = pd.factorize(d["participant_group_id"].astype(str))[0]
            fit = OrderedModel(y, x, distr="logit").fit(
                method="bfgs", maxiter=300, disp=False,
                cov_type="cluster", cov_kwds={"groups": groups},
            )
            reason = _fit_gate(fit)
            if reason:
                failures.append(_failure("q2", feature_id, outcome, family, d, reason))
                continue
            for term in ("ocular_within_z", "ocular_between_z"):
                if term not in fit.params:
                    continue
                est, se = float(fit.params[term]), float(fit.bse[term])
                row = _base_result(
                    "q2", feature_id, col, outcome, family, term, est, se, float(fit.pvalues[term]), d,
                    effect_scale="proportional log-odds per 1 SD component",
                )
                row.update({
                    "odds_ratio": math.exp(est), "or_ci_low": math.exp(est - 1.96 * se),
                    "or_ci_high": math.exp(est + 1.96 * se),
                })
                rows.append(row)
        except Exception as exc:
            failures.append(_failure("q2", feature_id, outcome, family, d, f"{type(exc).__name__}: {exc}"))
    return add_family_fdr(pd.DataFrame(rows)), failures


def fit_behavior_link_models(
    table: pd.DataFrame, *, min_rows: int = MIN_MODEL_ROWS, min_groups: int = MIN_PARTICIPANT_GROUPS,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for feature_id, spec in FROZEN_OCULAR_FEATURES.items():
        col = spec["column"]
        if col not in table:
            failures.append(_failure("behavior_links", feature_id, "all", "none", table, "feature_column_missing"))
            continue
        for outcome, bspec in PRIMARY_BEHAVIOR_LINKS.items():
            family = bspec["family"]
            required = [outcome, "block_b2", "progression_centered"]
            if outcome not in table:
                failures.append(_failure("behavior_links", feature_id, outcome, family, table, "behavior_outcome_missing"))
                continue
            if family == "binomial_gee":
                required += [bspec["numerator"], bspec["denominator"]]
                if any(c not in table for c in required):
                    failures.append(_failure("behavior_links", feature_id, outcome, family, table, "binomial_count_columns_missing"))
                    continue
            d = _decomposed_model_frame(table, col, required)
            gate = _sample_gate(d, min_rows, min_groups)
            if gate:
                failures.append(_failure("behavior_links", feature_id, outcome, family, d, gate))
                continue
            try:
                x = pd.DataFrame({
                    "ocular_within_z": d["ocular_within_z"], "ocular_between_z": d["ocular_between_z"],
                    "block_b2": _numeric(d["block_b2"]), "progression_centered": _numeric(d["progression_centered"]),
                }, index=d.index)
                x = sm.add_constant(x, has_constant="add").astype(float)
                groups = d["participant_group_id"].astype(str).to_numpy()
                if family == "gaussian_gee":
                    y = _numeric(d[outcome]).to_numpy(dtype=float)
                    if pd.Series(y).nunique() < 2:
                        raise ValueError("continuous_behavior_outcome_has_single_level")
                    fit = sm.GEE(
                        y, x, groups, family=sm.families.Gaussian(),
                        cov_struct=sm.cov_struct.Exchangeable(),
                    ).fit(maxiter=100)
                    effect_scale = f"{bspec['unit']} per 1 SD Ocular component"
                    model_family = "GEE_Gaussian_exchangeable"
                else:
                    num = _numeric(d[bspec["numerator"]])
                    den = _numeric(d[bspec["denominator"]])
                    if (den < 1).any() or (num < 0).any() or (num > den).any():
                        raise ValueError("invalid_binomial_numerator_or_denominator")
                    frac = num / den
                    if frac.nunique() < 2:
                        raise ValueError("behavior_rate_has_single_level")
                    endog = np.column_stack([num.to_numpy(float), (den - num).to_numpy(float)])
                    fit = sm.GEE(
                        endog, x, groups, family=sm.families.Binomial(),
                        cov_struct=sm.cov_struct.Exchangeable(),
                    ).fit(maxiter=100)
                    effect_scale = "log-odds per 1 SD Ocular component"
                    model_family = "GEE_Binomial_exchangeable"
                reason = _fit_gate(fit)
                if reason:
                    failures.append(_failure("behavior_links", feature_id, outcome, model_family, d, reason))
                    continue
                for term in ("ocular_within_z", "ocular_between_z"):
                    if term not in fit.params.index:
                        continue
                    est, se = float(fit.params[term]), float(fit.bse[term])
                    row = _base_result(
                        "behavior_links", feature_id, col, outcome, model_family, term,
                        est, se, float(fit.pvalues[term]), d, effect_scale=effect_scale,
                    )
                    if family == "binomial_gee":
                        row.update({
                            "odds_ratio": math.exp(est), "or_ci_low": math.exp(est - 1.96 * se),
                            "or_ci_high": math.exp(est + 1.96 * se),
                        })
                    rows.append(row)
            except Exception as exc:
                failures.append(_failure("behavior_links", feature_id, outcome, family, d, f"{type(exc).__name__}: {exc}"))
    return add_family_fdr(pd.DataFrame(rows)), failures


def feature_coverage_table(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(table)
    for feature_id, spec in FROZEN_OCULAR_FEATURES.items():
        col = spec["column"]
        x = _numeric(table[col]) if col in table else pd.Series(np.nan, index=table.index)
        finite = np.isfinite(x)
        rows.append({
            "feature_id": feature_id, "predictor_column": col, "label": spec["label"], "unit": spec["unit"],
            "probe_total_n": int(total), "finite_probe_n": int(finite.sum()),
            "finite_fraction": float(finite.mean()) if total else np.nan,
            "participant_group_n": int(table.loc[finite, "participant_group_id"].nunique()) if total else 0,
            "session_n": int(table.loc[finite, "session_id"].nunique()) if total else 0,
            "analysis_role": "first_round_frozen_main",
        })
    return pd.DataFrame(rows)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def build_ocular_postfreeze_science(inputs: AnalysisInputs, *, make_figures: bool = True) -> dict[str, Any]:
    if inputs.authoritative and not inputs.analysis_code_sha:
        raise ValueError("authoritative run requires analysis_code_sha")
    ocular_root = inputs.ocular_root.expanduser().resolve()
    if not ocular_root.is_dir():
        raise FileNotFoundError(ocular_root)
    for rel in ("tables", "figures/main", "figures/qualification", "figures/sensitivity", "figures/qc", "manifests"):
        (ocular_root / rel).mkdir(parents=True, exist_ok=True)

    ocular = _read_csv(inputs.ocular_wide)
    behavior = _read_csv(inputs.behavior_probe)
    table, identity_audit = prepare_analysis_table(
        ocular, behavior, require_exact_key_universe=inputs.authoritative
    )
    coverage = feature_coverage_table(table)
    task, task_fail = fit_task_progression_models(table)
    q1, q1_fail = fit_q1_models(table)
    q2, q2_fail = fit_q2_models(table)
    behavior_links, behavior_fail = fit_behavior_link_models(table)
    failures = pd.DataFrame(task_fail + q1_fail + q2_fail + behavior_fail)

    outputs = {
        "coverage": ocular_root / "tables/ocular_feature_analysis_coverage.csv",
        "task_progression": ocular_root / "tables/ocular_task_progression.csv",
        "q1": ocular_root / "tables/ocular_q1_models.csv",
        "q2": ocular_root / "tables/ocular_q2_models.csv",
        "behavior_links": ocular_root / "tables/ocular_behavior_links.csv",
        "model_failures": ocular_root / "tables/model_failures.csv",
    }
    for name, frame in (
        ("coverage", coverage), ("task_progression", task), ("q1", q1),
        ("q2", q2), ("behavior_links", behavior_links), ("model_failures", failures),
    ):
        _write_csv(frame, outputs[name])

    figure_summary: dict[str, Any] = {"status": "not_requested"}
    if make_figures:
        from .ocular_postfreeze_figures import build_ocular_postfreeze_figures
        figure_summary = build_ocular_postfreeze_figures(
            ocular_root=ocular_root, coverage=coverage, task=task, q1=q1, q2=q2,
            behavior_links=behavior_links, frozen_probe_table=table,
        )

    model_failures_n = int(len(failures))
    status = "complete_with_model_failures" if model_failures_n else "complete"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema": SCHEMA_VERSION, "status": status, "authoritative": bool(inputs.authoritative),
        "analysis_code_sha": inputs.analysis_code_sha, "scientific_modality": "ocular",
        "input_files": {
            "ocular_wide": {"path": str(inputs.ocular_wide), "sha256": _sha256(inputs.ocular_wide)},
            "behavior_probe": {"path": str(inputs.behavior_probe), "sha256": _sha256(inputs.behavior_probe)},
        },
        "identity_audit": identity_audit,
        "frozen_first_round_features": FROZEN_OCULAR_FEATURES,
        "main_behavior_links": PRIMARY_BEHAVIOR_LINKS,
        "behavior_sensitivity_only": list(SENSITIVITY_BEHAVIOR_LINKS),
        "rt_level_mean_median_researcher_freeze_respected": True,
        "task_progression_contract": {
            "outcome": "each frozen Ocular feature",
            "fixed_terms": ["block_b2", "progression_centered", "block_b2:progression_centered"],
            "progression_scale": "(probe_index_in_block - 1)/9 - 0.5",
            "model": "Gaussian GEE exchangeable",
            "cluster": "participant_group_id (highest repeated-measure level; sessions nested within participant)",
        },
        "q1_contract": {
            "outcome": "q1_nominal_4class", "reference_category": Q1_REFERENCE_CATEGORY,
            "model": "multinomial logit with participant-cluster robust covariance",
            "ocular_decomposition": "participant mean (between) + probe deviation (within), entered jointly",
            "nuisance_terms": ["block_b2", "progression_centered"],
        },
        "q2_contract": {
            "outcome": "q2_ordinal_4level",
            "model": "cumulative-logit OrderedModel with participant-cluster robust covariance",
            "ocular_decomposition": "participant mean (between) + probe deviation (within), entered jointly",
            "nuisance_terms": ["block_b2", "progression_centered"], "treated_as_continuous": False,
        },
        "behavior_link_contract": {
            "direction": "recent Behavior outcome ~ Ocular within + Ocular between + block + progression; association only, no causal claim",
            "continuous_model": "Gaussian GEE exchangeable",
            "rate_model": "binomial GEE from explicit numerator/denominator",
            "cluster": "participant_group_id", "one_ocular_feature_per_model": True,
            "joint_all_ocular_model": False,
        },
        "multiplicity": "Benjamini-Hochberg FDR reported separately within task/Q1/Q2/Behavior result families; never used to reselect Ocular features",
        "missingness": "complete-case per model; no zero fill; no imputation; governance identity retained",
        "sensitivity_boundary": "geometry, median, SD, NIR-only and other alternatives remain qualification/sensitivity/device-alternative only",
        "selection_policy": "Q1/Q2/Behavior significance does not alter the frozen Ocular feature set",
        "counts": {
            "task_effect_rows": int(len(task)), "q1_effect_rows": int(len(q1)),
            "q2_effect_rows": int(len(q2)), "behavior_effect_rows": int(len(behavior_links)),
            "model_failure_rows": model_failures_n,
        },
        "figure_summary": figure_summary,
        "outputs": {k: str(v) for k, v in outputs.items()},
        "stop_lines": {
            "final_feature_registry_mutated": False, "supervised_model_run": False,
            "multimodal_model_run": False, "performance_metric_computed": False,
            "nir_producer_rerun": False,
        },
    }
    manifest_path = ocular_root / "manifests/science_output_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    run_manifest = {
        "created_at_utc": manifest["created_at_utc"], "analysis_code_sha": inputs.analysis_code_sha,
        "status": status,
        "output_sha256": {
            str(path.relative_to(ocular_root)): _sha256(path)
            for path in outputs.values() if path.is_file()
        },
        "science_manifest": str(manifest_path),
    }
    (ocular_root / "manifests/run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
