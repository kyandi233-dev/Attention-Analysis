"""Block/session-level endpoint models: pupil state versus block behavior.

File: block_session_models.py
Version: nir-block-session-endpoint-models-v1
Purpose:
    Frozen-endpoint formal association at the block scale.  One row per
    session x block x metric.  The frozen primary pupil metrics
    (``pupil_geom_mean_diameter``, ``hard_pupil_fraction``) are summarized from
    the 10_analysis_ready candidate sidecar as raw baseline (median of valid
    raw frames) plus session-eye-centered summaries, then decomposed into
    participant mean (between) and within-participant deviation (within).
    Block behavior is aggregated from each session's trial-level table with the
    same canonical metric set as the behavior formal pipeline
    (``aggregate_behavior_metrics``): d', criterion c, beta, omission rate,
    commission rate, and Go-correct RT mean/median/SD/CV/Theil-Sen slope.

    Continuous outcomes use participant random-intercept LMM; omission and
    commission rates use binomial GEE with (successes, failures) two-column
    endog on exchangeable working correlation, clustered by participant.
    Omission and commission are always modeled as two separate outcomes and
    are never merged into a combined accuracy score.

    Model failures are always written as ``not_estimable`` rows; empty result
    tables are never used to fake success.

Usage:
    >>> from attention_pipeline.nir_formal_analysis.block_session_models import run_block_session_models
    >>> manifest = run_block_session_models("configs/nir_formal_analysis.yaml", subjects=["sub-041"])

Dependencies:
    numpy, pandas, statsmodels (mixedlm, GEE binomial)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from attention_pipeline.behavior_formal.science_v3 import (
    BehaviorScienceConfig,
    aggregate_behavior_metrics,
)
from attention_pipeline.config import Config, load_config
from .candidate_validation import (
    add_within_between as add_metric_within_between,
    summarize_session_block_candidates,
)
from .pupil_tables import selected_sessions

# ---- Frozen parameters (centralized; see configs/nir_formal_analysis.yaml) ----
BLOCK_MODEL_VERSION = "nir-block-session-endpoint-models-v1"
DEFAULT_PRIMARY_METRICS = ("pupil_geom_mean_diameter", "hard_pupil_fraction")
# Continuous block outcomes modeled with participant random-intercept LMM.
CONTINUOUS_OUTCOMES = (
    "dprime_loglinear",
    "criterion_c",
    "beta",
    "go_correct_rt_mean_ms",
    "go_correct_rt_median_ms",
    "go_correct_rt_sd_ms",
    "go_correct_rt_cv",
    "go_correct_rt_theilsen_slope_ms_per_s",
)
# Proportion outcomes modeled with binomial GEE: (outcome, numerator, denominator).
RATE_OUTCOMES = (
    ("omission_rate", "omission_numerator", "omission_denominator"),
    ("commission_rate", "commission_numerator", "commission_denominator"),
)
MIN_MODEL_ROWS = 24
MIN_PARTICIPANT_GROUPS = 6
# Sidecar columns required per metric for block summarization.
_METRIC_SIDECAR_SUFFIXES = ("__raw", "__centered_primary", "__valid_primary")


def _resolve(config: Config, key: str) -> Path:
    raw = config.section("paths").get(key)
    if raw in (None, ""):
        raise KeyError(f"formal pupil config missing paths.{key}")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (config.path.parent.parent / path).resolve()
    return path


def _safe_z(series: pd.Series) -> pd.Series:
    """Standardize a numeric series in-sample; zero/nonfinite variance yields NaN."""
    x = pd.to_numeric(series, errors="coerce")
    sd = float(x.std(ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=x.index, dtype=float)
    return (x - float(x.mean())) / sd


def _fit_gate(fit: Any) -> str | None:
    """Return a failure reason when a fitted model is not safe to report."""
    if getattr(fit, "converged", True) is False:
        return "model_did_not_converge"
    params = np.asarray(fit.params, dtype=float)
    bse = np.asarray(fit.bse, dtype=float)
    if params.size == 0 or not np.isfinite(params).all() or not np.isfinite(bse).all():
        return "nonfinite_parameter_or_se"
    return None


def _failure(
    *, model_name: str, family: str, outcome: str, metric: str,
    data: pd.DataFrame, reason: str,
) -> dict[str, Any]:
    """Build a single not_estimable failure record."""
    return {
        "model_name": model_name,
        "model_family": family,
        "outcome": outcome,
        "metric": metric,
        "status": "not_estimable",
        "reason": reason,
        "n_rows": int(len(data)),
        "participant_group_n": int(data.get("analysis_group_token", pd.Series(dtype=str)).astype(str).nunique()),
        "session_n": int(data.get("session_id", pd.Series(dtype=str)).astype(str).nunique()),
    }


def build_block_session_table(
    config: Config,
    sessions: Iterable[str],
    *,
    metrics: tuple[str, ...] = DEFAULT_PRIMARY_METRICS,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Build the session x block x metric endpoint table.

    Parameters
    ----------
    config : loaded formal NIR config.
    sessions : session ids to include.
    metrics : frozen primary pupil metrics.

    Returns
    -------
    (table, load_failures) - long-format rows with binocular raw/centered pupil
    summaries and canonical block behavior metrics; per-session load problems
    are recorded instead of aborting the cohort run.
    """
    output_root = _resolve(config, "output_root")
    candidate_root = _resolve(config, "analysis_ready_root") / "candidate_frame_level"
    load_failures: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for session_id in sessions:
        sidecar_path = candidate_root / session_id / f"{session_id}_nir_pupil_candidates.csv"
        trial_path = output_root / "sessions" / session_id / f"{session_id}_trial_level.csv"
        if not sidecar_path.is_file() or not trial_path.is_file():
            load_failures.append({
                "session_id": str(session_id),
                "reason": "candidate_sidecar_or_trial_level_missing",
                "sidecar_path": str(sidecar_path),
                "trial_path": str(trial_path),
            })
            continue
        try:
            sidecar_cols = [
                "session_id", "analysis_group_token", "phase", "eye",
                "pupil_valid_primary", "pupil_valid_strict",
            ]
            for metric in metrics:
                for suffix in _METRIC_SIDECAR_SUFFIXES:
                    sidecar_cols.append(f"{metric}{suffix}")
            header = pd.read_csv(sidecar_path, nrows=0, encoding="utf-8-sig")
            missing_sidecar = sorted(set(sidecar_cols) - set(header.columns))
            if missing_sidecar:
                raise ValueError(f"candidate_sidecar_schema_missing:{','.join(missing_sidecar)}")
            sidecar = pd.read_csv(
                sidecar_path, usecols=sidecar_cols, encoding="utf-8-sig", low_memory=False
            )
            summary = summarize_session_block_candidates(sidecar)
            summary = summary[summary["metric"].astype(str).isin(metrics)].copy()

            trials = pd.read_csv(trial_path, encoding="utf-8-sig", low_memory=False)
            # Only formal blocks carry a numeric block identity; rows without one
            # (if any future session produces them) cannot join pupil summaries.
            trials = trials[pd.to_numeric(trials["block_num"], errors="coerce").notna()].copy()
            behavior_rows: list[dict[str, Any]] = []
            for block_num, block in trials.groupby("block_num", sort=True, dropna=False):
                record = {
                    "session_id": str(session_id),
                    "block_num": int(block_num),
                }
                record.update(aggregate_behavior_metrics(block, BehaviorScienceConfig()))
                behavior_rows.append(record)
            behavior = pd.DataFrame(behavior_rows)

            merged = summary.merge(
                behavior, on=["session_id", "block_num"], how="left", validate="many_to_one"
            )
            frames.append(merged)
        except Exception as exc:
            load_failures.append({
                "session_id": str(session_id),
                "reason": f"{type(exc).__name__}: {exc}",
            })
    if not frames:
        return pd.DataFrame(), load_failures
    return pd.concat(frames, ignore_index=True), load_failures


def _gate_common(d: pd.DataFrame, *, min_rows: int, min_groups: int) -> str | None:
    """Shared sample-size and predictor-variance gate for block models."""
    if len(d) < min_rows or d["analysis_group_token"].astype(str).nunique() < min_groups:
        return "insufficient_rows_or_participant_groups"
    for col in ("pupil_within", "pupil_between"):
        if d[col].nunique(dropna=False) < 2:
            return f"{col}_has_single_level"
    return None


def _append_effect_rows(
    rows: list[dict[str, Any]],
    fit: Any,
    *,
    model_name: str,
    family: str,
    outcome: str,
    metric: str,
    data: pd.DataFrame,
) -> None:
    """Append pupil_within/pupil_between estimate rows for one fitted model."""
    params = pd.Series(fit.params)
    bse = pd.Series(fit.bse).reindex(params.index)
    p = pd.Series(fit.pvalues).reindex(params.index)
    for term in ("pupil_within", "pupil_between"):
        if term not in params:
            continue
        est = float(params[term])
        se = float(bse[term])
        rows.append({
            "model_name": model_name,
            "model_family": family,
            "outcome": outcome,
            "metric": metric,
            "pupil_term": term,
            "estimate": est,
            "se": se,
            "ci_low": est - 1.96 * se,
            "ci_high": est + 1.96 * se,
            "p_value_not_for_endpoint_selection": float(p[term]),
            "n_rows": int(len(data)),
            "participant_group_n": int(data["analysis_group_token"].astype(str).nunique()),
            "session_n": int(data["session_id"].astype(str).nunique()),
            "block_n": int(data["block_num"].astype(str).nunique()) if "block_num" in data else np.nan,
            "predictor_standardization": "z_score_within_fitted_sample_per_term",
            "status": "estimable",
        })


def _fit_continuous(
    d: pd.DataFrame,
    *,
    outcome: str,
    metric: str,
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    min_rows: int,
    min_groups: int,
) -> None:
    """One continuous outcome: participant random-intercept LMM."""
    model_name = f"block_{outcome}__{metric}"
    family = "LMM"
    d = d.dropna(subset=[outcome, "pupil_within", "pupil_between", "analysis_group_token", "session_id"]).copy()
    if d[outcome].nunique(dropna=False) < 2:
        failures.append(_failure(
            model_name=model_name, family=family, outcome=outcome, metric=metric,
            data=d, reason="continuous_outcome_has_single_level",
        ))
        return
    gate = _gate_common(d, min_rows=min_rows, min_groups=min_groups)
    if gate:
        failures.append(_failure(
            model_name=model_name, family=family, outcome=outcome, metric=metric, data=d, reason=gate,
        ))
        return
    try:
        d = d.copy()
        d["pupil_within"] = _safe_z(d["pupil_within"])
        d["pupil_between"] = _safe_z(d["pupil_between"])
        # Optimizer note: the statsmodels default lbfgs hard-crashes this
        # runtime inside scipy's L-BFGS-B extension (native 0xc06d007f), which
        # cannot be caught as a Python exception; bfgs is used instead.
        fit = smf.mixedlm(
            f"{outcome} ~ pupil_within + pupil_between",
            data=d,
            groups=d["analysis_group_token"].astype(str),
        ).fit(reml=False, method="bfgs", disp=False)
        reason = _fit_gate(fit)
        if reason:
            failures.append(_failure(
                model_name=model_name, family=family, outcome=outcome, metric=metric, data=d, reason=reason,
            ))
            return
        _append_effect_rows(
            results, fit, model_name=model_name, family=family, outcome=outcome, metric=metric, data=d
        )
    except Exception as exc:
        failures.append(_failure(
            model_name=model_name, family=family, outcome=outcome, metric=metric,
            data=d, reason=f"{type(exc).__name__}: {exc}",
        ))


def _fit_rate(
    d: pd.DataFrame,
    *,
    outcome: str,
    numerator_col: str,
    denominator_col: str,
    metric: str,
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    min_rows: int,
    min_groups: int,
) -> None:
    """One proportion outcome: binomial GEE with (successes, failures) endog."""
    model_name = f"block_{outcome}__{metric}"
    family = "GEE_binomial_exchangeable"
    d = d.dropna(subset=[
        outcome, numerator_col, denominator_col,
        "pupil_within", "pupil_between", "analysis_group_token", "session_id",
    ]).copy()
    denominator = pd.to_numeric(d[denominator_col], errors="coerce")
    numerator = pd.to_numeric(d[numerator_col], errors="coerce")
    if len(d) == 0 or (denominator < 1).any():
        failures.append(_failure(
            model_name=model_name, family=family, outcome=outcome, metric=metric,
            data=d, reason="rate_denominator_missing_or_zero",
        ))
        return
    fraction = numerator / denominator
    if not (fraction.gt(0).any() and fraction.lt(1).any()):
        failures.append(_failure(
            model_name=model_name, family=family, outcome=outcome, metric=metric,
            data=d, reason="rate_outcome_at_boundary_no_variation",
        ))
        return
    gate = _gate_common(d, min_rows=min_rows, min_groups=min_groups)
    if gate:
        failures.append(_failure(
            model_name=model_name, family=family, outcome=outcome, metric=metric, data=d, reason=gate,
        ))
        return
    try:
        endog = np.column_stack([
            numerator.to_numpy(dtype=float),
            (denominator - numerator).to_numpy(dtype=float),
        ])
        exog = pd.DataFrame({
            "pupil_within": _safe_z(d["pupil_within"]),
            "pupil_between": _safe_z(d["pupil_between"]),
        })
        fit = sm.GEE(
            endog,
            exog,
            d["analysis_group_token"].astype(str).to_numpy(),
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit(maxiter=100)
        reason = _fit_gate(fit)
        if reason:
            failures.append(_failure(
                model_name=model_name, family=family, outcome=outcome, metric=metric, data=d, reason=reason,
            ))
            return
        _append_effect_rows(
            results, fit, model_name=model_name, family=family, outcome=outcome, metric=metric, data=d
        )
    except Exception as exc:
        failures.append(_failure(
            model_name=model_name, family=family, outcome=outcome, metric=metric,
            data=d, reason=f"{type(exc).__name__}: {exc}",
        ))


def fit_block_session_models(
    table: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = DEFAULT_PRIMARY_METRICS,
    min_participant_groups: int = MIN_PARTICIPANT_GROUPS,
    min_rows: int = MIN_MODEL_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit block-level behavior ~ pupil between/within models per frozen metric.

    Parameters
    ----------
    table : long session x block x metric table from build_block_session_table
        (or equivalent synthetic table with metric, binocular_raw_median and
        canonical behavior columns).
    metrics : frozen primary pupil metrics.
    min_participant_groups / min_rows : minimum sample gates.

    Returns
    -------
    (results, failures) - estimable effect rows and not_estimable failure rows.
    """
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if table.empty:
        failures.append(_failure(
            model_name="block_session_models", family="none", outcome="all", metric="",
            data=table, reason="empty_block_session_table",
        ))
        return pd.DataFrame(), pd.DataFrame(failures)

    for metric in metrics:
        current = table[table["metric"].astype(str).eq(metric)].copy()
        if current.empty or "binocular_raw_median" not in current:
            failures.append(_failure(
                model_name=f"block_models_{metric}", family="none", outcome="all", metric=metric,
                data=current, reason="metric_rows_or_binocular_raw_median_missing",
            ))
            continue
        # Participant mean (between) and within-participant deviation (within)
        # over session x block observations, metric-aware (mirrors candidate_validation).
        decomposed = add_metric_within_between(current)
        decomposed["pupil_between"] = decomposed["participant_metric_mean"]
        decomposed["pupil_within"] = decomposed["within_participant_deviation"]

        for outcome in CONTINUOUS_OUTCOMES:
            if outcome not in decomposed:
                continue
            _fit_continuous(
                decomposed, outcome=outcome, metric=metric,
                results=results, failures=failures, min_rows=min_rows, min_groups=min_participant_groups,
            )
        for outcome, numerator_col, denominator_col in RATE_OUTCOMES:
            if not {outcome, numerator_col, denominator_col}.issubset(decomposed.columns):
                continue
            _fit_rate(
                decomposed, outcome=outcome, numerator_col=numerator_col,
                denominator_col=denominator_col, metric=metric,
                results=results, failures=failures, min_rows=min_rows, min_groups=min_participant_groups,
            )
    return pd.DataFrame(results), pd.DataFrame(failures)


def _primary_metrics_from_config(config: Config) -> tuple[str, ...]:
    raw = config.section("candidate_metrics").get("primary_metrics")
    if isinstance(raw, list) and raw:
        return tuple(str(m) for m in raw)
    return DEFAULT_PRIMARY_METRICS


def run_block_session_models(
    config_path: str | Path,
    *,
    subjects: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Pipeline step: build the block table, fit LMM/GEE models, write outputs.

    Parameters
    ----------
    config_path : path to configs/nir_formal_analysis.yaml (or equivalent).
    subjects : optional session-id override for smoke/subset runs.

    Returns
    -------
    Manifest dict with status complete|blocked, model counts and output paths.
    """
    config = load_config(config_path)
    sessions = selected_sessions(config, subjects)
    metrics = _primary_metrics_from_config(config)
    table, load_failures = build_block_session_table(config, sessions, metrics=metrics)
    results, failures = fit_block_session_models(table, metrics=metrics)

    root = _resolve(config, "output_root") / "block_session_models"
    root.mkdir(parents=True, exist_ok=True)
    table.to_csv(root / "block_session_model_table.csv", index=False, encoding="utf-8-sig")
    results.to_csv(root / "block_session_model_effects.csv", index=False, encoding="utf-8-sig")
    failures.to_csv(root / "block_session_model_failures.csv", index=False, encoding="utf-8-sig")

    status = "blocked" if table.empty else "complete"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "pipeline_version": BLOCK_MODEL_VERSION,
        "primary_metrics": list(metrics),
        "endpoint_freeze": config.section("candidate_metrics").get("final_endpoint_freeze"),
        "observation_unit": "session x block x metric",
        "pupil_summary": (
            "binocular_raw_median = fused per-eye median of valid raw frames per block "
            "(raw baseline); binocular_centered_median = session-eye centered summary kept for audit"
        ),
        "within_between_decomposition": (
            "participant_metric_mean (between) and within_participant_deviation (within) "
            "over session x block observations per metric"
        ),
        "continuous_outcomes": list(CONTINUOUS_OUTCOMES),
        "continuous_family": "LMM_participant_random_intercept",
        "continuous_family_note": (
            "mixedlm bfgs optimizer used because the statsmodels default lbfgs "
            "hard-crashes this runtime in scipy L-BFGS-B (native 0xc06d007f)"
        ),
        "rate_outcomes": [entry[0] for entry in RATE_OUTCOMES],
        "rate_family": "GEE_binomial_exchangeable_two_column_success_failure_endog",
        "omission_commission_policy": "modeled as two separate outcomes; never merged into a combined accuracy score",
        "predictor_standardization": "z_score_within_fitted_sample_per_term",
        "n_sessions_requested": len(sessions),
        "n_sessions_loaded": int(table["session_id"].astype(str).nunique()) if not table.empty else 0,
        "n_block_observations": int(len(table)),
        "n_effect_rows": int(len(results)),
        "n_failure_rows": int(len(failures)),
        "load_failures": load_failures,
        "scientific_inference_authorized_by_code_alone": False,
        "endpoint_selection_from_p_values_allowed": False,
        "outputs": {
            "model_table": str(root / "block_session_model_table.csv"),
            "effects": str(root / "block_session_model_effects.csv"),
            "failures": str(root / "block_session_model_failures.csv"),
        },
    }
    (root / "block_session_models_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
