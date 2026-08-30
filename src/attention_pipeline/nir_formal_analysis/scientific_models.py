"""Reference explanatory models for pupil-only NIR × Behavior associations.

These models validate the formal statistical interface at the trial scale.
They deliberately separate within-participant pupil deviation from
between-participant pupil mean and write model failures instead of silently
dropping them.  The current geom-mean track is a reference signal, not a
scientifically frozen winning pupil metric.

Endpoints are now frozen (configs/nir_formal_analysis.yaml candidate_metrics
final_endpoint_freeze = frozen_20260831_primary_plus_sensitivity) and the two
previously deferred layers are implemented in sibling modules: probe Q1/Q2 in
``probe_pupil_models.py`` and block/session behavior associations in
``block_session_models.py`` (see the deferred registration below).
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

from attention_pipeline.config import Config, load_config
from .pupil_tables import selected_sessions

MODEL_VERSION = "nir-reference-adjusted-models-v1"
REFERENCE_TRACK = "binocular_primary"
REFERENCE_TRIAL_WINDOW = "pre_200ms"
REFERENCE_PROBE_WINDOW = "pre_30s"
VISUAL_METRICS = (
    "central_rel_lum_mean",
    "central_rms_contrast",
    "fruit_visible_area_fraction_central_roi",
)


def _resolve(config: Config, key: str) -> Path:
    raw = config.section("paths").get(key)
    if raw in (None, ""):
        raise KeyError(f"formal pupil config missing paths.{key}")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (config.path.parent.parent / path).resolve()
    return path


def _visual_path(config: Config) -> Path | None:
    raw = config.section("paths").get("stimulus_visual_table")
    if raw in (None, ""):
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (config.path.parent.parent / path).resolve()
    return path


def _session_files(config: Config, session_id: str) -> tuple[Path, Path]:
    root = _resolve(config, "output_root") / "sessions" / session_id
    return (
        root / f"{session_id}_trial_level.csv",
        root / f"{session_id}_trial_pupil_windows.csv",
    )


def _safe_z(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    sd = float(x.std(ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=x.index, dtype=float)
    return (x - float(x.mean())) / sd


def add_pupil_within_between(
    frame: pd.DataFrame,
    *,
    value_col: str = "pupil_median",
    group_col: str = "analysis_group_token",
) -> pd.DataFrame:
    if group_col not in frame:
        raise ValueError(f"missing participant grouping column: {group_col}")
    out = frame.copy()
    value = pd.to_numeric(out[value_col], errors="coerce")
    out["pupil_between"] = value.groupby(out[group_col].astype(str)).transform("mean")
    out["pupil_within"] = value - out["pupil_between"]
    out["pupil_observation_n_within_group"] = value.notna().groupby(out[group_col].astype(str)).transform("sum")
    return out


def _load_visual(config: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _visual_path(config)
    if path is None:
        return pd.DataFrame(), {"status": "unavailable", "reason": "path_not_configured"}
    if not path.is_file():
        return pd.DataFrame(), {"status": "unavailable", "reason": "file_missing", "path": str(path)}
    visual = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = {"stimulus_name", "stimulus_size_pct"}
    missing = sorted(required - set(visual.columns))
    if missing:
        return pd.DataFrame(), {"status": "unavailable", "reason": "schema_missing", "missing": missing, "path": str(path)}
    metrics = [m for m in VISUAL_METRICS if m in visual.columns]
    keep = ["stimulus_name", "stimulus_size_pct", *metrics]
    visual = visual[keep].copy()
    visual["stimulus_size_pct"] = pd.to_numeric(visual["stimulus_size_pct"], errors="coerce")
    if visual.duplicated(["stimulus_name", "stimulus_size_pct"]).any():
        raise ValueError("visual table contains duplicate stimulus_name×stimulus_size_pct")
    return visual, {"status": "available", "path": str(path), "metrics": metrics, "rows": int(len(visual))}


def _attach_current_previous_visual(trials: pd.DataFrame, visual: pd.DataFrame) -> pd.DataFrame:
    out = trials.copy().sort_values(["session_id", "block_num", "trial_num"], kind="stable")
    if "prev_stimulus_name" not in out:
        out["prev_stimulus_name"] = out.groupby(["session_id", "block_num"])["stimulus_name"].shift(1)
    if "prev_stimulus_size" not in out:
        out["prev_stimulus_size"] = out.groupby(["session_id", "block_num"])["stimulus_size"].shift(1)
    out["stimulus_size"] = pd.to_numeric(out["stimulus_size"], errors="coerce")
    out["prev_stimulus_size"] = pd.to_numeric(out["prev_stimulus_size"], errors="coerce")
    metrics = [m for m in VISUAL_METRICS if m in visual.columns]
    current = visual.rename(columns={
        "stimulus_size_pct": "stimulus_size",
        **{m: f"current_{m}" for m in metrics},
    })
    previous = visual.rename(columns={
        "stimulus_name": "prev_stimulus_name",
        "stimulus_size_pct": "prev_stimulus_size",
        **{m: f"previous_{m}" for m in metrics},
    })
    out = out.merge(current, on=["stimulus_name", "stimulus_size"], how="left", validate="many_to_one")
    out = out.merge(previous, on=["prev_stimulus_name", "prev_stimulus_size"], how="left", validate="many_to_one")
    return out


def build_trial_reference_table(
    config: Config,
    sessions: Iterable[str],
    *,
    track: str = REFERENCE_TRACK,
    window_name: str = REFERENCE_TRIAL_WINDOW,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    visual, visual_status = _load_visual(config)
    rows: list[pd.DataFrame] = []
    for session_id in sessions:
        trial_path, window_path = _session_files(config, session_id)
        if not trial_path.is_file() or not window_path.is_file():
            raise FileNotFoundError(f"{session_id}: trial/window table missing")
        trials = pd.read_csv(trial_path, encoding="utf-8-sig", low_memory=False)
        windows = pd.read_csv(window_path, encoding="utf-8-sig", low_memory=False)
        windows = windows[
            windows["track"].astype(str).eq(track)
            & windows["window_name"].astype(str).eq(window_name)
        ].copy()
        keys = ["session_id", "block_num", "trial_num", "global_trial_index"]
        if windows.duplicated(keys).any():
            raise ValueError(f"{session_id}: duplicate reference trial-window key")
        keep = keys + [
            c for c in (
                "analysis_group_token", "is_no_go", "correct", "commission", "omission",
                "rt", "time_in_block_sec", "stimulus_name", "stimulus_size",
                "prev_stimulus_name", "prev_stimulus_size",
            ) if c in trials.columns
        ]
        trial = trials[keep].copy()
        if not visual.empty:
            trial = _attach_current_previous_visual(trial, visual)
        merge_keep = keys + [
            c for c in (
                "pupil_median", "pupil_valid_fraction", "internal_coverage_fraction",
                "source_mode_binocular_fraction", "source_mode_left_only_fraction",
                "source_mode_right_only_fraction",
            ) if c in windows.columns
        ]
        current = trial.merge(windows[merge_keep], on=keys, how="left", validate="one_to_one")
        rows.append(current)
    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if table.empty:
        return table, visual_status
    table = add_pupil_within_between(table)
    table["time_in_block_z"] = table.groupby("analysis_group_token")["time_in_block_sec"].transform(_safe_z)
    if {"source_mode_left_only_fraction", "source_mode_right_only_fraction"}.issubset(table.columns):
        table["source_mode_single_eye_fraction"] = (
            pd.to_numeric(table["source_mode_left_only_fraction"], errors="coerce").fillna(0)
            + pd.to_numeric(table["source_mode_right_only_fraction"], errors="coerce").fillna(0)
        )
    return table, visual_status


def _fit_gate(fit: Any) -> str | None:
    if getattr(fit, "converged", True) is False:
        return "model_did_not_converge"
    params = np.asarray(fit.params, dtype=float)
    bse = np.asarray(fit.bse, dtype=float)
    if params.size == 0 or not np.isfinite(params).all() or not np.isfinite(bse).all():
        return "nonfinite_parameter_or_se"
    return None


def _result_rows(
    fit: Any,
    *,
    model_name: str,
    family: str,
    outcome: str,
    adjusted: bool,
    data: pd.DataFrame,
    covariates: list[str],
) -> list[dict[str, Any]]:
    params = pd.Series(fit.params)
    bse = pd.Series(fit.bse).reindex(params.index)
    p = pd.Series(fit.pvalues).reindex(params.index)
    rows: list[dict[str, Any]] = []
    for term in ("pupil_within", "pupil_between"):
        if term not in params:
            continue
        est = float(params[term])
        se = float(bse[term])
        rows.append({
            "model_name": model_name,
            "model_family": family,
            "outcome": outcome,
            "adjusted": adjusted,
            "pupil_term": term,
            "estimate": est,
            "se": se,
            "ci_low": est - 1.96 * se,
            "ci_high": est + 1.96 * se,
            "p_value_not_for_endpoint_selection": float(p[term]),
            "n_rows": int(len(data)),
            "participant_group_n": int(data["analysis_group_token"].astype(str).nunique()),
            "session_n": int(data["session_id"].astype(str).nunique()),
            "visual_support_fraction": (
                float(pd.to_numeric(data["visual_support"], errors="coerce").mean())
                if "visual_support" in data else np.nan
            ),
            "covariates": ";".join(covariates),
            "status": "estimable",
            "reference_signal_only": True,
        })
    return rows


def _failure(
    *, model_name: str, family: str, outcome: str, adjusted: bool,
    data: pd.DataFrame, reason: str, covariates: list[str],
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "model_family": family,
        "outcome": outcome,
        "adjusted": adjusted,
        "status": "not_estimable",
        "reason": reason,
        "n_rows": int(len(data)),
        "participant_group_n": int(data.get("analysis_group_token", pd.Series(dtype=str)).astype(str).nunique()),
        "session_n": int(data.get("session_id", pd.Series(dtype=str)).astype(str).nunique()),
        "covariates": ";".join(covariates),
        "reference_signal_only": True,
    }


def _available_adjustment_covariates(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "time_in_block_z",
        "pupil_valid_fraction",
        "internal_coverage_fraction",
        "source_mode_binocular_fraction",
        "source_mode_single_eye_fraction",
        "stimulus_size",
        "prev_stimulus_size",
    ]
    for prefix in ("current_", "previous_"):
        candidates.extend(f"{prefix}{m}" for m in VISUAL_METRICS)
    usable: list[str] = []
    for col in candidates:
        if col not in frame:
            continue
        x = pd.to_numeric(frame[col], errors="coerce")
        if x.notna().sum() >= 3 and x.nunique(dropna=True) >= 2:
            usable.append(col)
    return usable


def fit_trial_reference_models(
    table: pd.DataFrame,
    *,
    min_participant_groups: int = 6,
    min_rows: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if table.empty:
        return pd.DataFrame(), pd.DataFrame([_failure(
            model_name="trial_reference", family="none", outcome="all", adjusted=False,
            data=table, reason="empty_trial_reference_table", covariates=[]
        )])

    visual_cols = [c for c in table.columns if c.startswith("current_") or c.startswith("previous_")]
    if visual_cols:
        table = table.copy()
        table["visual_support"] = table[visual_cols].notna().any(axis=1)
    else:
        table = table.copy()
        table["visual_support"] = False

    specs = (
        ("go_correct_rt", "LMM", "rt", lambda d: pd.to_numeric(d["is_no_go"], errors="coerce").eq(0) & pd.to_numeric(d["correct"], errors="coerce").eq(1)),
        ("go_omission", "GEE_binomial", "omission", lambda d: pd.to_numeric(d["is_no_go"], errors="coerce").eq(0)),
        ("nogo_commission", "GEE_binomial", "commission", lambda d: pd.to_numeric(d["is_no_go"], errors="coerce").eq(1)),
    )

    for label, family, outcome, selector in specs:
        base = table[selector(table)].copy()
        base[outcome] = pd.to_numeric(base[outcome], errors="coerce")
        for col in ("pupil_within", "pupil_between"):
            base[col] = pd.to_numeric(base[col], errors="coerce")
        for adjusted in (False, True):
            covariates = _available_adjustment_covariates(base) if adjusted else []
            required = [outcome, "pupil_within", "pupil_between", "analysis_group_token", "session_id", *covariates]
            d = base.dropna(subset=required).copy()
            model_name = f"{label}__{'adjusted' if adjusted else 'unadjusted'}"
            if len(d) < min_rows or d["analysis_group_token"].nunique() < min_participant_groups:
                failures.append(_failure(
                    model_name=model_name, family=family, outcome=outcome, adjusted=adjusted,
                    data=d, reason="insufficient_rows_or_participant_groups", covariates=covariates,
                ))
                continue
            if family == "GEE_binomial" and d[outcome].nunique() < 2:
                failures.append(_failure(
                    model_name=model_name, family=family, outcome=outcome, adjusted=adjusted,
                    data=d, reason="binary_outcome_has_single_level", covariates=covariates,
                ))
                continue
            try:
                # Scale nuisance covariates within the fitted sample; pupil terms
                # retain their natural centered units for interpretable estimates.
                for cov in covariates:
                    d[cov] = _safe_z(d[cov])
                formula = f"{outcome} ~ pupil_within + pupil_between"
                if covariates:
                    formula += " + " + " + ".join(covariates)
                if family == "LMM":
                    fit = smf.mixedlm(
                        formula,
                        data=d,
                        groups=d["analysis_group_token"].astype(str),
                    ).fit(reml=False, method="lbfgs", disp=False)
                else:
                    fit = sm.GEE.from_formula(
                        formula,
                        groups="analysis_group_token",
                        data=d,
                        family=sm.families.Binomial(),
                    ).fit(maxiter=100)
                reason = _fit_gate(fit)
                if reason:
                    failures.append(_failure(
                        model_name=model_name, family=family, outcome=outcome, adjusted=adjusted,
                        data=d, reason=reason, covariates=covariates,
                    ))
                    continue
                results.extend(_result_rows(
                    fit,
                    model_name=model_name,
                    family=family,
                    outcome=outcome,
                    adjusted=adjusted,
                    data=d,
                    covariates=covariates,
                ))
            except Exception as exc:
                failures.append(_failure(
                    model_name=model_name, family=family, outcome=outcome, adjusted=adjusted,
                    data=d, reason=f"{type(exc).__name__}: {exc}", covariates=covariates,
                ))
    return pd.DataFrame(results), pd.DataFrame(failures)


def run_reference_adjusted_models(
    config_path: str | Path,
    *,
    subjects: Iterable[str] | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    sessions = selected_sessions(config, subjects)
    root = _resolve(config, "output_root") / "reference_adjusted_models"
    root.mkdir(parents=True, exist_ok=True)
    try:
        trial, visual_status = build_trial_reference_table(config, sessions)
        results, failures = fit_trial_reference_models(trial)
        status = "complete" if not trial.empty else "blocked"
        load_failure = None
    except Exception as exc:
        trial = pd.DataFrame()
        results = pd.DataFrame()
        failures = pd.DataFrame([{
            "model_name": "reference_adjusted_models",
            "model_family": "pipeline",
            "outcome": "all",
            "adjusted": False,
            "status": "not_estimable",
            "reason": f"{type(exc).__name__}: {exc}",
        }])
        visual_status = {"status": "unknown"}
        status = "blocked"
        load_failure = str(exc)

    trial.to_csv(root / "trial_reference_model_table.csv", index=False, encoding="utf-8-sig")
    results.to_csv(root / "trial_unadjusted_adjusted_effects.csv", index=False, encoding="utf-8-sig")
    failures.to_csv(root / "model_failures.csv", index=False, encoding="utf-8-sig")
    deferred = pd.DataFrame([
        {
            "analysis_level": "probe",
            "outcomes": "Q1 nominal; Q2 ordinal",
            "status": "frozen_and_implemented",
            "implemented_step": "probe_pupil_models",
            "module": "nir_formal_analysis.probe_pupil_models",
        },
        {
            "analysis_level": "block_session",
            "outcomes": "dprime;c;beta;omission;commission;RT level;RT variability;RT slope",
            "status": "frozen_and_implemented",
            "implemented_step": "block_session_models",
            "module": "nir_formal_analysis.block_session_models",
        },
    ])
    deferred.to_csv(root / "deferred_endpoint_models.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "pipeline_version": MODEL_VERSION,
        "reference_track": REFERENCE_TRACK,
        "reference_trial_window": REFERENCE_TRIAL_WINDOW,
        "reference_signal_is_final_endpoint": False,
        "within_between_decomposition": True,
        "trial_outcomes": ["go_correct_rt", "go_omission", "nogo_commission"],
        "visual_status": visual_status,
        "current_visual_temporal_role": "allowed only as post-stimulus behavior-outcome nuisance; never used to construct the pre-stimulus pupil predictor",
        "previous_visual_temporal_role": "eligible pre-stimulus pupil/behavior nuisance",
        "scientific_inference_authorized_by_code_alone": False,
        "endpoint_selection_from_p_values_allowed": False,
        "load_failure": load_failure,
        "n_effect_rows": int(len(results)),
        "n_failure_rows": int(len(failures)),
    }
    (root / "reference_adjusted_models_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
