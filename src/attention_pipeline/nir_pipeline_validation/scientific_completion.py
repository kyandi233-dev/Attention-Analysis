"""Scientific-completeness helpers for pupil-only NIR formal analysis.

This layer deliberately does not implement probe visual exposure itself.  The
single authoritative implementation is ``nir_formal_analysis.probe_contract``.
Here we keep only window-level within/between decomposition and participant-
clustered visual-confound models.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .scientific_contract import MODEL_FAILURE_COLUMNS


@dataclass(frozen=True)
class VisualAdjustmentConfig:
    min_rows: int = 24
    min_participant_groups: int = 6
    primary_track: str = "binocular_primary"
    primary_window: str = "pre_5s"


def decompose_pupil_within_between(
    frame: pd.DataFrame,
    *,
    value_columns: Sequence[str] = ("pupil_median",),
    participant_col: str = "analysis_group_token",
    session_col: str = "session_id",
) -> pd.DataFrame:
    """Separate stable participant means from momentary deviations.

    No participant identity mapping or visit order is inferred here.  The
    upstream grouping token is consumed as-is until the verified participant
    registry adapter is supplied.
    """
    if participant_col not in frame:
        raise ValueError(f"missing participant column: {participant_col}")
    if frame[participant_col].isna().any():
        raise ValueError(f"{participant_col} contains missing values")

    out = frame.copy()
    for column in value_columns:
        if column not in out:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        participant_mean = values.groupby(out[participant_col]).transform("mean")
        out[f"{column}__participant_mean"] = participant_mean
        out[f"{column}__within_participant"] = values - participant_mean
        if session_col in out:
            session_mean = values.groupby(out[session_col]).transform("mean")
            out[f"{column}__session_mean"] = session_mean
            out[f"{column}__within_session"] = values - session_mean
    return out


def _visual_predictors(frame: pd.DataFrame) -> list[str]:
    tokens = ("brightness", "lumin", "lum_", "contrast", "size", "area", "visible")
    candidates = [
        c
        for c in frame.columns
        if c.startswith("previous_visual__")
        and any(token in c.lower() for token in tokens)
    ]
    usable: list[str] = []
    for column in candidates:
        x = pd.to_numeric(frame[column], errors="coerce")
        if x.notna().sum() >= 3 and x.nunique(dropna=True) >= 2:
            usable.append(column)
    return usable


def _failure(
    *,
    target: str,
    model_stage: str,
    failure_type: str,
    failure_detail: str,
) -> dict[str, Any]:
    return {
        "analysis_question": "visual_confound_adjustment",
        "target": target,
        "model_stage": model_stage,
        "fold": pd.NA,
        "model_name": "GEE_Gaussian_exchangeable",
        "status": "not_estimable",
        "failure_type": failure_type,
        "failure_detail": failure_detail,
    }


def fit_visual_adjustment_models(
    visual_trial_windows: pd.DataFrame,
    trials: pd.DataFrame,
    *,
    config: VisualAdjustmentConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare unadjusted and visual/time/quality-adjusted pupil associations.

    For a strictly pre-stimulus pupil window, current-stimulus properties are
    causally unavailable and therefore never enter the model.  Previous-
    stimulus visual properties are focal predictors.  GEE (generalized
    estimating equations) uses the participant grouping token as the cluster.
    """
    cfg = config or VisualAdjustmentConfig()
    work = visual_trial_windows.copy()
    if "track" in work:
        work = work[work["track"].astype(str).eq(cfg.primary_track)]
    if "window_name" in work:
        work = work[work["window_name"].astype(str).eq(cfg.primary_window)]
    if work.empty:
        row = _failure(
            target="pupil_median",
            model_stage="all",
            failure_type="empty_primary_window",
            failure_detail=f"no rows for {cfg.primary_track}/{cfg.primary_window}",
        )
        return pd.DataFrame(), pd.DataFrame([row], columns=MODEL_FAILURE_COLUMNS)

    keys = [
        c
        for c in ("session_id", "block_num", "trial_num", "global_trial_index")
        if c in work.columns and c in trials.columns
    ]
    trial_covars = [
        c
        for c in ("is_no_go", "time_in_block_sec", "global_trial_index")
        if c in trials.columns and c not in keys
    ]
    if keys and trial_covars:
        right = trials[[*keys, *trial_covars]].drop_duplicates(keys)
        duplicate = [c for c in trial_covars if c in work.columns and c not in keys]
        if duplicate:
            work = work.drop(columns=duplicate)
        work = work.merge(right, on=keys, how="left", validate="many_to_one")

    required = {"analysis_group_token", "session_id", "pupil_median"}
    missing = sorted(required - set(work.columns))
    if missing:
        row = _failure(
            target="pupil_median",
            model_stage="all",
            failure_type="missing_required_columns",
            failure_detail=f"missing {missing}",
        )
        return pd.DataFrame(), pd.DataFrame([row], columns=MODEL_FAILURE_COLUMNS)

    predictors = _visual_predictors(work)
    if not predictors:
        row = _failure(
            target="pupil_median",
            model_stage="all",
            failure_type="no_visual_predictor",
            failure_detail="no previous-stimulus brightness/contrast/size/area predictor with support",
        )
        return pd.DataFrame(), pd.DataFrame([row], columns=MODEL_FAILURE_COLUMNS)

    controls = [
        c
        for c in (
            "global_trial_index",
            "time_in_block_sec",
            "is_no_go",
            "pupil_valid_fraction",
            "internal_coverage_fraction",
            "source_mode_binocular_fraction",
            "source_mode_left_only_fraction",
            "source_mode_right_only_fraction",
        )
        if c in work.columns
    ]

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for focal in predictors:
        for stage, requested in (
            ("unadjusted", [focal]),
            ("visual_time_quality_adjusted", [focal, *controls]),
        ):
            try:
                import statsmodels.api as sm

                requested = list(dict.fromkeys(requested))
                d = work[["pupil_median", "analysis_group_token", "session_id", *requested]].copy()
                d["pupil_median"] = pd.to_numeric(d["pupil_median"], errors="coerce")
                for column in requested:
                    d[column] = pd.to_numeric(d[column], errors="coerce")
                d = d.replace([np.inf, -np.inf], np.nan).dropna(
                    subset=["pupil_median", "analysis_group_token", focal]
                )
                if len(d) < cfg.min_rows:
                    raise ValueError(f"rows {len(d)} < minimum {cfg.min_rows}")
                group_n = int(d["analysis_group_token"].astype(str).nunique())
                if group_n < cfg.min_participant_groups:
                    raise ValueError(
                        f"participant groups {group_n} < minimum {cfg.min_participant_groups}"
                    )

                for control in [c for c in requested if c != focal]:
                    median = d[control].median()
                    if pd.isna(median):
                        d = d.drop(columns=[control])
                    else:
                        d[control] = d[control].fillna(median)
                used = [c for c in requested if c in d.columns and d[c].nunique(dropna=True) >= 2]
                if focal not in used:
                    raise ValueError("focal visual predictor has no variance")

                x = d[used].copy()
                for column in used:
                    mean = float(x[column].mean())
                    sd = float(x[column].std(ddof=0))
                    x[column] = (x[column] - mean) / sd if np.isfinite(sd) and sd > 0 else x[column] - mean
                x = sm.add_constant(x, has_constant="add")
                fit = sm.GEE(
                    d["pupil_median"].to_numpy(dtype=float),
                    x.to_numpy(dtype=float),
                    groups=d["analysis_group_token"].astype(str).to_numpy(),
                    family=sm.families.Gaussian(),
                    cov_struct=sm.cov_struct.Exchangeable(),
                ).fit()
                index = list(x.columns).index(focal)
                estimate = float(np.asarray(fit.params, dtype=float)[index])
                se = float(np.asarray(fit.bse, dtype=float)[index])
                if not np.isfinite(estimate) or not np.isfinite(se):
                    raise ValueError("non-finite focal estimate or SE")
                results.append(
                    {
                        "analysis_question": "visual_confound_adjustment",
                        "outcome": "pupil_median",
                        "visual_predictor": focal,
                        "model_stage": stage,
                        "estimate_per_predictor_sd": estimate,
                        "se": se,
                        "ci95_low": estimate - 1.96 * se,
                        "ci95_high": estimate + 1.96 * se,
                        "n_rows": int(len(d)),
                        "participant_group_n": group_n,
                        "session_n": int(d["session_id"].astype(str).nunique()),
                        "observation_unit": f"trial window {cfg.primary_window}",
                        "track": cfg.primary_track,
                        "controls": ";".join(c for c in used if c != focal),
                        "status": "complete",
                        "current_trial_visual_used": False,
                        "interpretation_boundary": "association with pre-stimulus pupil; adjusted model does not prove causality",
                    }
                )
            except Exception as exc:
                failures.append(
                    _failure(
                        target=focal,
                        model_stage=stage,
                        failure_type=type(exc).__name__,
                        failure_detail=str(exc),
                    )
                )
    return pd.DataFrame(results), pd.DataFrame(failures, columns=MODEL_FAILURE_COLUMNS)


def dynamic_feature_admission_registry() -> pd.DataFrame:
    """Record admitted dynamics and explicit fail-closed exclusions."""
    return pd.DataFrame(
        [
            {
                "feature": "pupil_peak_to_trough",
                "status": "admitted_candidate",
                "minimum_support": ">=2 valid samples",
                "unit": "same units as pupil track",
                "meaning": "window maximum minus minimum",
            },
            {
                "feature": "pupil_dilation_velocity_median_per_sec",
                "status": "admitted_candidate",
                "minimum_support": "dynamic velocity family requires >=2 valid successive-rate pairs",
                "unit": "pupil units/s",
                "meaning": "median positive first derivative when present",
            },
            {
                "feature": "pupil_constriction_velocity_median_per_sec",
                "status": "admitted_candidate",
                "minimum_support": "dynamic velocity family requires >=2 valid successive-rate pairs",
                "unit": "pupil units/s (positive magnitude)",
                "meaning": "median absolute negative first derivative when present",
            },
            {
                "feature": "recovery_magnitude_or_time",
                "status": "not_admitted",
                "minimum_support": "requires an independently valid recovery window",
                "unit": "not applicable",
                "meaning": "do not invent recovery across overlapping SART events",
            },
            {
                "feature": "frequency_domain_pupil",
                "status": "not_admitted",
                "minimum_support": "requires timestamp/gap/sampling audit and prespecified band",
                "unit": "not applicable",
                "meaning": "fail closed until temporal sampling support is qualified",
            },
        ]
    )
