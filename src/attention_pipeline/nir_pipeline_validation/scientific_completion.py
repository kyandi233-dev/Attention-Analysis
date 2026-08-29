"""Scientific-completeness helpers for pupil-only NIR formal analysis.

This module extends the existing validation contract without changing the
session/participant identity adapter.  It is intentionally fail-closed:
visual exposure is strictly pre-probe, participant structure is explicit, and
model failures are returned as rows rather than silently dropped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .scientific_contract import MODEL_FAILURE_COLUMNS, attach_causal_visual_covariates


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
    """Add between-participant and within-participant pupil components.

    For each value x, ``x__participant_mean`` is the stable between-participant
    component and ``x__within_participant`` is the deviation from that mean.
    When session ids are available, session means/deviations are emitted too.
    No visit order is required or inferred.
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


def _probe_identity_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "session_id",
        "analysis_group_token",
        "block_num",
        "probe_index_global",
        "probe_index_in_block",
        "probe_trial_num",
        "probe_onset_ms",
        "window_name",
        "window_start_ms",
        "window_end_ms",
    ]
    return [c for c in candidates if c in frame.columns]


def aggregate_probe_visual_exposure(
    probe_windows: pd.DataFrame,
    trials: pd.DataFrame,
    visual: pd.DataFrame,
    *,
    visual_key_cols: Sequence[str] = ("stimulus_name", "stimulus_size"),
) -> pd.DataFrame:
    """Aggregate actual visual exposure inside each strictly pre-probe window.

    The anchoring probe trial is excluded both by time (< probe onset) and, when
    trial numbers are present, by ``trial_num < probe_trial_num``.  No stimulus
    occurring at or after the probe can contribute to the exposure summary.
    The summary is trial-weighted (mean/median across experienced stimuli), not
    duration-weighted; trial count and matched coverage are retained explicitly.
    """
    required_probe = {
        "session_id",
        "block_num",
        "probe_onset_ms",
        "window_start_ms",
        "window_end_ms",
    }
    missing = sorted(required_probe - set(probe_windows.columns))
    if missing:
        raise ValueError(f"probe windows missing exposure fields: {missing}")
    required_trial = {"session_id", "block_num", "absolute_onset_time"}
    missing_trial = sorted(required_trial - set(trials.columns))
    if missing_trial:
        raise ValueError(f"trials missing exposure fields: {missing_trial}")

    linked = attach_causal_visual_covariates(
        trials, visual, key_cols=visual_key_cols
    )
    current_visual = [
        c
        for c in linked.columns
        if c.startswith("current_visual__")
        and pd.api.types.is_numeric_dtype(linked[c])
    ]
    identities = _probe_identity_columns(probe_windows)
    unique_probes = probe_windows[identities].drop_duplicates().copy()
    rows: list[dict[str, Any]] = []

    for probe in unique_probes.itertuples(index=False):
        probe_dict = probe._asdict()
        session_id = str(probe_dict["session_id"])
        block_num = float(probe_dict["block_num"])
        onset = float(probe_dict["probe_onset_ms"])
        start = float(probe_dict["window_start_ms"])
        end = min(float(probe_dict["window_end_ms"]), onset)

        onset_series = pd.to_numeric(linked["absolute_onset_time"], errors="coerce")
        block_series = pd.to_numeric(linked["block_num"], errors="coerce")
        current = linked[
            linked["session_id"].astype(str).eq(session_id)
            & block_series.eq(block_num)
            & onset_series.ge(start)
            & onset_series.lt(end)
            & onset_series.lt(onset)
        ].copy()

        probe_trial_num = probe_dict.get("probe_trial_num")
        if (
            probe_trial_num is not None
            and "trial_num" in current
            and pd.notna(probe_trial_num)
        ):
            current = current[
                pd.to_numeric(current["trial_num"], errors="coerce")
                < float(probe_trial_num)
            ]

        record = dict(probe_dict)
        record["visual_exposure_trial_n"] = int(len(current))
        matched = (
            current["current_visual_matched"].fillna(False).astype(bool)
            if "current_visual_matched" in current
            else pd.Series(False, index=current.index)
        )
        record["visual_exposure_matched_trial_n"] = int(matched.sum())
        record["visual_exposure_matched_fraction"] = (
            float(matched.mean()) if len(current) else np.nan
        )
        record["visual_exposure_latest_trial_onset_ms"] = (
            float(pd.to_numeric(current["absolute_onset_time"], errors="coerce").max())
            if len(current)
            else np.nan
        )
        record["strict_pre_probe_verified"] = bool(
            len(current) == 0
            or pd.to_numeric(current["absolute_onset_time"], errors="coerce")
            .lt(onset)
            .all()
        )
        record["anchoring_probe_trial_excluded"] = True
        record["visual_exposure_weighting"] = "trial_weighted_mean_and_median"
        record["visual_exposure_time_direction"] = (
            "trial onset inside [window_start, min(window_end, probe_onset)); "
            "anchoring/future stimuli excluded"
        )

        for column in current_visual:
            values = pd.to_numeric(current[column], errors="coerce")
            short = column.removeprefix("current_visual__")
            record[f"probe_exposure__{short}__mean"] = (
                float(values.mean()) if values.notna().any() else np.nan
            )
            record[f"probe_exposure__{short}__median"] = (
                float(values.median()) if values.notna().any() else np.nan
            )
        rows.append(record)

    result = pd.DataFrame(rows)
    if not result.empty and not result["strict_pre_probe_verified"].all():
        raise AssertionError("probe visual exposure contains post-probe stimulus leakage")
    return result


def _visual_predictors(frame: pd.DataFrame) -> list[str]:
    tokens = ("brightness", "lumin", "contrast", "size", "area", "visible")
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


def _model_failure(
    *,
    target: str,
    model_stage: str,
    model_name: str,
    failure_type: str,
    failure_detail: str,
) -> dict[str, Any]:
    return {
        "analysis_question": "visual_confound_adjustment",
        "target": target,
        "model_stage": model_stage,
        "fold": pd.NA,
        "model_name": model_name,
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
    """Fit participant-clustered unadjusted vs adjusted visual-pupil models.

    The dependent variable is pre-stimulus pupil state. Because the current
    stimulus has not yet occurred, only previous-stimulus visual covariates are
    eligible. Adjustment adds time-on-task/trial order, trial type and measured
    pupil coverage/source-mode covariates when available.
    """
    cfg = config or VisualAdjustmentConfig()
    work = visual_trial_windows.copy()
    if "track" in work:
        work = work[work["track"].astype(str).eq(cfg.primary_track)]
    if "window_name" in work:
        work = work[work["window_name"].astype(str).eq(cfg.primary_window)]
    if work.empty:
        failure = _model_failure(
            target="pupil_median",
            model_stage="all",
            model_name="GEE_Gaussian",
            failure_type="empty_primary_window",
            failure_detail=f"no rows for {cfg.primary_track}/{cfg.primary_window}",
        )
        return pd.DataFrame(), pd.DataFrame([failure], columns=MODEL_FAILURE_COLUMNS)

    keys = [
        c
        for c in ("session_id", "block_num", "trial_num", "global_trial_index")
        if c in work and c in trials
    ]
    trial_covars = [
        c
        for c in ("is_no_go", "time_in_block_sec", "global_trial_index")
        if c in trials and c not in keys
    ]
    if keys and trial_covars:
        right = trials[[*keys, *trial_covars]].drop_duplicates(keys)
        duplicate = [c for c in trial_covars if c in work and c not in keys]
        if duplicate:
            work = work.drop(columns=duplicate)
        work = work.merge(right, on=keys, how="left", validate="many_to_one")

    if "analysis_group_token" not in work or "pupil_median" not in work:
        failure = _model_failure(
            target="pupil_median",
            model_stage="all",
            model_name="GEE_Gaussian",
            failure_type="missing_required_columns",
            failure_detail="analysis_group_token or pupil_median missing",
        )
        return pd.DataFrame(), pd.DataFrame([failure], columns=MODEL_FAILURE_COLUMNS)

    predictors = _visual_predictors(work)
    if not predictors:
        failure = _model_failure(
            target="pupil_median",
            model_stage="all",
            model_name="GEE_Gaussian",
            failure_type="no_visual_predictor",
            failure_detail="no previous-stimulus brightness/contrast/size/area predictor with support",
        )
        return pd.DataFrame(), pd.DataFrame([failure], columns=MODEL_FAILURE_COLUMNS)

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
        if c in work
    ]

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for visual_predictor in predictors:
        for stage, columns in (
            ("unadjusted", [visual_predictor]),
            ("visual_time_quality_adjusted", [visual_predictor, *controls]),
        ):
            model_name = "GEE_Gaussian_exchangeable"
            try:
                import statsmodels.api as sm

                cols = list(dict.fromkeys(columns))
                d = work[
                    ["pupil_median", "analysis_group_token", "session_id", *cols]
                ].copy()
                d["pupil_median"] = pd.to_numeric(d["pupil_median"], errors="coerce")
                for column in cols:
                    d[column] = pd.to_numeric(d[column], errors="coerce")
                d = d.replace([np.inf, -np.inf], np.nan).dropna(
                    subset=["pupil_median", "analysis_group_token", visual_predictor]
                )
                if len(d) < cfg.min_rows:
                    raise ValueError(f"rows {len(d)} < minimum {cfg.min_rows}")
                group_n = int(d["analysis_group_token"].astype(str).nunique())
                if group_n < cfg.min_participant_groups:
                    raise ValueError(
                        f"participant groups {group_n} < minimum {cfg.min_participant_groups}"
                    )

                for control in [c for c in cols if c != visual_predictor]:
                    median = d[control].median()
                    if pd.isna(median):
                        d = d.drop(columns=[control])
                    else:
                        d[control] = d[control].fillna(median)
                used = [c for c in cols if c in d.columns]
                variable = [c for c in used if d[c].nunique(dropna=True) >= 2]
                if visual_predictor not in variable:
                    raise ValueError("focal visual predictor has no variance")
                x = d[variable].copy()
                for column in variable:
                    mean = float(x[column].mean())
                    sd = float(x[column].std(ddof=0))
                    x[column] = (
                        (x[column] - mean) / sd
                        if np.isfinite(sd) and sd > 0
                        else x[column] - mean
                    )
                x = sm.add_constant(x, has_constant="add")
                fit = sm.GEE(
                    d["pupil_median"].to_numpy(dtype=float),
                    x.to_numpy(dtype=float),
                    groups=d["analysis_group_token"].astype(str).to_numpy(),
                    family=sm.families.Gaussian(),
                    cov_struct=sm.cov_struct.Exchangeable(),
                ).fit()
                names = list(x.columns)
                index = names.index(visual_predictor)
                estimate = float(np.asarray(fit.params, dtype=float)[index])
                se = float(np.asarray(fit.bse, dtype=float)[index])
                if not np.isfinite(estimate) or not np.isfinite(se):
                    raise ValueError("non-finite focal estimate or SE")
                results.append(
                    {
                        "analysis_question": "visual_confound_adjustment",
                        "outcome": "pupil_median",
                        "visual_predictor": visual_predictor,
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
                        "controls": ";".join(c for c in variable if c != visual_predictor),
                        "status": "complete",
                        "current_trial_visual_used": False,
                        "interpretation_boundary": (
                            "association with pre-stimulus pupil; adjusted model does not prove causality"
                        ),
                    }
                )
            except Exception as exc:
                failures.append(
                    _model_failure(
                        target=visual_predictor,
                        model_stage=stage,
                        model_name=model_name,
                        failure_type=type(exc).__name__,
                        failure_detail=str(exc),
                    )
                )

    return pd.DataFrame(results), pd.DataFrame(failures, columns=MODEL_FAILURE_COLUMNS)


def dynamic_feature_admission_registry() -> pd.DataFrame:
    """Document dynamic pupil features admitted by the current timing contract."""
    return pd.DataFrame(
        [
            {
                "feature": "pupil_peak_to_trough",
                "status": "admitted_candidate",
                "minimum_support": ">=2 valid samples",
                "unit": "same units as pupil track",
                "meaning": "window maximum minus minimum; spike-sensitive amplitude candidate",
            },
            {
                "feature": "pupil_dilation_velocity_median_per_sec",
                "status": "admitted_candidate",
                "minimum_support": "dynamic velocity family requires >=2 valid successive-rate pairs",
                "unit": "pupil units/s",
                "meaning": "median positive first derivative within the window when present",
            },
            {
                "feature": "pupil_constriction_velocity_median_per_sec",
                "status": "admitted_candidate",
                "minimum_support": "dynamic velocity family requires >=2 valid successive-rate pairs",
                "unit": "pupil units/s (positive magnitude)",
                "meaning": "median absolute negative first derivative within the window when present",
            },
            {
                "feature": "recovery_magnitude_or_time",
                "status": "not_admitted",
                "minimum_support": "requires an independently valid post-response recovery window",
                "unit": "not applicable",
                "meaning": "current SART event windows reach the next-trial boundary; recovery is not invented across overlapping events",
            },
            {
                "feature": "frequency_domain_pupil",
                "status": "not_admitted",
                "minimum_support": "requires separate timestamp/gap/sampling audit and prespecified frequency band",
                "unit": "not applicable",
                "meaning": "fail closed until irregular sampling and temporal gaps are formally qualified",
            },
        ]
    )
