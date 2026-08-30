from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .analysis import add_within_between, trial_outcome_label


def _safe_z(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    sd = values.std(ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=values.index)
    return (values - values.mean()) / sd


def _model_table(model: Any, model_name: str) -> pd.DataFrame:
    params = pd.Series(model.params)
    bse = pd.Series(model.bse).reindex(params.index)
    pvalues = pd.Series(model.pvalues).reindex(params.index)
    return pd.DataFrame(
        {
            "model": model_name,
            "term": params.index.astype(str),
            "estimate": params.values,
            "se": bse.values,
            "p_value": pvalues.values,
        }
    )


def fit_extended_smoke_models(
    trial_level: pd.DataFrame,
    trial_windows: pd.DataFrame,
    probe_rt: pd.DataFrame,
    precursor: pd.DataFrame,
    visual_trial: pd.DataFrame,
    *,
    track: str,
    window_name: str,
    min_subjects: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    tables: list[pd.DataFrame] = []
    status: list[dict[str, Any]] = []

    def record(name: str, fn: Any) -> None:
        try:
            model = fn()
            tables.append(_model_table(model, name))
            status.append({"model": name, "status": "complete"})
        except Exception as exc:
            status.append(
                {
                    "model": name,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    windows = trial_windows[
        trial_windows["track"].astype(str).eq(track)
        & trial_windows["window_name"].astype(str).eq(window_name)
    ].copy()
    keys = [
        col
        for col in ("subject", "block_num", "trial_num", "global_trial_index")
        if col in windows.columns and col in trial_level.columns
    ]
    behavior = trial_level.copy()
    behavior["outcome"] = trial_outcome_label(behavior)
    keep = keys + [col for col in ("outcome", "time_in_block_sec") if col in behavior.columns]
    dynamic = windows.merge(behavior[keep], on=keys, how="left", validate="one_to_one")
    if "time_in_block_sec" in dynamic.columns:
        dynamic["time_z"] = _safe_z(dynamic["time_in_block_sec"])
    else:
        dynamic["time_z"] = np.nan

    for feature, model_name in (
        ("pupil_mad", "lmm_trial_pir_variability_by_outcome"),
        ("pupil_slope_per_sec", "lmm_trial_pir_slope_by_outcome"),
        ("pupil_diff_rate_mad_per_sec", "lmm_trial_pir_shortterm_instability_by_outcome"),
    ):
        if feature not in dynamic.columns:
            status.append({"model": model_name, "status": "skipped", "reason": "feature_missing"})
            continue
        current = dynamic.copy()
        current[feature] = pd.to_numeric(current[feature], errors="coerce")
        current = current.dropna(subset=[feature, "outcome", "subject", "block_num"])
        if current["subject"].nunique() >= min_subjects and current["outcome"].nunique() >= 2:
            record(
                model_name,
                lambda current=current, feature=feature: smf.mixedlm(
                    f"{feature} ~ C(outcome) + C(block_num)",
                    data=current,
                    groups=current["subject"],
                ).fit(reml=False, method="lbfgs", disp=False),
            )
        else:
            status.append({"model": model_name, "status": "skipped", "reason": "insufficient_rows_or_outcome_levels"})

    if not precursor.empty:
        pre = precursor[precursor["lag"].lt(0)].copy()
        pre["lag_numeric"] = pd.to_numeric(pre["lag"], errors="coerce")
        pre["go_rt_ms"] = pd.to_numeric(pre["go_rt_ms"], errors="coerce")
        pre["pupil_median"] = pd.to_numeric(pre["pupil_median"], errors="coerce")

        rt = pre.dropna(subset=["lag_numeric", "go_rt_ms", "event_outcome", "subject"])
        if rt["subject"].nunique() >= min_subjects and rt["event_outcome"].nunique() >= 2:
            record(
                "lmm_nogo_precursor_go_rt",
                lambda: smf.mixedlm(
                    "go_rt_ms ~ lag_numeric * C(event_outcome) + C(block_num)",
                    data=rt,
                    groups=rt["subject"],
                ).fit(reml=False, method="lbfgs", disp=False),
            )
        else:
            status.append({"model": "lmm_nogo_precursor_go_rt", "status": "skipped", "reason": "insufficient_precursor_rows"})

        pir = pre.dropna(subset=["lag_numeric", "pupil_median", "event_outcome", "subject"])
        if pir["subject"].nunique() >= min_subjects and pir["event_outcome"].nunique() >= 2:
            record(
                "lmm_nogo_precursor_pir",
                lambda: smf.mixedlm(
                    "pupil_median ~ lag_numeric * C(event_outcome) + C(block_num)",
                    data=pir,
                    groups=pir["subject"],
                ).fit(reml=False, method="lbfgs", disp=False),
            )
        else:
            status.append({"model": "lmm_nogo_precursor_pir", "status": "skipped", "reason": "insufficient_precursor_rows"})

    if not probe_rt.empty:
        if {"probe_rt", "probe_response", "subject", "block_num"}.issubset(probe_rt.columns):
            response_rt = probe_rt.copy()
            response_rt["probe_rt"] = pd.to_numeric(response_rt["probe_rt"], errors="coerce")
            response_rt = response_rt.dropna(subset=["probe_rt", "probe_response", "subject"])
            if response_rt["subject"].nunique() >= min_subjects and response_rt["probe_response"].nunique() >= 2:
                record(
                    "lmm_probe_response_rt",
                    lambda: smf.mixedlm(
                        "probe_rt ~ C(probe_response) + C(block_num)",
                        data=response_rt,
                        groups=response_rt["subject"],
                    ).fit(reml=False, method="lbfgs", disp=False),
                )
            else:
                status.append({"model": "lmm_probe_response_rt", "status": "skipped", "reason": "insufficient_probe_response_rt"})

        if {"probe_vigilance_rt", "probe_vigilance", "subject", "block_num"}.issubset(probe_rt.columns):
            vig_rt = probe_rt.copy()
            vig_rt["probe_vigilance_rt"] = pd.to_numeric(vig_rt["probe_vigilance_rt"], errors="coerce")
            vig_rt["probe_vigilance"] = pd.to_numeric(vig_rt["probe_vigilance"], errors="coerce")
            vig_rt = vig_rt.dropna(subset=["probe_vigilance_rt", "probe_vigilance", "subject"])
            if vig_rt["subject"].nunique() >= min_subjects:
                record(
                    "lmm_probe_vigilance_rt",
                    lambda: smf.mixedlm(
                        "probe_vigilance_rt ~ probe_vigilance + C(block_num)",
                        data=vig_rt,
                        groups=vig_rt["subject"],
                    ).fit(reml=False, method="lbfgs", disp=False),
                )
            else:
                status.append({"model": "lmm_probe_vigilance_rt", "status": "skipped", "reason": "insufficient_probe_vigilance_rt"})

    if not visual_trial.empty:
        required = {
            "subject",
            "block_num",
            "pupil_median",
            "current_central_rel_lum_mean",
            "previous_central_rel_lum_mean",
        }
        if required.issubset(visual_trial.columns):
            visual = visual_trial.copy()
            for col in ("pupil_median", "current_central_rel_lum_mean", "previous_central_rel_lum_mean"):
                visual[col] = pd.to_numeric(visual[col], errors="coerce")
            visual = visual.dropna(subset=list(required))
            if visual["subject"].nunique() >= min_subjects:
                record(
                    "lmm_pir_visual_covariates",
                    lambda: smf.mixedlm(
                        "pupil_median ~ current_central_rel_lum_mean + previous_central_rel_lum_mean + C(block_num)",
                        data=visual,
                        groups=visual["subject"],
                    ).fit(reml=False, method="lbfgs", disp=False),
                )
            else:
                status.append({"model": "lmm_pir_visual_covariates", "status": "skipped", "reason": "insufficient_visual_rows"})

        go_required = required | {"is_no_go", "correct", "rt", "time_in_block_sec"}
        if go_required.issubset(visual_trial.columns):
            visual_go = add_within_between(visual_trial, "pupil_median")
            visual_go["rt"] = pd.to_numeric(visual_go["rt"], errors="coerce")
            visual_go["time_z"] = _safe_z(visual_go["time_in_block_sec"])
            visual_go = visual_go[
                pd.to_numeric(visual_go["is_no_go"], errors="coerce").eq(0)
                & pd.to_numeric(visual_go["correct"], errors="coerce").eq(1)
            ].dropna(
                subset=[
                    "rt",
                    "pupil_median_within",
                    "pupil_median_between",
                    "current_central_rel_lum_mean",
                    "previous_central_rel_lum_mean",
                    "time_z",
                ]
            )
            if visual_go["subject"].nunique() >= min_subjects:
                record(
                    "lmm_go_rt_pir_visual_controlled",
                    lambda: smf.mixedlm(
                        "rt ~ pupil_median_within + pupil_median_between + current_central_rel_lum_mean + previous_central_rel_lum_mean + time_z + C(block_num)",
                        data=visual_go,
                        groups=visual_go["subject"],
                    ).fit(reml=False, method="lbfgs", disp=False),
                )
            else:
                status.append({"model": "lmm_go_rt_pir_visual_controlled", "status": "skipped", "reason": "insufficient_visual_go_rows"})

    combined = (
        pd.concat(tables, ignore_index=True)
        if tables
        else pd.DataFrame(columns=["model", "term", "estimate", "se", "p_value"])
    )
    return combined, status
