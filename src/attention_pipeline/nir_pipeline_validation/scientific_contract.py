from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

MODEL_FAILURE_COLUMNS = [
    "analysis_question",
    "target",
    "model_stage",
    "fold",
    "model_name",
    "status",
    "failure_type",
    "failure_detail",
]


@dataclass(frozen=True)
class ModelGate:
    accepted: bool
    status: str
    failure_type: str | None = None
    failure_detail: str | None = None


def split_sart_error_targets(trials: pd.DataFrame) -> pd.DataFrame:
    """Create separate Go-omission and NoGo-commission targets.

    A single generic ``error`` target is intentionally not emitted because the
    two SART failure modes have different denominators, prevalence and behavior.
    """
    result = trials.copy()
    is_nogo = pd.to_numeric(result["is_no_go"], errors="coerce").eq(1)
    omission = pd.to_numeric(result["omission"], errors="coerce")
    commission = pd.to_numeric(result["commission"], errors="coerce")
    result["go_omission_target"] = np.where(~is_nogo, omission, np.nan)
    result["nogo_commission_target"] = np.where(is_nogo, commission, np.nan)
    result["target_denominator"] = np.where(is_nogo, "nogo_trials", "go_trials")
    return result


def _binary_arrays(y_true: Sequence[Any], y_pred: Sequence[Any] | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    y = pd.to_numeric(pd.Series(y_true), errors="coerce")
    valid = y.isin([0, 1])
    truth = y[valid].to_numpy(dtype=int)
    if y_pred is None:
        return truth, None
    pred = pd.to_numeric(pd.Series(y_pred), errors="coerce")
    pred = pred.loc[valid]
    pred_valid = pred.isin([0, 1])
    truth = truth[pred_valid.to_numpy()]
    return truth, pred[pred_valid].to_numpy(dtype=int)


def _f1(y: np.ndarray, pred: np.ndarray, positive: int) -> float:
    tp = int(np.sum((y == positive) & (pred == positive)))
    fp = int(np.sum((y != positive) & (pred == positive)))
    fn = int(np.sum((y == positive) & (pred != positive)))
    denom = 2 * tp + fp + fn
    return (2.0 * tp / denom) if denom else float("nan")


def _auc_rank(y: np.ndarray, score: np.ndarray) -> float:
    valid = np.isfinite(score)
    y = y[valid]
    score = score[valid]
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy(dtype=float)
    rank_sum_pos = float(ranks[y == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def binary_classification_metrics(
    y_true: Sequence[Any],
    *,
    y_pred: Sequence[Any] | None = None,
    score: Sequence[Any] | None = None,
) -> dict[str, float | int]:
    y, pred = _binary_arrays(y_true, y_pred)
    if len(y) == 0:
        return {
            "n": 0,
            "positive_prevalence": float("nan"),
            "majority_accuracy": float("nan"),
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
            "auroc": float("nan"),
            "brier": float("nan"),
        }
    prevalence = float(np.mean(y))
    result: dict[str, float | int] = {
        "n": int(len(y)),
        "positive_prevalence": prevalence,
        "majority_accuracy": max(prevalence, 1.0 - prevalence),
        "accuracy": float("nan"),
        "balanced_accuracy": float("nan"),
        "macro_f1": float("nan"),
        "auroc": float("nan"),
        "brier": float("nan"),
    }
    if pred is not None and len(pred) == len(y):
        result["accuracy"] = float(np.mean(pred == y))
        sensitivity = float(np.mean(pred[y == 1] == 1)) if np.any(y == 1) else float("nan")
        specificity = float(np.mean(pred[y == 0] == 0)) if np.any(y == 0) else float("nan")
        result["balanced_accuracy"] = float(np.nanmean([sensitivity, specificity]))
        result["macro_f1"] = float(np.nanmean([_f1(y, pred, 0), _f1(y, pred, 1)]))
    if score is not None:
        raw_score = pd.to_numeric(pd.Series(score), errors="coerce").to_numpy(dtype=float)
        if len(raw_score) == len(y):
            result["auroc"] = _auc_rank(y, raw_score)
            finite = np.isfinite(raw_score)
            result["brier"] = (
                float(np.mean((raw_score[finite] - y[finite]) ** 2)) if finite.any() else float("nan")
            )
    return result


def participant_exclusive_outer_folds(
    frame: pd.DataFrame,
    *,
    group_col: str = "analysis_group_token",
    target_col: str,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Assign outer folds at analysis-group level with simple load/prevalence balance."""
    if group_col not in frame or target_col not in frame:
        raise KeyError(f"fold assignment requires {group_col!r} and {target_col!r}")
    work = frame[[group_col, target_col]].copy()
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work[work[target_col].isin([0, 1]) & work[group_col].notna()]
    summary = (
        work.groupby(group_col, as_index=False)[target_col]
        .agg(n="size", positives="sum")
    )
    n_groups = len(summary)
    if n_groups < 2:
        raise ValueError("participant-exclusive folds require at least two analysis groups")
    k = min(max(2, int(n_splits)), n_groups)
    summary["prevalence"] = summary["positives"] / summary["n"]
    summary = summary.sort_values(
        ["n", "prevalence", group_col], ascending=[False, False, True], kind="stable"
    ).reset_index(drop=True)
    fold_n = np.zeros(k, dtype=float)
    fold_pos = np.zeros(k, dtype=float)
    assignments: list[dict[str, Any]] = []
    global_prev = float(summary["positives"].sum() / max(1, summary["n"].sum()))
    for row in summary.itertuples(index=False):
        costs = []
        for fold in range(k):
            new_n = fold_n[fold] + float(row.n)
            new_pos = fold_pos[fold] + float(row.positives)
            new_prev = new_pos / max(1.0, new_n)
            size_cost = new_n / max(1.0, summary["n"].sum())
            prevalence_cost = abs(new_prev - global_prev)
            costs.append((size_cost + prevalence_cost, fold_n[fold], fold))
        fold = min(costs)[2]
        fold_n[fold] += float(row.n)
        fold_pos[fold] += float(row.positives)
        assignments.append(
            {
                group_col: getattr(row, group_col),
                "outer_fold": int(fold),
                "group_n_rows": int(row.n),
                "group_positive_n": int(row.positives),
                "group_positive_prevalence": float(row.prevalence),
            }
        )
    result = pd.DataFrame(assignments)
    if result[group_col].duplicated().any():
        raise AssertionError("one analysis group was assigned to multiple outer folds")
    return result.sort_values("outer_fold", kind="stable").reset_index(drop=True)


def assert_outer_fold_exclusive(
    rows: pd.DataFrame,
    *,
    fold: int,
    group_col: str = "analysis_group_token",
    fold_col: str = "outer_fold",
) -> None:
    train = set(rows.loc[rows[fold_col].ne(fold), group_col].dropna().astype(str))
    test = set(rows.loc[rows[fold_col].eq(fold), group_col].dropna().astype(str))
    overlap = train & test
    if overlap:
        raise AssertionError("analysis-group leakage across outer train/test fold")


def aggregate_visual_components(
    visual: pd.DataFrame,
    *,
    key_cols: Sequence[str] = ("stimulus_name", "stimulus_size"),
) -> pd.DataFrame:
    """Collapse one-to-many stimulus components without silently dropping coverage."""
    missing = [name for name in key_cols if name not in visual]
    if missing:
        raise ValueError(f"visual table missing key fields: {missing}")
    numeric = [
        name
        for name in visual.columns
        if name not in key_cols and pd.api.types.is_numeric_dtype(visual[name])
    ]
    rows: list[dict[str, Any]] = []
    for key, current in visual.groupby(list(key_cols), sort=False, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        record = dict(zip(key_cols, key))
        record["visual_component_count"] = int(len(current))
        for name in numeric:
            values = pd.to_numeric(current[name], errors="coerce")
            record[f"{name}__mean"] = float(values.mean()) if values.notna().any() else np.nan
            record[f"{name}__min"] = float(values.min()) if values.notna().any() else np.nan
            record[f"{name}__max"] = float(values.max()) if values.notna().any() else np.nan
        rows.append(record)
    return pd.DataFrame(rows)


def attach_causal_visual_covariates(
    trials: pd.DataFrame,
    visual: pd.DataFrame,
    *,
    session_col: str = "session_id",
    block_col: str = "block_num",
    key_cols: Sequence[str] = ("stimulus_name", "stimulus_size"),
) -> pd.DataFrame:
    """Join current and strictly previous stimulus properties with explicit direction."""
    work = trials.copy()
    for name in (session_col, block_col, *key_cols):
        if name not in work:
            raise ValueError(f"trial table missing visual join field: {name}")
    aggregated = aggregate_visual_components(visual, key_cols=key_cols)
    current = aggregated.rename(
        columns={name: f"current_visual__{name}" for name in aggregated.columns if name not in key_cols}
    )
    work = work.merge(current, on=list(key_cols), how="left", validate="many_to_one")
    work["current_visual_matched"] = work["current_visual__visual_component_count"].notna()

    ordered = work.sort_values(
        [session_col, block_col, "absolute_onset_time", "trial_num"], kind="stable"
    ).copy()
    previous_keys: list[str] = []
    for name in key_cols:
        previous = f"previous__{name}"
        ordered[previous] = ordered.groupby([session_col, block_col], sort=False)[name].shift(1)
        previous_keys.append(previous)
    previous_visual = aggregated.rename(
        columns={
            **{name: f"previous__{name}" for name in key_cols},
            **{
                name: f"previous_visual__{name}"
                for name in aggregated.columns
                if name not in key_cols
            },
        }
    )
    ordered = ordered.merge(
        previous_visual,
        on=previous_keys,
        how="left",
        validate="many_to_one",
    )
    ordered["previous_visual_matched"] = ordered[
        "previous_visual__visual_component_count"
    ].notna()
    ordered["visual_time_direction"] = "current=trial stimulus; previous=strictly prior trial within same session/block"
    ordered["visual_multiple_component_policy"] = "aggregate all matched components and retain component count/min/max/mean"
    return ordered.sort_index()


def build_phasic_pupil_features(
    trial_windows: pd.DataFrame,
    *,
    baseline_window: str = "pre_200ms",
    response_window: str = "post_200_1150ms",
) -> pd.DataFrame:
    keys = [
        "session_id",
        "analysis_group_token",
        "block_num",
        "trial_num",
        "global_trial_index",
        "track",
    ]
    missing = [name for name in [*keys, "window_name", "pupil_median"] if name not in trial_windows]
    if missing:
        raise ValueError(f"trial windows missing phasic fields: {missing}")
    base = trial_windows[trial_windows["window_name"].astype(str).eq(baseline_window)][
        [*keys, "pupil_median", "pupil_valid_fraction", "internal_coverage_fraction"]
    ].rename(
        columns={
            "pupil_median": "phasic_baseline_pupil_median",
            "pupil_valid_fraction": "phasic_baseline_valid_fraction",
            "internal_coverage_fraction": "phasic_baseline_internal_coverage",
        }
    )
    response = trial_windows[trial_windows["window_name"].astype(str).eq(response_window)][
        [*keys, "pupil_median", "pupil_valid_fraction", "internal_coverage_fraction"]
    ].rename(
        columns={
            "pupil_median": "phasic_response_pupil_median",
            "pupil_valid_fraction": "phasic_response_valid_fraction",
            "internal_coverage_fraction": "phasic_response_internal_coverage",
        }
    )
    merged = base.merge(response, on=keys, how="inner", validate="one_to_one")
    merged["phasic_pupil_delta"] = (
        pd.to_numeric(merged["phasic_response_pupil_median"], errors="coerce")
        - pd.to_numeric(merged["phasic_baseline_pupil_median"], errors="coerce")
    )
    merged["phasic_baseline_window"] = baseline_window
    merged["phasic_response_window"] = response_window
    merged["feature_family"] = "phasic"
    return merged


def feature_family_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "family": "tonic",
                "features": "pupil_median;pupil_mean",
                "meaning": "state level within prespecified pre-event/probe window",
                "baseline_required": False,
            },
            {
                "family": "variability",
                "features": "pupil_mad;pupil_iqr;pupil_sd;pupil_p10;pupil_p90",
                "meaning": "within-window dispersion/distribution",
                "baseline_required": False,
            },
            {
                "family": "derivative",
                "features": "pupil_slope_per_sec;pupil_diff_mad;pupil_diff_rate_mad_per_sec",
                "meaning": "within-window rate/change descriptors",
                "baseline_required": False,
            },
            {
                "family": "phasic",
                "features": "phasic_pupil_delta",
                "meaning": "post-event response minus explicit pre-event baseline",
                "baseline_required": True,
            },
        ]
    )


def audit_feature_family_columns(frame: pd.DataFrame) -> pd.DataFrame:
    registry = feature_family_registry()
    rows: list[dict[str, Any]] = []
    columns = set(frame.columns)
    for row in registry.itertuples(index=False):
        features = str(row.features).split(";")
        present = [name for name in features if name in columns]
        baseline_metadata = {
            "phasic_baseline_window",
            "phasic_response_window",
        }.issubset(columns)
        rows.append(
            {
                "family": row.family,
                "expected_features": row.features,
                "n_present": len(present),
                "present_features": ";".join(present),
                "baseline_required": bool(row.baseline_required),
                "baseline_metadata_present": bool(
                    baseline_metadata if row.baseline_required else True
                ),
                "contract_ok": bool(
                    len(present) > 0
                    and (baseline_metadata if row.baseline_required else True)
                ),
            }
        )
    return pd.DataFrame(rows)


def audit_model_result(model: Any, *, model_name: str) -> ModelGate:
    """Reject singular/non-converged/empty model objects before result publication."""
    if model is None:
        return ModelGate(False, "failed", "empty_result", f"{model_name}: model result is None")
    converged = getattr(model, "converged", True)
    if converged is False:
        return ModelGate(False, "failed", "not_converged", f"{model_name}: converged=False")
    try:
        params = np.asarray(pd.Series(model.params), dtype=float)
        bse = np.asarray(pd.Series(model.bse), dtype=float)
    except Exception as exc:
        return ModelGate(False, "failed", "invalid_result_table", f"{type(exc).__name__}: {exc}")
    if params.size == 0:
        return ModelGate(False, "failed", "empty_result", f"{model_name}: no parameters")
    if not np.isfinite(params).all() or not np.isfinite(bse).all():
        return ModelGate(False, "failed", "nonfinite_estimates", f"{model_name}: non-finite parameter or SE")
    cov_re = getattr(model, "cov_re", None)
    if cov_re is not None:
        try:
            matrix = np.atleast_2d(np.asarray(cov_re, dtype=float))
            if not np.isfinite(matrix).all():
                return ModelGate(False, "failed", "singular_random_effects", f"{model_name}: non-finite random-effect covariance")
            eigen = np.linalg.eigvalsh(matrix)
            if eigen.size and float(np.min(eigen)) <= 1e-10:
                return ModelGate(False, "failed", "singular_random_effects", f"{model_name}: random-effect covariance is singular/boundary")
        except np.linalg.LinAlgError as exc:
            return ModelGate(False, "failed", "singular_random_effects", f"{type(exc).__name__}: {exc}")
    return ModelGate(True, "complete")


def repeat_session_descriptive_summary(
    time_on_task: pd.DataFrame,
    *,
    track: str = "binocular_primary",
) -> pd.DataFrame:
    """Describe B1/B2 repeat-session agreement without claiming reliability."""
    work = time_on_task[time_on_task["track"].astype(str).eq(track)].copy()
    work["pupil_median"] = pd.to_numeric(work["pupil_median"], errors="coerce")
    session = (
        work.groupby(["analysis_group_token", "session_id", "block_num"], as_index=False)
        .agg(pupil_median=("pupil_median", "median"), n_bins=("pupil_median", "size"))
    )
    counts = session.groupby("analysis_group_token")["session_id"].nunique()
    repeat_tokens = set(counts[counts.eq(2)].index.astype(str))
    rows: list[dict[str, Any]] = []
    for token, current in session[
        session["analysis_group_token"].astype(str).isin(repeat_tokens)
    ].groupby("analysis_group_token", sort=True):
        sessions = sorted(current["session_id"].astype(str).unique())
        for block_num in sorted(pd.to_numeric(current["block_num"], errors="coerce").dropna().astype(int).unique()):
            block = current[pd.to_numeric(current["block_num"], errors="coerce").eq(block_num)]
            values = block.set_index("session_id")["pupil_median"]
            if len(sessions) == 2 and all(name in values.index for name in sessions):
                a, b = float(values[sessions[0]]), float(values[sessions[1]])
                rows.append(
                    {
                        "analysis_group_token": token,
                        "block_num": int(block_num),
                        "session_pair_count": 1,
                        "session_a_pupil_median": a,
                        "session_b_pupil_median": b,
                        "absolute_difference": abs(a - b),
                        "signed_difference": b - a,
                        "interpretation": "descriptive_repeat_session_difference_only; B1/B2 are insufficient for broad stability/validity claims",
                    }
                )
    return pd.DataFrame(rows)


def qc_count_axes(
    *,
    sessions: Iterable[Any],
    analysis_groups: Iterable[Any],
    eye_rows: int,
    timepoints: int,
    trial_windows: int,
    probe_windows: int,
    failures: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"axis": "sessions", "count": len(set(map(str, sessions)))},
            {"axis": "analysis_groups", "count": len(set(map(str, analysis_groups)))},
            {"axis": "eye_rows", "count": int(eye_rows)},
            {"axis": "timepoints", "count": int(timepoints)},
            {"axis": "trial_windows", "count": int(trial_windows)},
            {"axis": "probe_windows", "count": int(probe_windows)},
            {"axis": "failures", "count": int(failures)},
        ]
    )


def report_admission(
    *,
    figure_names: Iterable[str],
    trial_windows: pd.DataFrame,
    probe_windows: pd.DataFrame,
    model_failures: pd.DataFrame,
    failure_tables_written: bool,
    topology: Mapping[str, int],
) -> dict[str, Any]:
    figures = set(map(str, figure_names))
    required_figures = {f"Figure{index:02d}" for index in range(1, 11)}
    track_values = set(trial_windows.get("track", pd.Series(dtype=str)).dropna().astype(str))
    hierarchy_ok = {
        "session_id",
        "analysis_group_token",
        "block_num",
    }.issubset(trial_windows.columns)
    q1_ok = {
        "go_omission_target",
        "nogo_commission_target",
    }.issubset(trial_windows.columns)
    # Q2 is the probe-state stream; we avoid inventing a stronger construct label.
    q2_ok = (
        not probe_windows.empty
        and "probe_response" in probe_windows
        and "probe_vigilance" in probe_windows
    )
    eyes_ok = {"left_primary", "right_primary"}.issubset(track_values)
    model_ok = model_failures.empty or not model_failures["status"].astype(str).eq("published_as_complete").any()
    topology_ok = topology == {
        "n_sessions": 44,
        "n_analysis_groups": 38,
        "n_double_session_repeat_groups": 6,
    }
    gates = {
        "figure01_10_complete": required_figures.issubset(figures),
        "q1_separate_error_targets_present": q1_ok,
        "q2_probe_stream_present": q2_ok,
        "left_right_tracks_present": eyes_ok,
        "session_analysis_group_block_hierarchy_present": hierarchy_ok,
        "model_failure_gate_active": model_ok,
        "failure_tables_written": bool(failure_tables_written),
        "topology_count_axes_match_current_contract": topology_ok,
    }
    package_admitted = all(gates.values())
    return {
        "report_package_admitted": package_admitted,
        "scientific_inference_authorized": False,
        "measurement_validity_claim_authorized": False,
        "formal_statistics_claim_authorized": False,
        "gates": gates,
        "interpretation_boundary": (
            "report admission is a completeness/contract gate only; it is not a measurement-validity "
            "or formal inferential conclusion"
        ),
    }
