"""Probe-level endpoint models: Q1 nominal response and Q2 ordinal vigilance.

File: probe_pupil_models.py
Version: nir-probe-endpoint-models-v1
Purpose:
    Frozen-endpoint formal association at the subjective-state anchor scale
    (probe).  One row per probe event, primary window ``pre_30s`` on the
    ``binocular_primary`` track.  Each frozen primary pupil metric
    (``pupil_geom_mean_diameter``, ``hard_pupil_fraction``) is decomposed into
    participant-mean (between) and deviation (within) components, then used as
    standardized predictors of:

    * Q1 ``probe_response`` - four-class nominal outcome, multinomial logit
      with reference category 1 and participant-cluster robust covariance;
    * Q2 ``probe_vigilance`` - four-level ordinal outcome.

    Q2 is implemented as a cumulative-logit ``OrderedModel`` with
    participant-cluster robust covariance because ``statsmodels``
    ``OrdinalGEE`` (the behavior-side reference implementation) hard-crashes
    the current runtime environment inside scipy's Cholesky factorization
    (native ``0xc06d007f``), which cannot be caught as a Python exception.
    The substitution is declared in every output row and in the manifest.

    Model failures are always written as ``not_estimable`` rows; empty result
    tables are never used to fake success.

Usage:
    >>> from attention_pipeline.nir_formal_analysis.probe_pupil_models import run_probe_pupil_models
    >>> manifest = run_probe_pupil_models("configs/nir_formal_analysis.yaml", subjects=["sub-041"])

Dependencies:
    numpy, pandas, statsmodels (MNLogit, miscmodels.ordinal_model.OrderedModel)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

from attention_pipeline.config import Config, load_config
from .pupil_tables import selected_sessions
from .scientific_models import add_pupil_within_between

# ---- Frozen parameters (centralized; see configs/nir_formal_analysis.yaml) ----
PROBE_MODEL_VERSION = "nir-probe-endpoint-models-v1"
PRIMARY_TRACK = "binocular_primary"
PRIMARY_PROBE_WINDOW = "pre_30s"
# Frozen primary pupil metrics from candidate_metrics.final_endpoint_freeze.
DEFAULT_PRIMARY_METRICS = ("pupil_geom_mean_diameter", "hard_pupil_fraction")
Q1_REFERENCE_CATEGORY = 1
Q1_CATEGORIES = (1, 2, 3, 4)
Q2_LEVELS = (1, 2, 3, 4)
# Minimum sample gates mirroring the trial-level reference model layer.
MIN_MODEL_ROWS = 24
MIN_PARTICIPANT_GROUPS = 6
Q2_FAMILY = "OrderedModel_cumulative_logit_cluster_robust"
Q2_FAMILY_NOTE = (
    "OrdinalGEE (behavior-side reference) unavailable in this runtime: native "
    "scipy cholesky crash 0xc06d007f kills the process and cannot be caught; "
    "cumulative-logit OrderedModel with participant-cluster robust covariance used instead"
)
# Sidecar columns required per metric for probe-window projection.
_METRIC_SIDECAR_SUFFIXES = ("__raw", "__valid_primary")


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


def _fuse_eye_values(left: float, right: float) -> tuple[float, str]:
    """Fuse per-eye window medians into one binocular value with source label."""
    l = np.isfinite(left)
    r = np.isfinite(right)
    if l and r:
        return float((left + right) / 2.0), "binocular"
    if l:
        return float(left), "left_only"
    if r:
        return float(right), "right_only"
    return np.nan, "missing"


def _project_windows(
    sidecar: pd.DataFrame,
    windows: pd.DataFrame,
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    """Project per-metric valid-frame medians from the candidate sidecar into probe windows.

    Parameters
    ----------
    sidecar : frame-level candidate sidecar with unix_ms/eye and metric raw+valid columns.
    windows : one row per probe window with window_start_ms/window_end_ms on the same unix axis.
    metrics : candidate metric names to project.

    Returns
    -------
    DataFrame with one row per window and per-metric median/valid-fraction/source columns.
    """
    # Pre-sort each eye's rows once so window slicing is a pair of binary searches.
    eye_views: dict[str, dict[str, Any]] = {}
    for eye, group in sidecar.groupby("eye", sort=True):
        group = group.sort_values("unix_ms", kind="stable")
        eye_views[str(eye)] = {
            "times": pd.to_numeric(group["unix_ms"], errors="coerce").to_numpy(dtype=float),
            "frame": group.reset_index(drop=True),
        }
    rows: list[dict[str, Any]] = []
    for window in windows.itertuples(index=False):
        start = float(window.window_start_ms)
        end = float(window.window_end_ms)
        record: dict[str, Any] = {}
        for metric in metrics:
            raw_col = f"{metric}__raw"
            valid_col = f"{metric}__valid_primary"
            fused: dict[str, float] = {}
            valid_fracs: list[float] = []
            n_valid: list[int] = []
            for eye, view in eye_views.items():
                left_edge = int(np.searchsorted(view["times"], start, side="left"))
                right_edge = int(np.searchsorted(view["times"], end, side="left"))
                segment = view["frame"].iloc[left_edge:right_edge]
                valid = segment[valid_col].fillna(False).astype(bool)
                values = pd.to_numeric(segment[raw_col], errors="coerce")
                eye_values = values[valid]
                fused[eye] = float(eye_values.median()) if len(eye_values) else np.nan
                n_valid.append(int(valid.sum()))
                valid_fracs.append(float(valid.mean()) if len(segment) else np.nan)
            value, source = _fuse_eye_values(fused.get("left", np.nan), fused.get("right", np.nan))
            record[metric] = value
            record[f"{metric}_valid_fraction"] = float(np.mean(valid_fracs)) if valid_fracs else np.nan
            record[f"{metric}_n_valid"] = int(np.sum(n_valid))
            record[f"{metric}_eye_source"] = source
        rows.append(record)
    return pd.DataFrame(rows)


def build_probe_endpoint_table(
    config: Config,
    sessions: Iterable[str],
    *,
    metrics: tuple[str, ...] = DEFAULT_PRIMARY_METRICS,
    window_name: str = PRIMARY_PROBE_WINDOW,
    track: str = PRIMARY_TRACK,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Build the probe-level frozen-metric table (one row per probe event).

    Parameters
    ----------
    config : loaded formal NIR config.
    sessions : session ids to include.
    metrics : frozen primary pupil metrics to project into each probe window.
    window_name : primary probe window; track : primary signal track.

    Returns
    -------
    (table, load_failures) - table rows are probe events carrying probe_response,
    probe_vigilance and per-metric window medians plus validity fractions; load
    failures are recorded per session instead of aborting the cohort run.
    """
    output_root = _resolve(config, "output_root")
    candidate_root = _resolve(config, "analysis_ready_root") / "candidate_frame_level"
    load_failures: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for session_id in sessions:
        probe_path = output_root / "sessions" / session_id / f"{session_id}_probe_pupil_windows.csv"
        sidecar_path = candidate_root / session_id / f"{session_id}_nir_pupil_candidates.csv"
        if not probe_path.is_file() or not sidecar_path.is_file():
            load_failures.append({
                "session_id": str(session_id),
                "reason": "probe_window_or_candidate_sidecar_missing",
                "probe_path": str(probe_path),
                "sidecar_path": str(sidecar_path),
            })
            continue
        probes = pd.read_csv(probe_path, encoding="utf-8-sig", low_memory=False)
        probes = probes[
            probes["track"].astype(str).eq(track)
            & probes["window_name"].astype(str).eq(window_name)
        ].copy()
        required = {
            "session_id", "analysis_group_token", "block_num", "probe_index_global",
            "probe_response", "probe_vigilance", "window_start_ms", "window_end_ms",
        }
        missing = sorted(required - set(probes.columns))
        if missing:
            load_failures.append({
                "session_id": str(session_id),
                "reason": f"probe_window_schema_missing:{','.join(missing)}",
            })
            continue
        sidecar_cols = ["eye", "unix_ms"]
        for metric in metrics:
            for suffix in _METRIC_SIDECAR_SUFFIXES:
                sidecar_cols.append(f"{metric}{suffix}")
        header = pd.read_csv(sidecar_path, nrows=0, encoding="utf-8-sig")
        missing_sidecar = sorted(set(sidecar_cols) - set(header.columns))
        if missing_sidecar:
            load_failures.append({
                "session_id": str(session_id),
                "reason": f"candidate_sidecar_schema_missing:{','.join(missing_sidecar)}",
            })
            continue
        sidecar = pd.read_csv(
            sidecar_path, usecols=sidecar_cols, encoding="utf-8-sig", low_memory=False
        )
        projected = _project_windows(sidecar, probes, metrics)
        probes = probes.reset_index(drop=True)
        combined = pd.concat([probes, projected.reset_index(drop=True)], axis=1)
        frames.append(combined)
    if not frames:
        return pd.DataFrame(), load_failures
    return pd.concat(frames, ignore_index=True), load_failures


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


def _effect_rows(
    fit: Any,
    *,
    model_name: str,
    family: str,
    outcome: str,
    metric: str,
    data: pd.DataFrame,
    family_note: str = "",
) -> list[dict[str, Any]]:
    """Extract pupil_within/pupil_between estimate rows from a fitted model."""
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
            "metric": metric,
            "pupil_term": term,
            "contrast_category": np.nan,
            "reference_category": np.nan,
            "estimate": est,
            "se": se,
            "ci_low": est - 1.96 * se,
            "ci_high": est + 1.96 * se,
            "p_value_not_for_endpoint_selection": float(p[term]),
            "n_rows": int(len(data)),
            "participant_group_n": int(data["analysis_group_token"].astype(str).nunique()),
            "session_n": int(data["session_id"].astype(str).nunique()),
            "predictor_standardization": "z_score_within_fitted_sample_per_term",
            "family_note": family_note,
            "status": "estimable",
        })
    return rows


def _gate_common(d: pd.DataFrame, *, min_rows: int, min_groups: int) -> str | None:
    """Shared sample-size and predictor-variance gate for probe models."""
    if len(d) < min_rows or d["analysis_group_token"].astype(str).nunique() < min_groups:
        return "insufficient_rows_or_participant_groups"
    for col in ("pupil_within", "pupil_between"):
        if d[col].nunique(dropna=False) < 2:
            return f"{col}_has_single_level"
    return None


def fit_probe_endpoint_models(
    table: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = DEFAULT_PRIMARY_METRICS,
    q1_reference_category: int = Q1_REFERENCE_CATEGORY,
    min_participant_groups: int = MIN_PARTICIPANT_GROUPS,
    min_rows: int = MIN_MODEL_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit Q1 multinomial and Q2 ordinal models for every frozen primary metric.

    Parameters
    ----------
    table : probe-level table from build_probe_endpoint_table (or equivalent
        synthetic table with per-metric columns, probe_response, probe_vigilance,
        session_id, analysis_group_token).
    metrics : frozen primary pupil metrics.
    q1_reference_category : Q1 reference category (frozen to 1).
    min_participant_groups / min_rows : minimum sample gates.

    Returns
    -------
    (results, failures) - estimable effect rows and not_estimable failure rows.
    """
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if table.empty:
        failures.append(_failure(
            model_name="probe_endpoint_models", family="none", outcome="all", metric="",
            data=table, reason="empty_probe_endpoint_table",
        ))
        return pd.DataFrame(), pd.DataFrame(failures)

    for metric in metrics:
        if metric not in table:
            failures.append(_failure(
                model_name=f"probe_{metric}", family="none", outcome="all", metric=metric,
                data=table, reason="metric_column_missing",
            ))
            continue
        decomposed = add_pupil_within_between(
            table, value_col=metric, group_col="analysis_group_token"
        )
        _fit_q1(decomposed, metric, results, failures, q1_reference_category, min_rows, min_participant_groups)
        _fit_q2(decomposed, metric, results, failures, min_rows, min_participant_groups)
    return pd.DataFrame(results), pd.DataFrame(failures)


def _fit_q1(
    decomposed: pd.DataFrame,
    metric: str,
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    reference_category: int,
    min_rows: int,
    min_groups: int,
) -> None:
    """Q1 four-class nominal response model: multinomial logit, cluster-robust."""
    model_name = f"Q1_{metric}"
    family = "MNLogit_cluster_robust"
    outcome = "probe_response"
    required = ["probe_response", "pupil_within", "pupil_between", "analysis_group_token", "session_id"]
    d = decomposed.dropna(subset=required).copy()
    levels = sorted(pd.to_numeric(d[outcome], errors="coerce").dropna().astype(int).unique())
    if set(levels) != set(Q1_CATEGORIES) or reference_category not in levels:
        failures.append(_failure(
            model_name=model_name, family=family, outcome=outcome, metric=metric,
            data=d, reason=f"all_Q1_categories_1_4_and_reference_{reference_category}_required",
        ))
        return
    gate = _gate_common(d, min_rows=min_rows, min_groups=min_groups)
    if gate:
        failures.append(_failure(
            model_name=model_name, family=family, outcome=outcome, metric=metric, data=d, reason=gate,
        ))
        return
    try:
        ordered = [reference_category] + [x for x in levels if x != reference_category]
        mapping = {level: idx for idx, level in enumerate(ordered)}
        y = pd.to_numeric(d[outcome], errors="coerce").astype(int).map(mapping).astype(int)
        x = pd.DataFrame({
            "pupil_within": _safe_z(d["pupil_within"]),
            "pupil_between": _safe_z(d["pupil_between"]),
        })
        x = sm.add_constant(x, has_constant="add")
        group_codes = pd.factorize(d["analysis_group_token"].astype(str))[0]
        model = sm.MNLogit(y, x)
        fit = model.fit(
            method="newton", maxiter=300, disp=False,
            cov_type="cluster", cov_kwds={"groups": group_codes},
        )
        reason = _fit_gate(fit)
        if reason:
            failures.append(_failure(
                model_name=model_name, family=family, outcome=outcome, metric=metric, data=d, reason=reason,
            ))
            return
        params = pd.DataFrame(fit.params)
        bse = pd.DataFrame(fit.bse)
        pvalues = pd.DataFrame(fit.pvalues)
        for equation_index, category in enumerate(ordered[1:]):
            if equation_index not in params.columns:
                continue
            for term in ("pupil_within", "pupil_between"):
                estimate = float(params.loc[term, equation_index])
                se = float(bse.loc[term, equation_index])
                results.append({
                    "model_name": model_name,
                    "model_family": family,
                    "outcome": outcome,
                    "metric": metric,
                    "pupil_term": term,
                    "contrast_category": int(category),
                    "reference_category": reference_category,
                    "estimate": estimate,
                    "se": se,
                    "ci_low": estimate - 1.96 * se,
                    "ci_high": estimate + 1.96 * se,
                    "p_value_not_for_endpoint_selection": float(pvalues.loc[term, equation_index]),
                    "n_rows": int(len(d)),
                    "participant_group_n": int(d["analysis_group_token"].astype(str).nunique()),
                    "session_n": int(d["session_id"].astype(str).nunique()),
                    "predictor_standardization": "z_score_within_fitted_sample_per_term",
                    "family_note": "",
                    "status": "estimable",
                })
    except Exception as exc:
        failures.append(_failure(
            model_name=model_name, family=family, outcome=outcome, metric=metric,
            data=d, reason=f"{type(exc).__name__}: {exc}",
        ))


def _fit_q2(
    decomposed: pd.DataFrame,
    metric: str,
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    min_rows: int,
    min_groups: int,
) -> None:
    """Q2 ordinal vigilance model: cumulative logit, cluster-robust covariance."""
    model_name = f"Q2_{metric}"
    family = Q2_FAMILY
    outcome = "probe_vigilance"
    required = ["probe_vigilance", "pupil_within", "pupil_between", "analysis_group_token", "session_id"]
    d = decomposed.dropna(subset=required).copy()
    levels = sorted(pd.to_numeric(d[outcome], errors="coerce").dropna().astype(int).unique())
    if not set(levels).issubset(set(Q2_LEVELS)) or len(levels) < 3:
        failures.append(_failure(
            model_name=model_name, family=family, outcome=outcome, metric=metric,
            data=d, reason="Q2_requires_at_least_three_observed_levels_within_1_4",
        ))
        return
    gate = _gate_common(d, min_rows=min_rows, min_groups=min_groups)
    if gate:
        failures.append(_failure(
            model_name=model_name, family=family, outcome=outcome, metric=metric, data=d, reason=gate,
        ))
        return
    try:
        x = pd.DataFrame({
            "pupil_within": _safe_z(d["pupil_within"]),
            "pupil_between": _safe_z(d["pupil_between"]),
        })
        y = pd.to_numeric(d[outcome], errors="coerce").astype(int)
        group_codes = pd.factorize(d["analysis_group_token"].astype(str))[0]
        model = OrderedModel(y, x, distr="logit")
        fit = model.fit(
            method="bfgs", maxiter=300, disp=False,
            cov_type="cluster", cov_kwds={"groups": group_codes},
        )
        reason = _fit_gate(fit)
        if reason:
            failures.append(_failure(
                model_name=model_name, family=family, outcome=outcome, metric=metric, data=d, reason=reason,
            ))
            return
        results.extend(_effect_rows(
            fit,
            model_name=model_name,
            family=family,
            outcome=outcome,
            metric=metric,
            data=d,
            family_note=Q2_FAMILY_NOTE,
        ))
    except Exception as exc:
        failures.append(_failure(
            model_name=model_name, family=family, outcome=outcome, metric=metric,
            data=d, reason=f"{type(exc).__name__}: {exc}",
        ))


def _primary_metrics_from_config(config: Config) -> tuple[str, ...]:
    raw = config.section("candidate_metrics").get("primary_metrics")
    if isinstance(raw, list) and raw:
        return tuple(str(m) for m in raw)
    return DEFAULT_PRIMARY_METRICS


def _primary_probe_window_from_config(config: Config) -> str:
    raw = config.section("candidate_metrics").get("primary_probe_window")
    return str(raw) if raw not in (None, "") else PRIMARY_PROBE_WINDOW


def run_probe_pupil_models(
    config_path: str | Path,
    *,
    subjects: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Pipeline step: build the probe endpoint table, fit Q1/Q2, write outputs.

    Parameters
    ----------
    config_path : path to configs/nir_formal_analysis.yaml (or equivalent).
    subjects : optional session-id override for smoke/subset runs.

    Returns
    -------
    Manifest dict with status/complete|blocked, model counts and output paths.
    """
    config = load_config(config_path)
    sessions = selected_sessions(config, subjects)
    metrics = _primary_metrics_from_config(config)
    window_name = _primary_probe_window_from_config(config)
    table, load_failures = build_probe_endpoint_table(
        config, sessions, metrics=metrics, window_name=window_name, track=PRIMARY_TRACK
    )
    results, failures = fit_probe_endpoint_models(table, metrics=metrics)

    root = _resolve(config, "output_root") / "probe_pupil_models"
    root.mkdir(parents=True, exist_ok=True)
    table.to_csv(root / "probe_pupil_model_table.csv", index=False, encoding="utf-8-sig")
    results.to_csv(root / "probe_pupil_model_effects.csv", index=False, encoding="utf-8-sig")
    failures.to_csv(root / "probe_pupil_model_failures.csv", index=False, encoding="utf-8-sig")

    status = "blocked" if table.empty else "complete"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "pipeline_version": PROBE_MODEL_VERSION,
        "primary_track": PRIMARY_TRACK,
        "primary_probe_window": window_name,
        "primary_metrics": list(metrics),
        "endpoint_freeze": config.section("candidate_metrics").get("final_endpoint_freeze"),
        "q1_model": {
            "outcome": "probe_response",
            "family": "MNLogit_cluster_robust",
            "reference_category": Q1_REFERENCE_CATEGORY,
            "category_gate": "all four categories 1-4 required",
        },
        "q2_model": {
            "outcome": "probe_vigilance",
            "family": Q2_FAMILY,
            "level_gate": "at least three observed levels within 1-4",
            "implementation_note": Q2_FAMILY_NOTE,
        },
        "within_between_decomposition": "participant-group mean (between) and deviation (within) per probe observation",
        "predictor_standardization": "z_score_within_fitted_sample_per_term",
        "pupil_projection": (
            "frozen primary metrics projected from 10_analysis_ready candidate sidecar "
            "into each probe window on the shared unix_ms axis; per-eye valid-frame "
            "median then binocular/left-only/right-only fusion"
        ),
        "n_sessions_requested": len(sessions),
        "n_sessions_loaded": int(table["session_id"].astype(str).nunique()) if not table.empty else 0,
        "n_probe_observations": int(len(table)),
        "n_effect_rows": int(len(results)),
        "n_failure_rows": int(len(failures)),
        "load_failures": load_failures,
        "scientific_inference_authorized_by_code_alone": False,
        "endpoint_selection_from_p_values_allowed": False,
        "outputs": {
            "model_table": str(root / "probe_pupil_model_table.csv"),
            "effects": str(root / "probe_pupil_model_effects.csv"),
            "failures": str(root / "probe_pupil_model_failures.csv"),
        },
    }
    (root / "probe_pupil_models_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
