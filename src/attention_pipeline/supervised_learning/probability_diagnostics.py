"""Descriptive probability diagnostics for frozen out-of-fold predictions.

Why a separate module instead of extending ``evaluation.py``
-----------------------------------------------------------
The frozen evaluation layer defines the *selection* and *primary* metric
(``participant_macro_log_loss``) and the D10 participant-cluster bootstrap. Those rules
must not be touched. AUROC, the Brier score and calibration are explicitly approved as
**supplementary / diagnostic** quantities (report section 4.6 and ``分析设计/1.16.9``),
and they are pure functions of already-archived out-of-fold predictions. Keeping them in
an additive, read-only post-processing module therefore

* guarantees the frozen selection metric and the C/candidate search are untouched;
* never retrains a prediction model; and
* lets the report's required columns be filled without re-running any LOSO fold.

Sign and definition conventions
-------------------------------
* Positives are ``q1_binary == 1`` (Q1 = 1, "focused on the current classification task").
* ``participant_macro_*`` aggregate the frozen way: probe-equal inside a participant, then
  participant-equal across participants (D2/D3).
* AUROC is only defined for a participant whose probes contain **both** classes. Single
  class participants are excluded from the value and reported separately as
  not-estimable, never imputed.
* ``calibration_intercept`` and ``calibration_slope`` come from the standard calibration
  model ``logit(p(y=1)) = intercept + slope * logit(p_hat)``. Slope 1 / intercept 0 means
  perfect calibration. Their interval comes from a participant-cluster bootstrap that
  **refits this small diagnostic model** on each resample; the prediction model is never
  refitted, which is recorded explicitly in every output row.
* The slope is only *interpretable* when the score discriminates. ``calibration_slope_reporting``
  carries that judgement (see ``CALIBRATION_SLOPE_REPORTABLE``) and the raw estimate is kept
  in every row regardless, so the guard hides nothing.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .evaluation import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE_LEVEL,
    fixed_oof_participant_bootstrap,
)

DIAGNOSTIC_SCHEMA_VERSION = "1.16.10-probability-diagnostics-v1"
PARTICIPANT_COLUMN = "participant_group_id"
MODEL_COLUMN = "model_id"
RUN_COLUMN = "run_id"
LABEL_COLUMN = "q1_binary"
PROBABILITY_COLUMN = "p_q1_equals_1"
FAILED_COLUMN = "model_failed"
DEFAULT_CALIBRATION_BINS = 10
_LOGIT_EPSILON = 1e-6

DIAGNOSTIC_METRICS = (
    "participant_macro_auroc",
    "participant_macro_brier",
    "participant_macro_calibration_in_the_large",
    "calibration_intercept",
    "calibration_slope",
)

#: A calibration slope is only interpretable when the score actually discriminates. With an
#: (effectively) unpenalised logistic fit on ``logit(p_hat)``, a near-chance score produces a
#: slope that is numerically enormous and sign-unstable: the observed sensor-only models
#: return slopes between -0.2 and -43.9 with bootstrap intervals up to 72 units wide. Reading
#: those numbers as "the probabilities are inverted" would be a substantive claim the data
#: cannot support. The guard is therefore stated as a rule and applied uniformly: the slope
#: is reportable only when the participant-macro AUROC interval excludes 0.5. Raw estimates
#: are always retained so nothing is hidden, and the flag is descriptive only.
CALIBRATION_SLOPE_REPORTABLE = "reportable"
CALIBRATION_SLOPE_NOT_REPORTABLE_LOW_DISCRIMINATION = "not_reportable_low_discrimination"
CALIBRATION_SLOPE_NOT_ESTIMABLE = "not_estimable"


class ProbabilityDiagnosticsError(ValueError):
    """Raised when predictions cannot produce an unambiguous diagnostic."""


def _valid_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Keep only successful, finite, correctly coded out-of-fold rows."""
    required = [
        MODEL_COLUMN,
        PARTICIPANT_COLUMN,
        LABEL_COLUMN,
        PROBABILITY_COLUMN,
    ]
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise ProbabilityDiagnosticsError(f"predictions missing required columns: {missing}")

    frame = predictions.copy()
    if FAILED_COLUMN in frame.columns:
        frame = frame[~frame[FAILED_COLUMN].astype(bool)]
    frame[LABEL_COLUMN] = pd.to_numeric(frame[LABEL_COLUMN], errors="coerce")
    frame[PROBABILITY_COLUMN] = pd.to_numeric(frame[PROBABILITY_COLUMN], errors="coerce")
    frame = frame.dropna(subset=[LABEL_COLUMN, PROBABILITY_COLUMN, PARTICIPANT_COLUMN])
    frame = frame[frame[LABEL_COLUMN].isin([0, 1])]
    frame = frame[
        np.isfinite(frame[PROBABILITY_COLUMN]) & frame[PROBABILITY_COLUMN].between(0.0, 1.0)
    ]
    if frame.empty:
        raise ProbabilityDiagnosticsError("no usable out-of-fold predictions remain")
    return frame


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities.astype(float), _LOGIT_EPSILON, 1.0 - _LOGIT_EPSILON)
    return np.log(clipped / (1.0 - clipped))


def participant_auroc_values(frame: pd.DataFrame) -> tuple[np.ndarray, int, int]:
    """Per-participant AUROC plus the not-estimable (single-class) count.

    Returns
    -------
    (values, n_estimable, n_single_class)
    """
    values: list[float] = []
    single_class = 0
    for _, group in frame.groupby(PARTICIPANT_COLUMN, sort=True):
        labels = group[LABEL_COLUMN].to_numpy(dtype=int)
        if len(np.unique(labels)) < 2:
            single_class += 1
            continue
        values.append(float(roc_auc_score(labels, group[PROBABILITY_COLUMN].to_numpy(float))))
    return np.asarray(values, dtype=float), len(values), single_class


def participant_brier_values(frame: pd.DataFrame) -> np.ndarray:
    """Per-participant mean Brier score (probe-equal inside a participant)."""
    labels = frame[LABEL_COLUMN].to_numpy(float)
    probabilities = frame[PROBABILITY_COLUMN].to_numpy(float)
    squared = (probabilities - labels) ** 2
    working = frame[[PARTICIPANT_COLUMN]].copy()
    working["__squared"] = squared
    return (
        working.groupby(PARTICIPANT_COLUMN, sort=True)["__squared"].mean().to_numpy(dtype=float)
    )


def participant_calibration_in_the_large_values(frame: pd.DataFrame) -> np.ndarray:
    """Per-participant (mean predicted probability - observed positive rate)."""
    working = frame[[PARTICIPANT_COLUMN]].copy()
    working["__p"] = frame[PROBABILITY_COLUMN].to_numpy(float)
    working["__y"] = frame[LABEL_COLUMN].to_numpy(float)
    grouped = working.groupby(PARTICIPANT_COLUMN, sort=True)
    return (grouped["__p"].mean() - grouped["__y"].mean()).to_numpy(dtype=float)


def calibration_model(frame: pd.DataFrame) -> tuple[float, float, str]:
    """Fit ``logit(y) = intercept + slope * logit(p)`` and return (intercept, slope, status)."""
    labels = frame[LABEL_COLUMN].to_numpy(dtype=int)
    if len(np.unique(labels)) < 2:
        return math.nan, math.nan, "not_estimable_single_class"
    features = _logit(frame[PROBABILITY_COLUMN].to_numpy(float)).reshape(-1, 1)
    if not np.isfinite(features).all():
        return math.nan, math.nan, "not_estimable_nonfinite_logit"
    try:
        # Effectively unpenalised calibration model. A very large C is used instead of
        # ``penalty=None`` because the latter is deprecated in newer scikit-learn while
        # ``C=np.inf`` is not accepted by the older releases this repository supports.
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        model.fit(features, labels)
    except Exception:  # pragma: no cover - defensive: convergence pathologies
        return math.nan, math.nan, "not_estimable_calibration_fit_failed"
    return float(model.intercept_[0]), float(model.coef_[0][0]), "estimable"


def _cluster_bootstrap_calibration(
    frame: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, float | int]:
    """Participant-cluster bootstrap for the calibration model.

    The prediction model is never refitted; only the small calibration diagnostic is
    refitted on each resample, and that is recorded in the returned metadata.
    """
    participants = np.sort(frame[PARTICIPANT_COLUMN].astype(str).unique())
    if len(participants) < 2:
        return {
            "intercept_ci_lower": math.nan,
            "intercept_ci_upper": math.nan,
            "slope_ci_lower": math.nan,
            "slope_ci_upper": math.nan,
            "n_valid_replicates": 0,
        }
    blocks = {name: group for name, group in frame.groupby(frame[PARTICIPANT_COLUMN].astype(str))}
    rng = np.random.default_rng(int(seed))
    intercepts: list[float] = []
    slopes: list[float] = []
    for _ in range(int(replicates)):
        draw = rng.integers(0, len(participants), size=len(participants))
        resampled = pd.concat([blocks[participants[i]] for i in draw], ignore_index=True)
        intercept, slope, status = calibration_model(resampled)
        if status == "estimable" and math.isfinite(intercept) and math.isfinite(slope):
            intercepts.append(intercept)
            slopes.append(slope)
    if not intercepts:
        return {
            "intercept_ci_lower": math.nan,
            "intercept_ci_upper": math.nan,
            "slope_ci_lower": math.nan,
            "slope_ci_upper": math.nan,
            "n_valid_replicates": 0,
        }
    alpha = (1.0 - float(confidence_level)) / 2.0
    intercept_low, intercept_high = np.quantile(intercepts, [alpha, 1.0 - alpha])
    slope_low, slope_high = np.quantile(slopes, [alpha, 1.0 - alpha])
    return {
        "intercept_ci_lower": float(intercept_low),
        "intercept_ci_upper": float(intercept_high),
        "slope_ci_lower": float(slope_low),
        "slope_ci_upper": float(slope_high),
        "n_valid_replicates": len(intercepts),
    }


def _bootstrap_row(values: np.ndarray, *, replicates: int, seed: int, confidence_level: float):
    if len(values) == 0 or not np.isfinite(values).all():
        return {"point_estimate": math.nan, "ci_lower": math.nan, "ci_upper": math.nan,
                "n_participants": 0, "n_valid_replicates": 0}
    result = fixed_oof_participant_bootstrap(
        values, replicates=replicates, seed=seed, confidence_level=confidence_level
    )
    return result


def _calibration_reporting_status(
    *,
    calibration_status: str,
    auroc_ci_lower: float,
    auroc_ci_upper: float,
) -> tuple[str, str]:
    """Decide whether the calibration slope may be interpreted, and say why.

    Returns ``(status, explanation)``. The rule is deliberately independent of the slope
    estimate itself, so it cannot be tuned to produce a preferred answer.
    """
    if calibration_status != "estimable":
        return (
            CALIBRATION_SLOPE_NOT_ESTIMABLE,
            f"the calibration model itself was not estimable ({calibration_status})",
        )
    if not (math.isfinite(auroc_ci_lower) and math.isfinite(auroc_ci_upper)):
        return (
            CALIBRATION_SLOPE_NOT_REPORTABLE_LOW_DISCRIMINATION,
            "the participant-macro AUROC interval could not be estimated, so discrimination "
            "is unestablished",
        )
    if auroc_ci_lower > 0.5:
        return (
            CALIBRATION_SLOPE_REPORTABLE,
            "the participant-macro AUROC 95% interval excludes 0.5, so discrimination is "
            "established and the slope may be read against perfect calibration (slope 1, "
            "intercept 0); the slope interval must still be consulted, because weak "
            "discrimination makes the estimate imprecise",
        )
    return (
        CALIBRATION_SLOPE_NOT_REPORTABLE_LOW_DISCRIMINATION,
        "the participant-macro AUROC 95% interval includes 0.5; the unpenalised calibration "
        "slope is then numerically unstable and must not be read as over-confidence or as "
        "inverted probabilities",
    )


def build_probability_diagnostics(
    predictions: pd.DataFrame,
    *,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (diagnostics, calibration_bins) for every model in ``predictions``.

    ``diagnostics`` has one row per (model, run, analysis_set_id, membership_type).
    ``calibration_bins`` has one row per model per probability bin.
    """
    if int(n_bins) < 2:
        raise ProbabilityDiagnosticsError("n_bins must be at least 2")
    frame = _valid_predictions(predictions)

    diagnostic_rows: list[dict[str, object]] = []
    bin_rows: list[dict[str, object]] = []

    group_columns = [MODEL_COLUMN]
    # ``run_id`` is part of the identity on purpose. Two run directories can share the same
    # ``(analysis_set_id, model_id, membership_type)`` triple - the exploratory smoke run
    # ``smoke_go_omission__aware`` writes exactly the same ``analysis_set_id`` and
    # ``model_id`` as the formal ``AS.standalone__behavior.go_omission.raw.v1`` run - so
    # grouping without it would silently merge two different probe sets into one row and
    # every downstream count would be wrong while still looking plausible.
    for optional in (RUN_COLUMN, "analysis_set_id", "membership_type"):
        if optional in frame.columns:
            group_columns.append(optional)

    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identity = dict(zip(group_columns, keys))

        auroc_values, n_estimable, n_single_class = participant_auroc_values(group)
        brier_values = participant_brier_values(group)
        citl_values = participant_calibration_in_the_large_values(group)

        auroc_boot = _bootstrap_row(
            auroc_values, replicates=replicates, seed=seed, confidence_level=confidence_level
        )
        brier_boot = _bootstrap_row(
            brier_values, replicates=replicates, seed=seed, confidence_level=confidence_level
        )
        citl_boot = _bootstrap_row(
            citl_values, replicates=replicates, seed=seed, confidence_level=confidence_level
        )

        intercept, slope, calibration_status = calibration_model(group)
        calibration_boot = _cluster_bootstrap_calibration(
            group, replicates=replicates, seed=seed, confidence_level=confidence_level
        )
        slope_reporting, slope_reporting_note = _calibration_reporting_status(
            calibration_status=calibration_status,
            auroc_ci_lower=float(auroc_boot["ci_lower"]),
            auroc_ci_upper=float(auroc_boot["ci_upper"]),
        )

        labels = group[LABEL_COLUMN].to_numpy(float)
        diagnostic_rows.append(
            {
                **identity,
                "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
                "n_participants": int(group[PARTICIPANT_COLUMN].nunique()),
                "n_probes": int(len(group)),
                "positive_rate": float(labels.mean()),
                "participant_macro_auroc": auroc_boot["point_estimate"],
                "participant_macro_auroc_ci_lower": auroc_boot["ci_lower"],
                "participant_macro_auroc_ci_upper": auroc_boot["ci_upper"],
                "n_participants_auroc_estimable": int(n_estimable),
                "n_participants_auroc_single_class": int(n_single_class),
                "participant_macro_brier": brier_boot["point_estimate"],
                "participant_macro_brier_ci_lower": brier_boot["ci_lower"],
                "participant_macro_brier_ci_upper": brier_boot["ci_upper"],
                "participant_macro_calibration_in_the_large": citl_boot["point_estimate"],
                "participant_macro_calibration_in_the_large_ci_lower": citl_boot["ci_lower"],
                "participant_macro_calibration_in_the_large_ci_upper": citl_boot["ci_upper"],
                "calibration_intercept": intercept,
                "calibration_intercept_ci_lower": calibration_boot["intercept_ci_lower"],
                "calibration_intercept_ci_upper": calibration_boot["intercept_ci_upper"],
                "calibration_slope": slope,
                "calibration_slope_ci_lower": calibration_boot["slope_ci_lower"],
                "calibration_slope_ci_upper": calibration_boot["slope_ci_upper"],
                "calibration_status": calibration_status,
                "calibration_slope_reporting": slope_reporting,
                "calibration_slope_reporting_note": slope_reporting_note,
                "calibration_n_valid_replicates": int(calibration_boot["n_valid_replicates"]),
                "bootstrap_method": "fixed_oof_participant_cluster_percentile",
                "bootstrap_replicates": int(replicates),
                "bootstrap_seed": int(seed),
                "bootstrap_confidence_level": float(confidence_level),
                "prediction_model_retrained_in_bootstrap": False,
                "calibration_model_refitted_in_bootstrap": True,
                "used_for_model_or_c_selection": False,
                "used_for_feature_selection": False,
                "role": "supplementary_descriptive_diagnostic",
            }
        )

        edges = np.quantile(
            group[PROBABILITY_COLUMN].to_numpy(float), np.linspace(0.0, 1.0, int(n_bins) + 1)
        )
        edges = np.unique(edges)
        if len(edges) < 2:
            continue
        binned = pd.cut(
            group[PROBABILITY_COLUMN], bins=edges, include_lowest=True, duplicates="drop"
        )
        for interval, block in group.groupby(binned, observed=True):
            bin_rows.append(
                {
                    **identity,
                    "bin_index": int(list(binned.cat.categories).index(interval)),
                    "bin_lower": float(interval.left),
                    "bin_upper": float(interval.right),
                    "n_probes": int(len(block)),
                    "mean_predicted_probability": float(
                        block[PROBABILITY_COLUMN].astype(float).mean()
                    ),
                    "observed_positive_rate": float(block[LABEL_COLUMN].astype(float).mean()),
                }
            )

    diagnostics = pd.DataFrame(diagnostic_rows)
    calibration_bins = pd.DataFrame(bin_rows)
    return diagnostics, calibration_bins


def diagnostic_summary(diagnostics: pd.DataFrame) -> dict[str, object]:
    """Compact machine-readable summary for the run manifest / handoff.

    ``n_diagnostic_rows`` is deliberately not called "n_models": one model can appear in
    several analysis sets (the shared ``behavior_reference`` baseline is retrained inside
    each comparison), so rows and distinct models differ.
    """
    n_runs = int(diagnostics[RUN_COLUMN].nunique()) if RUN_COLUMN in diagnostics.columns else None
    if "calibration_slope_reporting" in diagnostics.columns:
        n_reportable = int(
            (diagnostics["calibration_slope_reporting"] == CALIBRATION_SLOPE_REPORTABLE).sum()
        )
        n_low_discrimination = int(
            (
                diagnostics["calibration_slope_reporting"]
                == CALIBRATION_SLOPE_NOT_REPORTABLE_LOW_DISCRIMINATION
            ).sum()
        )
    else:  # pragma: no cover - defensive: only if the column is dropped upstream
        n_reportable = None
        n_low_discrimination = None
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "n_diagnostic_rows": int(len(diagnostics)),
        "n_unique_models": int(diagnostics[MODEL_COLUMN].nunique()),
        "n_runs": n_runs,
        "n_calibration_slope_reportable": n_reportable,
        "n_calibration_slope_not_reportable_low_discrimination": n_low_discrimination,
        "metrics": list(DIAGNOSTIC_METRICS),
        "used_for_model_or_c_selection": False,
        "used_for_feature_selection": False,
        "prediction_model_retrained_in_bootstrap": False,
        "calibration_model_refitted_in_bootstrap": True,
        "aggregation": "participant_equal_within_participant_probe_equal",
        "single_class_rule": (
            "a participant whose probes contain one class only is excluded from the AUROC "
            "value and counted in n_participants_auroc_single_class; never imputed"
        ),
        "calibration_slope_rule": (
            "the calibration slope is reported as interpretable only when the "
            "participant-macro AUROC 95% interval excludes 0.5; raw estimates are always kept"
        ),
    }
