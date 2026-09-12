from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .behavior_error_taxonomy import (
    CURRENT_PRIMARY_OMISSION_ENDPOINT_METRICS,
    OMISSION_PARTITION_RATE_METRICS,
    OMISSION_QC_RATE_METRICS,
    TAXONOMY_RATE_METRICS,
)


def _participant_column(frame: pd.DataFrame) -> str | None:
    for column in ("participant_group_id", "repeat_participant_id", "participant_key"):
        if column in frame.columns:
            return column
    return None


def _omission_role(metric: str) -> str:
    if metric in CURRENT_PRIMARY_OMISSION_ENDPOINT_METRICS:
        return "current_primary_omission_endpoint"
    if metric in OMISSION_PARTITION_RATE_METRICS:
        return "descriptive_qc_sensitivity_partition"
    return "qc_or_timing_diagnostic"


def validate_omission_candidates(
    scale_tables: Mapping[str, pd.DataFrame],
    primary_probe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe omission coverage/distribution/redundancy without full-cohort selection.

    Raw is the current primary omission endpoint. Clean/timing and finer timing
    fields remain descriptive/QC/sensitivity information. This full-cohort audit
    has no authority to include/drop supervised predictors.
    """
    frames = {k: v for k, v in scale_tables.items() if v is not None}
    frames["probe"] = primary_probe
    validation_rows: list[dict[str, object]] = []
    redundancy_rows: list[dict[str, object]] = []

    for scale, frame in frames.items():
        if frame is None or frame.empty:
            continue
        participant = _participant_column(frame)
        available = [m for m in TAXONOMY_RATE_METRICS if m in frame.columns]
        for metric in TAXONOMY_RATE_METRICS:
            endpoint_role = _omission_role(metric)
            if metric not in frame.columns:
                validation_rows.append({
                    "scale": scale,
                    "metric": metric,
                    "n_rows": int(len(frame)),
                    "n_valid": 0,
                    "coverage": 0.0,
                    "below_historical_80pct_coverage_reference": True,
                    "participant_group_n": 0,
                    "session_n": int(frame["session_id"].nunique()) if "session_id" in frame else 0,
                    "floor_fraction": np.nan,
                    "ceiling_fraction": np.nan,
                    "between_participant_variance": np.nan,
                    "within_participant_variance": np.nan,
                    "within_participant_observation_n": 0,
                    "candidate_status": "not_computable",
                    "candidate_reasons": "missing_column",
                    "endpoint_role": endpoint_role,
                    "endpoint_status": "not_estimable_missing_column",
                    "selection_authority": "descriptive_only",
                    "automatic_drop_allowed": False,
                })
                continue
            value = pd.to_numeric(frame[metric], errors="coerce")
            finite = value[np.isfinite(value)]
            participant_group_n = 0
            between = np.nan
            within = pd.Series(dtype=float)
            if participant is not None:
                work = frame[[participant, metric]].copy()
                work[metric] = pd.to_numeric(work[metric], errors="coerce")
                work = work.dropna(subset=[participant, metric])
                participant_group_n = int(work[participant].nunique())
                if not work.empty:
                    means = work.groupby(participant)[metric].mean()
                    between = float(means.var(ddof=1)) if len(means) >= 2 else np.nan
                    work["_participant_mean"] = work.groupby(participant)[metric].transform("mean")
                    work["_within"] = work[metric] - work["_participant_mean"]
                    group_sizes = work.groupby(participant)[metric].transform("size")
                    within = work.loc[group_sizes.ge(2), "_within"]
            floor = float((finite <= 0.02).mean()) if len(finite) else np.nan
            ceiling = float((finite >= 0.98).mean()) if len(finite) else np.nan
            review_flags: list[str] = []
            coverage = float(len(finite) / len(frame)) if len(frame) else 0.0
            if coverage < 0.80:
                review_flags.append("below_historical_80pct_coverage_reference")
            if len(finite) and floor >= 0.80:
                review_flags.append("strong_floor_effect")
            if len(finite) and ceiling >= 0.80:
                review_flags.append("strong_ceiling_effect")
            if finite.nunique() < 3:
                review_flags.append("low_unique_values")

            validation_rows.append({
                "scale": scale,
                "metric": metric,
                "n_rows": int(len(frame)),
                "n_valid": int(len(finite)),
                "coverage": coverage,
                "below_historical_80pct_coverage_reference": bool(coverage < 0.80),
                "participant_group_n": participant_group_n,
                "session_n": int(frame.loc[value.notna(), "session_id"].nunique()) if "session_id" in frame else 0,
                "floor_fraction": floor,
                "ceiling_fraction": ceiling,
                "between_participant_variance": between,
                "within_participant_variance": float(within.var(ddof=1)) if len(within) >= 2 else np.nan,
                "within_participant_observation_n": int(len(within)),
                "candidate_status": "not_computable" if len(finite) == 0 else "descriptive_audit_only",
                "candidate_reasons": ";".join(review_flags),
                "endpoint_role": endpoint_role,
                "endpoint_status": (
                    "prespecified_not_pvalue_selected"
                    if endpoint_role == "current_primary_omission_endpoint"
                    else "not_a_primary_endpoint"
                ),
                "selection_authority": "descriptive_only",
                "automatic_drop_allowed": False,
                "selection_contract": (
                    "full-cohort coverage + floor/ceiling + within/between + redundancy are descriptive only; "
                    "never outcome p-value, coverage-threshold or redundancy auto-selection"
                ),
                "interpretation_guard": (
                    "clean means no detected prestimulus/carry-over ambiguity, not proven attentional lapse"
                    if metric == "clean_go_omission_rate"
                    else "timing-ambiguous is a complementary component of raw omission, not a separate opportunity denominator"
                    if metric == "timing_ambiguous_go_omission_rate"
                    else ""
                ),
            })

        if len(available) >= 2:
            work = frame[[c for c in [participant, "session_id", *available] if c is not None and c in frame.columns]].copy()
            ranked = work[available].apply(pd.to_numeric, errors="coerce").rank(axis=0, method="average")
            corr = ranked.corr(method="pearson", min_periods=3)
            for i, a in enumerate(available):
                for b in available[i + 1:]:
                    r = corr.loc[a, b] if a in corr.index and b in corr.columns else np.nan
                    structural_pair = (
                        a in OMISSION_PARTITION_RATE_METRICS
                        and b in OMISSION_PARTITION_RATE_METRICS
                    )
                    redundancy_rows.append({
                        "scale": scale,
                        "metric_a": a,
                        "metric_b": b,
                        "spearman_r": float(r) if np.isfinite(r) else np.nan,
                        "abs_r": abs(float(r)) if np.isfinite(r) else np.nan,
                        "high_redundancy_flag": bool(np.isfinite(r) and abs(float(r)) >= 0.90),
                        "threshold": 0.90,
                        "threshold_role": "historical_descriptive_reference_only",
                        "structural_same_denominator_pair": structural_pair,
                        "selection_authority": "descriptive_only",
                        "automatic_drop_allowed": False,
                        "redundancy_interpretation": (
                            "omission partition metrics share the same denominator and are structurally related; correlation is descriptive only"
                            if structural_pair
                            else "descriptive redundancy audit"
                        ),
                    })

    return pd.DataFrame(validation_rows), pd.DataFrame(redundancy_rows)
