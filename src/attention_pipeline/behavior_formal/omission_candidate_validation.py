from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .behavior_error_taxonomy import (
    FORMAL_OMISSION_ENDPOINT_METRICS,
    OMISSION_QC_RATE_METRICS,
    TAXONOMY_RATE_METRICS,
)


def _participant_column(frame: pd.DataFrame) -> str | None:
    for column in ("participant_group_id", "repeat_participant_id", "participant_key"):
        if column in frame.columns:
            return column
    return None


def validate_omission_candidates(
    scale_tables: Mapping[str, pd.DataFrame],
    primary_probe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit formal omission endpoints and timing/QC metrics without p-value selection.

    ``raw_go_omission_rate``, ``clean_go_omission_rate`` and
    ``timing_ambiguous_go_omission_rate`` are prespecified formal endpoints.
    They share the Go denominator and the latter two partition the raw rate, so
    redundancy between them is structural and must never be misread as three
    independent phenomena.  Finer motor-timing subtypes remain QC/descriptive.
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
            endpoint_role = (
                "prespecified_formal_endpoint"
                if metric in FORMAL_OMISSION_ENDPOINT_METRICS
                else "qc_or_timing_diagnostic"
            )
            if metric not in frame.columns:
                validation_rows.append({
                    "scale": scale,
                    "metric": metric,
                    "n_rows": int(len(frame)),
                    "n_valid": 0,
                    "coverage": 0.0,
                    "participant_group_n": 0,
                    "session_n": int(frame["session_id"].nunique()) if "session_id" in frame else 0,
                    "floor_fraction": np.nan,
                    "ceiling_fraction": np.nan,
                    "between_participant_variance": np.nan,
                    "within_participant_variance": np.nan,
                    "within_participant_observation_n": 0,
                    "candidate_status": "missing_column",
                    "endpoint_role": endpoint_role,
                    "endpoint_status": "not_estimable_missing_column",
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
            reasons: list[str] = []
            coverage = float(len(finite) / len(frame)) if len(frame) else 0.0
            if coverage < 0.80:
                reasons.append("low_coverage")
            if len(finite) and floor >= 0.80:
                reasons.append("strong_floor_effect")
            if len(finite) and ceiling >= 0.80:
                reasons.append("strong_ceiling_effect")
            if finite.nunique() < 3:
                reasons.append("low_unique_values")

            if endpoint_role == "prespecified_formal_endpoint":
                status = (
                    "formal_endpoint_needs_real_data_stability_review"
                    if reasons
                    else "formal_endpoint_ready_for_real_data_review"
                )
                endpoint_status = "prespecified_not_pvalue_selected"
            else:
                status = "qc_candidate_needs_review" if reasons else "qc_candidate_eligible_for_scientific_review"
                endpoint_status = "not_a_primary_endpoint"

            validation_rows.append({
                "scale": scale,
                "metric": metric,
                "n_rows": int(len(frame)),
                "n_valid": int(len(finite)),
                "coverage": coverage,
                "participant_group_n": participant_group_n,
                "session_n": int(frame.loc[value.notna(), "session_id"].nunique()) if "session_id" in frame else 0,
                "floor_fraction": floor,
                "ceiling_fraction": ceiling,
                "between_participant_variance": between,
                "within_participant_variance": float(within.var(ddof=1)) if len(within) >= 2 else np.nan,
                "within_participant_observation_n": int(len(within)),
                "candidate_status": status,
                "candidate_reasons": ";".join(reasons),
                "endpoint_role": endpoint_role,
                "endpoint_status": endpoint_status,
                "selection_contract": "prespecified role + coverage + floor/ceiling + within/between + redundancy; never outcome p-value screening",
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
                        a in FORMAL_OMISSION_ENDPOINT_METRICS
                        and b in FORMAL_OMISSION_ENDPOINT_METRICS
                    )
                    redundancy_rows.append({
                        "scale": scale,
                        "metric_a": a,
                        "metric_b": b,
                        "spearman_r": float(r) if np.isfinite(r) else np.nan,
                        "abs_r": abs(float(r)) if np.isfinite(r) else np.nan,
                        "high_redundancy_flag": bool(np.isfinite(r) and abs(float(r)) >= 0.90),
                        "threshold": 0.90,
                        "structural_same_denominator_pair": structural_pair,
                        "automatic_drop_allowed": False,
                        "redundancy_interpretation": (
                            "formal omission endpoints are algebraically/structurally related; correlation is descriptive only"
                            if structural_pair
                            else "descriptive redundancy audit"
                        ),
                    })

    return pd.DataFrame(validation_rows), pd.DataFrame(redundancy_rows)
