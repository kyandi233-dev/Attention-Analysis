from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .behavior_error_taxonomy import (
    CURRENT_PRIMARY_OMISSION_ENDPOINT_METRICS,
    OMISSION_PARTITION_RATE_METRICS,
)


def build_omission_b1_b2_pairs(block_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build historical B1-B2 rows for the full omission partition.

    Raw remains the current primary omission endpoint. Clean and timing-
    ambiguous are retained as descriptive/QC/sensitivity partition rows so the
    historical explanatory output remains reproducible without assigning all
    three the same current scientific role.
    """
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    required = {"repeat_participant_id", "session_id", "block_id"}
    if not required.issubset(block_metrics.columns):
        raise ValueError(f"block metrics missing {sorted(required - set(block_metrics.columns))}")

    for session, current in block_metrics.groupby("session_id", sort=True):
        participants = current["repeat_participant_id"].dropna().astype(str).unique()
        if len(participants) != 1:
            failures.append({
                "analysis": "B1_B2_formal_omission",
                "session_id": session,
                "status": "not_estimable",
                "reason": "session maps to multiple participant groups",
            })
            continue
        b1 = current[current["block_id"].astype(str).eq("B1")]
        b2 = current[current["block_id"].astype(str).eq("B2")]
        if len(b1) != 1 or len(b2) != 1:
            failures.append({
                "analysis": "B1_B2_formal_omission",
                "session_id": session,
                "status": "not_estimable",
                "reason": f"expected one B1/B2 row; got {len(b1)}/{len(b2)}",
            })
            continue
        for metric in OMISSION_PARTITION_RATE_METRICS:
            if metric not in current.columns:
                failures.append({
                    "analysis": "B1_B2_formal_omission",
                    "session_id": session,
                    "metric": metric,
                    "status": "not_estimable",
                    "reason": "omission partition metric missing from block table",
                })
                continue
            a = pd.to_numeric(pd.Series([b1.iloc[0][metric]]), errors="coerce").iloc[0]
            b = pd.to_numeric(pd.Series([b2.iloc[0][metric]]), errors="coerce").iloc[0]
            rows.append({
                "repeat_participant_id": participants[0],
                "session_id": str(session),
                "metric": metric,
                "b1_value": a,
                "b2_value": b,
                "b2_minus_b1": b - a if np.isfinite(a) and np.isfinite(b) else math.nan,
                "pairing_unit": "within_session",
                "endpoint_role": (
                    "current_primary_omission_endpoint"
                    if metric in CURRENT_PRIMARY_OMISSION_ENDPOINT_METRICS
                    else "descriptive_qc_sensitivity_partition"
                ),
                "denominator_contract": "B1 and B2 rates each use their own valid Go opportunities",
            })
    return pd.DataFrame(rows), pd.DataFrame(failures)


def formal_omission_partition_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Audit raw = clean + timing-ambiguous on any aggregated table."""
    required = set(OMISSION_PARTITION_RATE_METRICS)
    if not required.issubset(frame.columns):
        return pd.DataFrame([{
            "status": "not_estimable_missing_columns",
            "missing": ";".join(sorted(required - set(frame.columns))),
        }])
    raw = pd.to_numeric(frame["raw_go_omission_rate"], errors="coerce")
    clean = pd.to_numeric(frame["clean_go_omission_rate"], errors="coerce")
    ambiguous = pd.to_numeric(frame["timing_ambiguous_go_omission_rate"], errors="coerce")
    finite = raw.notna() & clean.notna() & ambiguous.notna()
    delta = raw - clean - ambiguous
    return pd.DataFrame([{
        "status": "complete" if bool(finite.any()) else "not_estimable_no_finite_rows",
        "n_rows": int(len(frame)),
        "n_finite": int(finite.sum()),
        "max_absolute_partition_error": float(delta.loc[finite].abs().max()) if finite.any() else math.nan,
        "partition_exact_within_1e_12": bool((delta.loc[finite].abs() <= 1e-12).all()) if finite.any() else False,
        "contract": "raw_go_omission_rate = clean_go_omission_rate + timing_ambiguous_go_omission_rate",
    }])
