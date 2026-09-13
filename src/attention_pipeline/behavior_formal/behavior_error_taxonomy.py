from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

# Current method authority (1.15.5/1.15.6): raw program Go omission is the
# primary omission endpoint for the first-round supervised line. Clean and
# timing-ambiguous remain a complementary partition for description/QC and
# sensitivity; they are not parallel current primary endpoints.
CURRENT_PRIMARY_OMISSION_ENDPOINT_METRICS = (
    "raw_go_omission_rate",
)

# All three rates are retained to audit the exact partition
# raw = clean + timing_ambiguous on the same Go-opportunity denominator.
OMISSION_PARTITION_RATE_METRICS = (
    "raw_go_omission_rate",
    "clean_go_omission_rate",
    "timing_ambiguous_go_omission_rate",
)

# Compatibility name used by existing current-code callers. Its semantics now
# follow the current method authority: only raw is a current primary endpoint.
FORMAL_OMISSION_ENDPOINT_METRICS = CURRENT_PRIMARY_OMISSION_ENDPOINT_METRICS

# Finer descriptive/QC decomposition retained for auditability. These are not
# automatically promoted to psychological endpoints or supervised predictors.
OMISSION_QC_RATE_METRICS = (
    "omission_prestimulus_only_ambiguity_rate",
    "omission_carryover_only_ambiguity_rate",
    "omission_prestimulus_and_carryover_ambiguity_rate",
    "late_go_response_candidate_rate",
    "anticipatory_go_response_candidate_rate",
)

OMISSION_DESCRIPTIVE_QC_RATE_METRICS = tuple(dict.fromkeys((
    "clean_go_omission_rate",
    "timing_ambiguous_go_omission_rate",
    *OMISSION_QC_RATE_METRICS,
)))

# Public inventory used by figure/validation code. It preserves the full
# partition and finer QC fields without assigning all of them endpoint status.
TAXONOMY_RATE_METRICS = OMISSION_PARTITION_RATE_METRICS + OMISSION_QC_RATE_METRICS


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].fillna(False).astype(bool)


def _with_block_id(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "block_id" not in out.columns:
        if "block_num" not in out.columns:
            raise ValueError("taxonomy requires block_id or block_num")
        block = pd.to_numeric(out["block_num"], errors="coerce").astype("Int64")
        out["block_id"] = "B" + block.astype(str)
    return out


def add_omission_taxonomy(trials: pd.DataFrame) -> pd.DataFrame:
    """Add non-destructive Go-omission partition plus motor-timing QC flags.

    Program scoring is authoritative and never overwritten. For every Go trial:

    * ``raw_go_omission_flag`` is the task-program omission definition
      (no in-window response captured by the program).
    * ``clean_go_omission_flag`` is a raw omission without either of the two
      prespecified motor-timing ambiguity flags currently available in the
      recorded data (prestimulus press or carry-over candidate).
    * ``timing_ambiguous_go_omission_flag`` is a raw omission with at least one
      such ambiguity flag.

    Thus ``raw == clean + timing_ambiguous`` by construction. ``clean`` does
    not mean a proven attentional lapse; it only means that the recorded
    motor-timing ambiguity screen did not fire. Under the current 1.15.5/1.15.6
    contract, raw is the primary omission endpoint while clean/timing and finer
    subtypes remain descriptive/QC/sensitivity information.
    """
    out = trials.copy()
    go = _numeric(out, "is_no_go").eq(0)
    raw_omission = go & _numeric(out, "omission").eq(1)
    prestimulus = _bool(out, "prestimulus_press_flag")
    carryover = _bool(out, "carryover_candidate_flag")
    anticipatory = _bool(out, "anticipatory_candidate_flag")

    # A response nominally later than the programmed 1.15-s trial is retained
    # only as a timing diagnostic. The formal task polls responses within the
    # stimulus+mask trial, so such rows require data/timing audit before any
    # psychological interpretation.
    late = go & _numeric(out, "response").eq(1) & _numeric(out, "rt").gt(1150)

    timing_ambiguous = raw_omission & (prestimulus | carryover)
    clean = raw_omission & ~timing_ambiguous
    both = timing_ambiguous & prestimulus & carryover
    pre_only = timing_ambiguous & prestimulus & ~carryover
    carry_only = timing_ambiguous & ~prestimulus & carryover

    out["raw_go_omission_flag"] = raw_omission
    out["clean_go_omission_flag"] = clean
    out["timing_ambiguous_go_omission_flag"] = timing_ambiguous

    # Compatibility aliases retained so historical consumers do not silently
    # change numerical meaning.
    out["omission_no_detected_motor_timing_ambiguity_flag"] = clean
    out["omission_motor_timing_ambiguous_flag"] = timing_ambiguous

    out["omission_prestimulus_ambiguity_flag"] = raw_omission & prestimulus
    out["omission_carryover_ambiguity_flag"] = raw_omission & carryover
    out["omission_prestimulus_only_ambiguity_flag"] = pre_only
    out["omission_carryover_only_ambiguity_flag"] = carry_only
    out["omission_prestimulus_and_carryover_ambiguity_flag"] = both
    out["late_go_response_candidate_flag"] = late
    out["anticipatory_go_response_candidate_flag"] = go & anticipatory

    subtype = pd.Series(pd.NA, index=out.index, dtype="string")
    subtype.loc[both] = "timing_ambiguous_prestimulus_and_carryover"
    subtype.loc[pre_only] = "timing_ambiguous_prestimulus_only"
    subtype.loc[carry_only] = "timing_ambiguous_carryover_only"
    subtype.loc[clean] = "clean_go_omission"
    out["omission_subtype"] = subtype
    out["omission_taxonomy_contract"] = (
        "raw_go_omission is the current primary task-program endpoint; "
        "clean_go_omission and timing_ambiguous_go_omission are complementary "
        "descriptive/QC/sensitivity partition components using the same Go denominator; "
        "finer motor-timing subtypes remain QC only"
    )
    return out


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else math.nan


def summarize_error_taxonomy(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize the current primary omission plus its QC partition."""
    go = _numeric(frame, "is_no_go").eq(0)
    go_n = int(go.sum())
    if "raw_go_omission_flag" not in frame.columns:
        result: dict[str, Any] = {"omission_taxonomy_status": "not_available_qc_taxonomy_not_applied"}
        for metric in TAXONOMY_RATE_METRICS:
            stem = metric.removesuffix("_rate")
            result[f"{stem}_n"] = pd.NA
            result[metric] = math.nan
        result["omission_taxonomy_denominator"] = go_n
        return result

    flags = {
        "raw_go_omission": "raw_go_omission_flag",
        "clean_go_omission": "clean_go_omission_flag",
        "timing_ambiguous_go_omission": "timing_ambiguous_go_omission_flag",
        "omission_prestimulus_only_ambiguity": "omission_prestimulus_only_ambiguity_flag",
        "omission_carryover_only_ambiguity": "omission_carryover_only_ambiguity_flag",
        "omission_prestimulus_and_carryover_ambiguity": "omission_prestimulus_and_carryover_ambiguity_flag",
        "late_go_response_candidate": "late_go_response_candidate_flag",
        "anticipatory_go_response_candidate": "anticipatory_go_response_candidate_flag",
    }
    result: dict[str, Any] = {"omission_taxonomy_status": "available_current_primary_plus_qc_partition"}
    for name, column in flags.items():
        count = int(_bool(frame, column).sum())
        result[f"{name}_n"] = count
        result[f"{name}_rate"] = _rate(count, go_n)

    result["omission_no_detected_motor_timing_ambiguity_n"] = result["clean_go_omission_n"]
    result["omission_no_detected_motor_timing_ambiguity_rate"] = result["clean_go_omission_rate"]
    result["omission_motor_timing_ambiguous_n"] = result["timing_ambiguous_go_omission_n"]
    result["omission_motor_timing_ambiguous_rate"] = result["timing_ambiguous_go_omission_rate"]

    result["omission_taxonomy_denominator"] = go_n
    result["omission_taxonomy_denominator_contract"] = (
        "raw, clean, and timing-ambiguous Go omission all use all valid Go opportunities; "
        "No-Go commission keeps its separate No-Go denominator"
    )
    raw_n = int(result["raw_go_omission_n"])
    clean_n = int(result["clean_go_omission_n"])
    ambiguous_n = int(result["timing_ambiguous_go_omission_n"])
    result["omission_primary_partition_check"] = raw_n == clean_n + ambiguous_n
    result["omission_subtype_partition_check"] = (
        clean_n
        + int(_bool(frame, "omission_prestimulus_only_ambiguity_flag").sum())
        + int(_bool(frame, "omission_carryover_only_ambiguity_flag").sum())
        + int(_bool(frame, "omission_prestimulus_and_carryover_ambiguity_flag").sum())
        == raw_n
    )
    return result


def _aggregate(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, current in frame.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row.update(summarize_error_taxonomy(current))
        rows.append(row)
    return pd.DataFrame(rows)


def enrich_multiscale_taxonomy(
    trials: pd.DataFrame,
    scale_tables: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Merge the current omission endpoint/partition/QC onto scale metrics."""
    specs = {
        "session": ["repeat_participant_id", "session_id"],
        "block": ["repeat_participant_id", "session_id", "block_id"],
        "cycle": ["repeat_participant_id", "session_id", "block_id", "cycle_bin"],
    }
    source_all = _with_block_id(trials)
    result: dict[str, pd.DataFrame] = {}
    for scale, table in scale_tables.items():
        if table is None or table.empty or scale not in specs:
            result[scale] = table.copy() if table is not None else pd.DataFrame()
            continue
        group_cols = specs[scale]
        source = source_all.copy()
        if scale == "cycle":
            source = source.dropna(subset=["cycle_bin"])
        taxonomy = _aggregate(source, group_cols)
        result[scale] = table.merge(taxonomy, on=group_cols, how="left", validate="one_to_one")
    return result


def enrich_probe_taxonomy(
    trials: pd.DataFrame,
    probe_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    """Recompute omission endpoint/partition/QC inside each strict pre-probe window."""
    if probe_sensitivity is None or probe_sensitivity.empty:
        return probe_sensitivity.copy() if probe_sensitivity is not None else pd.DataFrame()
    source = _with_block_id(trials)
    rows: list[dict[str, Any]] = []
    for record in probe_sensitivity.itertuples(index=False):
        session = str(record.session_id)
        block = str(record.block_id)
        anchor_trial = int(record.anchor_trial_num)
        probe_time = float(record.probe_time_ms)
        seconds = int(record.window_seconds_nominal)
        lower = probe_time - seconds * 1000.0
        current = source[
            source["session_id"].astype(str).eq(session)
            & source["block_id"].astype(str).eq(block)
            & _numeric(source, "trial_num").lt(anchor_trial)
            & _numeric(source, "absolute_onset_time").lt(probe_time)
            & _numeric(source, "absolute_onset_time").ge(lower)
        ]
        row = {
            "probe_event_id": str(record.probe_event_id),
            "window_seconds_nominal": seconds,
        }
        row.update(summarize_error_taxonomy(current))
        rows.append(row)
    taxonomy = pd.DataFrame(rows)
    key = ["probe_event_id", "window_seconds_nominal"]
    if taxonomy.duplicated(key).any():
        raise ValueError("duplicate taxonomy probe/window key")
    return probe_sensitivity.merge(taxonomy, on=key, how="left", validate="one_to_one")


def build_taxonomy_validation(
    scale_tables: Mapping[str, pd.DataFrame],
    primary_probe: pd.DataFrame,
) -> pd.DataFrame:
    """Audit omission roles and QC candidates without p-value selection."""
    frames = {k: v for k, v in scale_tables.items() if v is not None}
    frames["probe"] = primary_probe
    rows: list[dict[str, Any]] = []
    for scale, frame in frames.items():
        if frame is None or frame.empty:
            continue
        for metric in TAXONOMY_RATE_METRICS:
            if metric in CURRENT_PRIMARY_OMISSION_ENDPOINT_METRICS:
                endpoint_role = "current_primary_omission_endpoint"
            elif metric in OMISSION_PARTITION_RATE_METRICS:
                endpoint_role = "descriptive_qc_sensitivity_partition"
            else:
                endpoint_role = "qc_or_timing_diagnostic"
            if metric not in frame.columns:
                rows.append({
                    "scale": scale, "metric": metric, "n_rows": int(len(frame)), "n_valid": 0,
                    "coverage": 0.0, "floor_fraction": np.nan, "ceiling_fraction": np.nan,
                    "candidate_status": "missing_column", "endpoint_role": endpoint_role,
                    "endpoint_status": "not_estimable_missing_column",
                })
                continue
            x = pd.to_numeric(frame[metric], errors="coerce")
            finite = x[np.isfinite(x)]
            rows.append({
                "scale": scale,
                "metric": metric,
                "n_rows": int(len(frame)),
                "n_valid": int(len(finite)),
                "coverage": float(len(finite) / len(frame)) if len(frame) else 0.0,
                "floor_fraction": float((finite <= 0.02).mean()) if len(finite) else np.nan,
                "ceiling_fraction": float((finite >= 0.98).mean()) if len(finite) else np.nan,
                "candidate_status": (
                    "current_primary_endpoint_descriptive_audit"
                    if endpoint_role == "current_primary_omission_endpoint"
                    else "qc_candidate_only"
                ),
                "endpoint_role": endpoint_role,
                "endpoint_status": (
                    "prespecified_not_pvalue_selected"
                    if endpoint_role == "current_primary_omission_endpoint"
                    else "not_a_primary_endpoint"
                ),
                "interpretation_guard": (
                    "clean means no detected motor-timing ambiguity, not proven attentional failure"
                    if metric == "clean_go_omission_rate"
                    else "timing-ambiguous is a recorded motor-timing subgroup, not an independent error mechanism"
                    if metric == "timing_ambiguous_go_omission_rate"
                    else ""
                ),
            })
    return pd.DataFrame(rows)
