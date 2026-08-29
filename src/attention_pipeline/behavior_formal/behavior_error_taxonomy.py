from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

TAXONOMY_RATE_METRICS = (
    "omission_no_detected_motor_timing_ambiguity_rate",
    "omission_motor_timing_ambiguous_rate",
    "omission_prestimulus_only_ambiguity_rate",
    "omission_carryover_only_ambiguity_rate",
    "omission_prestimulus_and_carryover_ambiguity_rate",
    "late_go_response_candidate_rate",
    "anticipatory_go_response_candidate_rate",
)


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
    """Add non-destructive omission/error mechanism labels.

    Raw task scoring is never overwritten. The available motor-timing evidence
    supports separating prestimulus/carry-over ambiguity, but it does not justify
    renaming the residual raw omission class as a unique attentional "true
    omission" endpoint before task-source and cohort-level validation.
    """
    out = trials.copy()
    go = _numeric(out, "is_no_go").eq(0)
    raw_omission = go & _numeric(out, "omission").eq(1)
    prestimulus = _bool(out, "prestimulus_press_flag")
    carryover = _bool(out, "carryover_candidate_flag")
    anticipatory = _bool(out, "anticipatory_candidate_flag")
    late = go & _numeric(out, "response").eq(1) & _numeric(out, "rt").gt(1150)

    both = raw_omission & prestimulus & carryover
    pre_only = raw_omission & prestimulus & ~carryover
    carry_only = raw_omission & ~prestimulus & carryover
    no_detected_ambiguity = raw_omission & ~prestimulus & ~carryover

    out["raw_go_omission_flag"] = raw_omission
    out["omission_prestimulus_ambiguity_flag"] = raw_omission & prestimulus
    out["omission_carryover_ambiguity_flag"] = raw_omission & carryover
    out["omission_motor_timing_ambiguous_flag"] = raw_omission & (prestimulus | carryover)
    out["omission_prestimulus_only_ambiguity_flag"] = pre_only
    out["omission_carryover_only_ambiguity_flag"] = carry_only
    out["omission_prestimulus_and_carryover_ambiguity_flag"] = both
    out["omission_no_detected_motor_timing_ambiguity_flag"] = no_detected_ambiguity
    out["late_go_response_candidate_flag"] = late
    out["anticipatory_go_response_candidate_flag"] = go & anticipatory

    subtype = pd.Series(pd.NA, index=out.index, dtype="string")
    subtype.loc[both] = "ambiguous_prestimulus_and_carryover"
    subtype.loc[pre_only] = "ambiguous_prestimulus_only"
    subtype.loc[carry_only] = "ambiguous_carryover_only"
    subtype.loc[no_detected_ambiguity] = "raw_omission_no_detected_motor_timing_ambiguity"
    out["omission_subtype"] = subtype
    out["omission_taxonomy_contract"] = (
        "raw scoring preserved; mutually exclusive omission subtypes reflect detected motor-timing ambiguity; "
        "the residual class is not automatically declared the sole attentional omission endpoint"
    )
    return out


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else math.nan


def summarize_error_taxonomy(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize omission subtypes and motor-timing candidates using Go opportunities."""
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
        "omission_no_detected_motor_timing_ambiguity": "omission_no_detected_motor_timing_ambiguity_flag",
        "omission_motor_timing_ambiguous": "omission_motor_timing_ambiguous_flag",
        "omission_prestimulus_only_ambiguity": "omission_prestimulus_only_ambiguity_flag",
        "omission_carryover_only_ambiguity": "omission_carryover_only_ambiguity_flag",
        "omission_prestimulus_and_carryover_ambiguity": "omission_prestimulus_and_carryover_ambiguity_flag",
        "late_go_response_candidate": "late_go_response_candidate_flag",
        "anticipatory_go_response_candidate": "anticipatory_go_response_candidate_flag",
    }
    result = {"omission_taxonomy_status": "available_candidate_taxonomy"}
    for name, column in flags.items():
        count = int(_bool(frame, column).sum())
        result[f"{name}_n"] = count
        result[f"{name}_rate"] = _rate(count, go_n)
    result["omission_taxonomy_denominator"] = go_n
    result["omission_taxonomy_denominator_contract"] = "all Go opportunities; raw task omission scoring unchanged"
    result["omission_subtype_partition_check"] = (
        int(_bool(frame, "omission_no_detected_motor_timing_ambiguity_flag").sum())
        + int(_bool(frame, "omission_prestimulus_only_ambiguity_flag").sum())
        + int(_bool(frame, "omission_carryover_only_ambiguity_flag").sum())
        + int(_bool(frame, "omission_prestimulus_and_carryover_ambiguity_flag").sum())
        == int(_bool(frame, "raw_go_omission_flag").sum())
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
    """Merge the same taxonomy onto session/block/cycle metrics."""
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
    """Recompute taxonomy inside each strict pre-probe window using the same anchor rules."""
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
    """Inventory calculability/floor/ceiling without promoting QC subclasses to endpoints."""
    frames = {k: v for k, v in scale_tables.items() if v is not None}
    frames["probe"] = primary_probe
    rows: list[dict[str, Any]] = []
    for scale, frame in frames.items():
        if frame is None or frame.empty:
            continue
        for metric in TAXONOMY_RATE_METRICS:
            if metric not in frame.columns:
                rows.append({
                    "scale": scale, "metric": metric, "n_rows": int(len(frame)), "n_valid": 0,
                    "coverage": 0.0, "floor_fraction": np.nan, "ceiling_fraction": np.nan,
                    "candidate_status": "missing_column", "endpoint_status": "not_frozen",
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
                "candidate_status": "qc_candidate_only",
                "endpoint_status": "pending_task_semantics_and_cohort_validation",
            })
    return pd.DataFrame(rows)
