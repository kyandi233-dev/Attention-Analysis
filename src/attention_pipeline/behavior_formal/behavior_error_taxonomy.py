from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].fillna(False).astype(bool)


def add_omission_taxonomy(trials: pd.DataFrame) -> pd.DataFrame:
    """Add non-destructive omission/error mechanism labels.

    Raw task scoring is never overwritten.  The current evidence supports
    distinguishing omission trials with prestimulus/carry-over motor timing
    ambiguity, but it does not yet justify renaming the remaining raw omission
    trials as a unique physiological/attentional "true omission" class.
    """
    out = trials.copy()
    go = _numeric(out, "is_no_go").eq(0)
    raw_omission = go & _numeric(out, "omission").eq(1)
    prestimulus = _bool(out, "prestimulus_press_flag")
    carryover = _bool(out, "carryover_candidate_flag")
    anticipatory = _bool(out, "anticipatory_candidate_flag")
    late = go & _numeric(out, "response").eq(1) & _numeric(out, "rt").gt(1150)

    out["raw_go_omission_flag"] = raw_omission
    out["omission_prestimulus_ambiguity_flag"] = raw_omission & prestimulus
    out["omission_carryover_ambiguity_flag"] = raw_omission & carryover
    out["omission_motor_timing_ambiguous_flag"] = raw_omission & (prestimulus | carryover)
    out["omission_no_detected_motor_timing_ambiguity_flag"] = raw_omission & ~(prestimulus | carryover)
    out["late_go_response_candidate_flag"] = late
    out["anticipatory_go_response_candidate_flag"] = go & anticipatory

    subtype = pd.Series(pd.NA, index=out.index, dtype="string")
    subtype.loc[raw_omission & prestimulus & carryover] = "ambiguous_prestimulus_and_carryover"
    subtype.loc[raw_omission & prestimulus & ~carryover] = "ambiguous_prestimulus"
    subtype.loc[raw_omission & ~prestimulus & carryover] = "ambiguous_carryover"
    subtype.loc[raw_omission & ~prestimulus & ~carryover] = "raw_omission_no_detected_motor_timing_ambiguity"
    out["omission_subtype"] = subtype
    out["omission_taxonomy_contract"] = (
        "raw scoring preserved; ambiguity subclasses are motor-timing QC candidates; "
        "no subclass is automatically declared the sole attentional omission endpoint"
    )
    return out


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else math.nan


def summarize_error_taxonomy(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize mutually interpretable error/QC candidates using Go opportunities."""
    go = _numeric(frame, "is_no_go").eq(0)
    go_n = int(go.sum())
    if "raw_go_omission_flag" not in frame.columns:
        return {
            "omission_taxonomy_status": "not_available_qc_taxonomy_not_applied",
            "omission_no_detected_motor_timing_ambiguity_n": pd.NA,
            "omission_no_detected_motor_timing_ambiguity_rate": math.nan,
            "omission_motor_timing_ambiguous_n": pd.NA,
            "omission_motor_timing_ambiguous_rate": math.nan,
            "omission_prestimulus_ambiguity_n": pd.NA,
            "omission_prestimulus_ambiguity_rate": math.nan,
            "omission_carryover_ambiguity_n": pd.NA,
            "omission_carryover_ambiguity_rate": math.nan,
            "late_go_response_candidate_n": pd.NA,
            "late_go_response_candidate_rate": math.nan,
            "anticipatory_go_response_candidate_n": pd.NA,
            "anticipatory_go_response_candidate_rate": math.nan,
        }

    no_ambiguity = _bool(frame, "omission_no_detected_motor_timing_ambiguity_flag")
    ambiguous = _bool(frame, "omission_motor_timing_ambiguous_flag")
    pre = _bool(frame, "omission_prestimulus_ambiguity_flag")
    carry = _bool(frame, "omission_carryover_ambiguity_flag")
    late = _bool(frame, "late_go_response_candidate_flag")
    anticipatory = _bool(frame, "anticipatory_go_response_candidate_flag")
    values = {
        "omission_no_detected_motor_timing_ambiguity": int(no_ambiguity.sum()),
        "omission_motor_timing_ambiguous": int(ambiguous.sum()),
        "omission_prestimulus_ambiguity": int(pre.sum()),
        "omission_carryover_ambiguity": int(carry.sum()),
        "late_go_response_candidate": int(late.sum()),
        "anticipatory_go_response_candidate": int(anticipatory.sum()),
    }
    result: dict[str, Any] = {"omission_taxonomy_status": "available_candidate_taxonomy"}
    for name, count in values.items():
        result[f"{name}_n"] = count
        result[f"{name}_rate"] = _rate(count, go_n)
    result["omission_taxonomy_denominator"] = go_n
    result["omission_taxonomy_denominator_contract"] = "all Go opportunities; raw omission scoring unchanged"
    return result
