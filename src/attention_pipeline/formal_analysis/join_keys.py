from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

# Canonical dtypes for fields that may participate in cross-source joins.
# The normalizer is intentionally independent of any machine path or modality.
STRING_JOIN_KEYS = {
    "anonymous_participant_group_id",
    "session_id",
    "subject",
    "phase",
    "phase_segment",
    "eye",
    "block_id",
    "trial_id",
    "probe_id",
    "window_name",
}
INTEGER_JOIN_KEYS = {
    "frame_idx",
    "block_num",
    "trial_num",
    "probe_num",
}
NUMERIC_TIME_KEYS = {
    "unix_ms",
    "absolute_onset_time",
    "block_onset_time",
    "probe_onset_time",
    "response_time",
}
KNOWN_JOIN_KEYS = STRING_JOIN_KEYS | INTEGER_JOIN_KEYS | NUMERIC_TIME_KEYS


def _required_set(required_non_null: Iterable[str]) -> set[str]:
    return {str(value) for value in required_non_null}


def normalize_known_join_dtypes(
    frame: pd.DataFrame,
    *,
    required_non_null: Iterable[str] = (),
) -> pd.DataFrame:
    """Return a copy with deterministic dtypes for known merge/timeline keys.

    This fixes the common CSV round-trip problem where the same key is read as
    object/string in one modality and int/float in another.  Unknown columns are
    preserved unchanged.  Required keys are also checked for missing values.
    """
    result = frame.copy()
    required = _required_set(required_non_null)
    missing_required = required - set(result.columns)
    if missing_required:
        raise ValueError(f"缺少要求的合并键: {sorted(missing_required)}")

    for column in sorted(KNOWN_JOIN_KEYS & set(result.columns)):
        series = result[column]
        if column in STRING_JOIN_KEYS:
            normalized = series.astype("string").str.strip()
            normalized = normalized.mask(normalized.eq(""))
            if column in required and normalized.isna().any():
                raise ValueError(f"合并键 {column} 存在缺失值")
            result[column] = normalized
            continue

        numeric = pd.to_numeric(series, errors="raise")
        if column in INTEGER_JOIN_KEYS:
            finite = numeric.dropna().astype(float)
            if len(finite) and not np.allclose(finite, np.round(finite), rtol=0.0, atol=1e-9):
                raise ValueError(f"整数合并键 {column} 含非整数值")
            normalized = numeric.round().astype("Int64")
            if column in required:
                if normalized.isna().any():
                    raise ValueError(f"合并键 {column} 存在缺失值")
                result[column] = normalized.astype("int64")
            else:
                result[column] = normalized
            continue

        normalized = numeric.astype("Float64")
        if column in required:
            if normalized.isna().any():
                raise ValueError(f"合并键 {column} 存在缺失值")
            result[column] = normalized.astype("float64")
        else:
            result[column] = normalized

    return result
