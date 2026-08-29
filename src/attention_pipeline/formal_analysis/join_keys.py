from __future__ import annotations

from collections.abc import Iterable, Mapping
import re

import numpy as np
import pandas as pd

# Canonical dtypes for fields that may participate in cross-source joins.
# All canonical time keys are integer UTC epoch milliseconds after normalization.
STRING_JOIN_KEYS = {
    "anonymous_participant_group_id",
    "repeat_participant_id",
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
TIME_JOIN_KEYS = {
    "unix_ms",
    "absolute_onset_time",
    "next_trial_onset_time",
    "block_onset_time",
    "probe_onset_time",
    "current_trial_onset_unix_ms",
    "current_next_trial_onset_unix_ms",
    "previous_trial_onset_unix_ms",
}
KNOWN_JOIN_KEYS = STRING_JOIN_KEYS | INTEGER_JOIN_KEYS | TIME_JOIN_KEYS
_SUPPORTED_TIME_UNITS = {"ms": 1.0, "s": 1000.0}
_EXPLICIT_TZ_RE = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$", re.IGNORECASE)


def _required_set(required_non_null: Iterable[str]) -> set[str]:
    return {str(value) for value in required_non_null}


def _integer_like(series: pd.Series, *, label: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise")
    finite = numeric.dropna().astype(float)
    if len(finite) and not np.allclose(
        finite, np.round(finite), rtol=0.0, atol=1e-9
    ):
        raise ValueError(f"整数合并键 {label} 含非整数值")
    return numeric.round().astype("Int64")


def _numeric_time_to_ms(series: pd.Series, *, label: str, unit: str) -> pd.Series:
    if unit not in _SUPPORTED_TIME_UNITS:
        raise ValueError(
            f"时间键 {label} 的单位 {unit!r} 不受支持；仅允许 {sorted(_SUPPORTED_TIME_UNITS)}"
        )
    numeric = pd.to_numeric(series, errors="raise").astype("Float64")
    millis = numeric * _SUPPORTED_TIME_UNITS[unit]
    finite = millis.dropna().astype(float)
    if len(finite) and not np.allclose(
        finite, np.round(finite), rtol=0.0, atol=1e-6
    ):
        raise ValueError(f"时间键 {label} 不能无损规范化为整数毫秒")
    return millis.round().astype("Int64")


def _datetime_time_to_ms(
    series: pd.Series,
    *,
    label: str,
    naive_timezone: str | None,
) -> pd.Series:
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        parsed = series.dt.tz_convert("UTC")
    elif pd.api.types.is_datetime64_dtype(series.dtype):
        if naive_timezone is None:
            raise ValueError(
                f"时间键 {label} 为无时区 datetime；必须显式提供 naive_timezone"
            )
        parsed = series.dt.tz_localize(naive_timezone).dt.tz_convert("UTC")
    else:
        text = series.astype("string").str.strip().mask(lambda x: x.eq(""))
        non_missing = text.dropna()
        if non_missing.empty:
            return pd.Series(pd.NA, index=series.index, dtype="Int64")
        numeric_probe = pd.to_numeric(non_missing, errors="coerce")
        if numeric_probe.notna().any():
            raise ValueError(
                f"时间键 {label} 混合了数值时间与 datetime 文本，拒绝猜测单位"
            )
        explicit_tz = non_missing.str.contains(_EXPLICIT_TZ_RE)
        if explicit_tz.any() and not explicit_tz.all():
            raise ValueError(
                f"时间键 {label} 混合了带时区与无时区 datetime，拒绝静默解释"
            )
        if explicit_tz.all():
            parsed = pd.to_datetime(text, errors="raise", utc=True)
        else:
            if naive_timezone is None:
                raise ValueError(
                    f"时间键 {label} 的 datetime 文本没有时区；必须显式提供 naive_timezone"
                )
            parsed = pd.to_datetime(text, errors="raise")
            parsed = parsed.dt.tz_localize(naive_timezone).dt.tz_convert("UTC")

    result = pd.Series(pd.NA, index=series.index, dtype="Int64")
    valid = parsed.notna()
    if valid.any():
        # Do not divide the Series' raw integer storage: pandas 3 may preserve
        # microsecond resolution for parsed datetimes. Timestamp.value is
        # explicitly nanoseconds since epoch across datetime64 resolutions.
        result.loc[valid] = parsed.loc[valid].map(
            lambda value: int(value.value // 1_000_000)
        ).astype("int64")
    return result


def _normalize_time(
    series: pd.Series,
    *,
    label: str,
    unit: str,
    naive_timezone: str | None,
) -> pd.Series:
    if isinstance(series.dtype, pd.DatetimeTZDtype) or pd.api.types.is_datetime64_dtype(
        series.dtype
    ):
        return _datetime_time_to_ms(
            series, label=label, naive_timezone=naive_timezone
        )

    if pd.api.types.is_numeric_dtype(series.dtype):
        return _numeric_time_to_ms(series, label=label, unit=unit)

    text = series.astype("string").str.strip().mask(lambda x: x.eq(""))
    non_missing = text.dropna()
    if non_missing.empty:
        return pd.Series(pd.NA, index=series.index, dtype="Int64")
    numeric_probe = pd.to_numeric(non_missing, errors="coerce")
    if numeric_probe.notna().all():
        return _numeric_time_to_ms(text, label=label, unit=unit)
    if numeric_probe.notna().any():
        raise ValueError(
            f"时间键 {label} 混合了数值与 datetime 表达，拒绝静默单位推断"
        )
    return _datetime_time_to_ms(
        text, label=label, naive_timezone=naive_timezone
    )


def normalize_known_join_dtypes(
    frame: pd.DataFrame,
    *,
    required_non_null: Iterable[str] = (),
    time_units: Mapping[str, str] | None = None,
    naive_timezone: str | None = None,
) -> pd.DataFrame:
    """Return a copy with deterministic dtypes for known merge/timeline keys.

    Contract:
    - identifier keys use pandas ``string`` dtype; blank strings become missing;
    - integer keys use nullable ``Int64`` (or ``int64`` when required non-null);
    - time keys use UTC epoch milliseconds as nullable ``Int64`` (or ``int64``
      when required non-null). Numeric inputs default to milliseconds; callers
      must explicitly declare seconds via ``time_units``. Timezone-aware
      datetimes are converted to UTC. Naive datetimes fail closed unless
      ``naive_timezone`` is explicitly supplied.

    Unknown columns are preserved unchanged. Required keys must exist and may
    not contain missing values after normalization. Fields that are not merge or
    timeline keys (for example a response latency named ``response_time``) are
    intentionally left untouched.
    """
    result = frame.copy()
    required = _required_set(required_non_null)
    units = {str(key): str(value) for key, value in (time_units or {}).items()}
    missing_required = required - set(result.columns)
    if missing_required:
        raise ValueError(f"缺少要求的合并键: {sorted(missing_required)}")

    for column in sorted(KNOWN_JOIN_KEYS & set(result.columns)):
        series = result[column]
        if column in STRING_JOIN_KEYS:
            normalized = series.astype("string").str.strip()
            normalized = normalized.mask(normalized.eq(""))
        elif column in INTEGER_JOIN_KEYS:
            normalized = _integer_like(series, label=column)
        else:
            normalized = _normalize_time(
                series,
                label=column,
                unit=units.get(column, "ms"),
                naive_timezone=naive_timezone,
            )

        if column in required and normalized.isna().any():
            raise ValueError(f"合并键 {column} 存在缺失值")
        if column in required and column in INTEGER_JOIN_KEYS | TIME_JOIN_KEYS:
            normalized = normalized.astype("int64")
        result[column] = normalized

    return result
