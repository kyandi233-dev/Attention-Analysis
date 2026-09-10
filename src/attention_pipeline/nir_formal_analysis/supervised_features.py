from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from attention_pipeline.nir_behavior.features import summarize_signal


SUPERVISED_NIR_INTERFACE_VERSION: Final = "nir-supervised-pupil-v1"
BASE_SIGNAL: Final = "pupil_geom_mean_diameter"
RAW_VALUE_COLUMNS: Final = (
    "left_raw_pupil_diameter",
    "right_raw_pupil_diameter",
)
PRIMARY_VALID_COLUMNS: Final = (
    "left_pupil_valid_primary",
    "right_pupil_valid_primary",
)
PRIMARY_WINDOW_SEC: Final = 30
SENSITIVITY_WINDOW_SEC: Final = (10, 20)
ALLOWED_WINDOW_SEC: Final = (10, 20, 30)
LEVEL_CANDIDATES: Final = ("mean", "median")
VARIABILITY_CANDIDATES: Final = ("sd", "mad", "iqr")
TREND_CANDIDATES: Final = ("robust_binned_slope_per_sec",)
SUMMARY_SOURCE_FUNCTION: Final = "attention_pipeline.nir_behavior.features.summarize_signal"
TREND_ALGORITHM: Final = "robust_binned_slope_per_sec"


@dataclass(frozen=True)
class ProbeWindowContract:
    window_sec: int
    role: str

    @property
    def start_offset_ms(self) -> int:
        return -1000 * self.window_sec

    @property
    def end_offset_ms(self) -> int:
        return 0


WINDOW_CONTRACTS: Final = tuple(
    ProbeWindowContract(
        window_sec=value,
        role="primary" if value == PRIMARY_WINDOW_SEC else "window_sensitivity",
    )
    for value in ALLOWED_WINDOW_SEC
)


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes", "y"})
    )


def _finite_positive(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.isfinite(numeric.to_numpy(dtype=float)) & numeric.gt(0).to_numpy(dtype=bool),
        index=series.index,
    )


def build_raw_binocular_timepoints(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the zero-calibration binocular pupil series from raw eye values.

    The existing analysis-ready table also contains session×eye centered/robust-z
    columns. Those columns are intentionally ignored here because a new-participant
    probe feature must be computable from the current pre-probe window alone.
    """

    required = {
        "session_id",
        "block",
        "unix_ms",
        *RAW_VALUE_COLUMNS,
        *PRIMARY_VALID_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"analysis-ready frame missing supervised NIR columns: {missing}")

    out = pd.DataFrame(index=frame.index)
    for name in (
        "session_id",
        "subject",
        "analysis_group_token",
        "participant_group_id",
        "block",
        "phase",
        "phase_segment",
        "frame_idx",
        "unix_ms",
        "video_time_ms",
        "phase_time_ms",
    ):
        if name in frame.columns:
            out[name] = frame[name]

    left = pd.to_numeric(frame["left_raw_pupil_diameter"], errors="coerce")
    right = pd.to_numeric(frame["right_raw_pupil_diameter"], errors="coerce")
    left_valid = _bool_series(frame["left_pupil_valid_primary"]) & _finite_positive(left)
    right_valid = _bool_series(frame["right_pupil_valid_primary"]) & _finite_positive(right)

    both = left_valid & right_valid
    left_only = left_valid & ~right_valid
    right_only = ~left_valid & right_valid

    out["left_raw_pupil_diameter"] = left
    out["right_raw_pupil_diameter"] = right
    out["left_pupil_valid_primary"] = left_valid
    out["right_pupil_valid_primary"] = right_valid
    out["raw_binocular_pupil"] = np.select(
        [both, left_only, right_only],
        [(left + right) / 2.0, left, right],
        default=np.nan,
    ).astype(float)
    out["raw_binocular_source_mode"] = np.select(
        [both, left_only, right_only],
        ["binocular", "left_only", "right_only"],
        default="missing",
    )
    out["base_signal"] = BASE_SIGNAL
    out["zero_calibration_source"] = "raw_current_probe_window_only"

    out["unix_ms"] = pd.to_numeric(out["unix_ms"], errors="coerce")
    out["block"] = pd.to_numeric(out["block"], errors="coerce")
    if out[["unix_ms", "block"]].isna().any(axis=1).any():
        raise ValueError("analysis-ready frame has missing block/unix_ms for supervised NIR")

    key = ["session_id", "block", "unix_ms"]
    if "frame_idx" in out.columns:
        key.append("frame_idx")
    if out.duplicated(key, keep=False).any():
        raise ValueError(f"duplicate supervised NIR timepoint key: {key}")

    return out.sort_values(["session_id", "block", "unix_ms"], kind="stable").reset_index(drop=True)


def select_preprobe_window(
    timepoints: pd.DataFrame,
    *,
    session_id: str,
    block_num: int,
    probe_onset_ms: float,
    window_sec: int,
) -> pd.DataFrame:
    """Select exactly [probe-window, probe) within one session and block."""

    if window_sec not in ALLOWED_WINDOW_SEC:
        raise ValueError(
            f"window_sec must be one of {ALLOWED_WINDOW_SEC}, got {window_sec}"
        )
    if not np.isfinite(float(probe_onset_ms)):
        raise ValueError("probe_onset_ms must be finite")

    required = {"session_id", "block", "unix_ms", "raw_binocular_pupil"}
    missing = sorted(required - set(timepoints.columns))
    if missing:
        raise ValueError(f"supervised timepoints missing columns: {missing}")

    onset = float(probe_onset_ms)
    start = onset - float(window_sec) * 1000.0
    times = pd.to_numeric(timepoints["unix_ms"], errors="coerce")
    blocks = pd.to_numeric(timepoints["block"], errors="coerce")
    mask = (
        timepoints["session_id"].astype(str).eq(str(session_id))
        & blocks.eq(int(block_num))
        & times.ge(start)
        & times.lt(onset)
    )
    result = timepoints.loc[mask].copy()
    result["window_sec"] = int(window_sec)
    result["window_start_ms"] = start
    result["window_end_ms"] = onset
    result["window_end_exclusive"] = True
    return result.reset_index(drop=True)


def summarize_supervised_window(window: pd.DataFrame) -> dict[str, object]:
    """Return the frozen compact NIR candidate family for one pre-probe window.

    Numeric summaries reuse ``nir_behavior.features.summarize_signal`` so the
    supervised interface cannot silently drift to a different slope/statistic
    implementation.  Estimability is stricter than the historical generic
    summary: variability requires at least two valid samples, and the trend
    requires the existing robust-binned slope implementation to return a value.
    """

    required = {"unix_ms", "raw_binocular_pupil"}
    missing = sorted(required - set(window.columns))
    if missing:
        raise ValueError(f"supervised NIR window missing columns: {missing}")

    times = pd.to_numeric(window["unix_ms"], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(window["raw_binocular_pupil"], errors="coerce").to_numpy(dtype=float)
    generic = summarize_signal(times, values, prefix="pupil")
    n_valid = int(generic["n_pupil_valid"])

    result: dict[str, object] = {
        "base_signal": BASE_SIGNAL,
        "summary_source_function": SUMMARY_SOURCE_FUNCTION,
        "trend_algorithm": TREND_ALGORITHM,
        "n_valid_pupil_samples": n_valid,
        "pupil_level_mean": generic["pupil_mean"],
        "pupil_level_median": generic["pupil_median"],
        "pupil_variability_sd": generic["pupil_sd"] if n_valid >= 2 else None,
        "pupil_variability_mad": generic["pupil_mad"] if n_valid >= 2 else None,
        "pupil_variability_iqr": generic["pupil_iqr"] if n_valid >= 2 else None,
        "pupil_trend_robust_binned_slope_per_sec": generic["pupil_slope_per_sec"],
    }

    level_status = "computable" if n_valid >= 1 else "not_estimable_no_valid_samples"
    variability_status = (
        "computable" if n_valid >= 2 else
        "not_estimable_low_valid_samples" if n_valid == 1 else
        "not_estimable_no_valid_samples"
    )
    if n_valid < 3:
        trend_status = (
            "not_estimable_no_valid_samples" if n_valid == 0
            else "not_estimable_low_valid_samples"
        )
    elif generic["pupil_slope_per_sec"] is None:
        trend_status = "not_estimable_time_support"
    else:
        trend_status = "computable"

    for candidate in LEVEL_CANDIDATES:
        result[f"pupil_level_{candidate}_status"] = level_status
    for candidate in VARIABILITY_CANDIDATES:
        result[f"pupil_variability_{candidate}_status"] = variability_status
    result["pupil_trend_robust_binned_slope_per_sec_status"] = trend_status
    return result
