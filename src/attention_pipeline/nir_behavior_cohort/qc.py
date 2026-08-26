from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..nir_behavior.contract import (
    OAR_COLUMN,
    OAR_P90_COLUMN,
    PIR_COLUMN,
    PIR_VALID_COLUMN,
)


_AUXILIARY_SIGNAL_COLUMNS = (
    ("fullclass_ocular_fraction", "ocular_fraction"),
    ("fullclass_iris_outer_fraction", "iris_outer_fraction"),
    ("fullclass_pupil_fraction", "pupil_fraction"),
    (
        "fullclass_ocular_largest_component_fraction",
        "ocular_largest_component_fraction",
    ),
)


def _mad(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


def _signal_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = values[np.isfinite(values)]
    result: dict[str, float] = {f"{prefix}_n": int(values.size)}
    if values.size == 0:
        for name in (
            "mean",
            "median",
            "sd",
            "mad",
            "iqr",
            "p01",
            "p05",
            "p10",
            "p90",
            "p95",
            "p99",
            "min",
            "max",
        ):
            result[f"{prefix}_{name}"] = float("nan")
        return result
    q = np.quantile(
        values,
        [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
    )
    result.update(
        {
            f"{prefix}_mean": float(np.mean(values)),
            f"{prefix}_median": float(q[4]),
            f"{prefix}_sd": (
                float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
            ),
            f"{prefix}_mad": _mad(values),
            f"{prefix}_iqr": float(q[5] - q[3]),
            f"{prefix}_p01": float(q[0]),
            f"{prefix}_p05": float(q[1]),
            f"{prefix}_p10": float(q[2]),
            f"{prefix}_p90": float(q[6]),
            f"{prefix}_p95": float(q[7]),
            f"{prefix}_p99": float(q[8]),
            f"{prefix}_min": float(np.min(values)),
            f"{prefix}_max": float(np.max(values)),
        }
    )
    return result


def _optional_signal_stats(
    frame: pd.DataFrame,
    *,
    column: str,
    prefix: str,
) -> dict[str, float]:
    if column not in frame.columns or frame.empty:
        result = {f"{prefix}_available_fraction": float("nan")}
        result.update(_signal_stats(np.array([], dtype=float), prefix))
        return result

    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values)
    result = {f"{prefix}_available_fraction": float(np.mean(finite))}
    result.update(_signal_stats(values[finite], prefix))
    return result


def _robust_binned_slope_per_sec(
    times_ms: np.ndarray,
    values: np.ndarray,
    bin_sec: float,
) -> float:
    mask = np.isfinite(times_ms) & np.isfinite(values)
    if mask.sum() < 2:
        return float("nan")
    t = times_ms[mask]
    y = values[mask]
    origin = float(np.min(t))
    bins = np.floor((t - origin) / (float(bin_sec) * 1000.0)).astype(int)
    frame = pd.DataFrame({"bin": bins, "time_ms": t, "value": y})
    med = frame.groupby("bin", sort=True).agg(
        time_ms=("time_ms", "median"),
        value=("value", "median"),
    )
    if len(med) < 2:
        return float("nan")
    x_sec = (med["time_ms"].to_numpy(dtype=float) - origin) / 1000.0
    vals = med["value"].to_numpy(dtype=float)
    return float(np.polyfit(x_sec, vals, deg=1)[0])


def _successive_stats(
    times_ms: np.ndarray,
    values: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    mask = np.isfinite(times_ms) & np.isfinite(values)
    t = times_ms[mask]
    y = values[mask]
    if y.size < 2:
        return {
            f"{prefix}_successive_diff_mad": float("nan"),
            f"{prefix}_successive_rate_mad_per_sec": float("nan"),
        }
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    dt_sec = np.diff(t) / 1000.0
    dy = np.diff(y)
    positive = dt_sec > 0
    rates = dy[positive] / dt_sec[positive]
    return {
        f"{prefix}_successive_diff_mad": _mad(dy),
        f"{prefix}_successive_rate_mad_per_sec": _mad(rates),
    }


def _empty_eye_block_row(
    *,
    subject: str,
    block_num: int,
    eye: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "subject": subject,
        "block_num": int(block_num),
        "eye": eye,
        "missing_eye_block": True,
        "n_nir_rows": 0,
        "unique_unix_ms_count": 0,
        "observed_duration_sec": np.nan,
        "rows_per_sec_observed": np.nan,
        "sampling_rate_hz_estimate": np.nan,
        "internal_coverage_fraction_estimate": np.nan,
        "max_temporal_gap_sec": np.nan,
        "p95_temporal_gap_ms": np.nan,
        "p99_temporal_gap_ms": np.nan,
        "duplicate_unix_ms_count": 0,
        "pir_numeric_finite_fraction": np.nan,
        "pir_normalization_valid_fraction": np.nan,
        "pir_usable_fraction": np.nan,
        "oar_available_fraction": np.nan,
        "oar_p90_available_fraction": np.nan,
        "oar_pair_available_fraction": np.nan,
        "roi_clipped_fraction": np.nan,
        "ritnet_found_fraction": np.nan,
        "ocular_fragmented_candidate_fraction": np.nan,
    }
    for prefix in (
        "pir",
        "oar",
        "oar_p90",
        "oar_p90_minus_median",
        "ocular_fraction",
        "iris_outer_fraction",
        "pupil_fraction",
        "ocular_largest_component_fraction",
    ):
        row.update(_signal_stats(np.array([], dtype=float), prefix))
    for prefix in (
        "ocular_fraction",
        "iris_outer_fraction",
        "pupil_fraction",
        "ocular_largest_component_fraction",
    ):
        row[f"{prefix}_available_fraction"] = np.nan
    return row


def summarize_eye_block(
    frame: pd.DataFrame,
    *,
    subject: str,
    block_num: int,
    eye: str,
    slope_bin_sec: float = 5.0,
    fragmentation_component_count_gt: int = 1,
    fragmentation_largest_fraction_lt: float = 0.90,
) -> dict[str, Any]:
    if frame.empty:
        return _empty_eye_block_row(
            subject=subject,
            block_num=block_num,
            eye=eye,
        )

    row: dict[str, Any] = {
        "subject": subject,
        "block_num": int(block_num),
        "eye": eye,
        "missing_eye_block": False,
    }

    times = pd.to_numeric(frame["unix_ms"], errors="coerce").to_numpy(dtype=float)
    times = times[np.isfinite(times)]
    times.sort()
    unique_times = np.unique(times)
    row["n_nir_rows"] = int(len(frame))
    row["unique_unix_ms_count"] = int(unique_times.size)

    if unique_times.size > 1:
        observed_duration_sec = float((unique_times[-1] - unique_times[0]) / 1000.0)
        positive_dt = np.diff(unique_times)
    else:
        observed_duration_sec = 0.0
        positive_dt = np.array([], dtype=float)

    row["observed_duration_sec"] = observed_duration_sec
    row["rows_per_sec_observed"] = (
        float((unique_times.size - 1) / observed_duration_sec)
        if observed_duration_sec > 0 and unique_times.size > 1
        else np.nan
    )
    sampling_rate = (
        float(1000.0 / np.median(positive_dt))
        if positive_dt.size
        else np.nan
    )
    row["sampling_rate_hz_estimate"] = sampling_rate
    if (
        observed_duration_sec > 0
        and np.isfinite(sampling_rate)
        and sampling_rate > 0
        and unique_times.size > 1
    ):
        expected_intervals = observed_duration_sec * sampling_rate
        row["internal_coverage_fraction_estimate"] = float(
            min(1.0, (unique_times.size - 1) / expected_intervals)
        )
    else:
        row["internal_coverage_fraction_estimate"] = np.nan

    row["max_temporal_gap_sec"] = (
        float(np.max(positive_dt) / 1000.0) if positive_dt.size else np.nan
    )
    row["p95_temporal_gap_ms"] = (
        float(np.quantile(positive_dt, 0.95)) if positive_dt.size else np.nan
    )
    row["p99_temporal_gap_ms"] = (
        float(np.quantile(positive_dt, 0.99)) if positive_dt.size else np.nan
    )
    row["duplicate_unix_ms_count"] = (
        int(len(times) - unique_times.size) if times.size else 0
    )

    pir = pd.to_numeric(frame[PIR_COLUMN], errors="coerce").to_numpy(dtype=float)
    pir_valid = frame[PIR_VALID_COLUMN].fillna(False).to_numpy(dtype=bool)
    pir_finite = np.isfinite(pir)
    pir_usable_mask = pir_valid & pir_finite
    row["pir_numeric_finite_fraction"] = float(np.mean(pir_finite))
    row["pir_normalization_valid_fraction"] = float(np.mean(pir_valid))
    row["pir_usable_fraction"] = float(np.mean(pir_usable_mask))
    row.update(_signal_stats(pir[pir_usable_mask], "pir"))
    row["pir_robust_binned_slope_per_sec"] = _robust_binned_slope_per_sec(
        pd.to_numeric(
            frame.loc[pir_usable_mask, "unix_ms"],
            errors="coerce",
        ).to_numpy(dtype=float),
        pir[pir_usable_mask],
        slope_bin_sec,
    )
    row.update(
        _successive_stats(
            pd.to_numeric(
                frame.loc[pir_usable_mask, "unix_ms"],
                errors="coerce",
            ).to_numpy(dtype=float),
            pir[pir_usable_mask],
            "pir",
        )
    )

    oar = pd.to_numeric(frame[OAR_COLUMN], errors="coerce").to_numpy(dtype=float)
    oar_mask = np.isfinite(oar)
    row["oar_available_fraction"] = float(np.mean(oar_mask))
    row.update(_signal_stats(oar[oar_mask], "oar"))
    row["oar_robust_binned_slope_per_sec"] = _robust_binned_slope_per_sec(
        pd.to_numeric(
            frame.loc[oar_mask, "unix_ms"],
            errors="coerce",
        ).to_numpy(dtype=float),
        oar[oar_mask],
        slope_bin_sec,
    )
    row.update(
        _successive_stats(
            pd.to_numeric(
                frame.loc[oar_mask, "unix_ms"],
                errors="coerce",
            ).to_numpy(dtype=float),
            oar[oar_mask],
            "oar",
        )
    )

    oar_p90 = pd.to_numeric(
        frame[OAR_P90_COLUMN],
        errors="coerce",
    ).to_numpy(dtype=float)
    oar_p90_mask = np.isfinite(oar_p90)
    row["oar_p90_available_fraction"] = float(np.mean(oar_p90_mask))
    row.update(_signal_stats(oar_p90[oar_p90_mask], "oar_p90"))
    row["oar_p90_robust_binned_slope_per_sec"] = _robust_binned_slope_per_sec(
        pd.to_numeric(
            frame.loc[oar_p90_mask, "unix_ms"],
            errors="coerce",
        ).to_numpy(dtype=float),
        oar_p90[oar_p90_mask],
        slope_bin_sec,
    )
    row.update(
        _successive_stats(
            pd.to_numeric(
                frame.loc[oar_p90_mask, "unix_ms"],
                errors="coerce",
            ).to_numpy(dtype=float),
            oar_p90[oar_p90_mask],
            "oar_p90",
        )
    )

    oar_pair = oar_mask & oar_p90_mask
    row["oar_pair_available_fraction"] = float(np.mean(oar_pair))
    oar_gap = oar_p90[oar_pair] - oar[oar_pair]
    row.update(_signal_stats(oar_gap, "oar_p90_minus_median"))

    for column, prefix in _AUXILIARY_SIGNAL_COLUMNS:
        row.update(
            _optional_signal_stats(
                frame,
                column=column,
                prefix=prefix,
            )
        )

    for column, output in (
        ("roi_clipped", "roi_clipped_fraction"),
        ("ritnet_found", "ritnet_found_fraction"),
    ):
        row[output] = (
            float(frame[column].fillna(False).mean())
            if column in frame.columns
            else np.nan
        )

    if {
        "fullclass_ocular_component_count",
        "fullclass_ocular_largest_component_fraction",
    }.issubset(frame.columns):
        count = pd.to_numeric(
            frame["fullclass_ocular_component_count"],
            errors="coerce",
        )
        frac = pd.to_numeric(
            frame["fullclass_ocular_largest_component_fraction"],
            errors="coerce",
        )
        fragmented = (
            count.gt(fragmentation_component_count_gt)
            & frac.lt(fragmentation_largest_fraction_lt)
        )
        row["ocular_fragmented_candidate_fraction"] = float(
            fragmented.fillna(False).mean()
        )
    else:
        row["ocular_fragmented_candidate_fraction"] = np.nan
    return row


def summarize_behavior_block(frame: pd.DataFrame) -> dict[str, Any]:
    subject = str(frame["subject"].iloc[0])
    block_num = int(frame["block_num"].iloc[0])
    is_go = pd.to_numeric(frame["is_no_go"], errors="coerce").eq(0)
    is_nogo = pd.to_numeric(frame["is_no_go"], errors="coerce").eq(1)
    response = pd.to_numeric(frame["response"], errors="coerce")
    omission = pd.to_numeric(frame["omission"], errors="coerce")
    commission = pd.to_numeric(frame["commission"], errors="coerce")
    raw_omission = is_go & omission.eq(1)
    raw_commission = is_nogo & commission.eq(1)
    prestim = frame["prestimulus_press_flag"].fillna(False)
    carry = frame["carryover_candidate_flag"].fillna(False)
    ambiguous = frame["ambiguous_omission_flag"].fillna(False)

    go_n = int(is_go.sum())
    nogo_n = int(is_nogo.sum())
    raw_omission_n = int(raw_omission.sum())
    raw_commission_n = int(raw_commission.sum())

    result: dict[str, Any] = {
        "subject": subject,
        "block_num": block_num,
        "trials": int(len(frame)),
        "go_trials": go_n,
        "nogo_trials": nogo_n,
        "probes": int(
            pd.to_numeric(frame["is_probe"], errors="coerce").eq(1).sum()
        ),
        "raw_omission_count": raw_omission_n,
        "raw_omission_rate_go": (
            float(raw_omission_n / go_n) if go_n else np.nan
        ),
        "commission_count": raw_commission_n,
        "commission_rate_nogo": (
            float(raw_commission_n / nogo_n) if nogo_n else np.nan
        ),
        "prestimulus_press_count": int(prestim.sum()),
        "multiple_keypress_count": int(
            frame["multiple_keypress_flag"].fillna(False).sum()
        ),
        "carryover_candidate_count": int(carry.sum()),
        "ambiguous_omission_count": int(ambiguous.sum()),
        "clean_omission_candidate_count": int(
            (raw_omission & ~ambiguous).sum()
        ),
        "prestimulus_only_omission_candidate_count": int(
            (raw_omission & prestim & ~carry).sum()
        ),
        "carryover_only_omission_candidate_count": int(
            (raw_omission & carry & ~prestim).sum()
        ),
        "prestimulus_and_carryover_omission_candidate_count": int(
            (raw_omission & prestim & carry).sum()
        ),
        "anticipatory_candidate_count": int(
            frame["anticipatory_candidate_flag"].fillna(False).sum()
        ),
    }
    result["clean_omission_candidate_rate_go"] = (
        float(result["clean_omission_candidate_count"] / go_n)
        if go_n
        else np.nan
    )
    result["ambiguous_omission_fraction_of_raw_omissions"] = (
        float(result["ambiguous_omission_count"] / raw_omission_n)
        if raw_omission_n
        else np.nan
    )

    rt = pd.to_numeric(frame["rt"], errors="coerce").to_numpy(dtype=float)
    result.update(_signal_stats(rt[np.isfinite(rt)], "rt_all_ms"))

    go_response_mask = is_go & response.eq(1) & pd.Series(
        np.isfinite(rt),
        index=frame.index,
    )
    go_rt = rt[go_response_mask.to_numpy(dtype=bool)]
    result.update(_signal_stats(go_rt, "go_response_rt_ms"))
    if go_rt.size > 1:
        go_rt_mean = float(np.mean(go_rt))
        go_rt_sd = float(np.std(go_rt, ddof=1))
        result["go_response_rt_cv"] = (
            go_rt_sd / go_rt_mean if go_rt_mean > 0 else np.nan
        )
    else:
        result["go_response_rt_cv"] = np.nan

    commission_rt_mask = raw_commission & pd.Series(
        np.isfinite(rt),
        index=frame.index,
    )
    result.update(
        _signal_stats(
            rt[commission_rt_mask.to_numpy(dtype=bool)],
            "nogo_commission_rt_ms",
        )
    )

    for threshold in (100, 150, 200, 900, 1000, 1150):
        direction = "lt" if threshold <= 200 else "gt"
        column = f"rt_candidate_{direction}_{threshold}_flag"
        result[f"{column}_count"] = int(
            frame[column].fillna(False).sum()
        )

    for column in ("probe_response", "probe_vigilance"):
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        for level in (1, 2, 3, 4):
            result[f"{column}_{level}_count"] = int(values.eq(level).sum())

    return result


def add_review_scores(qc: pd.DataFrame) -> pd.DataFrame:
    """Add continuous robust-z review scores; no threshold or exclusion flag is applied."""
    out = qc.copy()
    for column in (
        "pir_usable_fraction",
        "internal_coverage_fraction_estimate",
        "max_temporal_gap_sec",
        "roi_clipped_fraction",
        "ritnet_found_fraction",
        "ocular_fragmented_candidate_fraction",
        "pir_median",
        "pir_mad",
        "oar_median",
        "oar_mad",
        "oar_p90_median",
        "oar_p90_minus_median_median",
        "ocular_fraction_median",
        "iris_outer_fraction_median",
        "ocular_largest_component_fraction_median",
    ):
        if column not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        finite = values[np.isfinite(values)]
        if finite.empty:
            out[f"{column}_robust_z"] = np.nan
            continue
        median = float(finite.median())
        mad = float(
            np.median(
                np.abs(finite.to_numpy(dtype=float) - median)
            )
        )
        scale = 1.4826 * mad
        out[f"{column}_robust_z"] = (
            (values - median) / scale if scale > 0 else np.nan
        )
    return out
