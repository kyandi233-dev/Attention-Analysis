from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CANDIDATE_GROUP = ["signal", "cleaning_track", "buffer_id", "bin_width_sec"]


def _require(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _q(series: pd.Series, value: float) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna()
    return float(x.quantile(value)) if len(x) else np.nan


def _median(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna()
    return float(x.median()) if len(x) else np.nan


def _mean(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna()
    return float(x.mean()) if len(x) else np.nan


def _spearman(a: pd.Series, b: pd.Series) -> tuple[int, float]:
    left = pd.to_numeric(a, errors="coerce")
    right = pd.to_numeric(b, errors="coerce")
    mask = np.isfinite(left) & np.isfinite(right)
    n = int(mask.sum())
    if n < 3 or left[mask].nunique() < 2 or right[mask].nunique() < 2:
        return n, np.nan
    return n, float(left[mask].corr(right[mask], method="spearman"))


def validate_probe_candidate_bin_counts(frame: pd.DataFrame) -> None:
    """Fail if a per-signal candidate row reports more valid bins than can exist.

    This guard is deliberately applied before any aggregation so signal rows cannot be
    accidentally summed into values such as 90 valid 1-second bins for a 30-second window.
    """
    _require(frame, [*CANDIDATE_GROUP, "window_sec", "n_valid_bins"], "probe candidates")
    window = pd.to_numeric(frame["window_sec"], errors="coerce")
    width = pd.to_numeric(frame["bin_width_sec"], errors="coerce")
    valid_bins = pd.to_numeric(frame["n_valid_bins"], errors="coerce")
    bad_numeric = ~np.isfinite(window) | ~np.isfinite(width) | width.le(0) | window.le(0)
    if bad_numeric.any():
        raise ValueError("probe candidates contain invalid window_sec/bin_width_sec")
    defined = np.ceil((window / width) - 1e-12).astype(int)
    bad = np.isfinite(valid_bins) & (valid_bins > defined)
    if bad.any():
        example = frame.loc[bad, [*CANDIDATE_GROUP, "window_sec", "n_valid_bins"]].head(5)
        raise ValueError(
            "n_valid_bins exceeds the per-signal defined-bin count; aggregation likely mixed signals: "
            + example.to_dict("records").__repr__()
        )


def summarize_probe_bin_support(frame: pd.DataFrame) -> pd.DataFrame:
    validate_probe_candidate_bin_counts(frame)
    _require(
        frame,
        [
            "session_id", "valid_fraction", "linear_status", "quadratic_status",
            "has_early_support", "has_late_support",
        ],
        "probe candidates",
    )
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(CANDIDATE_GROUP, sort=True, dropna=False):
        window = float(pd.to_numeric(group["window_sec"], errors="coerce").iloc[0])
        width = float(key[-1])
        defined = int(np.ceil((window / width) - 1e-12))
        n_valid_bins = pd.to_numeric(group["n_valid_bins"], errors="coerce")
        rows.append(
            {
                **dict(zip(CANDIDATE_GROUP, key)),
                "window_sec": window,
                "defined_bin_n": defined,
                "candidate_row_n": int(len(group)),
                "session_n": int(group["session_id"].astype(str).nunique()),
                "valid_fraction_median": _median(group["valid_fraction"]),
                "valid_fraction_p05": _q(group["valid_fraction"], 0.05),
                "valid_fraction_p95": _q(group["valid_fraction"], 0.95),
                "n_valid_bins_min": float(n_valid_bins.min()) if n_valid_bins.notna().any() else np.nan,
                "n_valid_bins_p05": _q(n_valid_bins, 0.05),
                "n_valid_bins_p25": _q(n_valid_bins, 0.25),
                "n_valid_bins_median": _median(n_valid_bins),
                "n_valid_bins_p75": _q(n_valid_bins, 0.75),
                "n_valid_bins_p95": _q(n_valid_bins, 0.95),
                "n_valid_bins_max": float(n_valid_bins.max()) if n_valid_bins.notna().any() else np.nan,
                "full_bin_support_fraction": float(n_valid_bins.eq(defined).mean()),
                "zero_bin_support_fraction": float(n_valid_bins.fillna(0).eq(0).mean()),
                "linear_computable_fraction": float(group["linear_status"].astype(str).eq("computable").mean()),
                "quadratic_computable_fraction": float(group["quadratic_status"].astype(str).eq("computable").mean()),
                "early_support_fraction": float(group["has_early_support"].fillna(False).astype(bool).mean()),
                "late_support_fraction": float(group["has_late_support"].fillna(False).astype(bool).mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_level_variability_agreement(frame: pd.DataFrame) -> pd.DataFrame:
    _require(
        frame,
        [*CANDIDATE_GROUP, "level_mean", "level_median", "variability_sd", "variability_mad", "variability_iqr"],
        "probe candidates",
    )
    pairs = [
        ("level_mean", "level_median"),
        ("variability_sd", "variability_mad"),
        ("variability_sd", "variability_iqr"),
        ("variability_mad", "variability_iqr"),
    ]
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(CANDIDATE_GROUP, sort=True, dropna=False):
        base = dict(zip(CANDIDATE_GROUP, key))
        for left_name, right_name in pairs:
            n_pair, rho = _spearman(group[left_name], group[right_name])
            left = pd.to_numeric(group[left_name], errors="coerce")
            right = pd.to_numeric(group[right_name], errors="coerce")
            mask = np.isfinite(left) & np.isfinite(right)
            abs_diff = (left[mask] - right[mask]).abs()
            rows.append(
                {
                    **base,
                    "representation_a": left_name,
                    "representation_b": right_name,
                    "n_pair": n_pair,
                    "spearman_rho": rho,
                    "absolute_difference_median": float(abs_diff.median()) if len(abs_diff) else np.nan,
                    "absolute_difference_p95": float(abs_diff.quantile(0.95)) if len(abs_diff) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _probe_identity_columns(frame: pd.DataFrame) -> list[str]:
    columns = ["session_id"]
    if "probe_event_id" in frame.columns and frame["probe_event_id"].notna().any():
        columns.append("probe_event_id")
    elif "probe_index_global" in frame.columns and frame["probe_index_global"].notna().any():
        columns.append("probe_index_global")
    else:
        for name in ("block_num", "probe_onset_ms"):
            if name in frame.columns:
                columns.append(name)
    if len(columns) == 1:
        raise ValueError("probe candidates need a probe identity column for bin-width stability")
    return columns


def summarize_bin_width_stability(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare dynamic estimates across candidate bin widths without mixing signals."""
    _require(frame, [*CANDIDATE_GROUP, "linear_slope_per_sec", "quadratic_curvature_per_sec2"], "probe candidates")
    identity = _probe_identity_columns(frame)
    base_group = ["signal", "cleaning_track", "buffer_id"]
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(base_group, sort=True, dropna=False):
        widths = sorted(pd.to_numeric(group["bin_width_sec"], errors="coerce").dropna().unique())
        for width_a, width_b in combinations(widths, 2):
            a = group[pd.to_numeric(group["bin_width_sec"], errors="coerce").eq(width_a)]
            b = group[pd.to_numeric(group["bin_width_sec"], errors="coerce").eq(width_b)]
            if a.duplicated(identity).any() or b.duplicated(identity).any():
                raise ValueError(
                    f"duplicate probe identity within bin-width comparison for {key}: identity={identity}"
                )
            paired = a[identity + ["linear_slope_per_sec", "quadratic_curvature_per_sec2"]].merge(
                b[identity + ["linear_slope_per_sec", "quadratic_curvature_per_sec2"]],
                on=identity,
                how="inner",
                suffixes=("__a", "__b"),
                validate="one_to_one",
            )
            base = dict(zip(base_group, key))
            for metric in ("linear_slope_per_sec", "quadratic_curvature_per_sec2"):
                n_pair, rho = _spearman(paired[f"{metric}__a"], paired[f"{metric}__b"])
                left = pd.to_numeric(paired[f"{metric}__a"], errors="coerce")
                right = pd.to_numeric(paired[f"{metric}__b"], errors="coerce")
                mask = np.isfinite(left) & np.isfinite(right)
                diff = (left[mask] - right[mask]).abs()
                rows.append(
                    {
                        **base,
                        "bin_width_a_sec": float(width_a),
                        "bin_width_b_sec": float(width_b),
                        "metric": metric,
                        "n_pair": n_pair,
                        "spearman_rho": rho,
                        "absolute_difference_median": float(diff.median()) if len(diff) else np.nan,
                        "absolute_difference_p95": float(diff.quantile(0.95)) if len(diff) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def summarize_binocular_sources(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate source counts within measurement_state; never sum fractions across states."""
    _require(
        frame,
        ["session_id", "signal", "measurement_state", "source_mode", "n_timepoints"],
        "binocular source audit",
    )
    work = frame.copy()
    work["n_timepoints"] = pd.to_numeric(work["n_timepoints"], errors="coerce")
    if work["n_timepoints"].isna().any() or work["n_timepoints"].lt(0).any():
        raise ValueError("binocular source audit contains invalid n_timepoints")
    group_cols = ["signal", "measurement_state", "source_mode"]
    counts = (
        work.groupby(group_cols, sort=True, dropna=False)
        .agg(n_timepoints=("n_timepoints", "sum"), session_n=("session_id", "nunique"))
        .reset_index()
    )
    denom = counts.groupby(["signal", "measurement_state"], dropna=False)["n_timepoints"].transform("sum")
    counts["fraction"] = np.where(denom.gt(0), counts["n_timepoints"] / denom, np.nan)
    totals = counts.groupby(["signal", "measurement_state"], dropna=False)["fraction"].sum(min_count=1)
    bad = totals.dropna()[~np.isclose(totals.dropna(), 1.0, atol=1e-9)]
    if len(bad):
        raise ValueError(f"binocular source fractions do not sum to 1 within measurement_state: {bad.to_dict()}")
    return counts


def summarize_buffer_loss(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    _require(frame, ["session_id", "buffer_id", "signal", "finite_lost_fraction"], "blink buffer loss")
    optional = [name for name in ("pre_buffer_ms", "post_buffer_ms") if name in frame.columns]
    group_cols = ["signal", "buffer_id", *optional]
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(group_cols, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "session_n": int(group["session_id"].astype(str).nunique()),
                "finite_lost_fraction_mean": _mean(group["finite_lost_fraction"]),
                "finite_lost_fraction_median": _median(group["finite_lost_fraction"]),
                "finite_lost_fraction_p95": _q(group["finite_lost_fraction"], 0.95),
                "finite_lost_fraction_max": float(pd.to_numeric(group["finite_lost_fraction"], errors="coerce").max()),
            }
        )
    return pd.DataFrame(rows)


def summarize_blink_recovery(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    _require(
        frame,
        [
            "session_id", "signal", "anchor", "relative_bin_center_ms",
            "raw_computable_fraction", "nir_qc_valid_fraction",
            "raw_value_median", "nir_qc_value_median",
        ],
        "blink recovery bins",
    )
    group_cols = ["signal", "anchor", "relative_bin_center_ms"]
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(group_cols, sort=True, dropna=False):
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "event_bin_row_n": int(len(group)),
                "session_n": int(group["session_id"].astype(str).nunique()),
                "raw_computable_fraction_median": _median(group["raw_computable_fraction"]),
                "raw_computable_fraction_p25": _q(group["raw_computable_fraction"], 0.25),
                "raw_computable_fraction_p75": _q(group["raw_computable_fraction"], 0.75),
                "nir_qc_valid_fraction_median": _median(group["nir_qc_valid_fraction"]),
                "nir_qc_valid_fraction_p25": _q(group["nir_qc_valid_fraction"], 0.25),
                "nir_qc_valid_fraction_p75": _q(group["nir_qc_valid_fraction"], 0.75),
                "raw_value_median_across_events": _median(group["raw_value_median"]),
                "nir_qc_value_median_across_events": _median(group["nir_qc_value_median"]),
            }
        )
    return pd.DataFrame(rows)


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def build_g1_decision_summaries(audit_root: str | Path) -> dict[str, pd.DataFrame]:
    """Build Q1-blind measurement summaries from an existing G1 audit output directory."""
    root = Path(audit_root).expanduser().resolve()
    candidates = _read_csv_or_empty(root / "probe_measurement_candidates.csv")
    binocular = _read_csv_or_empty(root / "binocular_source_mode_audit.csv")
    if candidates.empty:
        raise ValueError("probe_measurement_candidates.csv is missing or empty")
    if binocular.empty:
        raise ValueError("binocular_source_mode_audit.csv is missing or empty")
    buffer_loss = _read_csv_or_empty(root / "blink_buffer_loss_audit.csv")
    recovery = _read_csv_or_empty(root / "blink_recovery_bins.csv")
    return {
        "g1_signal_bin_support_summary.csv": summarize_probe_bin_support(candidates),
        "g1_bin_width_stability.csv": summarize_bin_width_stability(candidates),
        "g1_level_variability_agreement.csv": summarize_level_variability_agreement(candidates),
        "g1_binocular_source_summary.csv": summarize_binocular_sources(binocular),
        "g1_buffer_loss_summary.csv": summarize_buffer_loss(buffer_loss),
        "g1_blink_recovery_by_anchor.csv": summarize_blink_recovery(recovery),
    }
