from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .pupil_blink_measurement import GEOMETRY_SIGNAL, RSEG_HARD_SIGNAL


PAIR_METRICS: tuple[str, ...] = (
    "level_mean",
    "level_median",
    "variability_sd",
    "variability_mad",
    "variability_iqr",
    "linear_slope_per_sec",
    "quadratic_curvature_per_sec2",
)
DEFAULT_TEMPORAL_SPAN_GRID_SEC: tuple[float, ...] = (10.0, 15.0, 20.0, 25.0)


def _require(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _q(series: pd.Series, q: float) -> float:
    values = _numeric(series).dropna()
    return float(values.quantile(q)) if len(values) else np.nan


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
        raise ValueError("probe candidates need probe identity columns")
    return columns


def _spearman(left: pd.Series, right: pd.Series) -> tuple[int, float]:
    a = _numeric(left)
    b = _numeric(right)
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(mask.sum())
    if n < 3 or a[mask].nunique() < 2 or b[mask].nunique() < 2:
        return n, np.nan
    return n, float(a[mask].corr(b[mask], method="spearman"))


def summarize_cross_signal_representation(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare geometry and hard R_seg at matched probes/configurations.

    The two signals have different physical scales, so this audit deliberately reports
    rank agreement and matched availability rather than raw absolute differences or an
    identity-line agreement statistic.
    """
    required = [
        "session_id",
        "signal",
        "cleaning_track",
        "buffer_id",
        "bin_width_sec",
        *PAIR_METRICS,
    ]
    _require(frame, required, "probe candidates")
    identity = _probe_identity_columns(frame)
    config = ["cleaning_track", "buffer_id", "bin_width_sec"]
    work = frame[frame["signal"].astype(str).isin({GEOMETRY_SIGNAL, RSEG_HARD_SIGNAL})].copy()
    rows: list[dict[str, object]] = []

    for key, group in work.groupby(config, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        geometry = group[group["signal"].astype(str).eq(GEOMETRY_SIGNAL)].copy()
        rseg = group[group["signal"].astype(str).eq(RSEG_HARD_SIGNAL)].copy()
        if geometry.duplicated(identity).any() or rseg.duplicated(identity).any():
            raise ValueError(f"duplicate probe identity in cross-signal comparison: config={key}")
        paired = geometry[identity + list(PAIR_METRICS)].merge(
            rseg[identity + list(PAIR_METRICS)],
            on=identity,
            how="outer",
            suffixes=("__geometry", "__rseg"),
            indicator=True,
            validate="one_to_one",
        )
        base = dict(zip(config, key))
        for metric in PAIR_METRICS:
            left = _numeric(paired[f"{metric}__geometry"])
            right = _numeric(paired[f"{metric}__rseg"])
            left_ok = np.isfinite(left)
            right_ok = np.isfinite(right)
            both = left_ok & right_ok
            union = left_ok | right_ok
            n_pair, rho = _spearman(left, right)
            sign_agreement = np.nan
            if metric in {"linear_slope_per_sec", "quadratic_curvature_per_sec2"} and both.any():
                nonzero = both & left.ne(0) & right.ne(0)
                if nonzero.any():
                    sign_agreement = float(np.sign(left[nonzero]).eq(np.sign(right[nonzero])).mean())
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "geometry_available_n": int(left_ok.sum()),
                    "rseg_available_n": int(right_ok.sum()),
                    "paired_available_n": int(both.sum()),
                    "available_union_n": int(union.sum()),
                    "paired_fraction_of_union": float(both.sum() / union.sum()) if union.any() else np.nan,
                    "spearman_rho": rho,
                    "dynamic_sign_agreement_fraction": sign_agreement,
                    "agreement_interpretation": "cross_representation_rank_agreement_not_same_scale_identity",
                }
            )
    return pd.DataFrame(rows)


def summarize_temporal_support(
    frame: pd.DataFrame,
    *,
    span_grid_sec: Sequence[float] = DEFAULT_TEMPORAL_SPAN_GRID_SEC,
) -> pd.DataFrame:
    """Summarize observed time support and descriptive retention at candidate spans.

    The span grid is a sensitivity grid only. This function never declares a formal
    threshold and never uses Q1/Q2 or prediction performance.
    """
    required = [
        "session_id",
        "signal",
        "cleaning_track",
        "buffer_id",
        "bin_width_sec",
        "n_valid_bins",
        "t_min_sec",
        "t_max_sec",
        "temporal_span_sec",
        "has_early_support",
        "has_late_support",
        "linear_status",
        "quadratic_status",
    ]
    _require(frame, required, "probe candidates")
    group_cols = ["signal", "cleaning_track", "buffer_id", "bin_width_sec"]
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(group_cols, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        span = _numeric(group["temporal_span_sec"])
        early = group["has_early_support"].fillna(False).astype(bool)
        late = group["has_late_support"].fillna(False).astype(bool)
        both_halves = early & late
        linear = group["linear_status"].astype(str).eq("computable")
        quadratic = group["quadratic_status"].astype(str).eq("computable")
        finite_span = np.isfinite(span)
        base = {
            **dict(zip(group_cols, key)),
            "candidate_row_n": int(len(group)),
            "session_n": int(group["session_id"].astype(str).nunique()),
            "temporal_span_p01_sec": _q(span, 0.01),
            "temporal_span_p05_sec": _q(span, 0.05),
            "temporal_span_p10_sec": _q(span, 0.10),
            "temporal_span_p25_sec": _q(span, 0.25),
            "temporal_span_median_sec": _q(span, 0.50),
            "temporal_span_p75_sec": _q(span, 0.75),
            "both_halves_support_fraction": float(both_halves.mean()),
            "linear_computable_fraction": float(linear.mean()),
            "quadratic_computable_fraction": float(quadratic.mean()),
            "linear_with_both_halves_fraction": float((linear & both_halves).mean()),
            "quadratic_with_both_halves_fraction": float((quadratic & both_halves).mean()),
        }
        for threshold in span_grid_sec:
            label = str(float(threshold)).rstrip("0").rstrip(".").replace(".", "p")
            support = finite_span & span.ge(float(threshold)) & both_halves
            base[f"support_span_ge_{label}s_fraction"] = float(support.mean())
            base[f"linear_span_ge_{label}s_fraction"] = float((support & linear).mean())
            base[f"quadratic_span_ge_{label}s_fraction"] = float((support & quadratic).mean())
        rows.append(base)
    return pd.DataFrame(rows)


def summarize_sync_semantics(frame: pd.DataFrame) -> pd.DataFrame:
    """Separate clock-boundary evidence from coverage-gap diagnostics.

    No automatic pass/fail threshold is introduced here. Nearest-frame residuals can
    become very large when one device has a coverage gap even if session boundaries are
    well aligned, so they must not be interpreted as clock offset by themselves.
    """
    if frame.empty:
        return pd.DataFrame()
    _require(frame, ["session_id", "sync_status"], "RGB-NIR sync audit")
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        available = bool(row.get("rgb_blink_source_available", row.get("sync_status") != "rgb_blink_source_unavailable"))
        start = pd.to_numeric(pd.Series([row.get("rgb_minus_nir_start_ms")]), errors="coerce").iloc[0]
        end = pd.to_numeric(pd.Series([row.get("rgb_minus_nir_end_ms")]), errors="coerce").iloc[0]
        nir_gap = pd.to_numeric(pd.Series([row.get("nir_frame_large_gap_n")]), errors="coerce").iloc[0]
        rgb_gap = pd.to_numeric(pd.Series([row.get("rgb_frame_large_gap_n")]), errors="coerce").iloc[0]
        boundary_available = bool(np.isfinite(start) and np.isfinite(end))
        rows.append(
            {
                "session_id": str(row["session_id"]),
                "rgb_blink_source_available": available,
                "legacy_sync_status": str(row.get("sync_status", "")),
                "sync_evidence_level": str(row.get("sync_evidence_level", "")),
                "clock_boundary_evidence_status": "available_descriptive_only" if boundary_available else "unavailable",
                "rgb_minus_nir_start_ms": float(start) if np.isfinite(start) else np.nan,
                "rgb_minus_nir_end_ms": float(end) if np.isfinite(end) else np.nan,
                "clock_boundary_max_abs_ms": float(max(abs(start), abs(end))) if boundary_available else np.nan,
                "clock_boundary_change_ms": float(end - start) if boundary_available else np.nan,
                "nir_large_gap_n": float(nir_gap) if np.isfinite(nir_gap) else np.nan,
                "rgb_large_gap_n": float(rgb_gap) if np.isfinite(rgb_gap) else np.nan,
                "nir_coverage_gap_status": (
                    "large_gap_present" if np.isfinite(nir_gap) and nir_gap > 0
                    else "no_large_gap_detected" if np.isfinite(nir_gap)
                    else "unavailable"
                ),
                "rgb_coverage_gap_status": (
                    "large_gap_present" if np.isfinite(rgb_gap) and rgb_gap > 0
                    else "no_large_gap_detected" if np.isfinite(rgb_gap)
                    else "unavailable"
                ),
                "event_boundary_residual_abs_p95_ms": row.get("nearest_residual_abs_p95_ms", np.nan),
                "frame_nearest_residual_abs_p95_ms": row.get("frame_nearest_residual_abs_p95_ms", np.nan),
                "frame_nearest_residual_abs_max_ms": row.get("frame_nearest_residual_abs_max_ms", np.nan),
                "interpretation": "clock_alignment_and_coverage_gaps_are_separate_diagnostics_no_auto_pass_fail",
            }
        )
    return pd.DataFrame(rows)


def summarize_source_mode_limitation(frame: pd.DataFrame) -> pd.DataFrame:
    """Condense source-mode composition and make the unresolved numerical-bias limit explicit."""
    if frame.empty:
        return pd.DataFrame()
    _require(frame, ["session_id", "signal", "measurement_state", "source_mode", "n_timepoints"], "binocular source audit")
    work = frame.copy()
    work["n_timepoints"] = _numeric(work["n_timepoints"])
    rows: list[dict[str, object]] = []
    for key, group in work.groupby(["signal", "measurement_state"], sort=True, dropna=False):
        total = float(group["n_timepoints"].sum())
        by_mode = group.groupby("source_mode", dropna=False)["n_timepoints"].sum().to_dict()
        left = float(by_mode.get("left_only", 0.0))
        right = float(by_mode.get("right_only", 0.0))
        binocular = float(by_mode.get("binocular", 0.0))
        missing = float(by_mode.get("missing", 0.0))
        rows.append(
            {
                "signal": key[0],
                "measurement_state": key[1],
                "session_n": int(group["session_id"].astype(str).nunique()),
                "binocular_fraction": binocular / total if total else np.nan,
                "single_eye_fraction": (left + right) / total if total else np.nan,
                "left_only_fraction": left / total if total else np.nan,
                "right_only_fraction": right / total if total else np.nan,
                "missing_fraction": missing / total if total else np.nan,
                "numeric_source_mode_bias_estimable_from_existing_g1_tables": False,
                "required_interpretation": "source_mode_composition_qc_only_carry_forward_as_sensitivity_limit",
            }
        )
    return pd.DataFrame(rows)


def _read_csv(root: Path, name: str, *, required: bool = True) -> pd.DataFrame:
    path = root / name
    if not path.exists() or path.stat().st_size == 0:
        if required:
            raise FileNotFoundError(f"required G1 audit table missing or empty: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        if required:
            raise ValueError(f"required G1 audit table has no columns: {path}")
        return pd.DataFrame()


def build_ocular_g1_freeze_support(audit_root: str | Path) -> dict[str, pd.DataFrame]:
    """Build supplemental Q1-blind freeze evidence from an existing G1 output.

    This reads only already-produced audit CSVs. It does not reopen NIR frame sources,
    rerun YOLO/RITnet, regenerate blink events, fit Q1/Q2 models, or mutate a feature registry.
    """
    root = Path(audit_root).expanduser().resolve()
    candidates = _read_csv(root, "probe_measurement_candidates.csv")
    sync = _read_csv(root, "rgb_nir_sync_audit.csv")
    source_mode = _read_csv(root, "binocular_source_mode_audit.csv")
    return {
        "g1_cross_signal_representation_summary.csv": summarize_cross_signal_representation(candidates),
        "g1_temporal_support_freeze_grid.csv": summarize_temporal_support(candidates),
        "g1_sync_semantics_split.csv": summarize_sync_semantics(sync),
        "g1_source_mode_limit_summary.csv": summarize_source_mode_limitation(source_mode),
    }
