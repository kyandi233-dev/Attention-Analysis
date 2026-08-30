from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

from attention_pipeline.config import Config
from attention_pipeline.nir_behavior.contract import normalize_subject
from attention_pipeline.nir_behavior.discovery import resolve_repo_path
from attention_pipeline.behavior_formal.metrics import fit_exgaussian

from .analysis import omission_qc_type, trial_outcome_label


DYNAMIC_PIR_FEATURES = (
    "pir_median",
    "pir_mean",
    "pir_mad",
    "pir_iqr",
    "pir_sd",
    "pir_p10",
    "pir_p90",
    "pir_slope_per_sec",
    "pir_diff_mad",
    "pir_diff_rate_mad_per_sec",
)

SOURCE_MODE_COLUMNS = (
    "source_mode_binocular_fraction",
    "source_mode_left_only_fraction",
    "source_mode_right_only_fraction",
    "source_mode_missing_fraction",
)

COVERAGE_COLUMNS = (
    "pir_valid_fraction_median",
    "available_duration_fraction_median",
    "internal_coverage_fraction_median",
    "boundary_truncated_fraction",
    "max_temporal_gap_sec_p95",
)

VISUAL_METRICS = (
    "central_rel_lum_mean",
    "central_rms_contrast",
    "fruit_support_rel_lum_mean",
    "fruit_support_rms_contrast",
    "fruit_visible_area_fraction_central_roi",
    "delta_central_rel_lum_vs_background",
    "delta_central_rel_lum_vs_mask",
)

_PRE_WINDOW_RE = re.compile(r"^pre_(\d+(?:\.\d+)?)s$")


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _bool_flag(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    series = frame[column]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes", "y"})
    )


def _median_mad(values: pd.Series) -> tuple[float, float]:
    x = _numeric(values).dropna().to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    med = float(np.median(x))
    return med, float(np.median(np.abs(x - med)))


def _corrected_rate(successes: int, opportunities: int) -> float:
    if opportunities <= 0:
        return np.nan
    return float((successes + 0.5) / (opportunities + 1.0))


def _sdt(hit_rate: float, fa_rate: float) -> dict[str, float]:
    if not (np.isfinite(hit_rate) and np.isfinite(fa_rate)):
        return {"dprime": np.nan, "criterion_c": np.nan, "beta": np.nan}
    z_h = stats.norm.ppf(hit_rate)
    z_f = stats.norm.ppf(fa_rate)
    dprime = z_h - z_f
    c = -(z_h + z_f) / 2.0
    denom = stats.norm.pdf(z_f)
    beta = stats.norm.pdf(z_h) / denom if denom > 0 else np.nan
    return {"dprime": float(dprime), "criterion_c": float(c), "beta": float(beta)}


def window_duration_sec(name: Any) -> float | None:
    match = _PRE_WINDOW_RE.match(str(name).strip())
    return float(match.group(1)) if match else None


def advanced_behavior_summary(trials: pd.DataFrame) -> pd.DataFrame:
    """Subject×Block behavior summary for high-order SART indices.

    This complements, rather than replaces, the program-scoring and QC-aware
    omission summaries in analysis.py.
    """
    df = trials.copy()
    for col in ("is_no_go", "correct", "commission", "omission", "rt"):
        if col in df.columns:
            df[col] = _numeric(df[col])
    df["omission_qc_type"] = omission_qc_type(df)

    rows: list[dict[str, Any]] = []
    for (subject, block_num), block in df.groupby(["subject", "block_num"], sort=True):
        go = block[block["is_no_go"].eq(0)].copy()
        nogo = block[block["is_no_go"].eq(1)].copy()
        go_rt = _numeric(go.loc[go["correct"].eq(1), "rt"]).dropna()
        go_rt = go_rt[np.isfinite(go_rt)]

        rt_mean = float(go_rt.mean()) if len(go_rt) else np.nan
        rt_sd = float(go_rt.std(ddof=1)) if len(go_rt) > 1 else np.nan
        rt_med, rt_mad = _median_mad(go_rt)
        rt_cv = rt_sd / rt_mean if len(go_rt) >= 2 and np.isfinite(rt_sd) and rt_mean > 0 else np.nan
        exg = fit_exgaussian(go_rt)

        hits = int(go["correct"].eq(1).sum())
        false_alarms = int(nogo["commission"].eq(1).sum())
        hit_rate = _corrected_rate(hits, len(go))
        fa_rate = _corrected_rate(false_alarms, len(nogo))
        sdt = _sdt(hit_rate, fa_rate)

        subtype = go["omission_qc_type"].astype(str)
        clean = int(subtype.eq("clean_omission").sum())
        ambiguous = int(
            subtype.isin(
                {
                    "prestimulus_associated_omission",
                    "carryover_associated_omission",
                    "prestimulus_and_carryover_associated_omission",
                }
            ).sum()
        )

        rows.append(
            {
                "subject": subject,
                "block_num": int(block_num),
                "n_trials": int(len(block)),
                "n_go": int(len(go)),
                "n_nogo": int(len(nogo)),
                "go_hit_n": hits,
                "nogo_false_alarm_n": false_alarms,
                "hit_rate_loglinear": hit_rate,
                "false_alarm_rate_loglinear": fa_rate,
                **sdt,
                "go_rt_n": int(len(go_rt)),
                "go_rt_median_ms": rt_med,
                "go_rt_mean_ms": rt_mean,
                "go_rt_sd_ms": rt_sd,
                "go_rt_mad_ms": rt_mad,
                "go_rt_iqr_ms": float(go_rt.quantile(0.75) - go_rt.quantile(0.25)) if len(go_rt) else np.nan,
                "go_rt_p10_ms": float(go_rt.quantile(0.10)) if len(go_rt) else np.nan,
                "go_rt_p90_ms": float(go_rt.quantile(0.90)) if len(go_rt) else np.nan,
                "rt_cv": rt_cv,
                "go_rt_lt_100_rate": float(go_rt.lt(100).mean()) if len(go_rt) else np.nan,
                "go_rt_lt_150_rate": float(go_rt.lt(150).mean()) if len(go_rt) else np.nan,
                "go_rt_lt_200_rate": float(go_rt.lt(200).mean()) if len(go_rt) else np.nan,
                "commission_rate": float(nogo["commission"].mean()) if len(nogo) else np.nan,
                "program_omission_rate": float(go["omission"].mean()) if len(go) else np.nan,
                "clean_omission_rate": clean / len(go) if len(go) else np.nan,
                "ambiguous_omission_rate": ambiguous / len(go) if len(go) else np.nan,
                "anticipatory_candidate_rate": float(_bool_flag(go, "anticipatory_candidate_flag").mean()) if len(go) else np.nan,
                "multiple_keypress_rate": float(_bool_flag(block, "multiple_keypress_flag").mean()) if len(block) else np.nan,
                "exg_mu": exg.get("exg_mu"),
                "exg_sigma": exg.get("exg_sigma"),
                "exg_tau": exg.get("exg_tau"),
                "exg_n": exg.get("exg_n"),
            }
        )
    return pd.DataFrame(rows)


def _join_trial_behavior(trial_level: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    behavior = trial_level.copy()
    behavior["outcome"] = trial_outcome_label(behavior)
    behavior["omission_qc_type"] = omission_qc_type(behavior)
    keys = [
        col
        for col in ("subject", "block_num", "trial_num", "global_trial_index")
        if col in behavior.columns and col in windows.columns
    ]
    keep = keys + [
        col
        for col in (
            "outcome",
            "omission_qc_type",
            "is_no_go",
            "correct",
            "commission",
            "omission",
            "rt",
            "time_in_block_sec",
            "stimulus_name",
            "stimulus_size",
            "prev_stimulus_name",
            "prev_stimulus_size",
        )
        if col in behavior.columns
    ]
    return windows.merge(behavior[keep], on=keys, how="left", validate="many_to_one")


def trial_dynamic_feature_long(
    trial_level: pd.DataFrame,
    trial_windows: pd.DataFrame,
    *,
    tracks: Iterable[str],
    window_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    wanted_tracks = {str(x) for x in tracks}
    df = trial_windows[trial_windows["track"].astype(str).isin(wanted_tracks)].copy()
    if window_names is not None:
        wanted_windows = {str(x) for x in window_names}
        df = df[df["window_name"].astype(str).isin(wanted_windows)].copy()
    df = _join_trial_behavior(trial_level, df)

    features = [col for col in DYNAMIC_PIR_FEATURES if col in df.columns]
    if not features or df.empty:
        return pd.DataFrame(
            columns=[
                "subject",
                "block_num",
                "track",
                "window_name",
                "outcome",
                "feature",
                "value",
            ]
        )

    id_cols = ["subject", "block_num", "track", "window_name", "outcome"]
    subject_level = (
        df[id_cols + features]
        .groupby(id_cols, as_index=False)
        .median(numeric_only=True)
    )
    return subject_level.melt(
        id_vars=id_cols,
        value_vars=features,
        var_name="feature",
        value_name="value",
    )


def probe_dynamic_feature_long(
    probe_windows: pd.DataFrame,
    *,
    tracks: Iterable[str],
) -> pd.DataFrame:
    wanted_tracks = {str(x) for x in tracks}
    df = probe_windows[probe_windows["track"].astype(str).isin(wanted_tracks)].copy()
    if df.empty:
        return pd.DataFrame()
    features = [col for col in DYNAMIC_PIR_FEATURES if col in df.columns]
    ids = [
        col
        for col in (
            "subject",
            "block_num",
            "track",
            "window_name",
            "probe_index_global",
            "probe_index_in_block",
            "probe_response",
            "probe_vigilance",
        )
        if col in df.columns
    ]
    if not features:
        return pd.DataFrame(columns=[*ids, "feature", "value"])
    result = df[ids + features].copy()
    return result.melt(
        id_vars=ids,
        value_vars=features,
        var_name="feature",
        value_name="value",
    )


def trial_multiscale_trajectory(
    trial_level: pd.DataFrame,
    trial_windows: pd.DataFrame,
    *,
    track: str,
    feature: str = "pir_median",
) -> pd.DataFrame:
    if feature not in trial_windows.columns:
        return pd.DataFrame()
    df = trial_windows[trial_windows["track"].astype(str).eq(track)].copy()
    df["seconds_before_event"] = df["window_name"].map(window_duration_sec)
    df = df[df["seconds_before_event"].notna()].copy()
    df = _join_trial_behavior(trial_level, df)
    df[feature] = _numeric(df[feature])
    return (
        df.groupby(
            ["subject", "block_num", "outcome", "window_name", "seconds_before_event"],
            as_index=False,
        )[feature]
        .median()
    )


def nogo_precursor_trajectory(
    trial_level: pd.DataFrame,
    trial_windows: pd.DataFrame,
    *,
    track: str,
    window_name: str,
    n_preceding_go: int,
) -> pd.DataFrame:
    pir = trial_windows[
        trial_windows["track"].astype(str).eq(track)
        & trial_windows["window_name"].astype(str).eq(window_name)
    ].copy()
    keys = [
        col
        for col in ("subject", "block_num", "trial_num", "global_trial_index")
        if col in pir.columns and col in trial_level.columns
    ]
    pir = pir[keys + [col for col in ("pir_median", "pir_slope_per_sec", "pir_mad") if col in pir.columns]]
    df = trial_level.merge(pir, on=keys, how="left", validate="one_to_one")
    for col in ("is_no_go", "correct", "commission", "rt"):
        if col in df.columns:
            df[col] = _numeric(df[col])
    df = df.sort_values(["subject", "block_num", "trial_num"]).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for (subject, block_num), block in df.groupby(["subject", "block_num"], sort=True):
        block = block.reset_index(drop=True)
        for idx, event in block[block["is_no_go"].eq(1)].iterrows():
            event_outcome = "commission" if event.get("commission") == 1 else "correct_inhibition"
            preceding = block.loc[: idx - 1]
            preceding = preceding[
                preceding["is_no_go"].eq(0) & preceding["correct"].eq(1)
            ].tail(int(n_preceding_go))
            n = len(preceding)
            for pos, (_, row) in enumerate(preceding.iterrows()):
                lag = -(n - pos)
                rows.append(
                    {
                        "subject": subject,
                        "block_num": int(block_num),
                        "event_trial_num": int(event["trial_num"]),
                        "event_outcome": event_outcome,
                        "lag": int(lag),
                        "source_trial_num": int(row["trial_num"]),
                        "go_rt_ms": row.get("rt"),
                        "pir_median": row.get("pir_median"),
                        "pir_slope_per_sec": row.get("pir_slope_per_sec"),
                        "pir_mad": row.get("pir_mad"),
                    }
                )
            rows.append(
                {
                    "subject": subject,
                    "block_num": int(block_num),
                    "event_trial_num": int(event["trial_num"]),
                    "event_outcome": event_outcome,
                    "lag": 0,
                    "source_trial_num": int(event["trial_num"]),
                    "go_rt_ms": np.nan,
                    "pir_median": event.get("pir_median"),
                    "pir_slope_per_sec": event.get("pir_slope_per_sec"),
                    "pir_mad": event.get("pir_mad"),
                }
            )
    return pd.DataFrame(rows)


def probe_rt_summary(probe_windows: pd.DataFrame, *, track: str) -> pd.DataFrame:
    df = probe_windows[probe_windows["track"].astype(str).eq(track)].copy()
    if df.empty:
        return pd.DataFrame()
    keys = [
        col
        for col in (
            "subject",
            "block_num",
            "probe_index_global",
            "probe_index_in_block",
            "probe_onset_ms",
        )
        if col in df.columns
    ]
    cols = keys + [
        col
        for col in (
            "probe_response",
            "probe_rt",
            "probe_vigilance",
            "probe_vigilance_rt",
        )
        if col in df.columns
    ]
    events = df[cols].drop_duplicates(keys).copy() if keys else df[cols].drop_duplicates().copy()
    for col in ("probe_response", "probe_rt", "probe_vigilance", "probe_vigilance_rt"):
        if col in events.columns:
            events[col] = _numeric(events[col])
    return events


def probe_behavior_multiscale(probe_windows: pd.DataFrame, *, track: str) -> pd.DataFrame:
    df = probe_windows[probe_windows["track"].astype(str).eq(track)].copy()
    if df.empty:
        return pd.DataFrame()
    keep = [
        col
        for col in (
            "subject",
            "block_num",
            "probe_index_global",
            "probe_index_in_block",
            "probe_response",
            "probe_vigilance",
            "window_name",
            "n_trials",
            "n_go",
            "n_nogo",
            "n_commission",
            "n_omission",
            "n_prestimulus_press",
            "n_ambiguous_omission",
            "n_anticipatory_candidate",
            "go_rt_median_ms",
            "go_rt_mad_ms",
            "go_rt_iqr_ms",
            "go_rt_cv",
        )
        if col in df.columns
    ]
    return df[keep].copy()


def track_robustness(
    trial_windows: pd.DataFrame,
    *,
    window_name: str,
    main_track: str,
    tracks: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wanted = {str(x) for x in tracks}
    df = trial_windows[
        trial_windows["window_name"].astype(str).eq(window_name)
        & trial_windows["track"].astype(str).isin(wanted)
    ].copy()
    if df.empty or "pir_median" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    keys = [
        col
        for col in ("subject", "block_num", "trial_num", "global_trial_index")
        if col in df.columns
    ]
    df["pir_median"] = _numeric(df["pir_median"])
    wide = df.pivot_table(index=keys, columns="track", values="pir_median", aggfunc="first")
    corr = wide.corr(min_periods=3)
    # ``stack(dropna=False)`` is rejected by the new pandas stack
    # implementation.  Melt the complete correlation matrix instead so NaN
    # comparisons remain explicit and the output schema is unchanged.
    corr_long = (
        corr.rename_axis(index="track_a", columns="track_b")
        .reset_index()
        .melt(id_vars="track_a", var_name="track_b", value_name="correlation")
    )

    agreement_rows: list[dict[str, Any]] = []
    if main_track in wide.columns:
        main = wide[main_track]
        for track in wide.columns:
            pair = pd.concat([main, wide[track]], axis=1, keys=["main", "other"]).dropna()
            agreement_rows.append(
                {
                    "main_track": main_track,
                    "comparison_track": str(track),
                    "n_pairs": int(len(pair)),
                    "pearson_r": float(pair["main"].corr(pair["other"])) if len(pair) >= 3 else np.nan,
                    "median_absolute_difference": float(np.median(np.abs(pair["main"] - pair["other"]))) if len(pair) else np.nan,
                    "median_signed_difference": float(np.median(pair["other"] - pair["main"])) if len(pair) else np.nan,
                }
            )
    return corr_long, pd.DataFrame(agreement_rows)


def source_mode_qc(trial_windows: pd.DataFrame) -> pd.DataFrame:
    columns = [col for col in SOURCE_MODE_COLUMNS if col in trial_windows.columns]
    if not columns:
        return pd.DataFrame()
    ids = ["subject", "block_num", "track", "window_name"]
    return (
        trial_windows[ids + columns]
        .groupby(ids, as_index=False)
        .mean(numeric_only=True)
    )


def multidimensional_coverage(
    trial_coverage: pd.DataFrame,
    probe_coverage: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for level, frame in (("trial", trial_coverage), ("probe", probe_coverage)):
        if frame.empty:
            continue
        current = frame.copy()
        current["analysis_level"] = level
        frames.append(current)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    metrics = [col for col in COVERAGE_COLUMNS if col in combined.columns]
    ids = [
        col
        for col in ("analysis_level", "subject", "block_num", "track", "window_name")
        if col in combined.columns
    ]
    return combined[ids + metrics].melt(
        id_vars=ids,
        value_vars=metrics,
        var_name="coverage_metric",
        value_name="coverage_value",
    )


def _resolve_optional_path(config: Config, key: str) -> Path | None:
    raw = config.section("paths").get(key)
    if raw in (None, ""):
        return None
    return resolve_repo_path(config, raw)


def load_visual_covariates(config: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _resolve_optional_path(config, "stimulus_visual_table")
    if path is None:
        return pd.DataFrame(), {"status": "unavailable", "reason": "path_not_configured"}
    if not path.is_file():
        return pd.DataFrame(), {"status": "unavailable", "reason": "file_missing", "path": str(path)}
    table = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = {"stimulus_name", "stimulus_size_pct"}
    missing = sorted(required - set(table.columns))
    if missing:
        return pd.DataFrame(), {"status": "unavailable", "reason": "schema_missing", "missing": missing, "path": str(path)}
    return table, {"status": "available", "path": str(path), "rows": int(len(table))}


def attach_visual_covariates(
    trial_level: pd.DataFrame,
    trial_windows: pd.DataFrame,
    visual: pd.DataFrame,
    *,
    track: str,
    window_name: str,
) -> pd.DataFrame:
    if visual.empty:
        return pd.DataFrame()
    windows = trial_windows[
        trial_windows["track"].astype(str).eq(track)
        & trial_windows["window_name"].astype(str).eq(window_name)
    ].copy()
    keys = [
        col
        for col in ("subject", "block_num", "trial_num", "global_trial_index")
        if col in windows.columns and col in trial_level.columns
    ]
    trial_keep = keys + [
        col
        for col in (
            "stimulus_name",
            "stimulus_size",
            "prev_stimulus_name",
            "prev_stimulus_size",
            "is_no_go",
            "correct",
            "commission",
            "omission",
            "rt",
            "time_in_block_sec",
        )
        if col in trial_level.columns
    ]
    merged = windows.merge(trial_level[trial_keep], on=keys, how="left", validate="one_to_one")

    if "prev_stimulus_name" not in merged.columns:
        merged["prev_stimulus_name"] = merged.groupby(["subject", "block_num"])["stimulus_name"].shift(1)
    if "prev_stimulus_size" not in merged.columns and "stimulus_size" in merged.columns:
        merged["prev_stimulus_size"] = merged.groupby(["subject", "block_num"])["stimulus_size"].shift(1)

    metrics = [col for col in VISUAL_METRICS if col in visual.columns]
    visual_keep = ["stimulus_name", "stimulus_size_pct", *metrics]
    current = visual[visual_keep].rename(
        columns={
            "stimulus_size_pct": "stimulus_size",
            **{col: f"current_{col}" for col in metrics},
        }
    )
    previous = visual[visual_keep].rename(
        columns={
            "stimulus_name": "prev_stimulus_name",
            "stimulus_size_pct": "prev_stimulus_size",
            **{col: f"previous_{col}" for col in metrics},
        }
    )
    merged["stimulus_size"] = _numeric(merged["stimulus_size"]) if "stimulus_size" in merged.columns else np.nan
    merged["prev_stimulus_size"] = _numeric(merged["prev_stimulus_size"])
    merged = merged.merge(current, on=["stimulus_name", "stimulus_size"], how="left", validate="many_to_one")
    merged = merged.merge(previous, on=["prev_stimulus_name", "prev_stimulus_size"], how="left", validate="many_to_one")
    return merged


def visual_covariate_correlation_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "pir_median" not in frame.columns:
        return pd.DataFrame()
    candidates = [col for col in frame.columns if col.startswith("current_") or col.startswith("previous_")]
    rows: list[dict[str, Any]] = []
    y = _numeric(frame["pir_median"])
    for col in candidates:
        x = _numeric(frame[col])
        ok = x.notna() & y.notna()
        if int(ok.sum()) < 5:
            continue
        rho, p = stats.spearmanr(x[ok], y[ok])
        rows.append(
            {
                "covariate": col,
                "n": int(ok.sum()),
                "spearman_rho_with_pir": float(rho),
                "p_value_validation_only": float(p),
            }
        )
    return pd.DataFrame(rows)


def raw_between_person_summary(
    config: Config,
    subjects: Iterable[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = _resolve_optional_path(config, "analysis_ready_root")
    if root is None:
        return pd.DataFrame(), {"status": "unavailable", "reason": "analysis_ready_root_not_configured"}

    rows: list[dict[str, Any]] = []
    missing_subjects: list[str] = []
    for subject_raw in subjects:
        subject = normalize_subject(subject_raw)
        path = root / "frame_level" / subject / f"{subject}_nir_analysis_ready.csv"
        if not path.is_file():
            missing_subjects.append(subject)
            continue
        header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
        required = {"subject", "block", "left_raw_PIR", "right_raw_PIR", "left_valid_primary", "right_valid_primary"}
        if not required.issubset(header.columns):
            missing_subjects.append(subject)
            continue
        df = pd.read_csv(path, usecols=sorted(required), encoding="utf-8-sig", low_memory=False)
        left = _numeric(df["left_raw_PIR"])
        right = _numeric(df["right_raw_PIR"])
        l_ok = _bool_flag(df, "left_valid_primary") & left.notna()
        r_ok = _bool_flag(df, "right_valid_primary") & right.notna()
        raw_binocular = np.select(
            [l_ok & r_ok, l_ok & ~r_ok, ~l_ok & r_ok],
            [(left + right) / 2.0, left, right],
            default=np.nan,
        )
        df["raw_binocular_PIR"] = raw_binocular
        record: dict[str, Any] = {
            "subject": subject,
            "raw_PIR_subject_median": float(np.nanmedian(raw_binocular)) if np.isfinite(raw_binocular).any() else np.nan,
            "raw_left_PIR_subject_median": float(left[l_ok].median()) if bool(l_ok.any()) else np.nan,
            "raw_right_PIR_subject_median": float(right[r_ok].median()) if bool(r_ok.any()) else np.nan,
            "raw_PIR_valid_fraction": float(np.isfinite(raw_binocular).mean()),
        }
        for block_num in sorted(_numeric(df["block"]).dropna().astype(int).unique()):
            values = df.loc[_numeric(df["block"]).eq(block_num), "raw_binocular_PIR"]
            record[f"block{int(block_num)}_raw_PIR_median"] = float(values.median()) if values.notna().any() else np.nan
        rows.append(record)

    status = {
        "status": "available" if rows else "unavailable",
        "root": str(root),
        "n_subjects": len(rows),
        "missing_or_schema_incompatible_subjects": missing_subjects,
        "semantics": "between-person raw PIR baseline characteristics; do not derive between-person effects from already-centered PIR",
    }
    return pd.DataFrame(rows), status


def subject_level_summary(
    advanced_behavior: pd.DataFrame,
    raw_pir: pd.DataFrame,
) -> pd.DataFrame:
    if advanced_behavior.empty and raw_pir.empty:
        return pd.DataFrame()
    behavior = advanced_behavior.copy()
    if not behavior.empty:
        numeric_cols = [col for col in behavior.select_dtypes(include=[np.number]).columns if col != "block_num"]
        behavior = behavior.groupby("subject", as_index=False)[numeric_cols].mean()
    if raw_pir.empty:
        return behavior
    if behavior.empty:
        return raw_pir.copy()
    return raw_pir.merge(behavior, on="subject", how="outer", validate="one_to_one")


def questionnaire_correlations(
    config: Config,
    subject_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _resolve_optional_path(config, "questionnaire_csv")
    if path is None:
        return pd.DataFrame(), {"status": "unavailable", "reason": "questionnaire_path_not_configured"}
    if not path.is_file():
        return pd.DataFrame(), {"status": "unavailable", "reason": "questionnaire_file_missing", "path": str(path)}
    q = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    subject_col = next((col for col in ("subject", "subject_id", "participant_id") if col in q.columns), None)
    if subject_col is None:
        return pd.DataFrame(), {"status": "unavailable", "reason": "no_subject_column", "path": str(path)}
    q = q.copy()
    q["subject"] = q[subject_col].map(normalize_subject)
    merged = subject_summary.merge(q, on="subject", how="inner")
    if merged.empty:
        return pd.DataFrame(), {"status": "unavailable", "reason": "no_subject_overlap", "path": str(path)}

    q_numeric = [col for col in q.select_dtypes(include=[np.number]).columns if col != subject_col]
    summary_numeric = [col for col in subject_summary.select_dtypes(include=[np.number]).columns]
    rows: list[dict[str, Any]] = []
    for qcol in q_numeric:
        for metric in summary_numeric:
            x = _numeric(merged[qcol])
            y = _numeric(merged[metric])
            ok = x.notna() & y.notna()
            if int(ok.sum()) < 5:
                continue
            rho, p = stats.spearmanr(x[ok], y[ok])
            rows.append(
                {
                    "questionnaire_variable": qcol,
                    "nir_behavior_metric": metric,
                    "n": int(ok.sum()),
                    "spearman_rho": float(rho),
                    "p_value_validation_only": float(p),
                }
            )
    return pd.DataFrame(rows), {"status": "available", "path": str(path), "n_subjects_overlap": int(merged["subject"].nunique())}


def extension_readiness(
    config: Config,
    subjects: Iterable[str],
) -> dict[str, Any]:
    root = _resolve_optional_path(config, "analysis_ready_root")
    oar_status: dict[str, Any] = {"status": "blocked_by_analysis_ready_schema"}
    first = next(iter(subjects), None)
    if root is not None and first is not None:
        subject = normalize_subject(first)
        path = root / "frame_level" / subject / f"{subject}_nir_analysis_ready.csv"
        if path.is_file():
            header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
            oar_columns = [
                col
                for col in (
                    "fullclass_ocular_aperture_ratio_median",
                    "fullclass_ocular_aperture_ratio_p90",
                    "left_ocular_aperture_ratio",
                    "right_ocular_aperture_ratio",
                )
                if col in header.columns
            ]
            if oar_columns:
                oar_status = {"status": "available_in_analysis_ready", "columns": oar_columns}
            else:
                oar_status = {
                    "status": "blocked_by_analysis_ready_schema",
                    "reason": "OAR is not yet an analysis-ready passthrough; formal downstream code must not bypass 10_analysis_ready",
                }

    visual_path = _resolve_optional_path(config, "stimulus_visual_table")
    questionnaire_path = _resolve_optional_path(config, "questionnaire_csv")
    return {
        "oar": oar_status,
        "visual_covariates": {
            "status": "available" if visual_path is not None and visual_path.is_file() else "unavailable",
            "path": str(visual_path) if visual_path is not None else None,
        },
        "questionnaire": {
            "status": "available" if questionnaire_path is not None and questionnaire_path.is_file() else "unavailable",
            "path": str(questionnaire_path) if questionnaire_path is not None else None,
        },
        "rgb_multimodal": {
            "status": "future_interface_only",
            "reason": "NIR validation remains modality-specific; RGB integration should consume frozen subject/time/trial keys later rather than enter this validation layer directly",
        },
    }
