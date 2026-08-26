from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from attention_pipeline.config import Config
from attention_pipeline.nir_behavior.contract import normalize_subject
from attention_pipeline.nir_behavior.discovery import resolve_repo_path

from .analysis import omission_qc_type, trial_outcome_label
from .extended import DYNAMIC_PIR_FEATURES, window_duration_sec


TRACK_TO_CONTINUOUS_VALUE = {
    "binocular_primary": "binocular_PIR",
    "left_primary": "left_centered_PIR",
    "right_primary": "right_centered_PIR",
    "binocular_strict": "binocular_strict_PIR",
    "left_strict": "left_strict_centered_PIR",
    "right_strict": "right_strict_centered_PIR",
}


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _analysis_ready_root(config: Config) -> Path:
    raw = config.section("paths").get("analysis_ready_root")
    if raw in (None, ""):
        raise KeyError("validation config missing paths.analysis_ready_root")
    return resolve_repo_path(config, raw)


def load_continuous_analysis_ready(
    config: Config,
    subjects: Iterable[str],
    *,
    track: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read one centered PIR track from 10_analysis_ready without touching production."""
    value_col = TRACK_TO_CONTINUOUS_VALUE.get(str(track))
    if value_col is None:
        return pd.DataFrame(), {
            "status": "unavailable",
            "reason": "unsupported_track",
            "track": track,
        }

    root = _analysis_ready_root(config)
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for raw_subject in subjects:
        subject = normalize_subject(raw_subject)
        path = root / "frame_level" / subject / f"{subject}_nir_analysis_ready.csv"
        if not path.is_file():
            missing.append(subject)
            continue
        header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
        required = {"subject", "block", "unix_ms", value_col}
        if not required.issubset(header.columns):
            missing.append(subject)
            continue
        usecols = ["subject", "block", "unix_ms", value_col]
        frame = pd.read_csv(
            path,
            usecols=usecols,
            encoding="utf-8-sig",
            low_memory=False,
        )
        frame["subject"] = frame["subject"].map(normalize_subject)
        frame["block_num"] = _numeric(frame["block"])
        frame["unix_ms"] = _numeric(frame["unix_ms"])
        frame["pir"] = _numeric(frame[value_col])
        frame = frame[["subject", "block_num", "unix_ms", "pir"]]
        frame = frame.dropna(subset=["block_num", "unix_ms"]).sort_values(
            ["block_num", "unix_ms"], kind="stable"
        )
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    status = {
        "status": "available" if frames else "unavailable",
        "track": track,
        "value_column": value_col,
        "analysis_ready_root": str(root),
        "n_subjects": len(frames),
        "missing_or_incompatible_subjects": missing,
        "source_boundary": "read-only 10_analysis_ready; production is never read",
    }
    return result, status


def global_pir_trajectory(
    time_on_task: pd.DataFrame,
    *,
    track: str,
    display_gap_sec: float = 60.0,
    summary_bin_sec: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build whole-experiment and aligned Block trajectories from 1-s summaries."""
    df = time_on_task[time_on_task["track"].astype(str).eq(track)].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df["block_num"] = _numeric(df["block_num"])
    df["time_in_block_sec"] = _numeric(df["time_in_block_mid_sec"])
    df["pir_median"] = _numeric(df["pir_median"])
    rows: list[pd.DataFrame] = []
    for subject, frame in df.groupby("subject", sort=True):
        current = frame.copy()
        b1 = current[current["block_num"].eq(1)]
        if b1.empty:
            b1_end = 0.0
        else:
            b1_end = float(b1["time_in_block_sec"].max()) + 0.5
        current["global_time_sec"] = np.where(
            current["block_num"].eq(1),
            current["time_in_block_sec"],
            b1_end + float(display_gap_sec) + current["time_in_block_sec"],
        )
        current["block1_display_end_sec"] = b1_end
        current["block2_display_start_sec"] = b1_end + float(display_gap_sec)
        rows.append(current)
    detail = pd.concat(rows, ignore_index=True)
    width = max(float(summary_bin_sec), 1.0)
    detail["global_bin_sec"] = np.floor(detail["global_time_sec"] / width) * width + width / 2.0
    summary = (
        detail.groupby(["block_num", "global_bin_sec"], as_index=False)["pir_median"]
        .agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            mean="mean",
            n="count",
        )
    )
    return detail, summary


def global_pir_distribution(time_on_task: pd.DataFrame, *, track: str) -> pd.DataFrame:
    df = time_on_task[time_on_task["track"].astype(str).eq(track)].copy()
    if df.empty:
        return pd.DataFrame()
    df["pir_median"] = _numeric(df["pir_median"])
    return (
        df.groupby(["subject", "block_num"], as_index=False)["pir_median"]
        .agg(
            pir_median="median",
            pir_mean="mean",
            pir_sd="std",
            pir_q25=lambda x: x.quantile(0.25),
            pir_q75=lambda x: x.quantile(0.75),
        )
    )


def trial_condition_summary(
    trial_level: pd.DataFrame,
    trial_windows: pd.DataFrame,
    *,
    track: str,
    window_name: str,
) -> pd.DataFrame:
    windows = trial_windows[
        trial_windows["track"].astype(str).eq(track)
        & trial_windows["window_name"].astype(str).eq(window_name)
    ].copy()
    if windows.empty:
        return pd.DataFrame()
    behavior = trial_level.copy()
    behavior["outcome"] = trial_outcome_label(behavior)
    behavior["omission_qc_type"] = omission_qc_type(behavior)
    keys = [
        col
        for col in ("subject", "block_num", "trial_num", "global_trial_index")
        if col in windows.columns and col in behavior.columns
    ]
    bcols = keys + [
        col
        for col in (
            "outcome",
            "omission_qc_type",
            "is_no_go",
            "commission",
            "omission",
            "rt",
            "stimulus_name",
            "stimulus_size",
        )
        if col in behavior.columns
    ]
    features = [col for col in DYNAMIC_PIR_FEATURES if col in windows.columns]
    merged = windows[keys + features].merge(
        behavior[bcols], on=keys, how="left", validate="one_to_one"
    )
    return merged


def _deduplicate_probe_events(probe_windows: pd.DataFrame, *, track: str) -> pd.DataFrame:
    df = probe_windows[probe_windows["track"].astype(str).eq(track)].copy()
    if df.empty:
        return pd.DataFrame()
    ids = [
        col
        for col in (
            "subject",
            "block_num",
            "probe_index_global",
            "probe_index_in_block",
            "probe_onset_ms",
            "probe_response",
            "probe_vigilance",
            "probe_rt",
            "probe_vigilance_rt",
        )
        if col in df.columns
    ]
    event_keys = [
        col
        for col in (
            "subject",
            "block_num",
            "probe_index_global",
            "probe_index_in_block",
            "probe_onset_ms",
        )
        if col in ids
    ]
    return df[ids].drop_duplicates(event_keys or None).copy()


def build_event_catalogs(
    trial_level: pd.DataFrame,
    probe_windows: pd.DataFrame,
    *,
    track: str,
    max_go_reference_per_subject_block: int = 60,
) -> dict[str, pd.DataFrame]:
    trials = trial_level.copy()
    trials["outcome"] = trial_outcome_label(trials)
    trials["omission_qc_type"] = omission_qc_type(trials)
    for col in ("is_no_go", "commission", "omission", "absolute_onset_time"):
        if col in trials.columns:
            trials[col] = _numeric(trials[col])

    onset_col = "absolute_onset_time" if "absolute_onset_time" in trials.columns else None
    if onset_col is None:
        nogo = pd.DataFrame()
        omission = pd.DataFrame()
    else:
        nogo = trials[trials["is_no_go"].eq(1)].copy()
        nogo["event_condition"] = np.where(
            nogo["commission"].eq(1), "commission", "correct_inhibition"
        )
        nogo = nogo.rename(columns={onset_col: "event_onset_ms"})

        omission_parts: list[pd.DataFrame] = []
        for (subject, block_num), frame in trials[trials["is_no_go"].eq(0)].groupby(
            ["subject", "block_num"], sort=True
        ):
            go = frame.copy()
            om = go[go["omission"].eq(1)].copy()
            if not om.empty:
                om["event_condition"] = np.where(
                    om["omission_qc_type"].astype(str).eq("clean_omission"),
                    "clean_omission",
                    "ambiguous_omission",
                )
                omission_parts.append(om)
            correct = go[go["omission"].ne(1)].copy()
            limit = max(0, int(max_go_reference_per_subject_block))
            if limit and len(correct) > limit:
                indices = np.linspace(0, len(correct) - 1, num=limit).round().astype(int)
                correct = correct.iloc[np.unique(indices)]
            if not correct.empty:
                correct["event_condition"] = "go_correct_reference"
                omission_parts.append(correct)
        omission = pd.concat(omission_parts, ignore_index=True) if omission_parts else pd.DataFrame()
        if not omission.empty:
            omission = omission.rename(columns={onset_col: "event_onset_ms"})

    probe = _deduplicate_probe_events(probe_windows, track=track)
    if not probe.empty and "probe_onset_ms" in probe.columns:
        probe = probe.rename(columns={"probe_onset_ms": "event_onset_ms"})
        response = probe.get("probe_response", pd.Series(pd.NA, index=probe.index))
        probe["event_condition"] = "response_" + response.astype("Int64").astype(str)

    keep_common = ["subject", "block_num", "event_onset_ms", "event_condition"]
    for name, frame in (("nogo", nogo), ("omission", omission), ("probe", probe)):
        if frame.empty:
            continue
        frame["event_id"] = [f"{name}_{i:07d}" for i in range(len(frame))]

    return {"nogo": nogo, "omission": omission, "probe": probe}


def continuous_event_trajectory(
    continuous: pd.DataFrame,
    events: pd.DataFrame,
    *,
    start_sec: float,
    end_sec: float,
    bin_sec: float,
    extra_event_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Align continuous centered PIR to events and summarize true small time bins."""
    if continuous.empty or events.empty:
        return pd.DataFrame()
    bin_sec = float(bin_sec)
    if bin_sec <= 0 or end_sec <= start_sec:
        raise ValueError("continuous event trajectory requires bin_sec>0 and end_sec>start_sec")
    edges = np.arange(float(start_sec), float(end_sec) + bin_sec * 0.5, bin_sec)
    if len(edges) < 2:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    event_extra = [col for col in extra_event_columns if col in events.columns]
    for (subject, block_num), event_block in events.groupby(["subject", "block_num"], sort=True):
        signal = continuous[
            continuous["subject"].astype(str).eq(str(subject))
            & _numeric(continuous["block_num"]).eq(float(block_num))
        ].copy()
        if signal.empty:
            continue
        signal = signal.sort_values("unix_ms", kind="stable")
        times = _numeric(signal["unix_ms"]).to_numpy(dtype=float)
        values = _numeric(signal["pir"]).to_numpy(dtype=float)
        for event in event_block.itertuples(index=False):
            onset = float(getattr(event, "event_onset_ms"))
            if not np.isfinite(onset):
                continue
            left = int(np.searchsorted(times, onset + start_sec * 1000.0, side="left"))
            right = int(np.searchsorted(times, onset + end_sec * 1000.0, side="left"))
            if right <= left:
                continue
            rel = (times[left:right] - onset) / 1000.0
            vals = values[left:right]
            bins = np.digitize(rel, edges, right=False) - 1
            for bin_idx in range(len(edges) - 1):
                mask = bins == bin_idx
                if not np.any(mask):
                    continue
                finite = mask & np.isfinite(vals)
                record: dict[str, Any] = {
                    "subject": subject,
                    "block_num": int(block_num),
                    "event_id": getattr(event, "event_id", None),
                    "event_condition": getattr(event, "event_condition", None),
                    "event_onset_ms": onset,
                    "time_bin_start_sec": float(edges[bin_idx]),
                    "time_bin_end_sec": float(edges[bin_idx + 1]),
                    "time_bin_mid_sec": float((edges[bin_idx] + edges[bin_idx + 1]) / 2.0),
                    "n_rows": int(mask.sum()),
                    "n_valid": int(finite.sum()),
                    "valid_fraction": float(finite.sum() / mask.sum()),
                    "pir_median": float(np.nanmedian(vals[mask])) if np.isfinite(vals[mask]).any() else np.nan,
                    "pir_mean": float(np.nanmean(vals[mask])) if np.isfinite(vals[mask]).any() else np.nan,
                    "pir_sd": float(np.nanstd(vals[mask], ddof=1)) if np.isfinite(vals[mask]).sum() >= 2 else np.nan,
                }
                for col in event_extra:
                    record[col] = getattr(event, col, None)
                rows.append(record)
    return pd.DataFrame(rows)


def feature_redundancy(
    trial_windows: pd.DataFrame,
    *,
    track: str,
    window_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = trial_windows[
        trial_windows["track"].astype(str).eq(track)
        & trial_windows["window_name"].astype(str).eq(window_name)
    ].copy()
    features = [col for col in DYNAMIC_PIR_FEATURES if col in df.columns]
    if df.empty or len(features) < 2:
        return pd.DataFrame(), pd.DataFrame()
    values = df[["subject", *features]].copy()
    for col in features:
        values[col] = _numeric(values[col])
        values[f"{col}__within"] = values[col] - values.groupby("subject")[col].transform("mean")
    within_cols = [f"{col}__within" for col in features]
    corr = values[within_cols].corr(min_periods=5)
    corr.index = features
    corr.columns = features
    long = corr.rename_axis(index="feature_a", columns="feature_b").stack(dropna=False).rename("r").reset_index()
    subject_means = values.groupby("subject", as_index=False)[features].mean()
    between = subject_means[features].corr(min_periods=3)
    between_long = between.rename_axis(index="feature_a", columns="feature_b").stack(dropna=False).rename("r").reset_index()
    return long, between_long


def within_between_correlation_tables(
    trial_level: pd.DataFrame,
    trial_windows: pd.DataFrame,
    subject_summary: pd.DataFrame,
    *,
    track: str,
    window_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial = trial_condition_summary(
        trial_level, trial_windows, track=track, window_name=window_name
    )
    if trial.empty:
        within_long = pd.DataFrame()
    else:
        candidates = [
            col
            for col in (
                "pir_median",
                "pir_mad",
                "pir_slope_per_sec",
                "pir_diff_rate_mad_per_sec",
                "rt",
                "commission",
                "omission",
            )
            if col in trial.columns
        ]
        work = trial[["subject", *candidates]].copy()
        centered_cols: list[str] = []
        for col in candidates:
            work[col] = _numeric(work[col])
            name = f"{col}__within"
            work[name] = work[col] - work.groupby("subject")[col].transform("mean")
            centered_cols.append(name)
        corr = work[centered_cols].corr(min_periods=5)
        labels = [col.replace("__within", "") for col in centered_cols]
        corr.index = labels
        corr.columns = labels
        within_long = corr.rename_axis(index="metric_a", columns="metric_b").stack(dropna=False).rename("r").reset_index()

    if subject_summary.empty:
        between_long = pd.DataFrame()
    else:
        preferred = [
            col
            for col in (
                "raw_PIR_subject_median",
                "raw_PIR_valid_fraction",
                "go_rt_median_ms",
                "rt_cv",
                "exg_tau",
                "dprime",
                "commission_rate",
                "program_omission_rate",
                "clean_omission_rate",
                "ambiguous_omission_rate",
            )
            if col in subject_summary.columns
        ]
        between = subject_summary[preferred].apply(pd.to_numeric, errors="coerce").corr(min_periods=3)
        between_long = between.rename_axis(index="metric_a", columns="metric_b").stack(dropna=False).rename("r").reset_index()
    return within_long, between_long


def window_effect_stability(
    trial_level: pd.DataFrame,
    trial_windows: pd.DataFrame,
    probe_windows: pd.DataFrame,
    *,
    track: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial = trial_condition_summary(trial_level, trial_windows, track=track, window_name="")
    # trial_condition_summary accepts a fixed window, so build directly here for all state windows.
    windows = trial_windows[trial_windows["track"].astype(str).eq(track)].copy()
    windows["window_sec"] = windows["window_name"].map(window_duration_sec)
    windows = windows[windows["window_sec"].notna()].copy()
    if windows.empty:
        subject_effects = pd.DataFrame()
    else:
        behavior = trial_level.copy()
        behavior["outcome"] = trial_outcome_label(behavior)
        keys = [
            col
            for col in ("subject", "block_num", "trial_num", "global_trial_index")
            if col in windows.columns and col in behavior.columns
        ]
        merged = windows.merge(
            behavior[keys + ["outcome"]], on=keys, how="left", validate="many_to_one"
        )
        merged["pir_median"] = _numeric(merged["pir_median"])
        med = merged.groupby(
            ["subject", "block_num", "window_name", "window_sec", "outcome"], as_index=False
        )["pir_median"].median()
        rows: list[dict[str, Any]] = []
        for (subject, block_num, window_name, window_sec), frame in med.groupby(
            ["subject", "block_num", "window_name", "window_sec"], sort=True
        ):
            values = dict(zip(frame["outcome"].astype(str), frame["pir_median"]))
            if "nogo_commission" in values and "nogo_correct" in values:
                rows.append({
                    "subject": subject,
                    "block_num": int(block_num),
                    "window_name": window_name,
                    "window_sec": float(window_sec),
                    "contrast": "commission_minus_correct_inhibition",
                    "effect": values["nogo_commission"] - values["nogo_correct"],
                })
            if "go_omission_program" in values and "go_correct" in values:
                rows.append({
                    "subject": subject,
                    "block_num": int(block_num),
                    "window_name": window_name,
                    "window_sec": float(window_sec),
                    "contrast": "program_omission_minus_go_correct",
                    "effect": values["go_omission_program"] - values["go_correct"],
                })
        subject_effects = pd.DataFrame(rows)

    probe = probe_windows[probe_windows["track"].astype(str).eq(track)].copy()
    if not probe.empty:
        probe["window_sec"] = probe["window_name"].map(window_duration_sec)
        probe["pir_median"] = _numeric(probe["pir_median"])
        probe["probe_vigilance_num"] = _numeric(probe["probe_vigilance"]) if "probe_vigilance" in probe.columns else np.nan
        probe_rows: list[dict[str, Any]] = []
        for (subject, block_num, window_name, window_sec), frame in probe.dropna(subset=["window_sec"]).groupby(
            ["subject", "block_num", "window_name", "window_sec"], sort=True
        ):
            ok = frame["pir_median"].notna() & frame["probe_vigilance_num"].notna()
            if int(ok.sum()) >= 3 and frame.loc[ok, "probe_vigilance_num"].nunique() >= 2:
                slope = np.polyfit(
                    frame.loc[ok, "probe_vigilance_num"].to_numpy(dtype=float),
                    frame.loc[ok, "pir_median"].to_numpy(dtype=float),
                    1,
                )[0]
                probe_rows.append({
                    "subject": subject,
                    "block_num": int(block_num),
                    "window_name": window_name,
                    "window_sec": float(window_sec),
                    "contrast": "pir_per_probe_vigilance_unit",
                    "effect": float(slope),
                })
        if probe_rows:
            probe_effects = pd.DataFrame(probe_rows)
            subject_effects = pd.concat([subject_effects, probe_effects], ignore_index=True) if not subject_effects.empty else probe_effects

    if subject_effects.empty:
        return subject_effects, pd.DataFrame()
    cohort = (
        subject_effects.groupby(["contrast", "window_name", "window_sec"], as_index=False)["effect"]
        .agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            n="count",
        )
    )
    return subject_effects, cohort


def block_transition_recovery(
    time_on_task: pd.DataFrame,
    *,
    track: str,
    transition_window_sec: float = 120.0,
    recovery_summary_sec: float = 60.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = time_on_task[time_on_task["track"].astype(str).eq(track)].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df["time"] = _numeric(df["time_in_block_mid_sec"])
    df["pir_median"] = _numeric(df["pir_median"])
    traj_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for subject, frame in df.groupby("subject", sort=True):
        b1 = frame[frame["block_num"].eq(1)].copy()
        b2 = frame[frame["block_num"].eq(2)].copy()
        if b1.empty or b2.empty:
            continue
        b1_end = float(b1["time"].max())
        b1["transition_time_sec"] = b1["time"] - b1_end
        b2_start = float(b2["time"].min())
        b2["transition_time_sec"] = b2["time"] - b2_start
        b1 = b1[b1["transition_time_sec"].ge(-float(transition_window_sec))]
        b2 = b2[b2["transition_time_sec"].le(float(transition_window_sec))]
        traj_rows.extend([b1, b2])
        tail = b1[b1["transition_time_sec"].ge(-float(recovery_summary_sec))]["pir_median"]
        head = b2[b2["transition_time_sec"].le(float(recovery_summary_sec))]["pir_median"]
        summary_rows.append({
            "subject": subject,
            "block1_last_window_median": float(tail.median()) if tail.notna().any() else np.nan,
            "block2_first_window_median": float(head.median()) if head.notna().any() else np.nan,
            "recovery_delta_block2_minus_block1": float(head.median() - tail.median()) if head.notna().any() and tail.notna().any() else np.nan,
        })
    trajectory = pd.concat(traj_rows, ignore_index=True) if traj_rows else pd.DataFrame()
    return trajectory, pd.DataFrame(summary_rows)


def probe_transition_table(
    probe_windows: pd.DataFrame,
    *,
    track: str,
    window_name: str,
) -> pd.DataFrame:
    df = probe_windows[
        probe_windows["track"].astype(str).eq(track)
        & probe_windows["window_name"].astype(str).eq(window_name)
    ].copy()
    if df.empty:
        return pd.DataFrame()
    sort_cols = [col for col in ("subject", "block_num", "probe_index_in_block", "probe_onset_ms") if col in df.columns]
    df = df.sort_values(sort_cols, kind="stable")
    for col in ("probe_response", "probe_vigilance", "pir_median"):
        if col in df.columns:
            df[col] = _numeric(df[col])
    group = df.groupby(["subject", "block_num"], sort=False)
    df["previous_probe_response"] = group["probe_response"].shift(1) if "probe_response" in df.columns else np.nan
    df["previous_probe_vigilance"] = group["probe_vigilance"].shift(1) if "probe_vigilance" in df.columns else np.nan
    df["previous_probe_pir"] = group["pir_median"].shift(1) if "pir_median" in df.columns else np.nan
    df["delta_probe_vigilance"] = df.get("probe_vigilance", np.nan) - df["previous_probe_vigilance"]
    df["delta_probe_pir"] = df.get("pir_median", np.nan) - df["previous_probe_pir"]
    df["response_transition"] = (
        df["previous_probe_response"].astype("Int64").astype(str)
        + "→"
        + df.get("probe_response", pd.Series(pd.NA, index=df.index)).astype("Int64").astype(str)
    )
    return df[df["previous_probe_response"].notna()].copy()


def individual_heterogeneity(
    time_on_task: pd.DataFrame,
    block_recovery: pd.DataFrame,
    window_effects: pd.DataFrame,
    *,
    track: str,
) -> pd.DataFrame:
    df = time_on_task[time_on_task["track"].astype(str).eq(track)].copy()
    rows: list[dict[str, Any]] = []
    for subject, frame in df.groupby("subject", sort=True):
        record: dict[str, Any] = {"subject": subject}
        for block_num, block in frame.groupby("block_num", sort=True):
            x = _numeric(block["time_in_block_mid_sec"])
            y = _numeric(block["pir_median"])
            ok = x.notna() & y.notna()
            slope = np.polyfit(x[ok], y[ok], 1)[0] if int(ok.sum()) >= 3 and x[ok].nunique() >= 2 else np.nan
            record[f"block{int(block_num)}_time_slope_per_sec"] = float(slope) if np.isfinite(slope) else np.nan
            record[f"block{int(block_num)}_pir_median"] = float(y[ok].median()) if bool(ok.any()) else np.nan
        rows.append(record)
    result = pd.DataFrame(rows)
    if not block_recovery.empty:
        result = result.merge(block_recovery, on="subject", how="left", validate="one_to_one")
    if not window_effects.empty:
        wide = (
            window_effects.groupby(["subject", "contrast"], as_index=False)["effect"].median()
            .pivot(index="subject", columns="contrast", values="effect")
            .add_prefix("median_effect__")
            .reset_index()
        )
        result = result.merge(wide, on="subject", how="left", validate="one_to_one")
    return result


def stimulus_condition_summary(visual_trial: pd.DataFrame) -> pd.DataFrame:
    if visual_trial.empty:
        return pd.DataFrame()
    df = visual_trial.copy()
    for col in ("pir_median", "stimulus_size", "is_no_go"):
        if col in df.columns:
            df[col] = _numeric(df[col])
    ids = [col for col in ("subject", "block_num", "is_no_go", "stimulus_size") if col in df.columns]
    metrics = [
        col
        for col in (
            "pir_median",
            "current_central_rel_lum_mean",
            "current_central_rms_contrast",
            "current_fruit_visible_area_fraction_central_roi",
        )
        if col in df.columns
    ]
    if not ids or not metrics:
        return pd.DataFrame()
    return df.groupby(ids, as_index=False)[metrics].median(numeric_only=True)
