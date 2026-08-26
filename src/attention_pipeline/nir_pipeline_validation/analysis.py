from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from attention_pipeline.config import Config
from attention_pipeline.nir_behavior.contract import normalize_subject, parse_subject_list
from attention_pipeline.nir_behavior.discovery import resolve_repo_path

PIPELINE_VERSION = "nir-pipeline-validation-v1"
SCHEMA_VERSION = 1
VALIDATION_LABEL = "PIPELINE VALIDATION ONLY — CURRENT NIR VALUES KNOWN INVALID"


@dataclass(frozen=True)
class SubjectTables:
    subject: str
    trial_level: Path
    trial_windows: Path
    probe_windows: Path
    time_on_task: Path
    trial_coverage: Path
    probe_coverage: Path
    completion: Path


def _root(config: Config, key: str) -> Path:
    raw = config.section("paths").get(key)
    if raw is None:
        raise KeyError(f"validation config missing paths.{key}")
    return resolve_repo_path(config, raw)


def analysis_tables_root(config: Config) -> Path:
    return _root(config, "analysis_tables_root")


def output_root(config: Config) -> Path:
    return _root(config, "output_root")


def subject_paths(config: Config, subject: str) -> SubjectTables:
    subject = normalize_subject(subject)
    base = analysis_tables_root(config) / "subjects" / subject
    return SubjectTables(
        subject=subject,
        trial_level=base / f"{subject}_trial_level.csv",
        trial_windows=base / f"{subject}_trial_pir_windows.csv",
        probe_windows=base / f"{subject}_probe_pir_windows.csv",
        time_on_task=base / f"{subject}_time_on_task_1s.csv",
        trial_coverage=base / f"{subject}_trial_window_coverage.csv",
        probe_coverage=base / f"{subject}_probe_window_coverage.csv",
        completion=base / f"{subject}_analysis_tables_completion.json",
    )


def valid_completion(paths: SubjectTables) -> bool:
    if not paths.completion.is_file():
        return False
    try:
        payload = json.loads(paths.completion.read_text(encoding="utf-8"))
    except Exception:
        return False
    if payload.get("status") != "complete":
        return False
    required = (
        paths.trial_level,
        paths.trial_windows,
        paths.probe_windows,
        paths.time_on_task,
        paths.trial_coverage,
        paths.probe_coverage,
    )
    return all(path.is_file() for path in required)


def discover_subjects(config: Config) -> list[str]:
    root = analysis_tables_root(config) / "subjects"
    if not root.is_dir():
        return []
    found: list[str] = []
    for path in root.glob("sub-*"):
        if not path.is_dir():
            continue
        try:
            subject = normalize_subject(path.name)
        except ValueError:
            continue
        if valid_completion(subject_paths(config, subject)):
            found.append(subject)
    return sorted(set(found), key=lambda value: int(value.split("-")[1]))


def selected_subjects(config: Config, override: Iterable[str] | None = None) -> list[str]:
    if override:
        return parse_subject_list(list(override))
    raw = config.section("subjects").get("include", [])
    if raw:
        return parse_subject_list(raw)
    return discover_subjects(config)


def load_cohort_tables(config: Config, subjects: Iterable[str]) -> dict[str, pd.DataFrame]:
    buckets: dict[str, list[pd.DataFrame]] = {
        "trial_level": [],
        "trial_windows": [],
        "probe_windows": [],
        "time_on_task": [],
        "trial_coverage": [],
        "probe_coverage": [],
    }
    for subject in subjects:
        paths = subject_paths(config, subject)
        if not valid_completion(paths):
            raise FileNotFoundError(f"{subject}: completion or required 11_analysis_tables artifacts missing")
        for name in buckets:
            buckets[name].append(
                pd.read_csv(getattr(paths, name), encoding="utf-8-sig", low_memory=False)
            )
    return {
        name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for name, frames in buckets.items()
    }


def trial_outcome_label(frame: pd.DataFrame) -> pd.Series:
    is_nogo = pd.to_numeric(frame["is_no_go"], errors="coerce").eq(1)
    commission = pd.to_numeric(frame["commission"], errors="coerce").eq(1)
    omission = pd.to_numeric(frame["omission"], errors="coerce").eq(1)
    values = np.select(
        [
            (~is_nogo) & (~omission),
            (~is_nogo) & omission,
            is_nogo & (~commission),
            is_nogo & commission,
        ],
        ["go_correct", "go_omission", "nogo_correct", "nogo_commission"],
        default="other",
    )
    return pd.Series(values, index=frame.index, dtype="string")


def behavior_subject_summary(trials: pd.DataFrame) -> pd.DataFrame:
    df = trials.copy()
    for column in ("is_no_go", "correct", "commission", "omission", "rt"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    rows: list[dict[str, Any]] = []
    for (subject, block_num), frame in df.groupby(["subject", "block_num"], sort=True):
        go = frame[frame["is_no_go"].eq(0)]
        nogo = frame[frame["is_no_go"].eq(1)]
        go_rt = pd.to_numeric(
            go.loc[go["correct"].eq(1), "rt"], errors="coerce"
        ).dropna()
        probe_count = (
            int(pd.to_numeric(frame["is_probe"], errors="coerce").eq(1).sum())
            if "is_probe" in frame.columns
            else 0
        )
        rows.append(
            {
                "subject": subject,
                "block_num": int(block_num),
                "n_trials": int(len(frame)),
                "accuracy": float(frame["correct"].mean()) if len(frame) else np.nan,
                "go_rt_median_ms": float(go_rt.median()) if len(go_rt) else np.nan,
                "go_rt_mad_ms": (
                    float(np.median(np.abs(go_rt - np.median(go_rt))))
                    if len(go_rt)
                    else np.nan
                ),
                "commission_rate": float(nogo["commission"].mean()) if len(nogo) else np.nan,
                "omission_rate": float(go["omission"].mean()) if len(go) else np.nan,
                "n_probes": probe_count,
            }
        )
    return pd.DataFrame(rows)


def block_pir_summary(time_on_task: pd.DataFrame, track: str) -> pd.DataFrame:
    df = time_on_task[time_on_task["track"].astype(str).eq(track)].copy()
    df["pir_median"] = pd.to_numeric(df["pir_median"], errors="coerce")
    df["pir_valid_fraction"] = pd.to_numeric(df["pir_valid_fraction"], errors="coerce")
    return (
        df.groupby(["subject", "block_num"], as_index=False)
        .agg(
            pir_median=("pir_median", "median"),
            pir_mean=("pir_median", "mean"),
            pir_valid_fraction=("pir_valid_fraction", "mean"),
            n_bins=("pir_median", "size"),
        )
    )


def coarse_time_on_task(
    time_on_task: pd.DataFrame, *, track: str, bin_sec: int
) -> pd.DataFrame:
    if bin_sec <= 0:
        raise ValueError("bin_sec must be > 0")
    df = time_on_task[time_on_task["track"].astype(str).eq(track)].copy()
    df["pir_median"] = pd.to_numeric(df["pir_median"], errors="coerce")
    time = pd.to_numeric(df["time_in_block_mid_sec"], errors="coerce")
    df["coarse_bin_start_sec"] = np.floor(time / bin_sec) * bin_sec
    return (
        df.groupby(
            ["subject", "block_num", "coarse_bin_start_sec"], as_index=False
        )
        .agg(
            pir_median=("pir_median", "median"),
            pir_valid_fraction=("pir_valid_fraction", "mean"),
            n_1s_bins=("pir_median", "size"),
        )
    )


def trial_analysis_table(
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
    window_cols = [
        "subject",
        "block_num",
        "trial_num",
        "global_trial_index",
        "pir_median",
        "pir_mean",
        "pir_valid_fraction",
        "internal_coverage_fraction",
        "max_temporal_gap_sec",
    ]
    windows = windows[[column for column in window_cols if column in windows.columns]]
    behavior_cols = [
        "subject",
        "block_num",
        "trial_num",
        "global_trial_index",
        "is_no_go",
        "correct",
        "commission",
        "omission",
        "rt",
        "time_in_block_sec",
    ]
    behavior = trial_level[
        [column for column in behavior_cols if column in trial_level.columns]
    ].copy()
    keys = [
        column
        for column in ("subject", "block_num", "trial_num", "global_trial_index")
        if column in windows.columns and column in behavior.columns
    ]
    merged = behavior.merge(windows, on=keys, how="inner", validate="one_to_one")
    merged["outcome"] = trial_outcome_label(merged)
    return merged


def probe_analysis_table(
    probe_windows: pd.DataFrame,
    *,
    track: str,
    window_name: str,
) -> pd.DataFrame:
    df = probe_windows[
        probe_windows["track"].astype(str).eq(track)
        & probe_windows["window_name"].astype(str).eq(window_name)
    ].copy()
    for column in (
        "pir_median",
        "pir_valid_fraction",
        "probe_vigilance",
        "seconds_since_previous_probe",
    ):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def add_within_between(frame: pd.DataFrame, value_col: str = "pir_median") -> pd.DataFrame:
    df = frame.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    subject_mean = df.groupby("subject")[value_col].transform("mean")
    df[f"{value_col}_between"] = subject_mean
    df[f"{value_col}_within"] = df[value_col] - subject_mean
    return df


def _safe_z(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if not np.isfinite(std) or std <= 0:
        return pd.Series(np.nan, index=values.index)
    return (values - values.mean()) / std


def _model_table(model: Any, model_name: str) -> pd.DataFrame:
    params = pd.Series(model.params)
    bse = pd.Series(model.bse).reindex(params.index)
    pvalues = pd.Series(model.pvalues).reindex(params.index)
    return pd.DataFrame(
        {
            "model": model_name,
            "term": params.index.astype(str),
            "estimate": params.values,
            "se": bse.values,
            "p_value": pvalues.values,
        }
    )


def fit_smoke_models(
    trial_table: pd.DataFrame,
    probe_table: pd.DataFrame,
    time_on_task: pd.DataFrame,
    *,
    track: str,
    min_subjects: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    tables: list[pd.DataFrame] = []
    status: list[dict[str, Any]] = []

    def record(name: str, fn: Any) -> None:
        try:
            model = fn()
            tables.append(_model_table(model, name))
            status.append({"model": name, "status": "complete"})
        except Exception as exc:
            status.append(
                {
                    "model": name,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    tot = time_on_task[time_on_task["track"].astype(str).eq(track)].copy()
    tot["pir_median"] = pd.to_numeric(tot["pir_median"], errors="coerce")
    tot["time_z"] = _safe_z(tot["time_in_block_mid_sec"])
    tot = tot.dropna(subset=["pir_median", "time_z", "subject", "block_num"])
    if tot["subject"].nunique() >= min_subjects:
        record(
            "lmm_time_on_task_pir",
            lambda: smf.mixedlm(
                "pir_median ~ time_z * C(block_num)",
                data=tot,
                groups=tot["subject"],
            ).fit(reml=False, method="lbfgs", disp=False),
        )
    else:
        status.append(
            {"model": "lmm_time_on_task_pir", "status": "skipped", "reason": "too_few_subjects"}
        )

    trial = add_within_between(trial_table)
    trial["rt"] = pd.to_numeric(trial["rt"], errors="coerce")
    trial["time_z"] = (
        _safe_z(trial["time_in_block_sec"])
        if "time_in_block_sec" in trial.columns
        else np.nan
    )
    go_correct = trial[
        pd.to_numeric(trial["is_no_go"], errors="coerce").eq(0)
        & pd.to_numeric(trial["correct"], errors="coerce").eq(1)
    ].dropna(
        subset=["rt", "pir_median_within", "pir_median_between", "time_z"]
    )
    if go_correct["subject"].nunique() >= min_subjects:
        record(
            "lmm_go_rt_pir_within_between",
            lambda: smf.mixedlm(
                "rt ~ pir_median_within + pir_median_between + time_z + C(block_num)",
                data=go_correct,
                groups=go_correct["subject"],
            ).fit(reml=False, method="lbfgs", disp=False),
        )
    else:
        status.append(
            {
                "model": "lmm_go_rt_pir_within_between",
                "status": "skipped",
                "reason": "insufficient_go_trials",
            }
        )

    nogo = trial[pd.to_numeric(trial["is_no_go"], errors="coerce").eq(1)].copy()
    nogo["commission"] = pd.to_numeric(nogo["commission"], errors="coerce")
    nogo = nogo.dropna(
        subset=["commission", "pir_median_within", "pir_median_between", "time_z"]
    )
    if nogo["subject"].nunique() >= min_subjects and nogo["commission"].nunique() == 2:
        record(
            "gee_nogo_commission_pir",
            lambda: smf.gee(
                "commission ~ pir_median_within + pir_median_between + time_z + C(block_num)",
                groups="subject",
                data=nogo,
                family=sm.families.Binomial(),
            ).fit(),
        )
    else:
        status.append(
            {
                "model": "gee_nogo_commission_pir",
                "status": "skipped",
                "reason": "binary_outcome_unavailable",
            }
        )

    go = trial[pd.to_numeric(trial["is_no_go"], errors="coerce").eq(0)].copy()
    go["omission"] = pd.to_numeric(go["omission"], errors="coerce")
    go = go.dropna(
        subset=["omission", "pir_median_within", "pir_median_between", "time_z"]
    )
    if go["subject"].nunique() >= min_subjects and go["omission"].nunique() == 2:
        record(
            "gee_go_omission_pir",
            lambda: smf.gee(
                "omission ~ pir_median_within + pir_median_between + time_z + C(block_num)",
                groups="subject",
                data=go,
                family=sm.families.Binomial(),
            ).fit(),
        )
    else:
        status.append(
            {
                "model": "gee_go_omission_pir",
                "status": "skipped",
                "reason": "binary_outcome_unavailable",
            }
        )

    probe = add_within_between(probe_table)
    if "probe_vigilance" in probe.columns:
        probe["probe_vigilance"] = pd.to_numeric(probe["probe_vigilance"], errors="coerce")
        probe = probe.dropna(
            subset=["probe_vigilance", "pir_median_within", "pir_median_between"]
        )
        if probe["subject"].nunique() >= min_subjects and len(probe) > probe["subject"].nunique():
            record(
                "lmm_probe_vigilance_pir",
                lambda: smf.mixedlm(
                    "probe_vigilance ~ pir_median_within + pir_median_between + C(block_num)",
                    data=probe,
                    groups=probe["subject"],
                ).fit(reml=False, method="lbfgs", disp=False),
            )
        else:
            status.append(
                {
                    "model": "lmm_probe_vigilance_pir",
                    "status": "skipped",
                    "reason": "insufficient_numeric_probe_rows",
                }
            )

    combined = (
        pd.concat(tables, ignore_index=True)
        if tables
        else pd.DataFrame(columns=["model", "term", "estimate", "se", "p_value"])
    )
    return combined, status
