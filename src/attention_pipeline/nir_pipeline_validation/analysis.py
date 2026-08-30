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

PIPELINE_VERSION = "nir-pipeline-validation-v1.1"
SCHEMA_VERSION = 2
VALIDATION_LABEL = "CANDIDATE ENDPOINTS NOT FROZEN — SCIENTIFIC REVIEW PENDING"


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
    base = analysis_tables_root(config) / "sessions" / subject
    return SubjectTables(
        subject=subject,
        trial_level=base / f"{subject}_trial_level.csv",
        trial_windows=base / f"{subject}_trial_pupil_windows.csv",
        probe_windows=base / f"{subject}_probe_pupil_windows.csv",
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
    root = analysis_tables_root(config) / "sessions"
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
    try:
        raw = config.section("subjects").get("include", [])
    except KeyError:
        raw = []
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
            raise FileNotFoundError(
                f"{subject}: completion or required 11_analysis_tables artifacts missing"
            )
        for name in buckets:
            buckets[name].append(
                pd.read_csv(getattr(paths, name), encoding="utf-8-sig", low_memory=False)
            )
    return {
        name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for name, frames in buckets.items()
    }


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


def omission_qc_type(frame: pd.DataFrame) -> pd.Series:
    """Return a descriptive, non-destructive omission subtype label.

    Program omission scoring remains authoritative. This label only separates
    omissions with prestimulus/carry-over motor timing evidence from omissions
    without those currently defined ambiguity flags.
    """
    is_go = ~pd.to_numeric(frame["is_no_go"], errors="coerce").eq(1)
    omission = pd.to_numeric(frame["omission"], errors="coerce").eq(1)
    pre = _bool_flag(frame, "prestimulus_press_flag")
    carry = _bool_flag(frame, "carryover_candidate_flag")
    qc_available = {
        "prestimulus_press_flag",
        "carryover_candidate_flag",
        "ambiguous_omission_flag",
    }.issubset(frame.columns)

    labels = np.full(len(frame), "not_go_omission", dtype=object)
    go_omission = is_go & omission
    if not qc_available:
        labels[go_omission.to_numpy()] = "go_omission_unclassified_qc_missing"
        return pd.Series(labels, index=frame.index, dtype="string")

    labels[(go_omission & ~pre & ~carry).to_numpy()] = "clean_omission"
    labels[(go_omission & pre & ~carry).to_numpy()] = "prestimulus_associated_omission"
    labels[(go_omission & ~pre & carry).to_numpy()] = "carryover_associated_omission"
    labels[(go_omission & pre & carry).to_numpy()] = (
        "prestimulus_and_carryover_associated_omission"
    )
    return pd.Series(labels, index=frame.index, dtype="string")


def trial_outcome_label(frame: pd.DataFrame) -> pd.Series:
    """Broad program-scoring outcome; omission QC subtype is kept separately."""
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
        ["go_correct", "go_omission_program", "nogo_correct", "nogo_commission"],
        default="other",
    )
    return pd.Series(values, index=frame.index, dtype="string")


def behavior_subject_summary(trials: pd.DataFrame) -> pd.DataFrame:
    df = trials.copy()
    for column in ("is_no_go", "correct", "commission", "omission", "rt"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["omission_qc_type"] = omission_qc_type(df)
    rows: list[dict[str, Any]] = []
    for (subject, block_num), frame in df.groupby(["subject", "block_num"], sort=True):
        go = frame[frame["is_no_go"].eq(0)].copy()
        nogo = frame[frame["is_no_go"].eq(1)].copy()
        go_rt = pd.to_numeric(
            go.loc[go["correct"].eq(1), "rt"], errors="coerce"
        ).dropna()
        probe_count = (
            int(pd.to_numeric(frame["is_probe"], errors="coerce").eq(1).sum())
            if "is_probe" in frame.columns
            else 0
        )
        go_n = max(1, len(go))
        subtype = go["omission_qc_type"].astype(str)
        n_program = int(go["omission"].eq(1).sum())
        n_clean = int(subtype.eq("clean_omission").sum())
        n_pre = int(subtype.eq("prestimulus_associated_omission").sum())
        n_carry = int(subtype.eq("carryover_associated_omission").sum())
        n_both = int(
            subtype.eq("prestimulus_and_carryover_associated_omission").sum()
        )
        n_unclassified = int(
            subtype.eq("go_omission_unclassified_qc_missing").sum()
        )
        n_ambiguous = n_pre + n_carry + n_both
        rows.append(
            {
                "subject": subject,
                "block_num": int(block_num),
                "n_trials": int(len(frame)),
                "n_go": int(len(go)),
                "n_nogo": int(len(nogo)),
                "accuracy": float(frame["correct"].mean()) if len(frame) else np.nan,
                "go_rt_median_ms": float(go_rt.median()) if len(go_rt) else np.nan,
                "go_rt_mad_ms": (
                    float(np.median(np.abs(go_rt - np.median(go_rt))))
                    if len(go_rt)
                    else np.nan
                ),
                "commission_rate": float(nogo["commission"].mean()) if len(nogo) else np.nan,
                "omission_program_n": n_program,
                "omission_program_rate": n_program / go_n,
                "clean_omission_n": n_clean,
                "clean_omission_rate": n_clean / go_n,
                "ambiguous_omission_n": n_ambiguous,
                "ambiguous_omission_rate": n_ambiguous / go_n,
                "prestimulus_associated_omission_n": n_pre,
                "carryover_associated_omission_n": n_carry,
                "prestimulus_and_carryover_omission_n": n_both,
                "unclassified_omission_qc_missing_n": n_unclassified,
                "anticipatory_candidate_rate": float(
                    _bool_flag(go, "anticipatory_candidate_flag").mean()
                )
                if len(go)
                else np.nan,
                "multiple_keypress_rate": float(
                    _bool_flag(frame, "multiple_keypress_flag").mean()
                )
                if len(frame)
                else np.nan,
                "n_probes": probe_count,
            }
        )
    return pd.DataFrame(rows)


def omission_subject_summary(trial_table: pd.DataFrame) -> pd.DataFrame:
    df = trial_table.copy()
    df["is_no_go"] = pd.to_numeric(df["is_no_go"], errors="coerce")
    df["omission"] = pd.to_numeric(df["omission"], errors="coerce")
    if "omission_qc_type" not in df.columns:
        df["omission_qc_type"] = omission_qc_type(df)
    go = df[df["is_no_go"].eq(0)].copy()
    rows: list[dict[str, Any]] = []
    categories = (
        "clean_omission",
        "prestimulus_associated_omission",
        "carryover_associated_omission",
        "prestimulus_and_carryover_associated_omission",
        "go_omission_unclassified_qc_missing",
    )
    for (subject, block_num), frame in go.groupby(["subject", "block_num"], sort=True):
        record: dict[str, Any] = {
            "subject": subject,
            "block_num": int(block_num),
            "n_go": int(len(frame)),
            "program_omission_n": int(frame["omission"].eq(1).sum()),
        }
        for category in categories:
            record[f"{category}_n"] = int(
                frame["omission_qc_type"].astype(str).eq(category).sum()
            )
        record["ambiguous_omission_n"] = (
            record["prestimulus_associated_omission_n"]
            + record["carryover_associated_omission_n"]
            + record["prestimulus_and_carryover_associated_omission_n"]
        )
        rows.append(record)
    return pd.DataFrame(rows)


def block_pir_summary(time_on_task: pd.DataFrame, track: str) -> pd.DataFrame:
    df = time_on_task[time_on_task["track"].astype(str).eq(track)].copy()
    df["pupil_median"] = pd.to_numeric(df["pupil_median"], errors="coerce")
    df["pupil_valid_fraction"] = pd.to_numeric(df["pupil_valid_fraction"], errors="coerce")
    return (
        df.groupby(["subject", "block_num"], as_index=False)
        .agg(
            pupil_median=("pupil_median", "median"),
            pupil_mean=("pupil_median", "mean"),
            pupil_valid_fraction=("pupil_valid_fraction", "mean"),
            n_bins=("pupil_median", "size"),
        )
    )


def coarse_time_on_task(
    time_on_task: pd.DataFrame, *, track: str, bin_sec: int
) -> pd.DataFrame:
    if bin_sec <= 0:
        raise ValueError("bin_sec must be > 0")
    df = time_on_task[time_on_task["track"].astype(str).eq(track)].copy()
    df["pupil_median"] = pd.to_numeric(df["pupil_median"], errors="coerce")
    time = pd.to_numeric(df["time_in_block_mid_sec"], errors="coerce")
    df["coarse_bin_start_sec"] = np.floor(time / bin_sec) * bin_sec
    return (
        df.groupby(
            ["subject", "block_num", "coarse_bin_start_sec"], as_index=False
        )
        .agg(
            pupil_median=("pupil_median", "median"),
            pupil_valid_fraction=("pupil_valid_fraction", "mean"),
            n_1s_bins=("pupil_median", "size"),
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
        "pupil_median",
        "pupil_mean",
        "pupil_valid_fraction",
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
        "response",
        "rt",
        "time_in_block_sec",
        "raw_keypresses",
        "prestimulus_press_ms",
        "n_raw_keypresses",
        "first_raw_keypress_ms",
        "second_raw_keypress_ms",
        "rt_reconstructed_ms",
        "rt_reconstruction_error_ms",
        "prestimulus_press_flag",
        "prestimulus_delta_to_onset_ms",
        "multiple_keypress_flag",
        "previous_second_press_to_current_onset_ms",
        "carryover_candidate_flag",
        "ambiguous_omission_flag",
        "anticipatory_candidate_flag",
        "rt_candidate_lt_100_flag",
        "rt_candidate_lt_150_flag",
        "rt_candidate_lt_200_flag",
        "rt_candidate_gt_900_flag",
        "rt_candidate_gt_1000_flag",
        "rt_candidate_gt_1150_flag",
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
    merged["omission_qc_type"] = omission_qc_type(merged)
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
        "pupil_median",
        "pupil_valid_fraction",
        "probe_vigilance",
        "seconds_since_previous_probe",
    ):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def add_within_between(frame: pd.DataFrame, value_col: str = "pupil_median") -> pd.DataFrame:
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
    tot["pupil_median"] = pd.to_numeric(tot["pupil_median"], errors="coerce")
    tot["time_z"] = _safe_z(tot["time_in_block_mid_sec"])
    tot = tot.dropna(subset=["pupil_median", "time_z", "subject", "block_num"])
    if tot["subject"].nunique() >= min_subjects:
        record(
            "lmm_time_on_task_pir",
            lambda: smf.mixedlm(
                "pupil_median ~ time_z * C(block_num)",
                data=tot,
                groups=tot["subject"],
            ).fit(reml=False, method="lbfgs", disp=False),
        )
    else:
        status.append(
            {
                "model": "lmm_time_on_task_pir",
                "status": "skipped",
                "reason": "too_few_subjects",
            }
        )

    trial = add_within_between(trial_table)
    trial["rt"] = pd.to_numeric(trial["rt"], errors="coerce")
    trial["time_z"] = (
        _safe_z(trial["time_in_block_sec"])
        if "time_in_block_sec" in trial.columns
        else np.nan
    )
    if "omission_qc_type" not in trial.columns:
        trial["omission_qc_type"] = omission_qc_type(trial)

    go_correct = trial[
        pd.to_numeric(trial["is_no_go"], errors="coerce").eq(0)
        & pd.to_numeric(trial["correct"], errors="coerce").eq(1)
    ].dropna(
        subset=["rt", "pupil_median_within", "pupil_median_between", "time_z"]
    )
    if go_correct["subject"].nunique() >= min_subjects:
        record(
            "lmm_go_rt_pir_within_between",
            lambda: smf.mixedlm(
                "rt ~ pupil_median_within + pupil_median_between + time_z + C(block_num)",
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
        subset=["commission", "pupil_median_within", "pupil_median_between", "time_z"]
    )
    if nogo["subject"].nunique() >= min_subjects and nogo["commission"].nunique() == 2:
        record(
            "gee_nogo_commission_pir",
            lambda: smf.gee(
                "commission ~ pupil_median_within + pupil_median_between + time_z + C(block_num)",
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
        subset=["omission", "pupil_median_within", "pupil_median_between", "time_z"]
    )
    if go["subject"].nunique() >= min_subjects and go["omission"].nunique() == 2:
        record(
            "gee_go_program_omission_pir",
            lambda: smf.gee(
                "omission ~ pupil_median_within + pupil_median_between + time_z + C(block_num)",
                groups="subject",
                data=go,
                family=sm.families.Binomial(),
            ).fit(),
        )
    else:
        status.append(
            {
                "model": "gee_go_program_omission_pir",
                "status": "skipped",
                "reason": "binary_outcome_unavailable",
            }
        )

    qc_go = go[
        ~go["omission_qc_type"].astype(str).isin(
            [
                "prestimulus_associated_omission",
                "carryover_associated_omission",
                "prestimulus_and_carryover_associated_omission",
                "go_omission_unclassified_qc_missing",
            ]
        )
    ].copy()
    if (
        qc_go["subject"].nunique() >= min_subjects
        and qc_go["omission"].nunique() == 2
    ):
        record(
            "gee_go_clean_omission_sensitivity",
            lambda: smf.gee(
                "omission ~ pupil_median_within + pupil_median_between + time_z + C(block_num)",
                groups="subject",
                data=qc_go,
                family=sm.families.Binomial(),
            ).fit(),
        )
    else:
        status.append(
            {
                "model": "gee_go_clean_omission_sensitivity",
                "status": "skipped",
                "reason": "clean_binary_outcome_unavailable",
            }
        )

    if "anticipatory_candidate_flag" in go.columns:
        go["anticipatory_candidate"] = _bool_flag(
            go, "anticipatory_candidate_flag"
        ).astype(int)
        if (
            go["subject"].nunique() >= min_subjects
            and go["anticipatory_candidate"].nunique() == 2
        ):
            record(
                "gee_go_anticipatory_candidate_pir",
                lambda: smf.gee(
                    "anticipatory_candidate ~ pupil_median_within + pupil_median_between + time_z + C(block_num)",
                    groups="subject",
                    data=go,
                    family=sm.families.Binomial(),
                ).fit(),
            )
        else:
            status.append(
                {
                    "model": "gee_go_anticipatory_candidate_pir",
                    "status": "skipped",
                    "reason": "anticipatory_binary_outcome_unavailable",
                }
            )

    probe = add_within_between(probe_table)
    if "probe_vigilance" in probe.columns:
        probe["probe_vigilance"] = pd.to_numeric(
            probe["probe_vigilance"], errors="coerce"
        )
        probe = probe.dropna(
            subset=["probe_vigilance", "pupil_median_within", "pupil_median_between"]
        )
        if (
            probe["subject"].nunique() >= min_subjects
            and len(probe) > probe["subject"].nunique()
        ):
            record(
                "lmm_probe_vigilance_pir",
                lambda: smf.mixedlm(
                    "probe_vigilance ~ pupil_median_within + pupil_median_between + C(block_num)",
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
