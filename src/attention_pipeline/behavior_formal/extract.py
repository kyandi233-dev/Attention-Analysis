"""Extraction and validation for the final FocusWave v3.1.3 BB behavior data."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config

FORMAL_COLUMNS = [
    "subject_id", "block_num", "condition", "trial_num", "cycle_num",
    "position_in_cycle", "stimulus_name", "stimulus_size", "is_no_go",
    "response", "rt", "response_time", "correct", "commission", "omission",
    "is_probe", "probe_response", "probe_rt", "probe_vigilance",
    "probe_vigilance_rt", "probe_onset_time", "probe_response_time",
    "absolute_onset_time", "block_onset_time", "raw_keypresses",
    "prestimulus_press_ms", "rest_duration",
]
REQUIRED_COLUMNS = set(FORMAL_COLUMNS)
NUMERIC_COLUMNS = [
    "block_num", "trial_num", "cycle_num", "position_in_cycle", "is_no_go",
    "response", "rt", "response_time", "correct", "commission", "omission",
    "is_probe", "probe_response", "probe_rt", "probe_vigilance",
    "probe_vigilance_rt", "probe_onset_time", "probe_response_time",
    "absolute_onset_time", "block_onset_time", "stimulus_size",
]
_SUBJECT_RE = re.compile(r"^sub-(\d+)_?$", re.IGNORECASE)


def _repo_root(config: Config) -> Path:
    return config.path.parent.parent


def data_roots(config: Config) -> list[Path]:
    roots = []
    for raw in config.section("data").get("roots", []):
        path = Path(str(raw))
        if not path.is_absolute():
            path = (_repo_root(config) / path).resolve()
        roots.append(path)
    if not roots:
        raise ValueError("behavior config requires data.roots")
    return roots


def _subject_number(name: str) -> int | None:
    match = _SUBJECT_RE.match(name)
    return int(match.group(1)) if match else None


def _subject_behavior_matches(config: Config, subject: str) -> list[Path]:
    number = int(subject.split("-")[1])
    names = (f"sub-{number:03d}_", f"sub-{number:03d}")
    behavior_dir = str(config.section("data").get("behavior_dir", "beh"))
    matches: list[Path] = []
    for root in data_roots(config):
        if not root.exists():
            continue
        for name in names:
            path = root / name / behavior_dir
            if path.is_dir():
                resolved = path.resolve()
                if resolved not in matches:
                    matches.append(resolved)
    return matches


def discover_subjects(config: Config) -> list[str]:
    """Discover final formal subjects with both B1/B2 behavior files."""
    data = config.section("data")
    include = [str(x) for x in data.get("include", [])]
    exclude = {str(x) for x in data.get("exclude", [])}
    min_number = int(data.get("min_subject_number", 31))

    if include:
        candidates = include
    else:
        found: set[str] = set()
        for root in data_roots(config):
            if not root.exists():
                continue
            for entry in root.glob("sub-*"):
                if not entry.is_dir():
                    continue
                number = _subject_number(entry.name)
                if number is not None and number >= min_number:
                    found.add(f"sub-{number:03d}")
        candidates = sorted(found, key=lambda x: int(x.split("-")[1]))

    subjects = [s for s in candidates if s not in exclude]
    complete = []
    missing = []
    for subject in subjects:
        try:
            formal_block_files(config, subject)
            complete.append(subject)
        except FileNotFoundError as exc:
            missing.append(str(exc))
    if missing:
        raise FileNotFoundError(
            "Incomplete final formal subject(s) discovered: " + "; ".join(missing)
        )
    if not complete:
        raise FileNotFoundError(
            "No complete final BB behavior subjects were found in data.roots"
        )
    return complete


def subject_behavior_dir(config: Config, subject: str) -> Path:
    matches = _subject_behavior_matches(config, subject)
    if not matches:
        raise FileNotFoundError(f"{subject}: behavior directory not found in data.roots")
    if len(matches) > 1:
        detail = ", ".join(str(path) for path in matches)
        raise RuntimeError(
            f"{subject}: duplicate behavior directories found across data.roots: {detail}"
        )
    return matches[0]


def formal_blocks(config: Config) -> list[dict]:
    blocks = list(config.section("formal").get("blocks", []))
    expected = int(config.section("formal").get("expected_blocks", 2))
    if len(blocks) != expected:
        raise ValueError(f"formal.blocks has {len(blocks)} entries; expected {expected}")
    numbers = [int(x["number"]) for x in blocks]
    if numbers != [1, 2]:
        raise ValueError(f"final formal behavior requires block numbers [1, 2], got {numbers}")
    return blocks


def formal_block_files(config: Config, subject: str) -> list[Path]:
    beh_dir = subject_behavior_dir(config, subject)
    result: list[Path] = []
    for spec in formal_blocks(config):
        block_num = int(spec["number"])
        condition = str(spec.get("condition", "B"))
        pattern = f"{subject}_Block{block_num}_{condition}_beh.csv"
        candidates = sorted(beh_dir.glob(pattern))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"{subject} Block{block_num}_{condition}: expected one `{pattern}`, found {len(candidates)}"
            )
        result.append(candidates[0])
    return result


def _add_derived(trials: pd.DataFrame, config: Config) -> pd.DataFrame:
    df = trials.copy()
    qc = config.section("behavior")["rt_qc_ms"]
    has_rt = df["rt"].notna()
    df["rt_qc_lt_100"] = has_rt & df["rt"].lt(qc["very_early"])
    df["rt_qc_lt_150"] = has_rt & df["rt"].lt(qc["early"])
    df["rt_qc_gt_1000"] = has_rt & df["rt"].gt(qc["long"])
    df["rt_qc_gt_1150"] = has_rt & df["rt"].gt(qc["beyond_nominal"])

    tol = float(config.section("validation").get("timestamp_tolerance_ms", 25))
    df["rt_timestamp_delta_ms"] = (
        df["response_time"] - df["absolute_onset_time"] - df["rt"]
    )
    df["rt_qc_timestamp_inconsistent"] = (
        has_rt
        & df["response_time"].notna()
        & df["absolute_onset_time"].notna()
        & df["rt_timestamp_delta_ms"].abs().gt(tol)
    )
    df["go_rt_valid"] = df["rt"].where(
        df["is_no_go"].eq(0) & df["correct"].eq(1)
    )

    labels = config.section("behavior").get("probe_labels", {})
    vigilance = config.section("behavior").get("vigilance_labels", {})
    df["probe_state_label"] = df["probe_response"].map(labels) if labels else pd.NA
    df["probe_vigilance_label"] = df["probe_vigilance"].map(vigilance) if vigilance else pd.NA

    df["time_in_block_sec"] = np.nan
    for (_, _block_num), idx in df.groupby(["subject", "block_num"]).groups.items():
        values = df.loc[idx, "absolute_onset_time"]
        origin = values.dropna().min()
        if pd.notna(origin):
            df.loc[idx, "time_in_block_sec"] = (values - origin) / 1000.0

    n_bins = int(config.section("behavior").get("cycle_bins", 6))
    df["cycle_bin"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    for (_, _block_num), idx in df.groupby(["subject", "block_num"]).groups.items():
        cycles = pd.to_numeric(df.loc[idx, "cycle_num"], errors="coerce")
        valid = cycles.notna()
        if valid.any():
            unique_n = int(cycles[valid].nunique())
            bins = max(1, min(n_bins, unique_n))
            labels_bin = np.arange(1, bins + 1)
            binned = pd.cut(cycles[valid], bins=bins, labels=labels_bin, include_lowest=True)
            df.loc[cycles[valid].index, "cycle_bin"] = binned.astype("Int64")
    return df


def extract_formal_trials(config: Config, subject: str) -> pd.DataFrame:
    frames = []
    for source in formal_block_files(config, subject):
        frame = pd.read_csv(source, encoding="utf-8-sig")
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{source}: missing required columns {sorted(missing)}")
        frame["source_file"] = str(source.resolve())
        frame["source_row"] = np.arange(2, len(frame) + 2)
        frames.append(frame)

    trials = pd.concat(frames, ignore_index=True)
    for column in NUMERIC_COLUMNS:
        trials[column] = pd.to_numeric(trials[column], errors="coerce")
    trials["condition"] = trials["condition"].astype(str).str.strip()
    trials.insert(0, "subject", subject)
    trials = trials.sort_values(["block_num", "trial_num"]).reset_index(drop=True)
    return _add_derived(trials, config)


def load_cohort(config: Config, subjects: list[str] | None = None) -> pd.DataFrame:
    subjects = subjects or discover_subjects(config)
    frames = [extract_formal_trials(config, subject) for subject in subjects]
    return pd.concat(frames, ignore_index=True)


def validate_formal(config: Config, trials: pd.DataFrame) -> pd.DataFrame:
    """Validate two-block structure without importing v3.0 BBB count assumptions."""
    validation = config.section("validation")
    expected_blocks = {1, 2}
    rows = []

    for subject, subject_df in trials.groupby("subject", sort=True):
        actual_blocks = set(subject_df["block_num"].dropna().astype(int).unique())
        if actual_blocks != expected_blocks:
            raise ValueError(f"{subject}: blocks={sorted(actual_blocks)}, expected [1, 2]")

        for block_num, block_df in subject_df.groupby("block_num", sort=True):
            condition_values = set(block_df["condition"].dropna().astype(str))
            if condition_values != {"B"}:
                raise ValueError(
                    f"{subject} block {int(block_num)}: condition={sorted(condition_values)}, expected ['B']"
                )
            trial_count = int(len(block_df))
            nogo_count = int(block_df["is_no_go"].fillna(0).astype(int).sum())
            probe_count = int(block_df["is_probe"].fillna(0).astype(int).sum())
            rows.append({
                "subject": subject,
                "block_num": int(block_num),
                "trial_count": trial_count,
                "nogo_count": nogo_count,
                "probe_count": probe_count,
            })

    report = pd.DataFrame(rows)
    if report.empty:
        raise ValueError("No formal behavior blocks were validated")

    for column, key in (
        ("trial_count", "expected_trials_per_block"),
        ("nogo_count", "expected_nogo_per_block"),
        ("probe_count", "expected_probes_per_block"),
    ):
        expected = validation.get(key)
        if expected is not None and not report[column].eq(int(expected)).all():
            bad = report.loc[~report[column].eq(int(expected)), ["subject", "block_num", column]]
            raise ValueError(f"{key} mismatch: {bad.to_dict(orient='records')}")

    if bool(validation.get("require_cross_subject_count_consistency", True)):
        for block_num, block_df in report.groupby("block_num"):
            for column in ("trial_count", "nogo_count", "probe_count"):
                if block_df[column].nunique(dropna=False) != 1:
                    raise ValueError(
                        f"block {int(block_num)} has inconsistent {column} across subjects: "
                        f"{block_df[['subject', column]].to_dict(orient='records')}"
                    )
    return report
