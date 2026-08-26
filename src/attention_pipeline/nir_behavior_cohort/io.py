from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..config import Config, load_config
from ..nir_behavior.alignment import _normalize_eye
from ..nir_behavior.behavior_qc import add_behavior_qc
from ..nir_behavior.contract import (
    OAR_COLUMN,
    OAR_P90_COLUMN,
    OPTIONAL_NIR_QC_COLUMNS,
    PIR_COLUMN,
    PIR_VALID_COLUMN,
    normalize_subject,
    parse_subject_list,
)
from ..nir_behavior.discovery import (
    discover_nir_subjects,
    find_nir_source,
    load_behavior_trials,
    resolve_repo_path,
)
from ..nir_behavior.features import coerce_bool_series

CORE_QC_COLUMNS = (
    "subject",
    "phase",
    "unix_ms",
    "eye",
    PIR_COLUMN,
    PIR_VALID_COLUMN,
    OAR_COLUMN,
    OAR_P90_COLUMN,
)


def alignment_config(config: Config) -> Config:
    raw = config.section("paths").get("alignment_config")
    if raw is None:
        raise KeyError("cohort config missing paths.alignment_config")
    return load_config(resolve_repo_path(config, raw))


def cohort_output_root(config: Config) -> Path:
    raw = config.section("paths").get("output_root")
    if raw is None:
        raise KeyError("cohort config missing paths.output_root")
    return resolve_repo_path(config, raw)


def selected_cohort_subjects(
    config: Config, override: Iterable[str] | None = None
) -> list[str]:
    if override:
        return parse_subject_list(override)
    raw = config.section("subjects").get("include", [])
    if raw:
        return parse_subject_list(raw)
    # Deliberately bypass the frozen alignment config's sub-031 prototype include gate.
    return discover_nir_subjects(alignment_config(config))


def load_nir_qc_frame(config: Config, subject: str) -> tuple[pd.DataFrame, object]:
    aconfig = alignment_config(config)
    source = find_nir_source(aconfig, subject)
    header = pd.read_csv(source.csv_path, nrows=0, encoding="utf-8-sig")
    available = set(header.columns)
    missing = set(CORE_QC_COLUMNS) - available
    if missing:
        raise ValueError(f"{source.csv_path}: missing cohort-QC columns {sorted(missing)}")

    usecols = [column for column in (*CORE_QC_COLUMNS, *OPTIONAL_NIR_QC_COLUMNS) if column in available]
    df = pd.read_csv(source.csv_path, usecols=usecols, encoding="utf-8-sig")
    df["subject"] = df["subject"].map(normalize_subject)
    expected_subject = normalize_subject(subject)
    actual_subjects = set(df["subject"].dropna().unique())
    if actual_subjects != {expected_subject}:
        raise ValueError(f"{source.csv_path}: unexpected subject identifiers {sorted(actual_subjects)}")

    df["eye"] = df["eye"].map(_normalize_eye)
    unexpected_eyes = sorted(set(df["eye"].dropna().unique()) - {"left", "right"})
    if unexpected_eyes:
        raise ValueError(f"{source.csv_path}: unexpected eye labels {unexpected_eyes}")

    numeric_columns = [
        "unix_ms",
        PIR_COLUMN,
        OAR_COLUMN,
        OAR_P90_COLUMN,
        "fullclass_ocular_component_count",
        "fullclass_ocular_largest_component_fraction",
        "fullclass_ocular_fraction",
        "fullclass_iris_outer_fraction",
        "fullclass_pupil_fraction",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df[PIR_VALID_COLUMN] = coerce_bool_series(df[PIR_VALID_COLUMN])
    for column in ("roi_clipped", "ritnet_found"):
        if column in df.columns:
            df[column] = coerce_bool_series(df[column])

    df = df[df["phase"].isin(["block1", "block2"])].copy()
    df["block_num"] = df["phase"].map({"block1": 1, "block2": 2}).astype(int)
    df = df[df["unix_ms"].notna()].copy()
    df = df.sort_values(["block_num", "eye", "unix_ms"]).reset_index(drop=True)
    return df, source


def load_behavior_qc_frame(config: Config, subject: str) -> pd.DataFrame:
    aconfig = alignment_config(config)
    trials = load_behavior_trials(aconfig, subject)
    carryover_ms = float(aconfig.section("behavior_qc").get("carryover_candidate_ms", 200.0))
    return add_behavior_qc(trials, carryover_ms=carryover_ms)
