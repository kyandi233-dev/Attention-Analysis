from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from attention_pipeline.config import Config, load_config
from attention_pipeline.nir_behavior.alignment import _normalize_eye
from attention_pipeline.nir_behavior.contract import (
    FULLCLASS_EXTENSION_VERSION,
    PIR_COLUMN,
    PIR_VALID_COLUMN,
    normalize_subject,
    parse_subject_list,
)
from attention_pipeline.nir_behavior.discovery import (
    find_nir_source,
    nir_source_roots,
    resolve_repo_path,
)
from attention_pipeline.nir_behavior.features import coerce_bool_series


ANALYSIS_READY_SCHEMA_VERSION = 1
ANALYSIS_READY_PIPELINE_VERSION = "nir-analysis-ready-v1"
ROBUST_Z_SCALE = 1.4826
FORMAL_PHASE_TO_BLOCK = {"block1": 1, "block2": 2}

PUPIL_FIT_COLUMN = "fullclass_pupil_fit_valid"
IRIS_FIT_COLUMN = "fullclass_iris_outer_fit_valid"
CENTER_IN_COLUMN = "fullclass_pupil_center_in_iris_outer"
PUPIL_DIAMETER_COLUMN = "fullclass_pupil_geom_mean_diameter"
IRIS_DIAMETER_COLUMN = "fullclass_iris_outer_geom_mean_diameter"

REQUIRED_SOURCE_COLUMNS = (
    "subject",
    "phase",
    "phase_segment",
    "frame_idx",
    "video_time_ms",
    "unix_ms",
    "phase_time_ms",
    "eye",
    PIR_COLUMN,
    PIR_VALID_COLUMN,
    PUPIL_FIT_COLUMN,
    IRIS_FIT_COLUMN,
    CENTER_IN_COLUMN,
    PUPIL_DIAMETER_COLUMN,
    IRIS_DIAMETER_COLUMN,
)

OPTIONAL_QC_COLUMNS = (
    "fullclass_pupil_touches_roi_edge",
    "fullclass_iris_outer_touches_roi_edge",
    "roi_clipped",
    "ritnet_found",
    "fullclass_pupil_confidence",
    "fullclass_ocular_component_count",
    "fullclass_ocular_largest_component_fraction",
)

BOOLEAN_COLUMNS = {
    PIR_VALID_COLUMN,
    PUPIL_FIT_COLUMN,
    IRIS_FIT_COLUMN,
    CENTER_IN_COLUMN,
    "fullclass_pupil_touches_roi_edge",
    "fullclass_iris_outer_touches_roi_edge",
    "roi_clipped",
    "ritnet_found",
}

TIME_COLUMNS = ("unix_ms", "video_time_ms", "phase_time_ms")
PAIR_KEY_COLUMNS = ("subject", "phase", "phase_segment", "frame_idx")


@dataclass(frozen=True)
class SubjectMaterialization:
    subject: str
    source_csv: Path
    completion_marker: Path
    output_csv: Path
    baseline_csv: Path
    n_source_formal_rows: int
    n_timepoints: int
    n_primary_valid: int
    n_strict_valid: int
    n_recovered: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_fraction(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return float("nan")
    return float(numerator) / float(denominator)


def _finite_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.isfinite(values.to_numpy(dtype=float)), index=series.index)


def _median_mad(values: pd.Series) -> tuple[float, float, float, bool]:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if numeric.empty:
        return float("nan"), float("nan"), float("nan"), False
    median = float(numeric.median())
    mad = float(np.median(np.abs(numeric.to_numpy(dtype=float) - median)))
    robust_sigma = ROBUST_Z_SCALE * mad
    denominator_valid = bool(np.isfinite(robust_sigma) and robust_sigma > 0)
    return median, mad, robust_sigma, denominator_valid


def derive_frame_validity(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive primary and strict PIR validity without mutating production data."""
    result = frame.copy()

    pupil_fit = coerce_bool_series(result[PUPIL_FIT_COLUMN]).fillna(False).astype(bool)
    iris_fit = coerce_bool_series(result[IRIS_FIT_COLUMN]).fillna(False).astype(bool)
    center_in = coerce_bool_series(result[CENTER_IN_COLUMN]).fillna(False).astype(bool)
    production_valid = coerce_bool_series(result[PIR_VALID_COLUMN]).fillna(False).astype(bool)

    pupil_d = pd.to_numeric(result[PUPIL_DIAMETER_COLUMN], errors="coerce")
    iris_d = pd.to_numeric(result[IRIS_DIAMETER_COLUMN], errors="coerce")
    pir_finite = _finite_numeric(result[PIR_COLUMN])
    iris_larger = (
        np.isfinite(pupil_d.to_numpy(dtype=float))
        & np.isfinite(iris_d.to_numpy(dtype=float))
        & (iris_d.to_numpy(dtype=float) > pupil_d.to_numpy(dtype=float))
    )
    iris_larger = pd.Series(iris_larger, index=result.index)

    result["pir_finite"] = pir_finite.astype(bool)
    result["pir_valid_primary"] = (
        pupil_fit & iris_fit & center_in & iris_larger & pir_finite
    ).astype(bool)
    result["pir_valid_strict"] = (production_valid & pir_finite).astype(bool)

    strict_not_primary = result["pir_valid_strict"] & ~result["pir_valid_primary"]
    if bool(strict_not_primary.any()):
        example_cols = [
            "subject",
            "phase",
            "phase_segment",
            "frame_idx",
            "eye",
            PIR_VALID_COLUMN,
            PUPIL_FIT_COLUMN,
            IRIS_FIT_COLUMN,
            CENTER_IN_COLUMN,
            PUPIL_DIAMETER_COLUMN,
            IRIS_DIAMETER_COLUMN,
            PIR_COLUMN,
        ]
        examples = result.loc[strict_not_primary, example_cols].head(5).to_dict("records")
        raise ValueError(
            "Validity contract violation: pir_valid_strict must be a subset of "
            f"pir_valid_primary; examples={examples}"
        )
    return result


def compute_subject_eye_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (subject, eye), group in frame.groupby(["subject", "eye"], sort=True):
        primary = group.loc[group["pir_valid_primary"], PIR_COLUMN]
        strict = group.loc[group["pir_valid_strict"], PIR_COLUMN]

        p_median, p_mad, p_sigma, p_sigma_valid = _median_mad(primary)
        s_median, s_mad, s_sigma, s_sigma_valid = _median_mad(strict)

        row: dict[str, Any] = {
            "subject": subject,
            "eye": eye,
            "n_formal_rows": int(len(group)),
            "n_primary_valid": int(group["pir_valid_primary"].sum()),
            "primary_valid_fraction": _safe_fraction(
                int(group["pir_valid_primary"].sum()), len(group)
            ),
            "primary_median_PIR": p_median,
            "primary_MAD_PIR": p_mad,
            "primary_robust_sigma_PIR": p_sigma,
            "primary_robust_sigma_valid": p_sigma_valid,
            "n_strict_valid": int(group["pir_valid_strict"].sum()),
            "strict_valid_fraction": _safe_fraction(
                int(group["pir_valid_strict"].sum()), len(group)
            ),
            "strict_median_PIR": s_median,
            "strict_MAD_PIR": s_mad,
            "strict_robust_sigma_PIR": s_sigma,
            "strict_robust_sigma_valid": s_sigma_valid,
        }
        for phase, block in FORMAL_PHASE_TO_BLOCK.items():
            subset = group[group["phase"] == phase]
            row[f"block{block}_n_rows"] = int(len(subset))
            row[f"block{block}_n_primary_valid"] = int(
                subset["pir_valid_primary"].sum()
            )
            row[f"block{block}_n_strict_valid"] = int(
                subset["pir_valid_strict"].sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def apply_subject_eye_standardization(
    frame: pd.DataFrame, baselines: pd.DataFrame
) -> pd.DataFrame:
    result = frame.merge(
        baselines[
            [
                "subject",
                "eye",
                "primary_median_PIR",
                "primary_robust_sigma_PIR",
                "primary_robust_sigma_valid",
                "strict_median_PIR",
                "strict_robust_sigma_PIR",
                "strict_robust_sigma_valid",
            ]
        ],
        on=["subject", "eye"],
        how="left",
        validate="many_to_one",
    )

    pir = pd.to_numeric(result[PIR_COLUMN], errors="coerce")
    result["pir_centered_primary"] = np.where(
        result["pir_valid_primary"],
        pir - result["primary_median_PIR"],
        np.nan,
    )
    result["pir_robust_z_primary"] = np.where(
        result["pir_valid_primary"] & result["primary_robust_sigma_valid"],
        (pir - result["primary_median_PIR"]) / result["primary_robust_sigma_PIR"],
        np.nan,
    )
    result["pir_centered_strict"] = np.where(
        result["pir_valid_strict"],
        pir - result["strict_median_PIR"],
        np.nan,
    )
    result["pir_robust_z_strict"] = np.where(
        result["pir_valid_strict"] & result["strict_robust_sigma_valid"],
        (pir - result["strict_median_PIR"]) / result["strict_robust_sigma_PIR"],
        np.nan,
    )
    return result


def _assert_pair_uniqueness(frame: pd.DataFrame) -> None:
    duplicate = frame.duplicated(list(PAIR_KEY_COLUMNS) + ["eye"], keep=False)
    if bool(duplicate.any()):
        examples = (
            frame.loc[duplicate, list(PAIR_KEY_COLUMNS) + ["eye", "source_row"]]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(f"Duplicate eye rows for one frame identity: {examples}")


def _assert_time_consistency(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    tolerance_ms: float,
) -> None:
    paired = left[list(PAIR_KEY_COLUMNS) + list(TIME_COLUMNS)].merge(
        right[list(PAIR_KEY_COLUMNS) + list(TIME_COLUMNS)],
        on=list(PAIR_KEY_COLUMNS),
        how="inner",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    for column in TIME_COLUMNS:
        a = pd.to_numeric(paired[f"{column}_left"], errors="coerce")
        b = pd.to_numeric(paired[f"{column}_right"], errors="coerce")
        both = np.isfinite(a.to_numpy(dtype=float)) & np.isfinite(b.to_numpy(dtype=float))
        delta = np.abs(a.to_numpy(dtype=float) - b.to_numpy(dtype=float))
        bad = both & (delta > tolerance_ms)
        if bool(np.any(bad)):
            rows = paired.loc[bad, list(PAIR_KEY_COLUMNS) + [f"{column}_left", f"{column}_right"]]
            raise ValueError(
                f"Left/right {column} mismatch exceeds {tolerance_ms} ms; "
                f"examples={rows.head(5).to_dict('records')}"
            )


def _wide_eye_columns(frame: pd.DataFrame, eye: str) -> pd.DataFrame:
    subset = frame[frame["eye"] == eye].copy()
    keep = [
        *PAIR_KEY_COLUMNS,
        *TIME_COLUMNS,
        "source_row",
        PIR_COLUMN,
        PIR_VALID_COLUMN,
        PUPIL_FIT_COLUMN,
        IRIS_FIT_COLUMN,
        CENTER_IN_COLUMN,
        PUPIL_DIAMETER_COLUMN,
        IRIS_DIAMETER_COLUMN,
        "pir_finite",
        "pir_valid_primary",
        "pir_valid_strict",
        "pir_centered_primary",
        "pir_robust_z_primary",
        "pir_centered_strict",
        "pir_robust_z_strict",
        "primary_robust_sigma_valid",
        "strict_robust_sigma_valid",
    ]
    keep += [column for column in OPTIONAL_QC_COLUMNS if column in subset.columns]
    subset = subset[keep]

    rename = {
        "source_row": f"{eye}_source_row",
        PIR_COLUMN: f"{eye}_raw_PIR",
        PIR_VALID_COLUMN: f"{eye}_production_normalization_valid",
        PUPIL_FIT_COLUMN: f"{eye}_pupil_fit_valid",
        IRIS_FIT_COLUMN: f"{eye}_iris_outer_fit_valid",
        CENTER_IN_COLUMN: f"{eye}_pupil_center_in_iris_outer",
        PUPIL_DIAMETER_COLUMN: f"{eye}_pupil_geom_mean_diameter",
        IRIS_DIAMETER_COLUMN: f"{eye}_iris_outer_geom_mean_diameter",
        "pir_finite": f"{eye}_pir_finite",
        "pir_valid_primary": f"{eye}_valid_primary",
        "pir_valid_strict": f"{eye}_valid_strict",
        "pir_centered_primary": f"{eye}_centered_PIR",
        "pir_robust_z_primary": f"{eye}_robust_z_PIR",
        "pir_centered_strict": f"{eye}_strict_centered_PIR",
        "pir_robust_z_strict": f"{eye}_strict_robust_z_PIR",
        "primary_robust_sigma_valid": f"{eye}_robust_sigma_valid",
        "strict_robust_sigma_valid": f"{eye}_strict_robust_sigma_valid",
    }
    for column in OPTIONAL_QC_COLUMNS:
        if column in subset.columns:
            rename[column] = f"{eye}_{column}"
    return subset.rename(columns=rename)


def _coalesce_time_columns(wide: pd.DataFrame) -> pd.DataFrame:
    result = wide.copy()
    for column in TIME_COLUMNS:
        left = f"{column}_left"
        right = f"{column}_right"
        result[column] = result[left].combine_first(result[right])
        result = result.drop(columns=[left, right])
    return result


def _source_mode(left_valid: pd.Series, right_valid: pd.Series) -> pd.Series:
    left_valid = left_valid.fillna(False).astype(bool)
    right_valid = right_valid.fillna(False).astype(bool)
    values = np.select(
        [
            left_valid & right_valid,
            left_valid & ~right_valid,
            ~left_valid & right_valid,
        ],
        ["binocular", "left_only", "right_only"],
        default="missing",
    )
    return pd.Series(values, index=left_valid.index, dtype="object")


def _fuse_values(
    left_value: pd.Series,
    right_value: pd.Series,
    left_valid: pd.Series,
    right_valid: pd.Series,
) -> pd.Series:
    lv = pd.to_numeric(left_value, errors="coerce")
    rv = pd.to_numeric(right_value, errors="coerce")
    l_ok = left_valid.fillna(False).astype(bool) & _finite_numeric(lv)
    r_ok = right_valid.fillna(False).astype(bool) & _finite_numeric(rv)
    values = np.select(
        [l_ok & r_ok, l_ok & ~r_ok, ~l_ok & r_ok],
        [(lv + rv) / 2.0, lv, rv],
        default=np.nan,
    )
    return pd.Series(values, index=left_value.index, dtype=float)


def build_wide_timepoints(
    frame: pd.DataFrame, *, time_tolerance_ms: float = 1.0
) -> pd.DataFrame:
    _assert_pair_uniqueness(frame)
    left_source = frame[frame["eye"] == "left"].copy()
    right_source = frame[frame["eye"] == "right"].copy()
    _assert_time_consistency(left_source, right_source, tolerance_ms=time_tolerance_ms)

    left = _wide_eye_columns(frame, "left")
    right = _wide_eye_columns(frame, "right")
    wide = left.merge(
        right,
        on=list(PAIR_KEY_COLUMNS),
        how="outer",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    wide = _coalesce_time_columns(wide)
    wide["block"] = wide["phase"].map(FORMAL_PHASE_TO_BLOCK).astype("Int64")

    for column in (
        "left_valid_primary",
        "right_valid_primary",
        "left_valid_strict",
        "right_valid_strict",
        "left_robust_sigma_valid",
        "right_robust_sigma_valid",
        "left_strict_robust_sigma_valid",
        "right_strict_robust_sigma_valid",
    ):
        if column not in wide.columns:
            wide[column] = False
        wide[column] = wide[column].fillna(False).astype(bool)

    wide["binocular_source_mode"] = _source_mode(
        wide["left_valid_primary"], wide["right_valid_primary"]
    )
    wide["binocular_centered_PIR"] = _fuse_values(
        wide["left_centered_PIR"],
        wide["right_centered_PIR"],
        wide["left_valid_primary"],
        wide["right_valid_primary"],
    )
    # Explicit alias required by the analysis contract.
    wide["binocular_PIR"] = wide["binocular_centered_PIR"]
    wide["binocular_robust_z_PIR"] = _fuse_values(
        wide["left_robust_z_PIR"],
        wide["right_robust_z_PIR"],
        wide["left_valid_primary"] & wide["left_robust_sigma_valid"],
        wide["right_valid_primary"] & wide["right_robust_sigma_valid"],
    )

    wide["binocular_strict_source_mode"] = _source_mode(
        wide["left_valid_strict"], wide["right_valid_strict"]
    )
    wide["binocular_strict_PIR"] = _fuse_values(
        wide["left_strict_centered_PIR"],
        wide["right_strict_centered_PIR"],
        wide["left_valid_strict"],
        wide["right_valid_strict"],
    )
    wide["binocular_strict_robust_z_PIR"] = _fuse_values(
        wide["left_strict_robust_z_PIR"],
        wide["right_strict_robust_z_PIR"],
        wide["left_valid_strict"] & wide["left_strict_robust_sigma_valid"],
        wide["right_valid_strict"] & wide["right_strict_robust_sigma_valid"],
    )

    primary_has_value = _finite_numeric(wide["binocular_PIR"])
    expected_primary_has_value = wide["binocular_source_mode"].ne("missing")
    if bool((primary_has_value != expected_primary_has_value).any()):
        raise ValueError("Binocular primary source_mode/value invariant failed")

    strict_has_value = _finite_numeric(wide["binocular_strict_PIR"])
    expected_strict_has_value = wide["binocular_strict_source_mode"].ne("missing")
    if bool((strict_has_value != expected_strict_has_value).any()):
        raise ValueError("Binocular strict source_mode/value invariant failed")

    preferred = [
        "subject",
        "block",
        "phase",
        "phase_segment",
        "frame_idx",
        "unix_ms",
        "video_time_ms",
        "phase_time_ms",
        "left_source_row",
        "right_source_row",
        "left_raw_PIR",
        "right_raw_PIR",
        "left_production_normalization_valid",
        "right_production_normalization_valid",
        "left_pir_finite",
        "right_pir_finite",
        "left_pupil_fit_valid",
        "right_pupil_fit_valid",
        "left_iris_outer_fit_valid",
        "right_iris_outer_fit_valid",
        "left_pupil_center_in_iris_outer",
        "right_pupil_center_in_iris_outer",
        "left_pupil_geom_mean_diameter",
        "right_pupil_geom_mean_diameter",
        "left_iris_outer_geom_mean_diameter",
        "right_iris_outer_geom_mean_diameter",
        "left_valid_primary",
        "right_valid_primary",
        "left_valid_strict",
        "right_valid_strict",
        "left_centered_PIR",
        "right_centered_PIR",
        "left_robust_z_PIR",
        "right_robust_z_PIR",
        "left_strict_centered_PIR",
        "right_strict_centered_PIR",
        "left_strict_robust_z_PIR",
        "right_strict_robust_z_PIR",
        "left_robust_sigma_valid",
        "right_robust_sigma_valid",
        "left_strict_robust_sigma_valid",
        "right_strict_robust_sigma_valid",
        "binocular_PIR",
        "binocular_centered_PIR",
        "binocular_robust_z_PIR",
        "binocular_source_mode",
        "binocular_strict_PIR",
        "binocular_strict_robust_z_PIR",
        "binocular_strict_source_mode",
    ]
    remaining = [column for column in wide.columns if column not in preferred]
    wide = wide[[column for column in preferred if column in wide.columns] + sorted(remaining)]
    return wide.sort_values(
        ["block", "phase_segment", "frame_idx"], kind="stable"
    ).reset_index(drop=True)


def subject_eye_block_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (subject, phase, eye), group in frame.groupby(
        ["subject", "phase", "eye"], sort=True
    ):
        n_total = int(len(group))
        production_valid = coerce_bool_series(group[PIR_VALID_COLUMN]).fillna(False).astype(bool)
        n_production = int(production_valid.sum())
        n_primary = int(group["pir_valid_primary"].sum())
        n_strict = int(group["pir_valid_strict"].sum())
        n_recovered_vs_production = int(
            (group["pir_valid_primary"] & ~production_valid).sum()
        )
        n_recovered_vs_strict = int(
            (group["pir_valid_primary"] & ~group["pir_valid_strict"]).sum()
        )
        old_production_invalid = n_total - n_production
        old_strict_invalid = n_total - n_strict
        rows.append(
            {
                "subject": subject,
                "block": FORMAL_PHASE_TO_BLOCK.get(phase),
                "phase": phase,
                "eye": eye,
                "n_rows": n_total,
                "n_production_normalization_valid": n_production,
                "production_normalization_valid_fraction": _safe_fraction(
                    n_production, n_total
                ),
                "n_primary_valid": n_primary,
                "primary_valid_fraction": _safe_fraction(n_primary, n_total),
                "n_strict_valid": n_strict,
                "strict_valid_fraction": _safe_fraction(n_strict, n_total),
                "n_recovered_primary_vs_production": n_recovered_vs_production,
                "recovered_vs_production_fraction_all_rows": _safe_fraction(
                    n_recovered_vs_production, n_total
                ),
                "recovered_fraction_old_production_invalid": _safe_fraction(
                    n_recovered_vs_production, old_production_invalid
                ),
                "n_recovered_primary_vs_strict": n_recovered_vs_strict,
                "recovered_vs_strict_fraction_all_rows": _safe_fraction(
                    n_recovered_vs_strict, n_total
                ),
                "recovered_fraction_old_strict_invalid": _safe_fraction(
                    n_recovered_vs_strict, old_strict_invalid
                ),
                "n_strict_not_primary": int(
                    (group["pir_valid_strict"] & ~group["pir_valid_primary"]).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _alignment_config(config: Config) -> Config:
    raw = config.section("paths").get("alignment_config")
    if raw is None:
        raise KeyError("analysis-ready config missing paths.alignment_config")
    return load_config(resolve_repo_path(config, raw))


def _selected_subjects(
    config: Config, override: Iterable[str] | None = None
) -> list[str]:
    if override:
        return parse_subject_list(override)
    raw = config.section("subjects").get("include", [])
    if raw:
        return parse_subject_list(raw)
    alignment = _alignment_config(config)
    subjects: set[str] = set()
    for root in nir_source_roots(alignment):
        if not root.is_dir():
            continue
        for marker in root.glob(
            "sub-*_formal_*/*_ritnet_fullclass_v1-2-fast-qc_completion.json"
        ):
            try:
                with marker.open(encoding="utf-8") as handle:
                    value = json.load(handle)
                if value.get("status") != "complete":
                    continue
                if value.get("extension_version") != FULLCLASS_EXTENSION_VERSION:
                    continue
                if bool(value.get("pupil_validation_mode")):
                    continue
                subject = value.get(
                    "subject", marker.parent.name.split("_formal_", 1)[0]
                )
                subjects.add(normalize_subject(subject))
            except Exception:
                continue
    return sorted(subjects, key=lambda value: int(value.split("-")[1]))


def _output_root(config: Config, override: str | Path | None = None) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()
    raw = config.section("paths").get("output_root")
    if raw is None:
        raise KeyError("analysis-ready config missing paths.output_root")
    return resolve_repo_path(config, raw)


def _guard_output_root(config: Config, output_root: Path) -> None:
    alignment = _alignment_config(config)
    output = output_root.resolve()
    for source_root in nir_source_roots(alignment):
        source = source_root.resolve()
        if output == source or source in output.parents:
            raise ValueError(
                f"Refusing analysis-ready output inside frozen production root: {output}"
            )


def _formal_phases(config: Config) -> list[str]:
    phases = config.section("materialization").get(
        "formal_phases", ["block1", "block2"]
    )
    result = [str(value).strip() for value in phases]
    invalid = [value for value in result if value not in FORMAL_PHASE_TO_BLOCK]
    if invalid:
        raise ValueError(f"Unsupported formal phases: {invalid}")
    return result


def _time_tolerance_ms(config: Config) -> float:
    value = float(
        config.section("materialization").get("max_eye_time_delta_ms", 1.0)
    )
    if value < 0:
        raise ValueError("max_eye_time_delta_ms must be >= 0")
    return value


def _validate_standardization_config(config: Config) -> None:
    scale = float(config.section("standardization").get("robust_z_scale", ROBUST_Z_SCALE))
    if not np.isclose(scale, ROBUST_Z_SCALE, rtol=0, atol=1e-12):
        raise ValueError(
            f"robust_z_scale must remain frozen at {ROBUST_Z_SCALE}; got {scale}"
        )
    if bool(config.section("standardization").get("block_centering_primary", False)):
        raise ValueError("Block-specific centering cannot be primary in this contract")


def _load_subject_frame(
    config: Config, subject: str
) -> tuple[pd.DataFrame, Any]:
    alignment = _alignment_config(config)
    source = find_nir_source(alignment, subject)

    header = pd.read_csv(source.csv_path, nrows=0, encoding="utf-8-sig")
    available = set(header.columns)
    missing = sorted(set(REQUIRED_SOURCE_COLUMNS) - available)
    if missing:
        raise ValueError(f"{source.csv_path}: missing required columns {missing}")
    usecols = list(REQUIRED_SOURCE_COLUMNS) + [
        column for column in OPTIONAL_QC_COLUMNS if column in available
    ]
    frame = pd.read_csv(
        source.csv_path,
        usecols=usecols,
        encoding="utf-8-sig",
        low_memory=False,
    )
    # 1-based CSV line number including the header line for direct source tracing.
    frame["source_row"] = np.arange(len(frame), dtype=np.int64) + 2

    expected_subject = normalize_subject(subject)
    frame["subject"] = frame["subject"].map(normalize_subject)
    actual = set(frame["subject"].dropna().unique())
    if actual != {expected_subject}:
        raise ValueError(
            f"{source.csv_path}: unexpected subject identifiers {sorted(actual)}"
        )

    frame["phase"] = frame["phase"].astype(str).str.strip()
    frame = frame[frame["phase"].isin(_formal_phases(config))].copy()
    frame["eye"] = frame["eye"].map(_normalize_eye)
    unexpected_eyes = sorted(
        set(frame["eye"].dropna().unique()) - {"left", "right"}
    )
    if unexpected_eyes:
        raise ValueError(f"{source.csv_path}: unexpected eyes {unexpected_eyes}")

    for column in frame.columns:
        if column in {"subject", "phase", "eye"}:
            continue
        if column in BOOLEAN_COLUMNS:
            frame[column] = coerce_bool_series(frame[column])
        elif column != "source_row":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    missing_identity = frame[
        ["phase_segment", "frame_idx", "eye", *TIME_COLUMNS]
    ].isna().any(axis=1)
    if bool(missing_identity.any()):
        examples = frame.loc[
            missing_identity,
            ["subject", "phase", "phase_segment", "frame_idx", "eye", *TIME_COLUMNS],
        ].head(5)
        raise ValueError(
            "Formal source contains missing frame/time identity values; "
            f"examples={examples.to_dict('records')}"
        )

    frame["phase_segment"] = frame["phase_segment"].astype(np.int64)
    frame["frame_idx"] = frame["frame_idx"].astype(np.int64)
    frame = frame.sort_values(
        ["phase", "phase_segment", "frame_idx", "eye"], kind="stable"
    ).reset_index(drop=True)
    return derive_frame_validity(frame), source


def _subject_paths(output_root: Path, subject: str) -> dict[str, Path]:
    frame_dir = output_root / "frame_level" / subject
    baseline_dir = output_root / "baselines"
    return {
        "frame_dir": frame_dir,
        "frame": frame_dir / f"{subject}_nir_analysis_ready.csv",
        "baseline": baseline_dir / f"{subject}_eye_baselines.csv",
    }


def _write_csv(
    frame: pd.DataFrame, path: Path, *, overwrite_derived: bool
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite_derived:
        raise FileExistsError(
            f"Derived output already exists: {path}; use --overwrite-derived to rebuild"
        )
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def materialize_subject(
    config: Config,
    subject: str,
    output_root: Path,
    *,
    overwrite_derived: bool = False,
) -> tuple[
    SubjectMaterialization,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    subject = normalize_subject(subject)
    frame, source = _load_subject_frame(config, subject)
    baselines = compute_subject_eye_baselines(frame)
    standardized = apply_subject_eye_standardization(frame, baselines)
    wide = build_wide_timepoints(
        standardized, time_tolerance_ms=_time_tolerance_ms(config)
    )
    inclusion = subject_eye_block_summary(standardized)

    paths = _subject_paths(output_root, subject)
    _write_csv(wide, paths["frame"], overwrite_derived=overwrite_derived)
    _write_csv(baselines, paths["baseline"], overwrite_derived=overwrite_derived)

    n_primary = int(standardized["pir_valid_primary"].sum())
    n_strict = int(standardized["pir_valid_strict"].sum())
    n_recovered = int(
        (standardized["pir_valid_primary"] & ~standardized["pir_valid_strict"]).sum()
    )

    provenance = {
        "subject": subject,
        "source_csv": str(source.csv_path),
        "source_csv_size_bytes": int(source.csv_path.stat().st_size),
        "source_csv_mtime_ns": int(source.csv_path.stat().st_mtime_ns),
        "completion_marker": str(source.completion_path),
        "completion_marker_sha256": _sha256(source.completion_path),
        "extension_version": source.completion.get("extension_version"),
        "alternative_run_dirs": [str(value) for value in source.alternatives],
        "derived_frame_csv": str(paths["frame"]),
        "derived_baseline_csv": str(paths["baseline"]),
    }
    if bool(config.section("provenance").get("hash_source_csv", False)):
        provenance["source_csv_sha256"] = _sha256(source.csv_path)
    else:
        provenance["source_csv_sha256"] = None

    result = SubjectMaterialization(
        subject=subject,
        source_csv=source.csv_path,
        completion_marker=source.completion_path,
        output_csv=paths["frame"],
        baseline_csv=paths["baseline"],
        n_source_formal_rows=int(len(standardized)),
        n_timepoints=int(len(wide)),
        n_primary_valid=n_primary,
        n_strict_valid=n_strict,
        n_recovered=n_recovered,
    )
    return result, baselines, inclusion, provenance


def _source_mode_summary(
    wide_paths: list[Path], *, mode_column: str
) -> dict[str, Any]:
    counts = {"binocular": 0, "left_only": 0, "right_only": 0, "missing": 0}
    total = 0
    for path in wide_paths:
        values = pd.read_csv(
            path, usecols=[mode_column], encoding="utf-8-sig", low_memory=False
        )[mode_column].fillna("missing")
        vc = values.value_counts()
        total += int(len(values))
        for key in counts:
            counts[key] += int(vc.get(key, 0))
    return {
        "n_timepoints": total,
        "counts": counts,
        "fractions": {
            key: _safe_fraction(value, total) for key, value in counts.items()
        },
    }


def _known_low_changes(
    inclusion: pd.DataFrame, subjects: Iterable[str]
) -> pd.DataFrame:
    wanted = {normalize_subject(value) for value in subjects}
    result = inclusion[inclusion["subject"].isin(wanted)].copy()
    if result.empty:
        return result
    return result.sort_values(["subject", "block", "eye"]).reset_index(drop=True)


def run_materialization(
    config_path: str | Path,
    *,
    subjects: Iterable[str] | None = None,
    output_root_override: str | Path | None = None,
    overwrite_derived: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    _validate_standardization_config(config)
    output_root = _output_root(config, output_root_override)
    _guard_output_root(config, output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    selected = _selected_subjects(config, subjects)
    if not selected:
        raise ValueError("No completed NIR subjects selected")

    results: list[SubjectMaterialization] = []
    baselines_all: list[pd.DataFrame] = []
    inclusion_all: list[pd.DataFrame] = []
    provenance_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for subject in selected:
        try:
            result, baselines, inclusion, provenance = materialize_subject(
                config,
                subject,
                output_root,
                overwrite_derived=overwrite_derived,
            )
            results.append(result)
            baselines_all.append(baselines)
            inclusion_all.append(inclusion)
            provenance_rows.append(provenance)
        except Exception as exc:
            failures.append(
                {
                    "subject": normalize_subject(subject),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    qc_dir = output_root / "qc"
    baseline_dir = output_root / "baselines"
    provenance_dir = output_root / "provenance"
    for directory in (qc_dir, baseline_dir, provenance_dir):
        directory.mkdir(parents=True, exist_ok=True)

    failures_df = pd.DataFrame(failures, columns=["subject", "error_type", "error"])
    failures_path = qc_dir / "subject_load_failures.csv"
    if failures_path.exists() and not overwrite_derived:
        raise FileExistsError(
            f"Derived output already exists: {failures_path}; use --overwrite-derived"
        )
    failures_df.to_csv(failures_path, index=False, encoding="utf-8-sig")

    if not results:
        raise RuntimeError(f"All selected subjects failed; see {failures_path}")

    baselines_df = pd.concat(baselines_all, ignore_index=True)
    inclusion_df = pd.concat(inclusion_all, ignore_index=True)
    source_df = pd.DataFrame(provenance_rows)

    cohort_baselines_path = baseline_dir / "subject_eye_baselines.csv"
    inclusion_path = qc_dir / "subject_eye_block_inclusion.csv"
    source_path = provenance_dir / "source_files.csv"
    for frame, path in (
        (baselines_df, cohort_baselines_path),
        (inclusion_df, inclusion_path),
        (source_df, source_path),
    ):
        if path.exists() and not overwrite_derived:
            raise FileExistsError(
                f"Derived output already exists: {path}; use --overwrite-derived"
            )
        frame.to_csv(path, index=False, encoding="utf-8-sig")

    n_rows = int(inclusion_df["n_rows"].sum())
    n_production = int(inclusion_df["n_production_normalization_valid"].sum())
    n_primary = int(inclusion_df["n_primary_valid"].sum())
    n_strict = int(inclusion_df["n_strict_valid"].sum())
    n_recovered_vs_production = int(
        inclusion_df["n_recovered_primary_vs_production"].sum()
    )
    n_recovered_vs_strict = int(
        inclusion_df["n_recovered_primary_vs_strict"].sum()
    )
    n_strict_not_primary = int(inclusion_df["n_strict_not_primary"].sum())
    if n_strict_not_primary:
        raise RuntimeError(
            f"Contract error after cohort aggregation: strict_not_primary={n_strict_not_primary}"
        )
    old_production_invalid = n_rows - n_production
    old_strict_invalid = n_rows - n_strict

    wide_paths = [result.output_csv for result in results]
    primary_modes = _source_mode_summary(
        wide_paths, mode_column="binocular_source_mode"
    )
    strict_modes = _source_mode_summary(
        wide_paths, mode_column="binocular_strict_source_mode"
    )

    known = config.section("reporting").get("known_low_usable_subjects", [])
    known_df = _known_low_changes(inclusion_df, known)
    known_path = qc_dir / "known_low_usable_subject_changes.csv"
    if known_path.exists() and not overwrite_derived:
        raise FileExistsError(
            f"Derived output already exists: {known_path}; use --overwrite-derived"
        )
    known_df.to_csv(known_path, index=False, encoding="utf-8-sig")

    summary = {
        "pipeline_version": ANALYSIS_READY_PIPELINE_VERSION,
        "schema_version": ANALYSIS_READY_SCHEMA_VERSION,
        "cohort_label": config.section("pipeline").get("cohort_label"),
        "exploratory_cohort_only": bool(
            config.section("analysis_policy").get("exploratory_cohort_only", True)
        ),
        "config_path": str(config.path),
        "config_digest": config.digest,
        "output_root": str(output_root),
        "n_subjects_requested": len(selected),
        "n_subjects_materialized": len(results),
        "n_subjects_failed": len(failures),
        "n_formal_eye_rows": n_rows,
        "production_normalization_valid": {
            "n_valid": n_production,
            "valid_fraction": _safe_fraction(n_production, n_rows),
        },
        "primary": {
            "n_valid": n_primary,
            "valid_fraction": _safe_fraction(n_primary, n_rows),
        },
        "strict": {
            "n_valid": n_strict,
            "valid_fraction": _safe_fraction(n_strict, n_rows),
        },
        "recovered_primary_vs_production": {
            "n_rows": n_recovered_vs_production,
            "fraction_all_rows": _safe_fraction(n_recovered_vs_production, n_rows),
            "fraction_old_production_invalid": _safe_fraction(
                n_recovered_vs_production, old_production_invalid
            ),
        },
        "recovered_primary_vs_strict": {
            "n_rows": n_recovered_vs_strict,
            "fraction_all_rows": _safe_fraction(n_recovered_vs_strict, n_rows),
            "fraction_old_strict_invalid": _safe_fraction(
                n_recovered_vs_strict, old_strict_invalid
            ),
        },
        "strict_not_primary_n": n_strict_not_primary,
        "primary_binocular_source_modes": primary_modes,
        "strict_binocular_source_modes": strict_modes,
    }
    summary_path = qc_dir / "cohort_inclusion_summary.json"
    if summary_path.exists() and not overwrite_derived:
        raise FileExistsError(
            f"Derived output already exists: {summary_path}; use --overwrite-derived"
        )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": ANALYSIS_READY_PIPELINE_VERSION,
        "schema_version": ANALYSIS_READY_SCHEMA_VERSION,
        "cohort_label": config.section("pipeline").get("cohort_label"),
        "config_path": str(config.path),
        "config_digest": config.digest,
        "source_extension_version": FULLCLASS_EXTENSION_VERSION,
        "production_read_only": True,
        "primary_validity_definition": [
            PUPIL_FIT_COLUMN,
            IRIS_FIT_COLUMN,
            CENTER_IN_COLUMN,
            f"{IRIS_DIAMETER_COLUMN} > {PUPIL_DIAMETER_COLUMN}",
            f"{PIR_COLUMN} finite",
        ],
        "strict_validity_definition": [
            f"{PIR_VALID_COLUMN} == True",
            f"{PIR_COLUMN} finite",
        ],
        "standardization": {
            "primary": "subject×eye median-centered across block1+block2 primary-valid PIR",
            "robust_z_scale": ROBUST_Z_SCALE,
            "strict_track_uses_strict_baseline": True,
        },
        "binocular_base_layer": {
            "input": "per-eye centered PIR",
            "both_valid": "equal-weight mean",
            "single_eye_fallback": True,
            "coverage_gate_applied": False,
            "concordance_gate_applied": False,
        },
        "n_subjects_requested": len(selected),
        "n_subjects_materialized": len(results),
        "n_subjects_failed": len(failures),
        "subjects_materialized": [result.subject for result in results],
        "subjects_failed": [item["subject"] for item in failures],
        "artifacts": {
            "subject_eye_baselines": str(cohort_baselines_path),
            "subject_eye_block_inclusion": str(inclusion_path),
            "known_low_usable_subject_changes": str(known_path),
            "subject_load_failures": str(failures_path),
            "source_files": str(source_path),
            "cohort_inclusion_summary": str(summary_path),
        },
    }
    manifest_path = provenance_dir / "analysis_ready_manifest.json"
    if manifest_path.exists() and not overwrite_derived:
        raise FileExistsError(
            f"Derived output already exists: {manifest_path}; use --overwrite-derived"
        )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "output_root": str(output_root),
        "summary_path": str(summary_path),
        "manifest_path": str(manifest_path),
        "failures_path": str(failures_path),
        "summary": summary,
    }
