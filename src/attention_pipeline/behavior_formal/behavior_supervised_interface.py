"""Materialize the probe-level Behavior -> B/A supervised interface.

The source is the authoritative 30-second primary behavior probe table. This
layer does not recompute behavior metrics, impute missing values, filter rows,
choose among unresolved feature representations, or generate ``analysis_set_id``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

from .behavior_supervised_contract import (
    FIRST_ROUND_OMISSION_DESCRIPTIVE_QC_ONLY,
    FIRST_ROUND_SUPERVISED_OMISSION_PREDICTORS,
    validate_first_round_omission_predictors,
)


INTERFACE_VERSION = "behavior-supervised-probe-v1"
PRIMARY_WINDOW_SECONDS = 30

IDENTITY_LOCATOR_COLUMNS = (
    "participant_group_id",
    "session_id",
    "block_id",
    "probe_event_id",
)

OUTCOME_COLUMNS = (
    "q1_nominal_4class",
    "q2_ordinal_4level",
)

RT_LEVEL_CANDIDATES = (
    "go_correct_rt_mean_ms",
    "go_correct_rt_median_ms",
)
RT_VARIABILITY_CANDIDATES = (
    "go_correct_rt_cv",
    "go_correct_rt_sd_ms",
    "go_correct_rt_mad_ms",
    "go_correct_rt_iqr_ms",
)
RT_TREND_CANDIDATES = (
    "go_correct_rt_theilsen_slope_ms_per_s",
)
ERROR_CONTROL_CANDIDATES = (
    "raw_go_omission_rate",
    "commission_rate",
    "dprime_loglinear",
)

FIRST_ROUND_CANDIDATE_POOL = tuple(dict.fromkeys(
    (*RT_LEVEL_CANDIDATES, *RT_VARIABILITY_CANDIDATES, *RT_TREND_CANDIDATES, *ERROR_CONTROL_CANDIDATES)
))

WINDOW_AUDIT_COLUMNS = (
    "window_seconds_nominal",
    "analysis_role",
    "formal_independent_sample",
    "anchor_trial_excluded",
    "window_crosses_block",
    "probe_order_in_block",
    "anchor_trial_num",
    "probe_time_ms",
)

OPPORTUNITY_AUDIT_COLUMNS = (
    "trial_opportunities",
    "go_opportunities",
    "nogo_opportunities",
    "correct_go_rt_opportunities",
    "omission_numerator",
    "omission_denominator",
    "commission_numerator",
    "commission_denominator",
    "raw_go_omission_n",
    "clean_go_omission_n",
    "timing_ambiguous_go_omission_n",
    "omission_taxonomy_denominator",
)

DESCRIPTIVE_QC_COLUMNS = FIRST_ROUND_OMISSION_DESCRIPTIVE_QC_ONLY


class BehaviorSupervisedInterfaceError(ValueError):
    """Raised when a primary probe table violates the Behavior interface contract."""


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise BehaviorSupervisedInterfaceError(f"behavior supervised interface missing required columns: {missing}")


def _nonempty_string(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def build_behavior_supervised_probe_table(primary_probe: pd.DataFrame) -> pd.DataFrame:
    """Return one unchanged-observation row per authoritative 30s probe.

    Missing feature cells are preserved exactly. No full-cohort coverage filter,
    imputation, scaling or empirical redundancy selection occurs here.
    """
    required = tuple(dict.fromkeys((*IDENTITY_LOCATOR_COLUMNS, *OUTCOME_COLUMNS, *FIRST_ROUND_CANDIDATE_POOL)))
    _require_columns(primary_probe, required)
    if primary_probe.empty:
        raise BehaviorSupervisedInterfaceError("primary behavior probe table is empty")
    for column in IDENTITY_LOCATOR_COLUMNS:
        if not _nonempty_string(primary_probe[column]).all():
            raise BehaviorSupervisedInterfaceError(f"identity/locator column contains missing or blank values: {column}")
    if primary_probe["probe_event_id"].astype(str).duplicated().any():
        raise BehaviorSupervisedInterfaceError("primary behavior probe_event_id must be unique")

    if "window_seconds_nominal" in primary_probe.columns:
        window = pd.to_numeric(primary_probe["window_seconds_nominal"], errors="coerce")
        if window.isna().any() or not window.eq(PRIMARY_WINDOW_SECONDS).all():
            raise BehaviorSupervisedInterfaceError("Behavior supervised interface accepts primary 30-second probes only")
    if "anchor_trial_excluded" in primary_probe.columns:
        if not primary_probe["anchor_trial_excluded"].fillna(False).astype(bool).all():
            raise BehaviorSupervisedInterfaceError("anchor trial exclusion contract failed")
    if "window_crosses_block" in primary_probe.columns:
        if primary_probe["window_crosses_block"].fillna(False).astype(bool).any():
            raise BehaviorSupervisedInterfaceError("probe window crossing block boundary is forbidden")

    validate_first_round_omission_predictors(FIRST_ROUND_CANDIDATE_POOL)
    ordered = list(IDENTITY_LOCATOR_COLUMNS)
    ordered.extend(c for c in WINDOW_AUDIT_COLUMNS if c in primary_probe.columns)
    ordered.extend(OUTCOME_COLUMNS)
    ordered.extend(FIRST_ROUND_CANDIDATE_POOL)
    ordered.extend(c for c in DESCRIPTIVE_QC_COLUMNS if c in primary_probe.columns)
    ordered.extend(c for c in OPPORTUNITY_AUDIT_COLUMNS if c in primary_probe.columns)
    ordered = list(dict.fromkeys(ordered))

    out = primary_probe.loc[:, ordered].copy().reset_index(drop=True)
    out["behavior_supervised_interface_version"] = INTERFACE_VERSION
    if len(out) != len(primary_probe):
        raise AssertionError("Behavior supervised interface changed source row count")
    if out["probe_event_id"].duplicated().any():
        raise AssertionError("Behavior supervised interface created duplicate probes")
    if "analysis_set_id" in out.columns:
        raise AssertionError("Task C must not generate analysis_set_id")
    return out


def _field_role(field: str) -> tuple[str, str]:
    if field == "q1_nominal_4class":
        return "supervised_target_source", "Q1=1 vs Q1=2/3/4 is encoded downstream by Task A"
    if field == "q2_ordinal_4level":
        return "interpretation_construct_only", "not a first-round Q1 predictor"
    if field in RT_LEVEL_CANDIDATES:
        return "candidate_scheme_rt_level", "mean vs median remains unresolved; do not enter both by default"
    if field in RT_VARIABILITY_CANDIDATES:
        return "candidate_scheme_rt_variability", "CV currently preferred; alternatives remain limited candidate representations"
    if field in RT_TREND_CANDIDATES:
        return "first_round_rt_trend_candidate", "behavior producer uses Theil-Sen slope"
    if field in FIRST_ROUND_SUPERVISED_OMISSION_PREDICTORS:
        return "first_round_supervised_omission_candidate", "raw program omission only"
    if field in ("commission_rate", "dprime_loglinear"):
        return "candidate_pending_final_scientific_freeze", "commission vs dprime joint retention remains unresolved"
    if field in DESCRIPTIVE_QC_COLUMNS:
        return "descriptive_qc_sensitivity_only", "not a first-round supervised predictor"
    return "audit_only", "opportunity/window/denominator provenance"


def build_behavior_supervised_feature_audit(interface: pd.DataFrame) -> pd.DataFrame:
    """Describe field coverage and scientific role without selecting from the full cohort."""
    rows: list[dict[str, Any]] = []
    fields = [
        *OUTCOME_COLUMNS,
        *FIRST_ROUND_CANDIDATE_POOL,
        *[c for c in DESCRIPTIVE_QC_COLUMNS if c in interface.columns],
        *[c for c in OPPORTUNITY_AUDIT_COLUMNS if c in interface.columns],
    ]
    for field in dict.fromkeys(fields):
        if field not in interface.columns:
            continue
        value = interface[field]
        if pd.api.types.is_numeric_dtype(value):
            valid = pd.to_numeric(value, errors="coerce").notna()
        else:
            valid = _nonempty_string(value)
        role, note = _field_role(field)
        rows.append({
            "field": field,
            "role": role,
            "n_rows": int(len(interface)),
            "n_valid": int(valid.sum()),
            "coverage": float(valid.mean()) if len(interface) else 0.0,
            "selection_authority": "prespecified_role_or_training_boundary_only",
            "automatic_drop_allowed": False,
            "note": note,
        })
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_behavior_supervised_interface(
    source_path: str | Path,
    output_root: str | Path,
    *,
    config_digest: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Write an immutable derived Behavior supervised interface directory."""
    source = Path(source_path).resolve()
    output = Path(output_root).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Behavior primary probe source not found: {source}")
    if output.exists():
        if not force:
            raise FileExistsError(f"Behavior supervised interface output already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=False)

    primary = pd.read_csv(source)
    interface = build_behavior_supervised_probe_table(primary)
    audit = build_behavior_supervised_feature_audit(interface)
    interface_path = output / "behavior_supervised_probe_30s.csv"
    audit_path = output / "behavior_supervised_feature_audit.csv"
    manifest_path = output / "behavior_supervised_interface_manifest.json"
    interface.to_csv(interface_path, index=False, encoding="utf-8-sig")
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")

    manifest: dict[str, Any] = {
        "interface_version": INTERFACE_VERSION,
        "status": "complete",
        "source_path": str(source),
        "source_sha256": _sha256(source),
        "config_digest": config_digest,
        "primary_window_seconds": PRIMARY_WINDOW_SECONDS,
        "n_rows": int(len(interface)),
        "n_participant_groups": int(interface["participant_group_id"].astype(str).nunique()),
        "n_sessions": int(interface["session_id"].astype(str).nunique()),
        "probe_event_id_unique": bool(interface["probe_event_id"].is_unique),
        "row_filter_applied": False,
        "imputation_applied": False,
        "scaling_applied": False,
        "full_cohort_empirical_feature_selection_applied": False,
        "analysis_set_id_generated": False,
        "q1_role": "supervised_target_source",
        "q2_role": "interpretation_construct_only_not_predictor",
        "first_round_candidate_pool": list(FIRST_ROUND_CANDIDATE_POOL),
        "descriptive_qc_only": list(DESCRIPTIVE_QC_COLUMNS),
        "files": {
            "probe_table": interface_path.name,
            "feature_audit": audit_path.name,
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
