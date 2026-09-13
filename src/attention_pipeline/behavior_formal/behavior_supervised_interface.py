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

from attention_pipeline.formal_analysis.identity_contract import assert_participant_group_contract

from .behavior_supervised_contract import (
    FIRST_ROUND_OMISSION_DESCRIPTIVE_QC_ONLY,
    FIRST_ROUND_SUPERVISED_OMISSION_PREDICTORS,
    validate_first_round_omission_predictors,
)


INTERFACE_VERSION = "behavior-supervised-probe-v1"
PRIMARY_WINDOW_SECONDS = 30
RT_CV_MATHEMATICAL_MIN_N = 2
OMISSION_PARTITION_TOLERANCE = 1e-12

IDENTITY_LOCATOR_COLUMNS = (
    "participant_group_id",
    "session_id",
    "block_id",
    "probe_event_id",
)

REQUIRED_IDENTITY_AUDIT_COLUMNS = (
    "repeat_participant_id",
    "participant_identity_source",
    "participant_identity_resolved_for_clustering",
)

OPTIONAL_IDENTITY_AUDIT_COLUMNS = (
    "participant_key",
    "legacy_repeat_participant_id",
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

REQUIRED_OMISSION_PARTITION_COLUMNS = (
    "omission_rate",
    "raw_go_omission_rate",
    "clean_go_omission_rate",
    "timing_ambiguous_go_omission_rate",
    "go_opportunities",
    "omission_numerator",
    "omission_denominator",
    "omission_taxonomy_denominator",
    "raw_go_omission_n",
    "clean_go_omission_n",
    "timing_ambiguous_go_omission_n",
)

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

COMPUTABILITY_AUDIT_COLUMNS = (
    "rt_variability_valid_n",
    "rt_cv_min_n",
    "rt_cv_status",
    "rt_slope_min_n",
    "rt_slope_status",
    "rt_slope_fit_scope",
    "sdt_status",
    "omission_taxonomy_status",
    "omission_primary_partition_check",
    "omission_subtype_partition_check",
    "omission_taxonomy_denominator_contract",
    "metric_units",
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

# The clean/timing partition plus finer motor-timing subtypes remain descriptive
# and QC-only. They are preserved for auditability but are never promoted into
# FIRST_ROUND_CANDIDATE_POOL.
DESCRIPTIVE_QC_COLUMNS = tuple(dict.fromkeys(FIRST_ROUND_OMISSION_DESCRIPTIVE_QC_ONLY))


class BehaviorSupervisedInterfaceError(ValueError):
    """Raised when a primary probe table violates the Behavior interface contract."""


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise BehaviorSupervisedInterfaceError(f"behavior supervised interface missing required columns: {missing}")


def _nonempty_string(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def _strict_boolean(series: pd.Series, *, column: str) -> pd.Series:
    """Parse a boolean contract without treating non-empty strings as True."""
    if series.isna().any():
        raise BehaviorSupervisedInterfaceError(f"{column} contains missing values")
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    parsed = normalized.map({"true": True, "false": False, "1": True, "0": False})
    if parsed.isna().any():
        bad = sorted(normalized.loc[parsed.isna()].unique().tolist())
        raise BehaviorSupervisedInterfaceError(f"{column} contains invalid boolean values: {bad}")
    return parsed.astype(bool)


def _nonnegative_integer(series: pd.Series, *, column: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise BehaviorSupervisedInterfaceError(f"{column} must contain finite nonnegative integer counts")
    if numeric.lt(0).any() or not np.allclose(numeric.to_numpy(dtype=float), np.round(numeric.to_numpy(dtype=float)), rtol=0.0, atol=1e-12):
        raise BehaviorSupervisedInterfaceError(f"{column} must contain finite nonnegative integer counts")
    return numeric.astype(float)


def _validate_rt_cv_handoff(frame: pd.DataFrame) -> None:
    """Reject primary tables that violate the mathematical RT-CV contract."""
    _require_columns(frame, ("rt_cv_min_n", "rt_cv_status", "correct_go_rt_opportunities", "go_correct_rt_cv"))
    minimum = pd.to_numeric(frame["rt_cv_min_n"], errors="coerce")
    if minimum.isna().any() or not minimum.eq(RT_CV_MATHEMATICAL_MIN_N).all():
        values = sorted(set(minimum.dropna().astype(float).tolist()))
        raise BehaviorSupervisedInterfaceError(
            "RT-CV handoff contract requires rt_cv_min_n=2 as the sample-SD mathematical condition; "
            f"observed={values}"
        )

    n_rt = _nonnegative_integer(frame["correct_go_rt_opportunities"], column="correct_go_rt_opportunities")
    cv = pd.to_numeric(frame["go_correct_rt_cv"], errors="coerce")
    status = frame["rt_cv_status"].astype(str).str.strip()
    should_be_estimable = n_rt.ge(RT_CV_MATHEMATICAL_MIN_N)
    wrongly_masked = should_be_estimable & cv.isna()
    wrong_estimable_status = should_be_estimable & ~status.eq("estimable")
    spuriously_present = ~should_be_estimable & cv.notna()
    spurious_status = ~should_be_estimable & status.eq("estimable")
    if wrongly_masked.any() or wrong_estimable_status.any():
        raise BehaviorSupervisedInterfaceError(
            "RT-CV handoff contract failed: rows with at least two valid correct-Go RTs must retain mathematically defined CV/status"
        )
    if spuriously_present.any() or spurious_status.any():
        raise BehaviorSupervisedInterfaceError(
            "RT-CV handoff contract failed: rows with fewer than two valid correct-Go RTs must remain non-estimable"
        )


def _validate_omission_handoff(frame: pd.DataFrame) -> None:
    """Verify program raw omission, partition counts/rates and shared denominator."""
    _require_columns(frame, REQUIRED_OMISSION_PARTITION_COLUMNS)
    alias = pd.to_numeric(frame["omission_rate"], errors="coerce")
    raw = pd.to_numeric(frame["raw_go_omission_rate"], errors="coerce")
    clean = pd.to_numeric(frame["clean_go_omission_rate"], errors="coerce")
    timing = pd.to_numeric(frame["timing_ambiguous_go_omission_rate"], errors="coerce")
    finite_count = pd.concat([raw.notna(), clean.notna(), timing.notna()], axis=1).sum(axis=1)
    if finite_count.isin([1, 2]).any():
        raise BehaviorSupervisedInterfaceError(
            "omission handoff contract failed: raw/clean/timing rates must be jointly present or jointly missing"
        )
    alias_mismatch = alias.notna().ne(raw.notna()) | (alias.notna() & raw.notna() & alias.sub(raw).abs().gt(OMISSION_PARTITION_TOLERANCE))
    if alias_mismatch.any():
        raise BehaviorSupervisedInterfaceError(
            "omission handoff contract failed: omission_rate compatibility alias must equal raw_go_omission_rate"
        )

    all_finite = finite_count.eq(3)
    if all_finite.any():
        error = (raw - clean - timing).abs()
        if error.loc[all_finite].gt(OMISSION_PARTITION_TOLERANCE).any():
            raise BehaviorSupervisedInterfaceError(
                "omission handoff contract failed: raw_go_omission_rate != clean + timing_ambiguous"
            )
        rate_block = pd.concat([raw, clean, timing], axis=1)
        if ((rate_block.loc[all_finite] < -OMISSION_PARTITION_TOLERANCE) | (rate_block.loc[all_finite] > 1 + OMISSION_PARTITION_TOLERANCE)).any().any():
            raise BehaviorSupervisedInterfaceError("omission handoff contract failed: omission rates must be proportions in [0, 1]")

    go_n = _nonnegative_integer(frame["go_opportunities"], column="go_opportunities")
    program_den = _nonnegative_integer(frame["omission_denominator"], column="omission_denominator")
    taxonomy_den = _nonnegative_integer(frame["omission_taxonomy_denominator"], column="omission_taxonomy_denominator")
    if not (go_n.eq(program_den) & go_n.eq(taxonomy_den)).all():
        raise BehaviorSupervisedInterfaceError(
            "omission handoff contract failed: raw/clean/timing must share the same Go opportunity denominator"
        )

    program_n = _nonnegative_integer(frame["omission_numerator"], column="omission_numerator")
    raw_n = _nonnegative_integer(frame["raw_go_omission_n"], column="raw_go_omission_n")
    clean_n = _nonnegative_integer(frame["clean_go_omission_n"], column="clean_go_omission_n")
    timing_n = _nonnegative_integer(frame["timing_ambiguous_go_omission_n"], column="timing_ambiguous_go_omission_n")
    if not program_n.eq(raw_n).all():
        raise BehaviorSupervisedInterfaceError(
            "omission handoff contract failed: program omission_numerator must equal raw_go_omission_n"
        )
    if not raw_n.eq(clean_n + timing_n).all():
        raise BehaviorSupervisedInterfaceError(
            "omission handoff contract failed: raw_go_omission_n != clean + timing_ambiguous counts"
        )
    if (raw_n.gt(go_n) | clean_n.gt(go_n) | timing_n.gt(go_n)).any():
        raise BehaviorSupervisedInterfaceError("omission handoff contract failed: omission counts cannot exceed Go opportunities")

    positive_den = go_n.gt(0)
    if positive_den.any():
        expected = pd.DataFrame({
            "raw": raw_n / go_n,
            "clean": clean_n / go_n,
            "timing": timing_n / go_n,
        })
        observed = pd.DataFrame({"raw": raw, "clean": clean, "timing": timing})
        mismatch = (observed.loc[positive_den] - expected.loc[positive_den]).abs().gt(OMISSION_PARTITION_TOLERANCE)
        if observed.loc[positive_den].isna().any().any() or mismatch.any().any():
            raise BehaviorSupervisedInterfaceError(
                "omission handoff contract failed: omission rates must equal counts divided by the shared Go denominator"
            )
    zero_den = go_n.eq(0)
    if zero_den.any():
        if raw_n.loc[zero_den].gt(0).any() or clean_n.loc[zero_den].gt(0).any() or timing_n.loc[zero_den].gt(0).any():
            raise BehaviorSupervisedInterfaceError("omission handoff contract failed: zero Go opportunities require zero omission counts")
        if pd.concat([raw.loc[zero_den], clean.loc[zero_den], timing.loc[zero_den]], axis=1).notna().any().any():
            raise BehaviorSupervisedInterfaceError("omission handoff contract failed: zero Go opportunities require missing omission rates")


def build_behavior_supervised_probe_table(primary_probe: pd.DataFrame) -> pd.DataFrame:
    """Return one unchanged-observation row per authoritative 30s probe.

    Missing feature cells are preserved exactly. No full-cohort coverage filter,
    imputation, scaling or empirical redundancy selection occurs here.
    """
    required = tuple(dict.fromkeys((
        *IDENTITY_LOCATOR_COLUMNS,
        *REQUIRED_IDENTITY_AUDIT_COLUMNS,
        *OUTCOME_COLUMNS,
        *FIRST_ROUND_CANDIDATE_POOL,
        *REQUIRED_OMISSION_PARTITION_COLUMNS,
        "window_seconds_nominal",
        "analysis_role",
        "formal_independent_sample",
        "anchor_trial_excluded",
        "window_crosses_block",
    )))
    _require_columns(primary_probe, required)
    if primary_probe.empty:
        raise BehaviorSupervisedInterfaceError("primary behavior probe table is empty")
    for column in (*IDENTITY_LOCATOR_COLUMNS, "repeat_participant_id", "participant_identity_source"):
        if not _nonempty_string(primary_probe[column]).all():
            raise BehaviorSupervisedInterfaceError(f"identity/locator column contains missing or blank values: {column}")

    try:
        assert_participant_group_contract(primary_probe, require_resolved=True)
    except ValueError as exc:
        raise BehaviorSupervisedInterfaceError(f"participant identity contract failed: {exc}") from exc

    if primary_probe["probe_event_id"].astype(str).duplicated().any():
        raise BehaviorSupervisedInterfaceError("primary behavior probe_event_id must be unique")

    window = pd.to_numeric(primary_probe["window_seconds_nominal"], errors="coerce")
    if window.isna().any() or not window.eq(PRIMARY_WINDOW_SECONDS).all():
        raise BehaviorSupervisedInterfaceError("Behavior supervised interface accepts primary 30-second probes only")

    role = primary_probe["analysis_role"].astype(str).str.strip()
    if not role.eq("primary_probe").all():
        bad = sorted(role.loc[~role.eq("primary_probe")].unique().tolist())
        raise BehaviorSupervisedInterfaceError(f"analysis_role must be primary_probe for every row; got {bad}")

    independent = _strict_boolean(primary_probe["formal_independent_sample"], column="formal_independent_sample")
    if not independent.all():
        raise BehaviorSupervisedInterfaceError("formal_independent_sample must be true for every primary probe row")

    anchor_excluded = _strict_boolean(primary_probe["anchor_trial_excluded"], column="anchor_trial_excluded")
    if not anchor_excluded.all():
        raise BehaviorSupervisedInterfaceError("anchor trial exclusion contract failed")

    crosses_block = _strict_boolean(primary_probe["window_crosses_block"], column="window_crosses_block")
    if crosses_block.any():
        raise BehaviorSupervisedInterfaceError("probe window crossing block boundary is forbidden")

    _validate_rt_cv_handoff(primary_probe)
    _validate_omission_handoff(primary_probe)
    validate_first_round_omission_predictors(FIRST_ROUND_CANDIDATE_POOL)
    ordered = list(IDENTITY_LOCATOR_COLUMNS)
    ordered.extend(REQUIRED_IDENTITY_AUDIT_COLUMNS)
    ordered.extend(c for c in OPTIONAL_IDENTITY_AUDIT_COLUMNS if c in primary_probe.columns)
    ordered.extend(c for c in WINDOW_AUDIT_COLUMNS if c in primary_probe.columns)
    ordered.extend(OUTCOME_COLUMNS)
    ordered.extend(FIRST_ROUND_CANDIDATE_POOL)
    ordered.extend(c for c in DESCRIPTIVE_QC_COLUMNS if c in primary_probe.columns)
    ordered.extend(c for c in COMPUTABILITY_AUDIT_COLUMNS if c in primary_probe.columns)
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
    return "audit_only", "identity/window/opportunity/computability provenance"


def build_behavior_supervised_feature_audit(interface: pd.DataFrame) -> pd.DataFrame:
    """Describe field coverage and scientific role without selecting from the full cohort."""
    rows: list[dict[str, Any]] = []
    fields = [
        *OUTCOME_COLUMNS,
        *FIRST_ROUND_CANDIDATE_POOL,
        *[c for c in DESCRIPTIVE_QC_COLUMNS if c in interface.columns],
        *[c for c in COMPUTABILITY_AUDIT_COLUMNS if c in interface.columns],
        *[c for c in OPPORTUNITY_AUDIT_COLUMNS if c in interface.columns],
    ]
    for field in dict.fromkeys(fields):
        if field not in interface.columns:
            continue
        value = interface[field]
        if pd.api.types.is_numeric_dtype(value):
            numeric = pd.to_numeric(value, errors="coerce")
            valid = pd.Series(np.isfinite(numeric.to_numpy(dtype=float)), index=value.index)
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


def _assert_safe_force_target(source: Path, output: Path) -> None:
    """Allow recursive replacement only for a recognized prior Task C interface."""
    if source == output or source.is_relative_to(output):
        raise BehaviorSupervisedInterfaceError(
            "refusing force replacement because the source file is inside the output directory"
        )
    marker = output / "behavior_supervised_interface_manifest.json"
    if not marker.is_file():
        raise BehaviorSupervisedInterfaceError(
            "refusing force replacement of an existing directory without a Task C interface manifest"
        )
    try:
        previous = json.loads(marker.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BehaviorSupervisedInterfaceError("existing Task C interface manifest is unreadable") from exc
    if previous.get("interface_version") != INTERFACE_VERSION:
        raise BehaviorSupervisedInterfaceError(
            "refusing force replacement because the existing directory is not the same Task C interface version"
        )


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
    output_exists = output.exists()
    if output_exists:
        if not force:
            raise FileExistsError(f"Behavior supervised interface output already exists: {output}")
        _assert_safe_force_target(source, output)

    # Validate and materialize in memory before replacing a recognized previous
    # interface. A bad new source must never destroy the last valid derived output.
    primary = pd.read_csv(source)
    interface = build_behavior_supervised_probe_table(primary)
    audit = build_behavior_supervised_feature_audit(interface)

    if output_exists:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=False)
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
        "rt_cv_mathematical_min_n": RT_CV_MATHEMATICAL_MIN_N,
        "n_rows": int(len(interface)),
        "n_participant_groups": int(interface["participant_group_id"].astype(str).nunique()),
        "n_sessions": int(interface["session_id"].astype(str).nunique()),
        "probe_event_id_unique": bool(interface["probe_event_id"].is_unique),
        "participant_identity_contract_checked": True,
        "participant_identity_audit_columns_required": list(REQUIRED_IDENTITY_AUDIT_COLUMNS),
        "primary_probe_role_contract_checked": True,
        "rt_cv_handoff_contract_checked": True,
        "omission_partition_handoff_contract_checked": True,
        "row_filter_applied": False,
        "imputation_applied": False,
        "scaling_applied": False,
        "full_cohort_empirical_feature_selection_applied": False,
        "analysis_set_id_generated": False,
        "q1_role": "supervised_target_source",
        "q2_role": "interpretation_construct_only_not_predictor",
        "first_round_candidate_pool": list(FIRST_ROUND_CANDIDATE_POOL),
        "descriptive_qc_only": list(DESCRIPTIVE_QC_COLUMNS),
        "computability_audit_fields": [c for c in COMPUTABILITY_AUDIT_COLUMNS if c in interface.columns],
        "files": {
            "probe_table": interface_path.name,
            "feature_audit": audit_path.name,
        },
        "output_sha256": {
            "probe_table": _sha256(interface_path),
            "feature_audit": _sha256(audit_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
