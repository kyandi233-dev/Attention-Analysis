"""Probe-level availability and feature-computability audit for Task B.

This module records data readiness facts. It does not train models, impute values,
standardize features, or decide scientific feature eligibility from full-sample
coverage/distribution statistics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .alignment import KEY_COLUMNS, PRIMARY_WINDOW

VERSION = "task-b-analysis-set-audit-v1.1.0"
KEYS = list(KEY_COLUMNS)
GROUP = "participant_group_id"
REVIEW_COVERAGE_REFERENCE = 0.80
FORMAL_METADATA_COLUMNS = (
    "probe_event_id",
    "probe_order_in_block",
    "q1_nominal_4class",
    "q2_ordinal_4level",
    "window_name",
)


def _true(series: pd.Series) -> pd.Series:
    return series.astype("string").str.lower().isin(["true", "1", "1.0", "yes"])


def _feature_map(feature_blocks: dict[str, Any]) -> dict[str, list[str]]:
    """Normalize feature config without performing feature selection."""
    result: dict[str, list[str]] = {}
    for modality, raw in feature_blocks.items():
        if modality == "nir_primary":
            if not isinstance(raw, dict):
                raise ValueError("nir_primary must be a mapping with metrics")
            result["nir"] = [str(x) for x in raw.get("metrics", [])]
        elif isinstance(raw, list):
            result[modality] = [str(x) for x in raw if isinstance(x, str)]
    return result


def validate_probe_keys(frame: pd.DataFrame, modality: str) -> None:
    required = KEYS + [GROUP]
    if not set(required) <= set(frame.columns):
        missing = sorted(set(required) - set(frame.columns))
        raise ValueError(f"{modality}: missing key/identity columns: {missing}")
    if frame[required].isna().any().any():
        raise ValueError(f"{modality}: null key/identity")
    text = frame[required].astype(str)
    if text.apply(lambda s: s.str.strip().isin(["", "nan", "None"]).any()).any():
        raise ValueError(f"{modality}: empty key/identity")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{modality}: duplicate probe keys")
    if frame.groupby("session_id")[GROUP].nunique().gt(1).any():
        raise ValueError(f"{modality}: session identity conflict")
    probe = pd.to_numeric(frame[KEYS[-1]], errors="coerce")
    legal = frame["block_id"].isin(["b1", "b2"]) & probe.gt(0) & probe.mod(1).eq(0)
    if not legal.all():
        raise ValueError(f"{modality}: illegal formal probe keys")


def _source_state(
    frame: pd.DataFrame, modality: str, matched: pd.Series
) -> tuple[pd.Series, pd.Series, str]:
    """Use explicit source evidence; finite feature values never prove source presence."""
    if modality == "behavior":
        return matched.copy(), matched.copy(), "formal_behavior_record"

    if modality == "mmwave" and {"mmwave_observed", "mmwave_loadable"} <= set(frame.columns):
        present = matched & _true(frame["mmwave_observed"])
        readable = present & _true(frame["mmwave_loadable"])
        return present, readable, "mmwave_observed+mmwave_loadable"

    if modality == "rgb" and "rgb_source_status" in frame.columns:
        status = frame["rgb_source_status"].astype("string")
        present = matched & status.notna() & ~status.isin(["missing", "absent", "unavailable"])
        readable = present & status.isin(["ok", "partial_no_blink"])
        return present, readable, "rgb_source_status"

    if modality == "nir" and "n_nir_rows" in frame.columns:
        observed = pd.to_numeric(frame["n_nir_rows"], errors="coerce").gt(0)
        present = matched & observed
        return present, present.copy(), "n_nir_rows>0"

    if "source_present" in frame.columns:
        present = matched & _true(frame["source_present"])
        readable = present & (
            _true(frame["source_readable"])
            if "source_readable" in frame.columns
            else True
        )
        return present, readable, "explicit_source_present"

    false = matched & False
    return false, false, "source_evidence_missing"


def _alignment_state(frame: pd.DataFrame, matched: pd.Series) -> tuple[pd.Series, str]:
    valid = matched.copy()
    evidence: list[str] = []

    if "window_name" in frame.columns:
        valid &= frame["window_name"].eq(PRIMARY_WINDOW)
        evidence.append("window_name")
    for col in ("anchor_trial_excluded", "anchoring_probe_trial_excluded"):
        if col in frame.columns:
            valid &= _true(frame[col])
            evidence.append(col)
    if "window_crosses_block" in frame.columns:
        valid &= frame["window_crosses_block"].notna() & ~_true(frame["window_crosses_block"])
        evidence.append("window_crosses_block")
    if "window_seconds_nominal" in frame.columns:
        valid &= pd.to_numeric(frame["window_seconds_nominal"], errors="coerce").eq(30)
        evidence.append("window_seconds_nominal")

    for start, end, probe in (
        ("window_start_ms", "window_end_ms", "probe_onset_ms"),
        ("window_start_unix_ms", "window_end_unix_ms", "probe_onset_unix_ms"),
    ):
        if {start, end, probe} <= set(frame.columns):
            s = pd.to_numeric(frame[start], errors="coerce")
            e = pd.to_numeric(frame[end], errors="coerce")
            p = pd.to_numeric(frame[probe], errors="coerce")
            valid &= s.lt(e) & e.le(p) & np.isclose(e.sub(s), 30000, atol=1)
            evidence.append(f"{start}:{end}:{probe}")

    bounds = {
        "window_effective_start_unix_ms",
        "block_start_unix_ms",
        "window_end_unix_ms",
        "block_end_unix_ms",
    }
    if bounds <= set(frame.columns):
        exported = frame["block_start_unix_ms"].notna() | frame["block_end_unix_ms"].notna()
        inherited = frame.get(
            "window_boundary_source", pd.Series("", index=frame.index)
        ).eq("probe_primary_30s")
        inside = (
            pd.to_numeric(frame["window_effective_start_unix_ms"], errors="coerce")
            .ge(pd.to_numeric(frame["block_start_unix_ms"], errors="coerce"))
            & pd.to_numeric(frame["window_end_unix_ms"], errors="coerce")
            .le(pd.to_numeric(frame["block_end_unix_ms"], errors="coerce"))
        )
        valid &= ((~exported & inherited) | (exported & inside))
        evidence.append("block_bounds_or_inherited_formal_window")

    return valid.fillna(False), ";".join(evidence) or "normalized_formal_probe_product"


def _native_qc_state(
    frame: pd.DataFrame, modality: str, feature: str
) -> tuple[pd.Series, str]:
    """Apply only explicit upstream modality QC; do not recreate empirical thresholds."""
    valid = pd.Series(True, index=frame.index, dtype=bool)
    evidence: list[str] = []

    if "native_qc_valid" in frame.columns:
        valid &= _true(frame["native_qc_valid"])
        evidence.append("native_qc_valid")

    feature_qc = f"{feature}_qc_valid"
    if feature_qc in frame.columns:
        valid &= _true(frame[feature_qc])
        evidence.append(feature_qc)

    if modality == "mmwave" and "mmwave_state" in frame.columns:
        state = frame["mmwave_state"].astype("string")
        known = state.notna()
        valid &= ~known | state.eq("OBSERVED")
        evidence.append("mmwave_state=OBSERVED")

    if modality == "rgb" and feature.startswith("blink_") and "rgb_source_status" in frame.columns:
        valid &= frame["rgb_source_status"].eq("ok")
        evidence.append("rgb_source_status=ok_for_blink")

    return valid.fillna(False), ";".join(evidence) or "no_additional_explicit_qc_gate"


def _feature_support_state(
    frame: pd.DataFrame, modality: str, feature: str
) -> tuple[pd.Series, str]:
    """Use producer-defined mathematical/estimability support, never empirical cutoffs.

    This layer intentionally does not implement historical rules such as RT-CV n>=20.
    It only prevents an explicitly non-estimable summary from being mislabeled as a
    residual single-feature missing value eligible for later imputation.
    """
    valid = pd.Series(True, index=frame.index, dtype=bool)
    evidence: list[str] = []

    if modality == "behavior":
        rt_n_col = "correct_go_rt_opportunities"
        if rt_n_col in frame.columns:
            n_rt = pd.to_numeric(frame[rt_n_col], errors="coerce")
            one_rt = {
                "go_correct_rt_mean_ms",
                "go_correct_rt_median_ms",
                "go_correct_rt_mad_ms",
                "go_correct_rt_iqr_ms",
            }
            two_rt = {
                "go_correct_rt_sd_ms",
                "go_correct_rt_cv",
                "go_correct_rt_theilsen_slope_ms_per_s",
            }
            if feature in one_rt:
                valid &= n_rt.ge(1)
                evidence.append(f"{rt_n_col}>=1")
            elif feature in two_rt:
                valid &= n_rt.ge(2)
                evidence.append(f"{rt_n_col}>=2")

        if feature in {"dprime_loglinear", "criterion_c", "beta"} and "sdt_status" in frame.columns:
            valid &= frame["sdt_status"].astype("string").eq("estimable")
            evidence.append("sdt_status=estimable")

        if feature in {
            "omission_rate",
            "raw_go_omission_rate",
            "clean_go_omission_rate",
            "timing_ambiguous_go_omission_rate",
        } and "omission_denominator" in frame.columns:
            valid &= pd.to_numeric(frame["omission_denominator"], errors="coerce").gt(0)
            evidence.append("omission_denominator>0")

        if feature == "commission_rate" and "commission_denominator" in frame.columns:
            valid &= pd.to_numeric(frame["commission_denominator"], errors="coerce").gt(0)
            evidence.append("commission_denominator>0")

    if modality == "nir" and feature.startswith("pupil_") and "n_pupil_valid" in frame.columns:
        n_valid = pd.to_numeric(frame["n_pupil_valid"], errors="coerce")
        minimum = 1
        if feature in {
            "pupil_diff_mad",
            "pupil_diff_rate_mad_per_sec",
            "pupil_peak_to_trough",
        }:
            minimum = 2
        elif feature == "pupil_slope_per_sec":
            minimum = 3
        valid &= n_valid.ge(minimum)
        evidence.append(f"n_pupil_valid>={minimum}:producer_math_support")

    return valid.fillna(False), ";".join(evidence) or "no_explicit_support_gate"


def audit_quality(
    tables: dict[str, pd.DataFrame],
    feature_blocks: dict[str, Any],
    rules: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Audit availability/computability against the behavior-defined formal denominator.

    Coverage and distribution diagnostics are descriptive. No threshold here promotes or
    rejects a scientific feature for supervised learning.
    """
    rules = rules or {}
    features = _feature_map(feature_blocks)
    if "behavior" not in tables or tables["behavior"].empty:
        raise ValueError("empty formal behavior denominator")

    behavior = tables["behavior"].reset_index(drop=True).copy()
    validate_probe_keys(behavior, "behavior")
    identity_columns = KEYS + [GROUP] + [
        column for column in FORMAL_METADATA_COLUMNS if column in behavior.columns
    ]
    identity = behavior[identity_columns].copy()

    feature_rows: list[pd.DataFrame] = []
    modality_rows: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []

    proportion_features = set(str(x) for x in rules.get("proportion_features", []))

    for modality, cols in features.items():
        source = tables.get(modality, pd.DataFrame()).copy()
        if source.empty:
            source = pd.DataFrame(columns=KEYS + [GROUP])
        else:
            validate_probe_keys(source, modality)
            identity_check = source.merge(
                identity[KEYS + [GROUP]],
                on=KEYS,
                how="left",
                suffixes=("", "_formal"),
                validate="one_to_one",
            )
            formal_group = f"{GROUP}_formal"
            mismatch = identity_check[formal_group].notna() & identity_check[GROUP].ne(
                identity_check[formal_group]
            )
            if mismatch.any():
                raise ValueError(f"{modality}: identity disagrees with behavior")

        merge_source = source.drop(columns=[GROUP], errors="ignore")
        frame = identity[KEYS + [GROUP]].merge(
            merge_source, on=KEYS, how="left", indicator=True, validate="one_to_one"
        )
        matched = frame["_merge"].eq("both")
        present, readable, source_basis = _source_state(frame, modality, matched)
        aligned, alignment_basis = _alignment_state(frame, matched)
        opportunity = present & readable & aligned

        modality_detail = identity.copy()
        modality_detail["modality"] = modality
        modality_detail["record_matched"] = matched
        modality_detail["source_present"] = present
        modality_detail["source_readable"] = readable
        modality_detail["aligned"] = aligned
        modality_detail["measurement_opportunity"] = opportunity
        modality_detail["source_evidence"] = source_basis
        modality_detail["alignment_evidence"] = alignment_basis
        modality_rows.append(modality_detail)

        for feature in cols:
            qc_valid, qc_basis = _native_qc_state(frame, modality, feature)
            support_valid, support_basis = _feature_support_state(frame, modality, feature)
            column_present = feature in frame.columns
            raw = (
                pd.to_numeric(frame[feature], errors="coerce")
                if column_present
                else pd.Series(np.nan, index=frame.index, dtype=float)
            )
            finite = pd.Series(np.isfinite(raw), index=frame.index)
            computable = opportunity & qc_valid & support_valid & finite
            missing_strategy_eligible = opportunity & qc_valid & support_valid & column_present

            reason = np.select(
                [
                    ~matched,
                    ~present,
                    ~readable,
                    ~aligned,
                    ~qc_valid,
                    ~support_valid,
                    pd.Series(not column_present, index=frame.index),
                    ~finite,
                ],
                [
                    "record_missing",
                    "structural_source_missing",
                    "structural_source_unreadable",
                    "structural_alignment_invalid",
                    "native_qc_invalid",
                    "feature_support_invalid",
                    "feature_column_missing",
                    "single_feature_missing",
                ],
                default="available",
            )

            detail = identity.copy()
            detail["modality"] = modality
            detail["feature"] = feature
            detail["record_matched"] = matched
            detail["source_present"] = present
            detail["source_readable"] = readable
            detail["aligned"] = aligned
            detail["measurement_opportunity"] = opportunity
            detail["native_qc_valid"] = qc_valid
            detail["feature_support_valid"] = support_valid
            detail["feature_column_present"] = column_present
            detail["feature_computable"] = computable
            detail["eligible_for_missing_strategy"] = missing_strategy_eligible
            detail["missing_kind"] = reason
            detail["source_evidence"] = source_basis
            detail["alignment_evidence"] = alignment_basis
            detail["native_qc_evidence"] = qc_basis
            detail["feature_support_evidence"] = support_basis
            detail["value"] = raw.where(computable)
            feature_rows.append(detail)

            x = raw[computable]
            opportunity_n = int(opportunity.sum())
            computable_n = int(computable.sum())
            coverage = computable_n / opportunity_n if opportunity_n else np.nan
            overall = computable_n / len(identity) if len(identity) else np.nan
            unique_n = int(x.nunique()) if computable_n else 0
            variance = float(x.var(ddof=1)) if computable_n >= 2 else np.nan
            floor = bool(
                computable_n
                and feature in proportion_features
                and float(x.le(0.02).mean()) >= 0.95
            )
            ceiling = bool(
                computable_n
                and feature in proportion_features
                and float(x.ge(0.98).mean()) >= 0.95
            )
            nonzero_rate = float(x.ne(0).mean()) if computable_n else np.nan

            coverage_rows.append(
                {
                    "modality": modality,
                    "feature": feature,
                    "formal_probe_n": int(len(identity)),
                    "opportunity_n": opportunity_n,
                    "computable_n": computable_n,
                    "modality_availability_rate": opportunity_n / len(identity),
                    "feature_computable_coverage": coverage,
                    "overall_effective_rate": overall,
                    "unique_valid_n": unique_n,
                    "valid_variance": variance,
                    "nonzero_valid_rate": nonzero_rate,
                    "zero_variance": bool(computable_n > 0 and unique_n == 1),
                    "fewer_than_3_unique_values_review": bool(
                        computable_n > 0 and unique_n < 3
                    ),
                    "severe_floor_review": floor,
                    "severe_ceiling_review": ceiling,
                    "coverage_below_0_80_review": bool(
                        np.isfinite(coverage) and coverage < REVIEW_COVERAGE_REFERENCE
                    ),
                    "automatic_exclusion": False,
                    "review_reference_only": REVIEW_COVERAGE_REFERENCE,
                    "native_qc_evidence": qc_basis,
                    "feature_support_evidence": support_basis,
                }
            )

    probe_feature_status = (
        pd.concat(feature_rows, ignore_index=True)
        if feature_rows
        else pd.DataFrame(columns=KEYS + [GROUP, "modality", "feature"])
    )
    modality_probe_status = (
        pd.concat(modality_rows, ignore_index=True)
        if modality_rows
        else pd.DataFrame(columns=KEYS + [GROUP, "modality"])
    )
    feature_coverage = pd.DataFrame(coverage_rows)

    modality_availability = []
    if not modality_probe_status.empty:
        for modality, group in modality_probe_status.groupby("modality", sort=False):
            opportunity_n = int(group["measurement_opportunity"].sum())
            modality_availability.append(
                {
                    "modality": modality,
                    "formal_probe_n": int(len(identity)),
                    "source_present_n": int(group["source_present"].sum()),
                    "source_readable_n": int(group["source_readable"].sum()),
                    "source_aligned_n": opportunity_n,
                    "modality_availability_rate": opportunity_n / len(identity),
                }
            )

    return {
        "formal_probe_identity": identity,
        "modality_probe_status": modality_probe_status,
        "probe_feature_status": probe_feature_status,
        "feature_coverage": feature_coverage,
        "modality_availability": pd.DataFrame(modality_availability),
    }


def write_quality_audit(
    output_dir: str | Path, result: dict[str, pd.DataFrame]
) -> dict[str, str]:
    """Write Task-B audit tables without producing a full-sample admitted-feature list."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, frame in result.items():
        if isinstance(frame, pd.DataFrame):
            path = output / f"{name}.csv"
            frame.to_csv(path, index=False, encoding="utf-8-sig")
            paths[name] = str(path)
    return paths
