"""Canonical mmWave snapshot -> Cardiopulmonary ingest/audit bridge.

This module consumes ``MMWAVE_INTEGRATION_SNAPSHOT_V1`` and answers only the
engineering-ingest question.  It never re-runs the radar estimator, upgrades
physiological validity, trains a supervised model, or writes the final feature
registry.

The bridge deliberately keeps three concepts separate:

* scientific modality: ``cardiopulmonary``;
* producer/source namespace and required device: ``mmwave``;
* producer availability/error state: AVAILABLE, SOURCE_UNAVAILABLE, or
  SOURCE_MALFORMED.

SOURCE_UNAVAILABLE and SOURCE_MALFORMED both retain missing HR/BR values, but
malformed rows remain explicit source errors rather than being relabelled as
structural source absence.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .alignment import KEY_COLUMNS, PRIMARY_WINDOW
from .analysis_sets import build_analysis_sets, write_analysis_sets
from .quality_admission import audit_quality, validate_probe_keys, write_quality_audit

ADAPTER_VERSION = "mmwave-cardiopulmonary-ingest-audit-v1"
CANONICAL_SNAPSHOT_VERSION = "mmwave_integration_snapshot_v1"
CANONICAL_PROBE_N = 2320
CANONICAL_SESSION_N = 116
CANONICAL_PARTICIPANT_GROUP_N = 61
CANONICAL_AVAILABLE_N = 2180
CANONICAL_SOURCE_UNAVAILABLE_N = 40
CANONICAL_SOURCE_MALFORMED_N = 100

SCIENTIFIC_MODALITY = "cardiopulmonary"
SOURCE_NAMESPACE = "mmwave"
REQUIRED_DEVICES = ("mmwave",)
PHYSIOLOGY_QUALIFICATION = "LIMITED_SUPPORTING_ONLY"

# Declared window-end / probe-onset identity guard.
#
# ``ENDPOINT_ATOL_MS_PROVISIONAL`` is a PROVISIONAL absolute tolerance, not a
# frozen methodological conclusion.  The snapshot stores integer Unix
# milliseconds, so ``window_end_unix_ms`` and ``probe_onset_unix_ms`` are expected
# to be exactly equal; whether the final contract is exact equality or a smaller
# justified constant is decided by the governed-cohort endpoint audit
# (``endpoint_delta_summary.json``) together with producer-side source evidence.
#
# ``rtol`` must remain 0.0.  These operands are Unix-epoch milliseconds
# (~1e12 ms), so NumPy's default relative tolerance (~1e-5) would widen the
# effective tolerance to ~1e7 ms -- hours -- and silently turn this guard into a
# no-op.
ENDPOINT_ATOL_MS_PROVISIONAL = 1.0
ENDPOINT_RTOL = 0.0

HR_COLUMN = "mmwave_hr_fused_bpm_median"
BR_COLUMN = "mmwave_breath_rate_breaths_per_min_median"
SCIENTIFIC_COLUMNS = (HR_COLUMN, BR_COLUMN)

AVAILABLE = "AVAILABLE"
SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
SOURCE_MALFORMED = "SOURCE_MALFORMED"
ALLOWED_INTEGRATION_STATES = frozenset(
    {AVAILABLE, SOURCE_UNAVAILABLE, SOURCE_MALFORMED}
)

TIME_LEGALITY_STATUS = "verified_pre_probe_only"
TIME_LEGALITY_EVIDENCE = (
    "MMWAVE_INTEGRATION_SNAPSHOT_V1 row audit + producer replacement/field-role "
    "contract: window_name=pre_30s; science clock=dll_host_receive_enqueue; "
    "nominal_start=probe_onset-30000ms; effective_start>=nominal_start and < probe; "
    "window_end=probe_onset; producer contract defines the end as right-exclusive "
    "and effective_start as formal-block-start truncated when required"
)

KEYS = list(KEY_COLUMNS)
GROUP = "participant_group_id"

REQUIRED_SNAPSHOT_COLUMNS = (
    GROUP,
    "repeat_participant_id",
    "session_id",
    "block_id",
    "probe_index_in_block",
    "probe_id",
    "window_name",
    "window_nominal_start_unix_ms",
    "window_effective_start_unix_ms",
    "window_end_unix_ms",
    "probe_onset_unix_ms",
    "alignment_clock_source",
    HR_COLUMN,
    BR_COLUMN,
    "source_availability_state",
    "source_readability_state",
    "estimability_state",
    "measurement_qc_state",
    "malformed_state",
    "integration_state",
    "producer_state",
    "missing_reason",
    "snapshot_version",
    "producer_commit",
    "producer_adapter_version",
    "source_run_id",
    "snapshot_run_id",
)

PROHIBITED_FORMAL_FEATURE_COLUMNS = frozenset(
    {
        "mmwave_motion_proxy_median",
        "mmwave_ibi_median_ms",
        "mmwave_rmssd_ms",
        "mmwave_sdnn_ms",
        "LF",
        "HF",
        "LF_HF",
    }
)


class MmwaveCardiopulmonaryIngestError(ValueError):
    """Raised when the canonical mmWave ingest contract is violated."""


@dataclass(frozen=True)
class MmwaveCardiopulmonaryIngestResult:
    taskb_source: pd.DataFrame
    ingest_audit: pd.DataFrame
    coverage: pd.DataFrame
    feature_handoff: pd.DataFrame
    quality_tables: dict[str, pd.DataFrame]
    analysis_sets: pd.DataFrame
    analysis_set_summary: pd.DataFrame
    manifest: dict[str, Any]
    endpoint_delta: pd.DataFrame
    endpoint_delta_nonzero: pd.DataFrame
    endpoint_delta_summary: dict[str, Any]


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise MmwaveCardiopulmonaryIngestError(
            f"{label}: missing required columns {missing}"
        )


def _normalize_identity(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    out = frame.copy()
    _require_columns(out, KEYS + [GROUP], label)
    out["session_id"] = out["session_id"].astype("string").str.strip()
    out["block_id"] = out["block_id"].astype("string").str.strip().str.lower()
    out[GROUP] = out[GROUP].astype("string").str.strip()
    probe = pd.to_numeric(out["probe_index_in_block"], errors="coerce")
    if probe.isna().any() or not np.isclose(probe, np.round(probe)).all():
        raise MmwaveCardiopulmonaryIngestError(
            f"{label}: probe_index_in_block must be finite integer-valued"
        )
    out["probe_index_in_block"] = probe.astype(int)
    return out


def _key_index(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame[KEYS], names=KEYS)


def _nonblank_unique(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame.columns:
        return []
    values = frame[column].dropna().astype(str).str.strip()
    return sorted(values[values.ne("")].unique().tolist())


def _validate_snapshot_provenance(snapshot: pd.DataFrame) -> None:
    versions = _nonblank_unique(snapshot, "snapshot_version")
    if versions != [CANONICAL_SNAPSHOT_VERSION]:
        raise MmwaveCardiopulmonaryIngestError(
            f"unexpected snapshot_version={versions}; "
            f"expected={[CANONICAL_SNAPSHOT_VERSION]}"
        )

    producer_commit = snapshot["producer_commit"].astype("string").str.strip()
    if producer_commit.isna().any() or producer_commit.eq("").any():
        raise MmwaveCardiopulmonaryIngestError(
            "canonical snapshot requires nonblank producer_commit on every row"
        )


def _validate_identity_and_keys(
    snapshot: pd.DataFrame,
    behavior: pd.DataFrame,
    *,
    strict_canonical_counts: bool,
) -> dict[str, int]:
    validate_probe_keys(behavior, "behavior")
    validate_probe_keys(snapshot, "mmwave_snapshot")

    snapshot_keys = _key_index(snapshot)
    behavior_keys = _key_index(behavior)
    missing = behavior_keys.difference(snapshot_keys)
    extra = snapshot_keys.difference(behavior_keys)
    if len(missing) or len(extra):
        raise MmwaveCardiopulmonaryIngestError(
            "mmwave snapshot key conservation failed: "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    merged = snapshot[KEYS + [GROUP]].merge(
        behavior[KEYS + [GROUP]],
        on=KEYS,
        how="inner",
        suffixes=("_mmwave", "_behavior"),
        validate="one_to_one",
    )
    mismatch = merged[f"{GROUP}_mmwave"].ne(merged[f"{GROUP}_behavior"])
    if mismatch.any():
        sample = merged.loc[
            mismatch,
            KEYS + [f"{GROUP}_mmwave", f"{GROUP}_behavior"],
        ].head(5)
        raise MmwaveCardiopulmonaryIngestError(
            "mmwave participant identity disagrees with Behavior authority; "
            f"sample={sample.to_dict(orient='records')}"
        )

    counts = {
        "governed_probe_n": int(len(behavior)),
        "governed_session_n": int(behavior["session_id"].nunique()),
        "participant_group_n": int(behavior[GROUP].nunique()),
        "duplicate_key_n": 0,
        "missing_key_n": 0,
        "extra_key_n": 0,
        "identity_mismatch_n": 0,
    }
    if strict_canonical_counts:
        observed = (
            counts["governed_probe_n"],
            counts["governed_session_n"],
            counts["participant_group_n"],
        )
        expected = (
            CANONICAL_PROBE_N,
            CANONICAL_SESSION_N,
            CANONICAL_PARTICIPANT_GROUP_N,
        )
        if observed != expected:
            raise MmwaveCardiopulmonaryIngestError(
                "canonical governed denominator mismatch: "
                f"observed probes/sessions/groups={observed}, expected={expected}"
            )
    return counts


def _validate_integration_states(snapshot: pd.DataFrame) -> pd.Series:
    state = snapshot["integration_state"].astype("string").str.strip().str.upper()
    unknown = sorted(set(state.dropna().tolist()) - ALLOWED_INTEGRATION_STATES)
    if state.isna().any() or unknown:
        raise MmwaveCardiopulmonaryIngestError(
            f"unknown/missing integration_state values: {unknown or ['<missing>']}"
        )

    hr = pd.to_numeric(snapshot[HR_COLUMN], errors="coerce")
    br = pd.to_numeric(snapshot[BR_COLUMN], errors="coerce")
    available = state.eq(AVAILABLE)
    retained_missing_value = state.isin([SOURCE_UNAVAILABLE, SOURCE_MALFORMED])

    if not (np.isfinite(hr[available]).all() and np.isfinite(br[available]).all()):
        raise MmwaveCardiopulmonaryIngestError(
            "AVAILABLE rows must contain finite fused HR and BR"
        )
    if (
        hr[retained_missing_value].notna().any()
        or br[retained_missing_value].notna().any()
    ):
        raise MmwaveCardiopulmonaryIngestError(
            "SOURCE_UNAVAILABLE/SOURCE_MALFORMED rows must preserve HR/BR as missing; "
            "zero-fill or stale numeric carry-forward is forbidden"
        )
    return state


def _validate_time_legality(snapshot: pd.DataFrame) -> pd.DataFrame:
    window = snapshot["window_name"].astype("string").str.strip()
    clock = snapshot["alignment_clock_source"].astype("string").str.strip()
    nominal = pd.to_numeric(
        snapshot["window_nominal_start_unix_ms"], errors="coerce"
    )
    effective = pd.to_numeric(
        snapshot["window_effective_start_unix_ms"], errors="coerce"
    )
    end = pd.to_numeric(snapshot["window_end_unix_ms"], errors="coerce")
    probe = pd.to_numeric(snapshot["probe_onset_unix_ms"], errors="coerce")

    finite_time = (
        np.isfinite(nominal)
        & np.isfinite(effective)
        & np.isfinite(end)
        & np.isfinite(probe)
    )
    nominal_30s = np.isclose(probe - nominal, 30000.0, atol=1.0)
    effective_not_before_nominal = effective.ge(nominal)
    effective_before_probe = effective.lt(probe)
    end_equals_probe = np.isclose(
        end,
        probe,
        atol=ENDPOINT_ATOL_MS_PROVISIONAL,
        rtol=ENDPOINT_RTOL,
    )
    correct_window = window.eq(PRIMARY_WINDOW)
    correct_clock = clock.eq("dll_host_receive_enqueue")

    legal = (
        finite_time
        & nominal_30s
        & effective_not_before_nominal
        & effective_before_probe
        & end_equals_probe
        & correct_window
        & correct_clock
    )
    if not bool(pd.Series(legal).all()):
        failed = pd.DataFrame(
            {
                **{key: snapshot[key] for key in KEYS},
                "finite_time": finite_time,
                "nominal_30s": nominal_30s,
                "effective_not_before_nominal": effective_not_before_nominal,
                "effective_before_probe": effective_before_probe,
                "end_equals_probe": end_equals_probe,
                "correct_window": correct_window,
                "correct_clock": correct_clock,
            }
        )
        failed = failed.loc[~pd.Series(legal, index=snapshot.index)].head(5)
        raise MmwaveCardiopulmonaryIngestError(
            "mmwave snapshot fails verified_pre_probe_only row audit; "
            f"sample={failed.to_dict(orient='records')}"
        )

    return pd.DataFrame(
        {
            "time_legal": True,
            "nominal_30s": nominal_30s,
            "effective_start_truncated": effective.gt(nominal),
            "right_exclusive_end_matches_probe": end_equals_probe,
            "alignment_clock_verified": correct_clock,
        },
        index=snapshot.index,
    )


def _endpoint_delta_audit(
    snapshot: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Characterise the declared ``window_end`` minus ``probe_onset`` difference.

    Scope limit (must not be overstated in any downstream report): this is a
    *metadata contract* audit only.  The snapshot carries no per-frame timestamps
    and no frame-membership evidence, so this function cannot and does not verify
    ``all(actual_frame_timestamp < probe_onset)``.  Frame-level evidence requires a
    later producer/snapshot contract extension.
    """
    end = pd.to_numeric(snapshot["window_end_unix_ms"], errors="coerce")
    probe = pd.to_numeric(snapshot["probe_onset_unix_ms"], errors="coerce")
    delta = end - probe

    per_probe = pd.DataFrame(
        {
            **{key: snapshot[key] for key in KEYS},
            GROUP: snapshot[GROUP],
            "repeat_participant_id": snapshot["repeat_participant_id"],
            "probe_id": snapshot["probe_id"],
            "window_end_unix_ms": end,
            "probe_onset_unix_ms": probe,
            "endpoint_delta_ms": delta,
        }
    )

    finite = delta.notna()
    zero = finite & delta.eq(0)
    nonzero = finite & delta.ne(0)
    nonzero_delta = delta.loc[nonzero]
    exceeds = finite & delta.abs().gt(ENDPOINT_ATOL_MS_PROVISIONAL)

    frequencies: dict[str, int] = {}
    if bool(nonzero.any()):
        counts = nonzero_delta.value_counts().sort_index()
        frequencies = {str(int(value)): int(count) for value, count in counts.items()}

    summary: dict[str, Any] = {
        "audit_scope": "metadata_contract_window_end_vs_probe_onset",
        "frame_membership_available": False,
        "frame_membership_note": (
            "The snapshot has no per-frame timestamp or frame-membership column, so "
            "actual entered-frame membership is NOT verified here; only the declared "
            "endpoint contract is audited."
        ),
        "provisional_atol_ms": ENDPOINT_ATOL_MS_PROVISIONAL,
        "provisional_atol_is_frozen_conclusion": False,
        "rtol": ENDPOINT_RTOL,
        "n_total": int(len(snapshot)),
        "n_finite": int(finite.sum()),
        "n_missing": int((~finite).sum()),
        "n_zero": int(zero.sum()),
        "n_nonzero": int(nonzero.sum()),
        "min_delta_ms": float(delta.min()) if bool(finite.any()) else None,
        "max_delta_ms": float(delta.max()) if bool(finite.any()) else None,
        "median_delta_ms": float(delta.median()) if bool(finite.any()) else None,
        "abs_max_delta_ms": float(delta.abs().max()) if bool(finite.any()) else None,
        "n_exceeding_provisional_atol": int(exceeds.sum()),
        "nonzero_delta_frequencies_ms": frequencies,
        "nonzero_session_n": int(snapshot.loc[nonzero, "session_id"].nunique()),
        "nonzero_participant_group_n": int(snapshot.loc[nonzero, GROUP].nunique()),
        "nonzero_sessions": sorted(
            snapshot.loc[nonzero, "session_id"].astype(str).unique().tolist()
        ),
        "nonzero_participant_groups": sorted(
            snapshot.loc[nonzero, GROUP].astype(str).unique().tolist()
        ),
    }
    return per_probe, per_probe.loc[nonzero].copy(), summary


def _feature_handoff() -> pd.DataFrame:
    common: dict[str, Any] = {
        "scientific_modality": SCIENTIFIC_MODALITY,
        "source_namespace": SOURCE_NAMESPACE,
        "required_devices": json.dumps(list(REQUIRED_DEVICES)),
        "preprocessing_dependencies": json.dumps([]),
        "temporal_anchor": "probe_time_ms",
        "temporal_scope": "pre_probe_only",
        "time_legality_status": TIME_LEGALITY_STATUS,
        "time_legality_evidence": TIME_LEGALITY_EVIDENCE,
        "physiology_qualification": PHYSIOLOGY_QUALIFICATION,
        "registry_ready": False,
        "researcher_freeze_required": True,
        "standalone_eligible": False,
        "behavior_increment_eligible": False,
        "modality_model_eligible": False,
        "full_model_eligible": False,
        "full_leave_one_out_eligible": False,
        "downstream_registry_eligibility_status": (
            "interface_contract_only_prediction_disabled_physiology_limited"
        ),
    }
    rows = [
        {
            **common,
            "feature_id": "mmwave_hr_fused_v1",
            "scientific_feature_id": "radar_derived_heart_rate",
            "predictor_column": HR_COLUMN,
            "feature_type": "heart_rate",
            "report_role": "supporting_only",
        },
        {
            **common,
            "feature_id": "mmwave_br_v1",
            "scientific_feature_id": "radar_derived_respiration_rate",
            "predictor_column": BR_COLUMN,
            "feature_type": "respiration_rate",
            "report_role": "supporting_only",
        },
    ]
    return pd.DataFrame(rows)


def _taskb_source(snapshot: pd.DataFrame, state: pd.Series) -> pd.DataFrame:
    keep = [
        *KEYS,
        GROUP,
        "repeat_participant_id",
        "probe_id",
        "window_name",
        "window_nominal_start_unix_ms",
        "window_effective_start_unix_ms",
        "window_end_unix_ms",
        "probe_onset_unix_ms",
        "alignment_clock_source",
        *SCIENTIFIC_COLUMNS,
        "source_availability_state",
        "source_readability_state",
        "estimability_state",
        "measurement_qc_state",
        "malformed_state",
        "integration_state",
        "producer_state",
        "missing_reason",
        "snapshot_version",
        "producer_commit",
        "producer_adapter_version",
        "source_run_id",
        "snapshot_run_id",
    ]
    out = snapshot[keep].copy()
    available = state.eq(AVAILABLE)
    malformed = state.eq(SOURCE_MALFORMED)
    # A malformed source existed but cannot be read/used. This distinction lets
    # quality_admission emit structural_source_unreadable instead of source missing.
    out["source_present"] = available | malformed
    out["source_readable"] = available
    out["native_qc_valid"] = available
    out["window_seconds_nominal"] = 30
    return out


def _coverage_table(
    snapshot: pd.DataFrame,
    state: pd.Series,
    counts: dict[str, int],
) -> pd.DataFrame:
    available = state.eq(AVAILABLE)
    unavailable = state.eq(SOURCE_UNAVAILABLE)
    malformed = state.eq(SOURCE_MALFORMED)
    base = {
        **counts,
        "available_probe_n": int(available.sum()),
        "source_unavailable_probe_n": int(unavailable.sum()),
        "source_malformed_probe_n": int(malformed.sum()),
        "retained_missing_or_error_probe_n": int((unavailable | malformed).sum()),
        "available_session_n": int(snapshot.loc[available, "session_id"].nunique()),
        "source_unavailable_session_n": int(
            snapshot.loc[unavailable, "session_id"].nunique()
        ),
        "source_malformed_session_n": int(
            snapshot.loc[malformed, "session_id"].nunique()
        ),
        "scientific_modality": SCIENTIFIC_MODALITY,
        "source_namespace": SOURCE_NAMESPACE,
        "required_devices": json.dumps(list(REQUIRED_DEVICES)),
        "time_legality_status": TIME_LEGALITY_STATUS,
        "physiology_qualification": PHYSIOLOGY_QUALIFICATION,
    }
    rows: list[dict[str, Any]] = []
    for feature_id, column in (
        ("mmwave_hr_fused_v1", HR_COLUMN),
        ("mmwave_br_v1", BR_COLUMN),
    ):
        numeric = pd.to_numeric(snapshot[column], errors="coerce")
        finite = np.isfinite(numeric)
        rows.append(
            {
                **base,
                "feature_id": feature_id,
                "predictor_column": column,
                "finite_probe_n": int(finite.sum()),
                "finite_rate_governed": (
                    float(finite.mean()) if len(finite) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _interface_analysis_sets(
    behavior: pd.DataFrame,
    taskb_source: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    quality = audit_quality(
        tables={"behavior": behavior, SOURCE_NAMESPACE: taskb_source},
        feature_blocks={SOURCE_NAMESPACE: list(SCIENTIFIC_COLUMNS)},
        rules={},
    )
    spec = {
        "cardiopulmonary_snapshot_v1_interface_only": {
            "models": ["cardiopulmonary_interface_only"],
            "required_features": {SCIENTIFIC_MODALITY: list(SCIENTIFIC_COLUMNS)},
            "required_feature_records": [
                {
                    "feature_id": "mmwave_hr_fused_v1",
                    "scientific_modality": SCIENTIFIC_MODALITY,
                    "source_namespace": SOURCE_NAMESPACE,
                    "predictor_column": HR_COLUMN,
                },
                {
                    "feature_id": "mmwave_br_v1",
                    "scientific_modality": SCIENTIFIC_MODALITY,
                    "source_namespace": SOURCE_NAMESPACE,
                    "predictor_column": BR_COLUMN,
                },
            ],
            # Interface smoke does not consume Q1/Q2. Prediction eligibility remains
            # a later Formal-method/researcher decision.
            "required_outcomes": [],
        }
    }
    sets, summary = build_analysis_sets(
        quality["formal_probe_identity"],
        quality["probe_feature_status"],
        spec,
    )
    return quality, sets, summary


def _analysis_membership_probe_n(summary: pd.DataFrame, membership: str) -> int:
    rows = summary.loc[summary["membership"].eq(membership), "probe_n"]
    if len(rows) != 1:
        raise MmwaveCardiopulmonaryIngestError(
            f"interface smoke expected one {membership} summary row, got {len(rows)}"
        )
    return int(rows.iloc[0])


def audit_mmwave_cardiopulmonary_snapshot(
    snapshot: pd.DataFrame,
    behavior_probe_table: pd.DataFrame,
    *,
    strict_canonical_counts: bool = True,
) -> MmwaveCardiopulmonaryIngestResult:
    """Validate one canonical snapshot against the Behavior probe authority."""
    _require_columns(snapshot, REQUIRED_SNAPSHOT_COLUMNS, "mmwave snapshot")
    _require_columns(behavior_probe_table, KEYS + [GROUP], "behavior authority")

    snapshot_n = _normalize_identity(snapshot, "mmwave snapshot")
    behavior_n = _normalize_identity(behavior_probe_table, "behavior authority")
    _validate_snapshot_provenance(snapshot_n)
    counts = _validate_identity_and_keys(
        snapshot_n,
        behavior_n,
        strict_canonical_counts=strict_canonical_counts,
    )
    state = _validate_integration_states(snapshot_n)
    time_audit = _validate_time_legality(snapshot_n)

    taskb_source = _taskb_source(snapshot_n, state)
    quality, analysis_sets, analysis_summary = _interface_analysis_sets(
        behavior_n,
        taskb_source,
    )

    available = state.eq(AVAILABLE)
    unavailable = state.eq(SOURCE_UNAVAILABLE)
    malformed = state.eq(SOURCE_MALFORMED)
    retained_missing_or_error = unavailable | malformed
    hr_finite = np.isfinite(pd.to_numeric(snapshot_n[HR_COLUMN], errors="coerce"))
    br_finite = np.isfinite(pd.to_numeric(snapshot_n[BR_COLUMN], errors="coerce"))

    complete_probe_n = _analysis_membership_probe_n(
        analysis_summary, "included_complete"
    )
    missing_aware_probe_n = _analysis_membership_probe_n(
        analysis_summary, "included_missing_aware"
    )

    ingest_audit = snapshot_n[
        KEYS
        + [
            GROUP,
            "repeat_participant_id",
            "probe_id",
            "integration_state",
            "source_availability_state",
            "source_readability_state",
            "estimability_state",
            "measurement_qc_state",
            "malformed_state",
            "producer_state",
            "missing_reason",
            "window_name",
            "window_nominal_start_unix_ms",
            "window_effective_start_unix_ms",
            "window_end_unix_ms",
            "probe_onset_unix_ms",
            "alignment_clock_source",
            "snapshot_version",
            "producer_commit",
            "producer_adapter_version",
            "source_run_id",
            "snapshot_run_id",
        ]
    ].copy()
    ingest_audit["scientific_modality"] = SCIENTIFIC_MODALITY
    ingest_audit["source_namespace"] = SOURCE_NAMESPACE
    ingest_audit["required_devices"] = json.dumps(list(REQUIRED_DEVICES))
    ingest_audit["source_present"] = available | malformed
    ingest_audit["source_readable"] = available
    ingest_audit["source_unavailable"] = unavailable
    ingest_audit["source_malformed"] = malformed
    ingest_audit["retained_missing_value"] = retained_missing_or_error
    ingest_audit["hr_available"] = hr_finite
    ingest_audit["br_available"] = br_finite
    for column in time_audit.columns:
        ingest_audit[column] = time_audit[column].to_numpy()
    ingest_audit["time_legality_status"] = TIME_LEGALITY_STATUS
    ingest_audit["time_legality_evidence"] = TIME_LEGALITY_EVIDENCE
    ingest_audit["physiology_qualification"] = PHYSIOLOGY_QUALIFICATION
    ingest_audit["downstream_registry_eligibility_status"] = (
        "interface_contract_only_prediction_disabled_physiology_limited"
    )

    coverage = _coverage_table(snapshot_n, state, counts)
    feature_handoff = _feature_handoff()
    endpoint_delta, endpoint_delta_nonzero, endpoint_delta_summary = (
        _endpoint_delta_audit(snapshot_n)
    )

    governed_real_complete = bool(strict_canonical_counts)
    manifest: dict[str, Any] = {
        "schema_version": ADAPTER_VERSION,
        "status": "READY" if governed_real_complete else "SMOKE_ONLY",
        "engineering_integration_qualification": (
            "READY" if governed_real_complete else "PENDING_GOVERNED_REAL_RUN"
        ),
        "governed_real_snapshot_rerun_complete": governed_real_complete,
        "physiology_qualification": PHYSIOLOGY_QUALIFICATION,
        "scientific_modality": SCIENTIFIC_MODALITY,
        "source_namespace": SOURCE_NAMESPACE,
        "required_devices": list(REQUIRED_DEVICES),
        **counts,
        "available_probe_n": int(available.sum()),
        "source_unavailable_probe_n": int(unavailable.sum()),
        "source_malformed_probe_n": int(malformed.sum()),
        "retained_missing_or_error_probe_n": int(retained_missing_or_error.sum()),
        "hr_available_n": int(hr_finite.sum()),
        "br_available_n": int(br_finite.sum()),
        "available_session_n": int(snapshot_n.loc[available, "session_id"].nunique()),
        "source_unavailable_session_n": int(
            snapshot_n.loc[unavailable, "session_id"].nunique()
        ),
        "source_malformed_session_n": int(
            snapshot_n.loc[malformed, "session_id"].nunique()
        ),
        "time_legality": {
            "temporal_anchor": "probe_time_ms",
            "temporal_scope": "pre_probe_only",
            "time_legality_status": TIME_LEGALITY_STATUS,
            "time_legality_evidence": TIME_LEGALITY_EVIDENCE,
            "right_exclusive_end": True,
            "alignment_clock_source": "dll_host_receive_enqueue",
            "truncated_window_n": int(
                time_audit["effective_start_truncated"].sum()
            ),
        },
        "endpoint_delta_audit": endpoint_delta_summary,
        "source_snapshot_version": CANONICAL_SNAPSHOT_VERSION,
        "source_producer_commits": _nonblank_unique(snapshot_n, "producer_commit"),
        "source_adapter_versions": _nonblank_unique(
            snapshot_n, "producer_adapter_version"
        ),
        "source_run_ids": _nonblank_unique(snapshot_n, "source_run_id"),
        "snapshot_run_ids": _nonblank_unique(snapshot_n, "snapshot_run_id"),
        "interface_smoke": {
            "included_complete_probe_n": complete_probe_n,
            "included_missing_aware_probe_n": missing_aware_probe_n,
            "required_scientific_modality": SCIENTIFIC_MODALITY,
            "source_namespace": SOURCE_NAMESPACE,
            "required_devices": list(REQUIRED_DEVICES),
            "required_outcomes": [],
        },
        "scientific_features": [
            {
                "feature_id": "mmwave_hr_fused_v1",
                "predictor_column": HR_COLUMN,
                "scientific_modality": SCIENTIFIC_MODALITY,
                "source_namespace": SOURCE_NAMESPACE,
                "required_devices": list(REQUIRED_DEVICES),
                "prediction_eligibility": False,
            },
            {
                "feature_id": "mmwave_br_v1",
                "predictor_column": BR_COLUMN,
                "scientific_modality": SCIENTIFIC_MODALITY,
                "source_namespace": SOURCE_NAMESPACE,
                "required_devices": list(REQUIRED_DEVICES),
                "prediction_eligibility": False,
            },
        ],
        "prohibited_promotions": sorted(PROHIBITED_FORMAL_FEATURE_COLUMNS),
        "final_feature_registry_modified": False,
        "models_trained": False,
        "q1_q2_used_for_ingest_decision": False,
        "source_unavailable_zero_imputed": False,
        "source_malformed_zero_imputed": False,
        "participant_identity_inferred_from_folder": False,
    }

    if strict_canonical_counts:
        expected = {
            "available": CANONICAL_AVAILABLE_N,
            "source_unavailable": CANONICAL_SOURCE_UNAVAILABLE_N,
            "source_malformed": CANONICAL_SOURCE_MALFORMED_N,
            "hr_available": CANONICAL_AVAILABLE_N,
            "br_available": CANONICAL_AVAILABLE_N,
            "analysis_complete": CANONICAL_AVAILABLE_N,
            "analysis_missing_aware": CANONICAL_AVAILABLE_N,
        }
        observed = {
            "available": manifest["available_probe_n"],
            "source_unavailable": manifest["source_unavailable_probe_n"],
            "source_malformed": manifest["source_malformed_probe_n"],
            "hr_available": manifest["hr_available_n"],
            "br_available": manifest["br_available_n"],
            "analysis_complete": complete_probe_n,
            "analysis_missing_aware": missing_aware_probe_n,
        }
        if observed != expected:
            raise MmwaveCardiopulmonaryIngestError(
                "canonical snapshot availability/interface denominator mismatch: "
                f"observed={observed}, expected={expected}"
            )

    return MmwaveCardiopulmonaryIngestResult(
        taskb_source=taskb_source,
        ingest_audit=ingest_audit,
        coverage=coverage,
        feature_handoff=feature_handoff,
        quality_tables=quality,
        analysis_sets=analysis_sets,
        analysis_set_summary=analysis_summary,
        manifest=manifest,
        endpoint_delta=endpoint_delta,
        endpoint_delta_nonzero=endpoint_delta_nonzero,
        endpoint_delta_summary=endpoint_delta_summary,
    )


def write_mmwave_cardiopulmonary_ingest_audit(
    output_dir: str | Path,
    result: MmwaveCardiopulmonaryIngestResult,
) -> dict[str, str]:
    """Write focused audit outputs plus reusable current Task-B interface tables."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    paths = {
        "ingest_audit": output / "mmwave_cardiopulmonary_ingest_audit.csv",
        "coverage": output / "mmwave_cardiopulmonary_coverage.csv",
        "feature_handoff": output / "mmwave_cardiopulmonary_feature_handoff.csv",
        "taskb_source": output / "mmwave_cardiopulmonary_taskb_source.csv",
        "manifest": output / "mmwave_cardiopulmonary_ingest_manifest.json",
        "endpoint_delta_ms": output / "endpoint_delta_ms.csv",
        "endpoint_delta_nonzero_ms": output / "endpoint_delta_nonzero_ms.csv",
        "endpoint_delta_summary": output / "endpoint_delta_summary.json",
    }
    result.ingest_audit.to_csv(
        paths["ingest_audit"], index=False, encoding="utf-8-sig"
    )
    result.coverage.to_csv(paths["coverage"], index=False, encoding="utf-8-sig")
    result.feature_handoff.to_csv(
        paths["feature_handoff"], index=False, encoding="utf-8-sig"
    )
    result.taskb_source.to_csv(
        paths["taskb_source"], index=False, encoding="utf-8-sig"
    )
    paths["manifest"].write_text(
        json.dumps(result.manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result.endpoint_delta.to_csv(
        paths["endpoint_delta_ms"], index=False, encoding="utf-8-sig"
    )
    result.endpoint_delta_nonzero.to_csv(
        paths["endpoint_delta_nonzero_ms"], index=False, encoding="utf-8-sig"
    )
    paths["endpoint_delta_summary"].write_text(
        json.dumps(result.endpoint_delta_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    standard_dir = output / "interface_smoke"
    quality_paths = write_quality_audit(standard_dir, result.quality_tables)
    analysis_paths = write_analysis_sets(
        standard_dir,
        result.analysis_sets,
        result.analysis_set_summary,
    )
    return {
        **{key: str(path) for key, path in paths.items()},
        **{f"quality_{key}": value for key, value in quality_paths.items()},
        **{f"analysis_{key}": value for key, value in analysis_paths.items()},
    }
