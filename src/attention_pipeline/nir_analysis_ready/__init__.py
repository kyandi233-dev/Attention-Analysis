"""Authoritative NIR analysis-ready interface.

The default materializer is pupil-only.  Canonical cohort membership and NIR
source availability are validated separately: the governed cohort is 116
sessions, while only current-contract NIR sources are materialized.  Missing or
legacy-incompatible NIR remains explicit missingness and never redefines the
cohort.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from attention_pipeline.config import load_config
from attention_pipeline.formal_analysis.cohort import included_cohort, load_cohort_manifest
from .candidate_metrics import (
    CANDIDATE_PIPELINE_VERSION,
    CANDIDATE_SCHEMA_VERSION,
    PUPIL_CANDIDATE_METRICS,
    apply_candidate_standardization,
    compute_candidate_baselines,
    run_candidate_materialization,
)
from .materialize import (
    apply_subject_eye_standardization as legacy_apply_subject_eye_standardization,
    build_wide_timepoints as legacy_build_wide_timepoints,
    compute_subject_eye_baselines as legacy_compute_subject_eye_baselines,
    derive_frame_validity as legacy_derive_frame_validity,
)
from .pupil_only import (
    ANALYSIS_READY_PIPELINE_VERSION,
    ANALYSIS_READY_SCHEMA_VERSION,
    apply_session_eye_standardization,
    build_wide_timepoints,
    compute_session_eye_baselines,
    load_source_manifest,
    run_materialization as _run_materialization,
)
from attention_pipeline.nir_pupil_only import cohort_topology_summary


def _normalized_nir_semantics(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out.pop("pir_oar_allowed", None)
    out.pop("pir_oar_refused", None)
    out["pir_iris_geometry_refused"] = True
    out["ocular_aperture_qc_preserved"] = True
    out["ocular_aperture_formal_endpoint"] = False
    out["ocular_aperture_interpretation"] = "producer_qc_not_ear_not_blink_not_perclos"
    return out


def _full_contract(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    cohort = included_cohort(load_cohort_manifest(config), require_groups=True)
    group_sizes = cohort.groupby("repeat_participant_id")["session_id"].nunique()
    distribution = {
        str(size): int(group_sizes.eq(size).sum())
        for size in sorted(group_sizes.astype(int).unique())
    }
    canonical = {
        "n_sessions": int(len(cohort)),
        "n_analysis_groups": int(group_sizes.size),
        "n_repeated_participant_groups": int(group_sizes.gt(1).sum()),
        "max_sessions_per_participant": int(group_sizes.max()) if len(group_sizes) else 0,
        "group_size_distribution": distribution,
    }
    expected_cfg = config.section("cohort_topology")
    expected = {
        "n_sessions": int(expected_cfg.get("sessions", 116)),
        "n_analysis_groups": int(expected_cfg.get("analysis_groups", 61)),
    }
    observed_required = {key: int(canonical[key]) for key in expected}
    if observed_required != expected:
        raise ValueError(
            f"canonical cohort topology mismatch: observed={observed_required}, expected={expected}"
        )

    payload, records = load_source_manifest(config)
    available = {str(row["session_id"]) for row in records}
    raw_unavailable = payload.get("unavailable_sessions", [])
    if not isinstance(raw_unavailable, list):
        raise ValueError("source manifest unavailable_sessions must be a list")
    unavailable_rows = [row for row in raw_unavailable if isinstance(row, dict)]
    if len(unavailable_rows) != len(raw_unavailable):
        raise TypeError("every unavailable NIR session entry must be an object")
    unavailable = {str(row.get("session_id", "")).strip() for row in unavailable_rows}
    if "" in unavailable:
        raise ValueError("unavailable NIR session missing session_id")
    if available & unavailable:
        raise ValueError(f"NIR session cannot be both available and unavailable: {sorted(available & unavailable)}")

    canonical_sessions = set(cohort["session_id"].astype(str))
    accounted = available | unavailable
    if accounted != canonical_sessions:
        raise ValueError(
            "NIR availability must account for the governed cohort exactly: "
            f"missing={sorted(canonical_sessions-accounted)}, outside={sorted(accounted-canonical_sessions)}"
        )
    if int(payload.get("session_count", len(available))) != len(available):
        raise ValueError("source manifest session_count does not match sessions list")
    if int(payload.get("unavailable_session_count", len(unavailable))) != len(unavailable):
        raise ValueError("source manifest unavailable_session_count does not match unavailable_sessions")

    group_size_map = group_sizes.astype(int).to_dict()
    group_map = cohort.set_index("session_id")["repeat_participant_id"].astype(str).to_dict()
    for row in records:
        session = str(row["session_id"])
        expected_group = str(group_map[session])
        if str(row["analysis_group_token"]) != expected_group:
            raise ValueError(f"{session}: NIR analysis_group_token disagrees with canonical cohort")
        expected_size = int(group_size_map[expected_group])
        if int(row.get("repeat_group_size", expected_size)) != expected_size:
            raise ValueError(f"{session}: repeat_group_size disagrees with canonical cohort")

    return {
        "canonical_cohort_topology": canonical,
        "nir_availability": {
            "n_available_sessions": len(available),
            "n_unavailable_sessions": len(unavailable),
            "n_canonical_sessions": len(canonical_sessions),
            "complete_accounting": True,
            "availability_does_not_redefine_cohort": True,
        },
        "source_available_subset_topology": cohort_topology_summary(records),
    }


def _apply_contract(payload: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    out = _normalized_nir_semantics(payload)
    prior_topology = out.get("topology")
    if prior_topology is not None and "source_available_subset_topology" not in out:
        out["source_available_subset_topology"] = prior_topology
    out["topology"] = contract["canonical_cohort_topology"]
    out["canonical_cohort_topology"] = contract["canonical_cohort_topology"]
    out["nir_availability"] = contract["nir_availability"]
    out["cohort_availability_contract"] = (
        "canonical cohort validated independently; only available current-contract NIR sources are materialized"
    )
    return out


def _rewrite_json_if_present(path_value: object, contract: dict[str, Any]) -> None:
    if path_value in (None, ""):
        return
    path = Path(str(path_value))
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    path.write_text(
        json.dumps(_apply_contract(payload, contract), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_materialization(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run pupil-only materialization with independent cohort/availability gates."""
    config_path = args[0] if args else kwargs.get("config_path")
    if config_path in (None, ""):
        raise TypeError("run_materialization requires config_path")
    contract = _full_contract(config_path)
    result = _run_materialization(*args, **kwargs)
    if not isinstance(result, dict):
        raise TypeError("pupil-only materializer must return a dict")
    summary = result.get("summary")
    if isinstance(summary, dict):
        result["summary"] = _apply_contract(summary, contract)
    _rewrite_json_if_present(result.get("summary_path"), contract)
    _rewrite_json_if_present(result.get("manifest_path"), contract)
    return result


__all__ = [
    "ANALYSIS_READY_PIPELINE_VERSION",
    "ANALYSIS_READY_SCHEMA_VERSION",
    "CANDIDATE_PIPELINE_VERSION",
    "CANDIDATE_SCHEMA_VERSION",
    "PUPIL_CANDIDATE_METRICS",
    "apply_candidate_standardization",
    "compute_candidate_baselines",
    "run_candidate_materialization",
    "apply_session_eye_standardization",
    "build_wide_timepoints",
    "compute_session_eye_baselines",
    "load_source_manifest",
    "run_materialization",
    "legacy_apply_subject_eye_standardization",
    "legacy_build_wide_timepoints",
    "legacy_compute_subject_eye_baselines",
    "legacy_derive_frame_validity",
]
