"""Authoritative NIR analysis-ready interface.

The default materializer is pupil-only. The historical PIR helpers remain
importable from ``nir_analysis_ready.materialize`` for provenance and legacy
tests, but they are not used by the authoritative downstream entry points.

The authoritative export also normalizes the persisted manifest semantics:
PIR / iris geometry remain refused, while producer-derived ocular-aperture
ratio fields are preserved strictly as eye-opening QC candidates. They are not
EAR, blink events, PERCLOS, or automatic formal endpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def _normalized_nir_semantics(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out.pop("pir_oar_allowed", None)
    out.pop("pir_oar_refused", None)
    out["pir_iris_geometry_refused"] = True
    out["ocular_aperture_qc_preserved"] = True
    out["ocular_aperture_formal_endpoint"] = False
    out["ocular_aperture_interpretation"] = "producer_qc_not_ear_not_blink_not_perclos"
    return out


def _rewrite_json_if_present(path_value: object) -> None:
    if path_value in (None, ""):
        return
    path = Path(str(path_value))
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    path.write_text(
        json.dumps(_normalized_nir_semantics(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_materialization(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run pupil-only materialization and normalize authoritative metadata.

    The underlying staged materializer is intentionally left intact for
    provenance-compatible tests. This public authoritative interface removes the
    obsolete PIR/OAR conflation from returned and persisted metadata.
    """
    result = _run_materialization(*args, **kwargs)
    if not isinstance(result, dict):
        raise TypeError("pupil-only materializer must return a dict")

    summary = result.get("summary")
    if isinstance(summary, dict):
        result["summary"] = _normalized_nir_semantics(summary)

    _rewrite_json_if_present(result.get("summary_path"))
    _rewrite_json_if_present(result.get("manifest_path"))
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
