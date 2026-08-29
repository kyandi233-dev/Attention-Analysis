"""Authoritative NIR analysis-ready interface.

The default materializer is pupil-only. The historical PIR helpers remain
importable from ``nir_analysis_ready.materialize`` for provenance and legacy
tests, but they are not used by the authoritative downstream entry points.
"""

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
    run_materialization,
)

__all__ = [
    "ANALYSIS_READY_PIPELINE_VERSION",
    "ANALYSIS_READY_SCHEMA_VERSION",
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
