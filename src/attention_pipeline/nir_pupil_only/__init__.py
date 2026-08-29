"""Versioned pupil-only adapter for fullclass-final NIR outputs."""

from .adapter import (
    ADAPTER_VERSION,
    OUTPUT_SCHEMA_VERSION,
    IrisGeometryUnavailableError,
    adapt_session,
    attach_behavior_and_visual,
    classify_quality_tracks,
    refuse_pir_without_iris_geometry,
)

__all__ = [
    "ADAPTER_VERSION",
    "OUTPUT_SCHEMA_VERSION",
    "IrisGeometryUnavailableError",
    "adapt_session",
    "attach_behavior_and_visual",
    "classify_quality_tracks",
    "refuse_pir_without_iris_geometry",
]
