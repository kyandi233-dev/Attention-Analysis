"""Versioned pupil-only adapter for fullclass-final NIR outputs."""

from .adapter import (
    ADAPTER_VERSION,
    OUTPUT_SCHEMA_VERSION,
    IrisGeometryUnavailableError,
    adapt_session,
    classify_quality_tracks,
    refuse_pir_without_iris_geometry,
)
from .join import attach_behavior_and_visual

__all__ = [
    "ADAPTER_VERSION",
    "OUTPUT_SCHEMA_VERSION",
    "IrisGeometryUnavailableError",
    "adapt_session",
    "attach_behavior_and_visual",
    "classify_quality_tracks",
    "refuse_pir_without_iris_geometry",
]
