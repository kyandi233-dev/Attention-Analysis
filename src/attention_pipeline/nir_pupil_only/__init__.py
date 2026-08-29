"""Versioned pupil-only NIR contracts and adapter APIs.

The package exposes two compatible layers:
- ``adapter`` / ``join``: the PR #27 canonical source adapter plus PR #28 join contract.
- ``contract``: the formal staged-analysis identity/topology contract used by
  ``10_analysis_ready -> 11_analysis_tables -> validation``.

The two SourceIdentity types intentionally have different responsibilities; the
formal staged contract is exported as ``SourceIdentity`` because the staged
materializer consumes it, while the original adapter identity remains available
as ``AdapterSourceIdentity`` for callers that need source-manifest path metadata.
"""

from .adapter import (
    ADAPTER_VERSION,
    OUTPUT_SCHEMA_VERSION as ADAPTER_OUTPUT_SCHEMA_VERSION,
    IrisGeometryUnavailableError as AdapterIrisGeometryUnavailableError,
    SourceIdentity as AdapterSourceIdentity,
    adapt_session,
    classify_quality_tracks,
    refuse_pir_without_iris_geometry,
)
from .contract import (
    OUTPUT_SCHEMA_VERSION,
    SUPPORTED_SOURCE_SCHEMAS,
    IrisGeometryUnavailableError,
    SourceIdentity,
    adapt_session_rows,
    cohort_topology_summary,
    refuse_iris_derived_metrics,
    validate_cohort_topology,
)
from .join import attach_behavior_and_visual

__all__ = [
    "ADAPTER_VERSION",
    "ADAPTER_OUTPUT_SCHEMA_VERSION",
    "AdapterIrisGeometryUnavailableError",
    "AdapterSourceIdentity",
    "OUTPUT_SCHEMA_VERSION",
    "SUPPORTED_SOURCE_SCHEMAS",
    "IrisGeometryUnavailableError",
    "SourceIdentity",
    "adapt_session",
    "adapt_session_rows",
    "attach_behavior_and_visual",
    "classify_quality_tracks",
    "cohort_topology_summary",
    "refuse_iris_derived_metrics",
    "refuse_pir_without_iris_geometry",
    "validate_cohort_topology",
]
