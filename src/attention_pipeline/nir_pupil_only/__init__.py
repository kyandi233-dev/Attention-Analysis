"""Versioned pupil-only NIR contracts and adapter APIs.

The package exposes two compatible layers:
- ``adapter`` / ``join``: the canonical source adapter and join contract.
- ``contract``: the formal staged-analysis identity/row contract used by
  ``10_analysis_ready -> 11_analysis_tables -> validation``.

Topology is exported from ``topology`` rather than the historical contract
implementation because repeat participants are not limited to one or two
sessions in the governed 116-session cohort.
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
    refuse_iris_derived_metrics,
)
from .join import attach_behavior_and_visual
from .topology import cohort_topology_summary, validate_cohort_topology

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
