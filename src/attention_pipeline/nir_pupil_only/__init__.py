"""Pupil-only NIR source contracts used by the authoritative downstream chain."""

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

__all__ = [
    "OUTPUT_SCHEMA_VERSION",
    "SUPPORTED_SOURCE_SCHEMAS",
    "IrisGeometryUnavailableError",
    "SourceIdentity",
    "adapt_session_rows",
    "cohort_topology_summary",
    "refuse_iris_derived_metrics",
    "validate_cohort_topology",
]
