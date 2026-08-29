"""Portable, audit-first formal multimodal analysis interfaces."""

from .cohort import (
    CohortSummary,
    attach_repeat_groups,
    included_cohort,
    load_cohort_manifest,
    summarize_cohort,
    validate_participant_disjoint_folds,
)
from .merge import validate_merge_ready
from .nir_adapter import adapt_nir_frame_table

__all__ = [
    "CohortSummary",
    "attach_repeat_groups",
    "included_cohort",
    "load_cohort_manifest",
    "summarize_cohort",
    "validate_participant_disjoint_folds",
    "validate_merge_ready",
    "adapt_nir_frame_table",
]
