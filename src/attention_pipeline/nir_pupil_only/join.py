from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from attention_pipeline.formal_analysis.join_keys import normalize_known_join_dtypes

from .adapter import attach_behavior_and_visual as _attach_behavior_and_visual


def attach_behavior_and_visual(
    pupil: pd.DataFrame,
    behavior: pd.DataFrame,
    visual_properties: pd.DataFrame,
    *,
    time_units: Mapping[str, str] | None = None,
    naive_timezone: str | None = None,
) -> pd.DataFrame:
    """Normalize timeline keys before the canonical pupil/behavior interval join.

    This is the package-level entry point. ``unix_ms`` and
    ``absolute_onset_time`` are normalized to UTC epoch milliseconds; phase and
    phase-segment identifiers are normalized to string dtype. Missing required
    join keys, mixed numeric/datetime representations, ambiguous time units, or
    timezone-naive datetimes fail closed before ``merge_asof`` is attempted.
    """
    pupil_required = ["subject", "phase", "phase_segment", "unix_ms"]
    pupil_normalized = normalize_known_join_dtypes(
        pupil,
        required_non_null=pupil_required,
        time_units=time_units,
        naive_timezone=naive_timezone,
    )

    behavior_required = ["subject", "absolute_onset_time"]
    behavior_normalized = normalize_known_join_dtypes(
        behavior,
        required_non_null=behavior_required,
        time_units=time_units,
        naive_timezone=naive_timezone,
    )
    return _attach_behavior_and_visual(
        pupil_normalized,
        behavior_normalized,
        visual_properties,
    )
