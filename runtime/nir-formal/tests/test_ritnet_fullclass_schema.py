from __future__ import annotations

import pytest

from ritnet_fullclass_schema import (
    EYE_METRIC_FIELDS,
    EYE_METRICS_SCHEMA_VERSION,
    FRAME_COVERAGE_FIELDS,
    FRAME_COVERAGE_SCHEMA_VERSION,
    project_row,
    validate_exact_schema,
)


def test_final_schemas_are_unique_and_versioned():
    assert len(EYE_METRIC_FIELDS) == len(set(EYE_METRIC_FIELDS))
    assert len(FRAME_COVERAGE_FIELDS) == len(set(FRAME_COVERAGE_FIELDS))
    assert EYE_METRICS_SCHEMA_VERSION == 1
    assert FRAME_COVERAGE_SCHEMA_VERSION == 1
    assert "source_pupil_confidence" not in EYE_METRIC_FIELDS
    assert "pupil_confidence" not in EYE_METRIC_FIELDS


def test_project_row_does_not_leak_historical_columns():
    row = {
        "subject": "sub-031",
        "frame_idx": 100,
        "pupil_confidence": 0.99,
        "old_pupil_axis_a": 42,
    }
    projected = project_row(row, EYE_METRIC_FIELDS)
    assert set(projected) == set(EYE_METRIC_FIELDS)
    assert projected["subject"] == "sub-031"
    assert "pupil_confidence" not in projected
    assert "old_pupil_axis_a" not in projected


def test_validate_exact_schema_rejects_extra_or_reordered_fields():
    validate_exact_schema(list(EYE_METRIC_FIELDS), EYE_METRIC_FIELDS)
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_exact_schema([*EYE_METRIC_FIELDS, "extra"], EYE_METRIC_FIELDS)
    reordered = list(EYE_METRIC_FIELDS)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_exact_schema(reordered, EYE_METRIC_FIELDS)
