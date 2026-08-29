from __future__ import annotations

import pytest

from ritnet_fullclass_schema import (
    ANALYSIS_DOMAIN_FIELDS,
    EYE_METRIC_FIELDS,
    EYE_METRICS_SCHEMA_VERSION,
    FRAME_COVERAGE_FIELDS,
    FRAME_COVERAGE_SCHEMA_VERSION,
    SOURCE_YOLO_FIELDS,
    UNCERTAINTY_BASE_FIELDS,
    VALIDATION_GEOMETRY_FIELDS,
    project_row,
    validate_exact_schema,
)


def test_shadow_schema_is_unique_and_versioned():
    assert len(EYE_METRIC_FIELDS) == len(set(EYE_METRIC_FIELDS))
    assert len(FRAME_COVERAGE_FIELDS) == len(set(FRAME_COVERAGE_FIELDS))
    assert EYE_METRICS_SCHEMA_VERSION == 7
    assert FRAME_COVERAGE_SCHEMA_VERSION == 2
    assert "source_detection_source" in SOURCE_YOLO_FIELDS
    assert "source_detection_source" in EYE_METRIC_FIELDS
    assert "source_pupil_confidence" not in EYE_METRIC_FIELDS
    assert "pupil_confidence" not in EYE_METRIC_FIELDS
    assert "soft_class_fraction_domain_version" in UNCERTAINTY_BASE_FIELDS


def test_shadow_schema_keeps_production_fields_and_adds_three_validation_paths():
    for name in ("background", "sclera", "iris", "pupil"):
        assert f"hard_{name}_pixels" in EYE_METRIC_FIELDS
        assert f"hard_{name}_fraction" in EYE_METRIC_FIELDS
        assert f"soft_{name}_fraction" in EYE_METRIC_FIELDS
    assert "pupil_geom_mean_diameter" in EYE_METRIC_FIELDS
    assert "pupil_center_x" in EYE_METRIC_FIELDS
    assert "iris_outer_fit_valid" not in EYE_METRIC_FIELDS
    assert "pupil_to_iris_diameter_ratio" not in EYE_METRIC_FIELDS
    assert "ocular_aperture_ratio_median" not in EYE_METRIC_FIELDS
    assert "boundary_entropy_p95" not in EYE_METRIC_FIELDS
    assert "whole_max_probability_p50" not in EYE_METRIC_FIELDS
    assert "ocular_entropy_mean" in EYE_METRIC_FIELDS

    assert "validation_geometry_version" in VALIDATION_GEOMETRY_FIELDS
    for method in ("legacy", "topology", "ellseg"):
        assert f"validation_{method}_pupil_fit_valid" in VALIDATION_GEOMETRY_FIELDS
        assert f"validation_{method}_pupil_center_x" in VALIDATION_GEOMETRY_FIELDS
        assert f"validation_{method}_pupil_geom_mean_diameter" in VALIDATION_GEOMETRY_FIELDS
        assert f"validation_{method}_pupil_geometry_method" in VALIDATION_GEOMETRY_FIELDS
        assert f"validation_{method}_pupil_geometry_failure_reason" in VALIDATION_GEOMETRY_FIELDS
    assert "validation_ellseg_pupil_valid_boundary_point_count" in VALIDATION_GEOMETRY_FIELDS
    assert "validation_ellseg_pupil_ransac_used" in VALIDATION_GEOMETRY_FIELDS
    assert set(VALIDATION_GEOMETRY_FIELDS).issubset(set(EYE_METRIC_FIELDS))


def test_analysis_domain_qc_facts_are_persisted_by_final_schema():
    expected = {
        "analysis_domain_version",
        "analysis_valid_pixel_count",
        "analysis_valid_pixel_fraction",
        "pupil_predicted_in_padding_pixels",
        "iris_outer_predicted_in_padding_pixels",
        "ocular_predicted_in_padding_pixels",
        "pupil_touches_valid_domain_edge",
        "iris_outer_touches_valid_domain_edge",
        "ocular_touches_valid_domain_edge",
    }
    assert set(ANALYSIS_DOMAIN_FIELDS) == expected
    assert expected.issubset(set(EYE_METRIC_FIELDS))

    source = {field: f"value:{field}" for field in ANALYSIS_DOMAIN_FIELDS}
    projected = project_row(source, EYE_METRIC_FIELDS)
    for field in ANALYSIS_DOMAIN_FIELDS:
        assert projected[field] == source[field]


def test_project_row_preserves_explicit_provenance_without_leaking_historical_columns():
    row = {
        "subject": "sub-031",
        "frame_idx": 100,
        "source_detection_source": "track",
        "pupil_confidence": 0.99,
        "old_pupil_axis_a": 42,
    }
    projected = project_row(row, EYE_METRIC_FIELDS)
    assert set(projected) == set(EYE_METRIC_FIELDS)
    assert projected["subject"] == "sub-031"
    assert projected["source_detection_source"] == "track"
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
