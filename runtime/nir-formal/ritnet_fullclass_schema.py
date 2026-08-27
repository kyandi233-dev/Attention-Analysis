"""Fixed lean schema for the final NIR RITnet full-class workflow.

Four-class hard/soft segmentation, pupil geometry, source/ROI provenance and
compact temporal/model-QC facts are retained. Expensive or scientifically weak
iris ellipse/PIR/OAR fields and obsolete empty uncertainty distributions are not
part of the production cohort schema.
"""
from __future__ import annotations

from typing import Any, Mapping


EYE_METRICS_SCHEMA_VERSION = 6
FRAME_COVERAGE_SCHEMA_VERSION = 2

IDENTITY_FIELDS = (
    "eye_metrics_schema_version",
    "subject",
    "frame_idx",
    "eye",
    "phase",
    "phase_segment",
    "video_time_ms",
    "unix_ms",
    "phase_time_ms",
)

SOURCE_YOLO_FIELDS = (
    "source_detection_source",
    "source_frame_status",
    "source_eye_status",
    "source_redetect_reason",
    "source_yolo_batch_size",
    "yolo_confidence",
    "yolo_bbox_x1",
    "yolo_bbox_y1",
    "yolo_bbox_x2",
    "yolo_bbox_y2",
)

ROI_FIELDS = (
    "roi_expanded_x1",
    "roi_expanded_y1",
    "roi_expanded_x2",
    "roi_expanded_y2",
    "roi_requested_x1",
    "roi_requested_y1",
    "roi_requested_x2",
    "roi_requested_y2",
    "roi_source_x1",
    "roi_source_y1",
    "roi_source_x2",
    "roi_source_y2",
    "roi_width",
    "roi_height",
    "roi_pad_left",
    "roi_pad_top",
    "roi_pad_right",
    "roi_pad_bottom",
    "roi_padding_mode",
    "roi_valid_content_fraction",
    "roi_resize_scale",
    "roi_algorithm_version",
)

RITNET_STATUS_FIELDS = (
    "ritnet_status",
    "ritnet_failure_reason",
)

# Preserve the four native classes plus two cheap unions. No geometry is implied.
HARD_CLASS_FIELDS = tuple(
    field
    for name in ("background", "sclera", "iris", "pupil", "iris_outer", "ocular")
    for field in (f"hard_{name}_pixels", f"hard_{name}_fraction")
)

ANALYSIS_DOMAIN_FIELDS = (
    "analysis_domain_version",
    "analysis_valid_pixel_count",
    "analysis_valid_pixel_fraction",
    "pupil_predicted_in_padding_pixels",
    "iris_outer_predicted_in_padding_pixels",
    "ocular_predicted_in_padding_pixels",
    "pupil_touches_valid_domain_edge",
    "iris_outer_touches_valid_domain_edge",
    "ocular_touches_valid_domain_edge",
)

# Connected-component QC is retained only for the pupil.
COMPONENT_FIELDS = (
    "pupil_component_count",
    "pupil_largest_component_fraction",
)

# Formal NIR geometry is pupil-only. Iris remains available as a hard/soft class.
GEOMETRY_FIELDS = tuple(
    f"pupil_{suffix}"
    for suffix in (
        "found",
        "fit_valid",
        "center_x",
        "center_y",
        "short_axis",
        "long_axis",
        "angle_deg",
        "contour_area",
        "ellipse_area",
        "equiv_diameter",
        "geom_mean_diameter",
        "whole_mask_touches_edge",
        "largest_contour_touches_edge",
    )
)

# Retained as named compatibility groups for callers; intentionally empty in v6.
RELATION_FIELDS: tuple[str, ...] = ()
OAR_FIELDS: tuple[str, ...] = ()

ATOMIC_QC_FIELDS = (
    "qc_pupil_fragmented",
)

UNCERTAINTY_BASE_FIELDS = (
    "uncertainty_algorithm_version",
    "uncertainty_domain_version",
    "soft_class_fraction_domain_version",
    "soft_background_fraction",
    "soft_sclera_fraction",
    "soft_iris_fraction",
    "soft_pupil_fraction",
    "uncertainty_ocular_pixel_count",
)

# Cohort production only needs these three scalar uncertainty summaries.
UNCERTAINTY_DISTRIBUTION_FIELDS = (
    "ocular_max_probability_mean",
    "ocular_top1_top2_margin_mean",
    "ocular_entropy_mean",
)
UNCERTAINTY_THRESHOLD_FIELDS: tuple[str, ...] = ()

TEMPORAL_FIELDS = (
    "temporal_qc_version",
    "temporal_prev_frame_idx",
    "temporal_frame_gap",
    "temporal_time_gap_ms",
    "temporal_reset_reason",
    "delta_hard_pupil_fraction",
    "delta_hard_ocular_fraction",
    "delta_pupil_center_x",
    "delta_pupil_center_y",
    "delta_pupil_center_distance_px",
    "delta_ocular_max_probability_mean",
    "delta_ocular_top1_top2_margin_mean",
    "delta_ocular_entropy_mean",
    "temporal_jump_score",
    "temporal_anomaly",
)

EYE_METRIC_FIELDS = (
    *IDENTITY_FIELDS,
    *SOURCE_YOLO_FIELDS,
    *ROI_FIELDS,
    *RITNET_STATUS_FIELDS,
    *HARD_CLASS_FIELDS,
    *ANALYSIS_DOMAIN_FIELDS,
    *COMPONENT_FIELDS,
    *GEOMETRY_FIELDS,
    *RELATION_FIELDS,
    *OAR_FIELDS,
    *ATOMIC_QC_FIELDS,
    *UNCERTAINTY_BASE_FIELDS,
    *UNCERTAINTY_DISTRIBUTION_FIELDS,
    *UNCERTAINTY_THRESHOLD_FIELDS,
    *TEMPORAL_FIELDS,
)

FRAME_COVERAGE_FIELDS = (
    "frame_coverage_schema_version",
    "subject",
    "phase",
    "phase_segment",
    "frame_idx",
    "video_time_ms",
    "unix_ms",
    "phase_time_ms",
    "source_frame_status",
    "source_raw_detection_count",
    "source_selected_eye_count",
    "source_left_eye_present",
    "source_right_eye_present",
    "left_ritnet_status",
    "left_failure_reason",
    "right_ritnet_status",
    "right_failure_reason",
    "ritnet_success_eye_count",
    "coverage_status",
    "fixed_qc_anchor",
)


def _assert_unique(fields: tuple[str, ...], name: str) -> None:
    duplicates = sorted({field for field in fields if fields.count(field) > 1})
    if duplicates:
        raise AssertionError(f"duplicate fields in {name}: {duplicates}")


_assert_unique(EYE_METRIC_FIELDS, "EYE_METRIC_FIELDS")
_assert_unique(FRAME_COVERAGE_FIELDS, "FRAME_COVERAGE_FIELDS")


def project_row(row: Mapping[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: row.get(field) for field in fields}


def validate_exact_schema(fieldnames: list[str] | tuple[str, ...], expected: tuple[str, ...]) -> None:
    actual = tuple(fieldnames)
    if actual != expected:
        missing = [field for field in expected if field not in actual]
        unexpected = [field for field in actual if field not in expected]
        raise ValueError(
            "schema mismatch: "
            f"missing={missing}, unexpected={unexpected}, order_matches={actual == expected}"
        )
