"""Fixed versioned output schemas for the final NIR RITnet full-class workflow.

The final tables intentionally do NOT inherit every historical ``eyes.csv``
column. Source data are mapped into a small explicit provenance set, while all
new scientific/QC fields have stable names and documented coordinate semantics.
"""
from __future__ import annotations

from typing import Any, Mapping


EYE_METRICS_SCHEMA_VERSION = 5
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

COMPONENT_FIELDS = tuple(
    field
    for name in ("pupil", "iris_outer", "ocular")
    for field in (f"{name}_component_count", f"{name}_largest_component_fraction")
)

GEOMETRY_FIELDS = tuple(
    field
    for name in ("pupil", "iris_outer")
    for field in (
        f"{name}_found",
        f"{name}_fit_valid",
        f"{name}_center_x",
        f"{name}_center_y",
        f"{name}_short_axis",
        f"{name}_long_axis",
        f"{name}_angle_deg",
        f"{name}_contour_area",
        f"{name}_ellipse_area",
        f"{name}_equiv_diameter",
        f"{name}_geom_mean_diameter",
        f"{name}_whole_mask_touches_edge",
        f"{name}_largest_contour_touches_edge",
    )
)

RELATION_FIELDS = (
    "pupil_to_iris_diameter_ratio",
    "pupil_to_iris_ellipse_area_ratio",
    "pupil_to_iris_contour_area_ratio",
    "pupil_center_offset_px",
    "pupil_center_offset_norm",
    "pupil_center_in_iris_outer",
    "iris_diameter_gt_pupil_diameter",
    "pir_finite",
    "iris_outer_fill_ratio",
)

OAR_FIELDS = (
    "ocular_bbox_width",
    "ocular_bbox_height",
    "ocular_aperture_height_median",
    "ocular_aperture_height_p90",
    "ocular_aperture_ratio_median",
    "ocular_aperture_ratio_p90",
    "ocular_whole_mask_touches_edge",
)

ATOMIC_QC_FIELDS = (
    "qc_pupil_fragmented",
    "qc_iris_outer_fragmented",
    "qc_ocular_fragmented",
)

UNCERTAINTY_BASE_FIELDS = (
    "uncertainty_algorithm_version",
    "uncertainty_domain_version",
    "soft_class_fraction_domain_version",
    "uncertainty_boundary_band_px",
    "soft_background_fraction",
    "soft_sclera_fraction",
    "soft_iris_fraction",
    "soft_pupil_fraction",
    "uncertainty_ocular_pixel_count",
    "uncertainty_boundary_pixel_count",
)
UNCERTAINTY_DISTRIBUTION_FIELDS = tuple(
    f"{domain}_{metric}_{stat}"
    for domain in ("whole", "ocular", "boundary")
    for metric in ("max_probability", "top1_top2_margin", "entropy")
    for stat in ("mean", "p05", "p25", "p50", "p75", "p95")
)
UNCERTAINTY_THRESHOLD_FIELDS = (
    "low_max_probability_threshold",
    "whole_low_max_probability_fraction",
    "ocular_low_max_probability_fraction",
    "boundary_low_max_probability_fraction",
)

TEMPORAL_FIELDS = (
    "temporal_qc_version",
    "temporal_prev_frame_idx",
    "temporal_frame_gap",
    "temporal_time_gap_ms",
    "temporal_reset_reason",
    "delta_hard_pupil_fraction",
    "delta_hard_iris_outer_fraction",
    "delta_hard_ocular_fraction",
    "delta_pupil_to_iris_diameter_ratio",
    "delta_ocular_aperture_ratio_median",
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
    """Return exactly the fixed schema fields; unknown inputs cannot leak through."""
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
