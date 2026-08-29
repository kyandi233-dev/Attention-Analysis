from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

SUPPORTED_SOURCE_SCHEMAS = {6, 7}
OUTPUT_SCHEMA_VERSION = 2
PRIMARY_KEY = ["session_id", "phase", "phase_segment", "frame_idx", "eye"]

# These fields require iris geometry or encode historical pupil/iris ratios and
# therefore remain forbidden in the pupil-only formal output. The producer's
# fullclass ocular-aperture ratios are intentionally NOT in this set: they are
# eye-opening QC candidates derived from the visible ocular mask, not iris diameter.
FORBIDDEN_IRIS_DERIVED_FIELDS = {
    "pir",
    "oar",
    "pupil_to_iris_ratio",
    "fullclass_pupil_to_iris_diameter_ratio",
}

OCULAR_APERTURE_QC_FIELDS = (
    "fullclass_ocular_aperture_ratio_median",
    "fullclass_ocular_aperture_ratio_p90",
)

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "pupil_axis_a": ("pupil_short_axis", "pupil_axis_a"),
    "pupil_axis_b": ("pupil_long_axis", "pupil_axis_b"),
    "pupil_equivalent_diameter": (
        "pupil_equiv_diameter",
        "pupil_equivalent_diameter",
    ),
    "pupil_geom_mean_diameter": (
        "pupil_geom_mean_diameter",
        "fullclass_pupil_geom_mean_diameter",
    ),
    "prev_frame_idx": ("temporal_prev_frame_idx", "prev_frame_idx"),
    "frame_gap": ("temporal_frame_gap", "frame_gap"),
    "time_gap_ms": ("temporal_time_gap_ms", "time_gap_ms"),
    "temporal_reset": ("temporal_reset_reason", "temporal_reset"),
    "center_jump": ("delta_pupil_center_distance_px", "center_jump"),
    "diameter_delta": ("delta_pupil_geom_mean_diameter", "diameter_delta"),
    "touches_roi_edge": (
        "pupil_touches_valid_domain_edge",
        "fullclass_pupil_touches_roi_edge",
        "touches_roi_edge",
    ),
    "soft_max_probability": ("ocular_max_probability_mean", "soft_max_probability"),
    "soft_margin": ("ocular_top1_top2_margin_mean", "soft_margin"),
    "soft_entropy": ("ocular_entropy_mean", "soft_entropy"),
}

COPY_FIELDS = (
    "eye_metrics_schema_version",
    "phase",
    "phase_segment",
    "frame_idx",
    "video_time_ms",
    "unix_ms",
    "phase_time_ms",
    "source_detection_source",
    "source_frame_status",
    "source_eye_status",
    "source_redetect_reason",
    "ritnet_status",
    "ritnet_failure_reason",
    "roi_valid_content_fraction",
    "analysis_domain_version",
    "analysis_valid_pixel_count",
    "analysis_valid_pixel_fraction",
    "pupil_predicted_in_padding_pixels",
    "pupil_found",
    "pupil_fit_valid",
    "pupil_center_x",
    "pupil_center_y",
    "pupil_angle_deg",
    "pupil_contour_area",
    "pupil_ellipse_area",
    "pupil_component_count",
    "pupil_largest_component_fraction",
    "qc_pupil_fragmented",
    "hard_pupil_fraction",
    "soft_pupil_fraction",
    "hard_iris_fraction",
    "soft_iris_fraction",
    *OCULAR_APERTURE_QC_FIELDS,
    "temporal_anomaly",
)


class IrisGeometryUnavailableError(ValueError):
    """PIR/iris-geometry metrics cannot be derived from the pupil-only contract."""


@dataclass(frozen=True)
class SourceIdentity:
    session_id: str
    analysis_group_token: str
    source_schema_version: int
    repeat_group_size: int = 1
    source_kind: str = "ritnet-fullclass-pupil-only"
    source_branch: str | None = None
    source_commit: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceIdentity":
        required = {"session_id", "analysis_group_token", "source_schema_version"}
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"source identity missing fields: {missing}")
        version = int(value["source_schema_version"])
        if version not in SUPPORTED_SOURCE_SCHEMAS:
            raise ValueError(f"unsupported eye metrics schema: {version}")
        repeat_size = int(value.get("repeat_group_size", 1))
        if repeat_size < 1:
            raise ValueError("repeat_group_size must be >= 1")
        return cls(
            session_id=str(value["session_id"]),
            analysis_group_token=str(value["analysis_group_token"]),
            source_schema_version=version,
            repeat_group_size=repeat_size,
            source_kind=str(value.get("source_kind", "ritnet-fullclass-pupil-only")),
            source_branch=_none_or_str(value.get("source_branch")),
            source_commit=_none_or_str(value.get("source_commit")),
        )


def _none_or_str(value: Any) -> str | None:
    return None if value is None or pd.isna(value) else str(value)


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes", "y", "t"})
    )


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _first_existing(frame: pd.DataFrame, aliases: Iterable[str]) -> pd.Series:
    for name in aliases:
        if name in frame:
            return frame[name].copy()
    return pd.Series(pd.NA, index=frame.index, dtype="object")


def _require_columns(frame: pd.DataFrame, names: Iterable[str], label: str) -> None:
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required fields: {missing}")


def _normalize_eye(raw: pd.Series) -> pd.Series:
    mapping = {
        "frame_left": "left",
        "frame_right": "right",
        "left": "left",
        "right": "right",
    }
    normalized = raw.astype("string").str.strip().str.lower().map(mapping)
    bad = sorted(raw[normalized.isna()].dropna().astype(str).unique())
    if bad:
        raise ValueError(f"unsupported raw eye labels: {bad}")
    return normalized


def _validate_source(frame: pd.DataFrame, identity: SourceIdentity) -> None:
    _require_columns(
        frame,
        (
            "eye_metrics_schema_version",
            "phase",
            "phase_segment",
            "frame_idx",
            "eye",
            "unix_ms",
            "video_time_ms",
            "phase_time_ms",
            "source_eye_status",
            "ritnet_status",
            "pupil_found",
            "pupil_fit_valid",
        ),
        "eye metrics",
    )
    versions = set(
        pd.to_numeric(frame["eye_metrics_schema_version"], errors="coerce")
        .dropna()
        .astype(int)
    )
    if versions != {identity.source_schema_version}:
        raise ValueError(
            "source schema mismatch: "
            f"manifest={identity.source_schema_version}, rows={sorted(versions)}"
        )


def classify_quality(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    eye_status = out["source_eye_status"].astype("string").str.strip().str.lower()
    ritnet_status = out["ritnet_status"].astype("string").str.strip().str.lower()

    out["source_observed"] = eye_status.eq("observed")
    out["source_missing"] = eye_status.isin(
        {"missing", "source_missing", "yolo_missing", "yolo_no_eye", "not_observed"}
    )
    out["ritnet_missing"] = eye_status.eq("ritnet_missing") | ~ritnet_status.eq("success")
    edge = _as_bool(out.get("touches_roi_edge", pd.Series(False, index=out.index)))
    padding = _numeric(out, "pupil_predicted_in_padding_pixels").fillna(0).gt(0)
    out["roi_clipped"] = eye_status.eq("roi_clipped") | edge | padding

    found = _as_bool(out["pupil_found"])
    fit = _as_bool(out["pupil_fit_valid"])
    diameter = _numeric(out, "pupil_geom_mean_diameter")
    center_x = _numeric(out, "pupil_center_x")
    center_y = _numeric(out, "pupil_center_y")
    geometry_ok = (
        found
        & fit
        & np.isfinite(diameter)
        & diameter.gt(0)
        & np.isfinite(center_x)
        & np.isfinite(center_y)
    )
    out["geometry_invalid"] = ~geometry_ok
    out["temporal_flagged"] = _as_bool(
        out.get("temporal_anomaly", pd.Series(False, index=out.index))
    )
    out["interpolation_only"] = _as_bool(
        out.get("interpolation_only", pd.Series(False, index=out.index))
    )

    out["pupil_valid_primary"] = (
        out["source_observed"]
        & ~out["ritnet_missing"]
        & ~out["geometry_invalid"]
        & ~out["interpolation_only"]
    )
    out["pupil_valid_strict"] = (
        out["pupil_valid_primary"] & ~out["roi_clipped"] & ~out["temporal_flagged"]
    )
    out["quality_track"] = np.select(
        [
            out["interpolation_only"],
            out["source_missing"],
            out["ritnet_missing"],
            out["roi_clipped"],
            out["geometry_invalid"],
            out["temporal_flagged"],
        ],
        [
            "interpolation_only",
            "source_missing",
            "ritnet_missing",
            "roi_clipped",
            "geometry_invalid",
            "temporal_flagged",
        ],
        default="observed",
    )
    return out


def _validate_key_and_time(frame: pd.DataFrame) -> None:
    if frame[PRIMARY_KEY].isna().any().any():
        raise ValueError("pupil-only primary key contains missing values")
    duplicate = frame.duplicated(PRIMARY_KEY, keep=False)
    if duplicate.any():
        raise ValueError("duplicate pupil-only primary key")
    for key, group in frame.groupby(
        ["session_id", "phase", "phase_segment", "eye"], sort=False, dropna=False
    ):
        times = pd.to_numeric(group["unix_ms"], errors="coerce")
        if times.isna().any():
            raise ValueError(f"non-numeric unix_ms in group {key}")
        if (times.diff().dropna() < 0).any():
            raise ValueError(f"non-monotonic unix_ms in source order for group {key}")


def adapt_session_rows(
    eye_metrics: pd.DataFrame,
    source_identity: Mapping[str, Any] | SourceIdentity,
) -> pd.DataFrame:
    """Project one schema-v6/v7 session to the canonical pupil-only row contract."""
    identity = (
        source_identity
        if isinstance(source_identity, SourceIdentity)
        else SourceIdentity.from_mapping(source_identity)
    )
    _validate_source(eye_metrics, identity)
    out = pd.DataFrame(index=eye_metrics.index)
    for field in COPY_FIELDS:
        out[field] = (
            eye_metrics[field].copy()
            if field in eye_metrics
            else pd.Series(pd.NA, index=eye_metrics.index)
        )
    for canonical, aliases in FIELD_ALIASES.items():
        out[canonical] = _first_existing(eye_metrics, aliases)

    out["eye_raw"] = eye_metrics["eye"].astype("string")
    out["eye"] = _normalize_eye(out["eye_raw"])
    out["session_id"] = identity.session_id
    # ``subject`` is retained only as the legacy session key expected by the
    # existing Behavior loader; it must not be interpreted as participant N.
    out["subject"] = identity.session_id
    out["analysis_group_token"] = identity.analysis_group_token
    out["repeat_group_size"] = identity.repeat_group_size
    out["is_repeat_session"] = identity.repeat_group_size > 1
    out["source_schema_version"] = identity.source_schema_version
    out["source_kind"] = identity.source_kind
    out["source_branch"] = identity.source_branch
    out["source_commit"] = identity.source_commit
    out["output_schema_version"] = OUTPUT_SCHEMA_VERSION

    aperture = pd.concat(
        [pd.to_numeric(out[name], errors="coerce") for name in OCULAR_APERTURE_QC_FIELDS],
        axis=1,
    )
    out["ocular_aperture_available"] = np.isfinite(aperture).any(axis=1)
    out["ocular_aperture_role"] = "nir_eye_opening_candidate_qc"
    out["ocular_aperture_interpretation"] = "not_ear_not_blink_not_perclos"

    forbidden = FORBIDDEN_IRIS_DERIVED_FIELDS & {name.lower() for name in out.columns}
    if forbidden:
        raise AssertionError(f"forbidden iris-derived columns in pupil-only output: {sorted(forbidden)}")
    out = classify_quality(out)
    _validate_key_and_time(out)
    return out


def cohort_topology_summary(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    rows = [SourceIdentity.from_mapping(value) for value in records]
    sessions = {row.session_id for row in rows}
    if len(sessions) != len(rows):
        raise ValueError("topology manifest contains duplicate session_id")
    groups: dict[str, list[str]] = {}
    for row in rows:
        groups.setdefault(row.analysis_group_token, []).append(row.session_id)
    invalid = {key: len(value) for key, value in groups.items() if len(value) not in {1, 2}}
    if invalid:
        raise ValueError("current topology supports only one- or two-session analysis groups")
    repeat_groups = sum(len(value) == 2 for value in groups.values())
    return {
        "n_sessions": len(sessions),
        "n_analysis_groups": len(groups),
        "n_double_session_repeat_groups": int(repeat_groups),
    }


def validate_cohort_topology(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_sessions: int = 44,
    expected_analysis_groups: int = 38,
    expected_double_session_repeat_groups: int = 6,
) -> dict[str, int]:
    summary = cohort_topology_summary(records)
    expected = {
        "n_sessions": int(expected_sessions),
        "n_analysis_groups": int(expected_analysis_groups),
        "n_double_session_repeat_groups": int(expected_double_session_repeat_groups),
    }
    if summary != expected:
        raise ValueError(f"cohort topology mismatch: observed={summary}, expected={expected}")
    return summary


def refuse_iris_derived_metrics(*_: Any, **__: Any) -> None:
    raise IrisGeometryUnavailableError(
        "PIR/iris-geometry metrics refused: accepted fullclass-final sources provide "
        "pupil geometry but no independently validated iris geometry. Iris class "
        "fractions are segmentation proportions and must not be treated as iris diameter. "
        "Producer ocular-aperture ratios are separate eye-opening QC candidates and are "
        "not EAR, blink events, or PERCLOS."
    )
