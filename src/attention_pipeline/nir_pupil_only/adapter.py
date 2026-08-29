"""Fail-closed adapter from RITnet fullclass-final to pupil-only analysis rows.

The producer CSVs are read by field name.  Source fields are never rewritten;
``frame_left``/``frame_right`` are normalized only in the returned table while
``eye_raw`` retains the producer value.  Iris class fractions remain ordinary
segmentation fields and are never promoted to iris geometry or PIR/OAR.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


ADAPTER_VERSION = "nir-pupil-only-adapter-v1.0.0"
OUTPUT_SCHEMA_VERSION = 1
SUPPORTED_SOURCE_SCHEMAS = {6, 7}
PRIMARY_KEY = ["subject", "phase", "phase_segment", "frame_idx", "eye"]
FORBIDDEN_DERIVED_COLUMNS = {
    "pir",
    "oar",
    "pupil_to_iris_ratio",
    "fullclass_pupil_to_iris_diameter_ratio",
    "fullclass_ocular_aperture_ratio_median",
    "fullclass_ocular_aperture_ratio_p90",
}

# Canonical names are stable downstream names.  Each tuple is ordered by
# preference and documents actual v6/v7 producer spellings before evidence-plan
# aliases.  Missing optional fields remain explicit NA values.
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "pupil_axis_a": ("pupil_short_axis", "pupil_axis_a"),
    "pupil_axis_b": ("pupil_long_axis", "pupil_axis_b"),
    "pupil_equivalent_diameter": (
        "pupil_equiv_diameter",
        "pupil_equivalent_diameter",
    ),
    "prev_frame_idx": ("temporal_prev_frame_idx", "prev_frame_idx"),
    "frame_gap": ("temporal_frame_gap", "frame_gap"),
    "time_gap_ms": ("temporal_time_gap_ms", "time_gap_ms"),
    "temporal_reset": ("temporal_reset_reason", "temporal_reset"),
    "center_jump": ("delta_pupil_center_distance_px", "center_jump"),
    "diameter_delta": ("delta_pupil_geom_mean_diameter", "diameter_delta"),
    "predicted_in_padding_fraction": (
        "predicted_in_padding_fraction",
        "pupil_predicted_in_padding_fraction",
    ),
    "touches_roi_edge": (
        "pupil_touches_valid_domain_edge",
        "touches_roi_edge",
    ),
    "soft_max_probability": (
        "ocular_max_probability_mean",
        "soft_max_probability",
    ),
    "soft_margin": ("ocular_top1_top2_margin_mean", "soft_margin"),
    "soft_entropy": ("ocular_entropy_mean", "soft_entropy"),
}

CORE_FIELDS = (
    "eye_metrics_schema_version",
    "subject",
    "frame_idx",
    "eye",
    "phase",
    "phase_segment",
    "video_time_ms",
    "unix_ms",
    "phase_time_ms",
    "source_detection_source",
    "source_frame_status",
    "source_eye_status",
    "source_redetect_reason",
    "source_yolo_batch_size",
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
    "pupil_geom_mean_diameter",
    "pupil_component_count",
    "pupil_largest_component_fraction",
    "qc_pupil_fragmented",
    "hard_pupil_fraction",
    "soft_pupil_fraction",
    "hard_iris_fraction",
    "soft_iris_fraction",
    "temporal_anomaly",
)


class IrisGeometryUnavailableError(ValueError):
    """Raised when PIR/OAR is requested without independent iris geometry."""


@dataclass(frozen=True)
class SourceIdentity:
    subject: str
    source_schema_version: int
    source_path: str
    source_manifest_path: str
    source_kind: str
    source_branch: str | None = None
    source_commit: str | None = None

    @classmethod
    def from_manifest(
        cls, manifest: Mapping[str, Any], manifest_path: str | Path
    ) -> "SourceIdentity":
        required = {"subject", "source_schema_version", "source_path", "source_kind"}
        missing = sorted(required - set(manifest))
        if missing:
            raise ValueError(f"source manifest missing fields: {missing}")
        version = int(manifest["source_schema_version"])
        if version not in SUPPORTED_SOURCE_SCHEMAS:
            raise ValueError(f"unsupported eye metrics schema: {version}")
        return cls(
            subject=str(manifest["subject"]),
            source_schema_version=version,
            source_path=str(manifest["source_path"]),
            source_manifest_path=str(manifest_path),
            source_kind=str(manifest["source_kind"]),
            source_branch=_none_or_str(manifest.get("source_branch")),
            source_commit=_none_or_str(manifest.get("source_commit")),
        )


def _none_or_str(value: Any) -> str | None:
    return None if value is None or pd.isna(value) else str(value)


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y", "t"})


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


def _validate_source_schema(frame: pd.DataFrame, identity: SourceIdentity) -> None:
    _require_columns(
        frame,
        [
            "eye_metrics_schema_version",
            "subject",
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
        ],
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
    subjects = set(frame["subject"].dropna().astype(str))
    if subjects != {identity.subject}:
        raise ValueError(
            f"source subject mismatch: manifest={identity.subject}, rows={sorted(subjects)}"
        )


def _normalize_eye(raw: pd.Series) -> pd.Series:
    mapping = {
        "frame_left": "left",
        "frame_right": "right",
        "left": "left",
        "right": "right",
    }
    normalized = raw.astype("string").map(mapping)
    bad = sorted(raw[normalized.isna()].dropna().astype(str).unique())
    if bad:
        raise ValueError(f"unsupported raw eye labels: {bad}")
    return normalized


def _validate_primary_key_and_time(frame: pd.DataFrame) -> None:
    if frame[PRIMARY_KEY].isna().any().any():
        raise ValueError("pupil-only primary key contains missing values")
    duplicate = frame.duplicated(PRIMARY_KEY, keep=False)
    if duplicate.any():
        example = frame.loc[duplicate, PRIMARY_KEY].head(3).to_dict("records")
        raise ValueError(f"duplicate pupil-only primary key: {example}")
    time_groups = ["subject", "phase", "phase_segment", "eye"]
    for key, group in frame.groupby(time_groups, sort=False, dropna=False):
        times = pd.to_numeric(group["unix_ms"], errors="coerce")
        if times.isna().any():
            raise ValueError(f"non-numeric unix_ms in group {key}")
        if (times.diff().dropna() < 0).any():
            raise ValueError(f"non-monotonic unix_ms in source order for group {key}")


def classify_quality_tracks(frame: pd.DataFrame) -> pd.DataFrame:
    """Add independent flags and one fail-closed exclusive analysis track."""
    out = frame.copy()
    eye_status = out.get("source_eye_status", pd.Series("", index=out.index)).astype(
        "string"
    ).str.lower()
    ritnet_status = out.get("ritnet_status", pd.Series("", index=out.index)).astype(
        "string"
    ).str.lower()

    out["source_observed"] = eye_status.eq("observed")
    out["source_missing"] = eye_status.isin(
        {"missing", "source_missing", "yolo_missing", "yolo_no_eye", "not_observed"}
    )
    out["ritnet_missing"] = eye_status.eq("ritnet_missing") | ~ritnet_status.eq(
        "success"
    )
    edge = _as_bool(out.get("touches_roi_edge", pd.Series(False, index=out.index)))
    padding_pixels = _numeric(out, "pupil_predicted_in_padding_pixels").fillna(0).gt(0)
    out["roi_clipped"] = eye_status.eq("roi_clipped") | edge | padding_pixels

    found = _as_bool(out.get("pupil_found", pd.Series(False, index=out.index)))
    fit = _as_bool(out.get("pupil_fit_valid", pd.Series(False, index=out.index)))
    required_geometry = [
        "pupil_center_x",
        "pupil_center_y",
        "pupil_axis_a",
        "pupil_axis_b",
        "pupil_geom_mean_diameter",
    ]
    geometry_present = pd.concat(
        [_numeric(out, name).notna() for name in required_geometry], axis=1
    ).all(axis=1)
    out["geometry_invalid"] = ~(found & fit & geometry_present)
    out["temporal_flagged"] = _as_bool(
        out.get("temporal_anomaly", pd.Series(False, index=out.index))
    )
    out["interpolation_only"] = _as_bool(
        out.get("interpolation_only", pd.Series(False, index=out.index))
    )

    conditions = [
        out["interpolation_only"],
        out["source_missing"],
        out["ritnet_missing"],
        out["roi_clipped"],
        out["geometry_invalid"],
        out["temporal_flagged"],
    ]
    labels = [
        "interpolation_only",
        "source_missing",
        "ritnet_missing",
        "roi_clipped",
        "geometry_invalid",
        "temporal_flagged",
    ]
    out["quality_track"] = np.select(conditions, labels, default="observed")
    return out


def adapt_session(
    eye_metrics: pd.DataFrame,
    source_manifest: Mapping[str, Any],
    *,
    source_manifest_path: str | Path,
) -> pd.DataFrame:
    """Project one v6/v7 session into the versioned pupil-only row schema."""
    identity = SourceIdentity.from_manifest(source_manifest, source_manifest_path)
    _validate_source_schema(eye_metrics, identity)

    out = pd.DataFrame(index=eye_metrics.index)
    for field in CORE_FIELDS:
        out[field] = (
            eye_metrics[field].copy()
            if field in eye_metrics
            else pd.Series(pd.NA, index=eye_metrics.index)
        )
    for canonical, aliases in FIELD_ALIASES.items():
        out[canonical] = _first_existing(eye_metrics, aliases)

    out["eye_raw"] = eye_metrics["eye"].astype("string")
    out["eye"] = _normalize_eye(out["eye_raw"])
    out["source_schema_version"] = identity.source_schema_version
    out["source_path"] = identity.source_path
    out["source_manifest_path"] = identity.source_manifest_path
    out["source_kind"] = identity.source_kind
    out["source_branch"] = identity.source_branch
    out["source_commit"] = identity.source_commit
    out["adapter_version"] = ADAPTER_VERSION
    out["output_schema_version"] = OUTPUT_SCHEMA_VERSION

    leaked = FORBIDDEN_DERIVED_COLUMNS & set(out.columns)
    if leaked:
        raise AssertionError(f"forbidden iris-derived columns in output: {sorted(leaked)}")
    _validate_primary_key_and_time(out)
    return classify_quality_tracks(out)


def _visual_columns(frame: pd.DataFrame) -> list[str]:
    identity = {
        "stimulus_name",
        "stimulus_code",
        "stimulus_size_pct",
        "base_size_px",
        "rendered_size_px",
        "stimulus_pos_x_psychopy_px",
        "stimulus_pos_y_psychopy_px",
        "stimulus_center_x_image_px",
        "stimulus_center_y_image_px",
    }
    numeric_visual = {
        name
        for name in frame.columns
        if "rel_lum" in name or "rms_contrast" in name
    }
    return [name for name in frame.columns if name in identity | numeric_visual]


def _validate_visual_keys(visual: pd.DataFrame) -> None:
    keys = ["stimulus_name", "stimulus_size_pct"]
    _require_columns(visual, keys, "visual properties")
    if visual.duplicated(keys).any():
        raise ValueError("visual property key is not unique")
    physical = [name for name in visual.columns if "cd_m2" in name.lower() or "cd/m" in name.lower()]
    if physical:
        raise ValueError(
            "relative luminance asset must not be labelled as physical cd/m²: "
            f"{physical}"
        )


def _prepare_behavior(behavior: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        behavior,
        ["subject", "absolute_onset_time", "stimulus_name", "stimulus_size"],
        "behavior trials",
    )
    out = behavior.copy()
    if "phase" not in out:
        if "block_num" not in out:
            raise ValueError("behavior trials require phase or block_num")
        out["phase"] = out["block_num"].map({1: "block1", 2: "block2"})
    if "phase_segment" not in out:
        out["phase_segment"] = out["phase"]
    out["absolute_onset_time"] = pd.to_numeric(
        out["absolute_onset_time"], errors="coerce"
    )
    if out["absolute_onset_time"].isna().any():
        raise ValueError("behavior absolute_onset_time contains non-numeric values")
    group = ["subject", "phase", "phase_segment"]
    out = out.sort_values(group + ["absolute_onset_time"]).reset_index(drop=True)
    if "next_trial_onset_time" not in out:
        out["next_trial_onset_time"] = out.groupby(group, dropna=False)[
            "absolute_onset_time"
        ].shift(-1)
    out["_previous_onset_time"] = out.groupby(group, dropna=False)[
        "absolute_onset_time"
    ].shift(1)
    if "prev_stimulus_name" not in out:
        out["prev_stimulus_name"] = out.groupby(group, dropna=False)[
            "stimulus_name"
        ].shift(1)
    if "prev_stimulus_size" not in out:
        out["prev_stimulus_size"] = out.groupby(group, dropna=False)[
            "stimulus_size"
        ].shift(1)
    if "prev_is_no_go" not in out and "is_no_go" in out:
        out["prev_is_no_go"] = out.groupby(group, dropna=False)["is_no_go"].shift(1)
    return out


def attach_behavior_and_visual(
    pupil: pd.DataFrame,
    behavior: pd.DataFrame,
    visual_properties: pd.DataFrame,
) -> pd.DataFrame:
    """Interval-link eye rows to trials with auditable current/previous visuals."""
    _validate_visual_keys(visual_properties)
    trials = _prepare_behavior(behavior)
    result_parts: list[pd.DataFrame] = []
    group_names = ["subject", "phase", "phase_segment"]

    for group_key, eyes in pupil.groupby(group_names, sort=False, dropna=False):
        mask = pd.Series(True, index=trials.index)
        for name, value in zip(group_names, group_key):
            mask &= trials[name].astype("string").eq(str(value))
        block_trials = trials.loc[mask].copy()
        left = eyes.sort_values("unix_ms").copy()
        if block_trials.empty:
            left["behavior_match_status"] = "no_behavior_group"
            left["behavior_match_failure_reason"] = "phase_or_segment_not_found"
            left["behavior_match_delta_ms"] = np.nan
            result_parts.append(left)
            continue
        joined = pd.merge_asof(
            left,
            block_trials.sort_values("absolute_onset_time"),
            left_on="unix_ms",
            right_on="absolute_onset_time",
            direction="backward",
            suffixes=("", "_behavior"),
        )
        after_onset = joined["absolute_onset_time"].notna()
        before_next = joined["next_trial_onset_time"].isna() | (
            pd.to_numeric(joined["unix_ms"]) < pd.to_numeric(joined["next_trial_onset_time"])
        )
        matched = after_onset & before_next
        joined["behavior_match_status"] = np.where(matched, "matched", "failed")
        joined["behavior_match_failure_reason"] = np.select(
            [~after_onset, after_onset & ~before_next],
            ["before_first_trial", "outside_trial_interval"],
            default=pd.NA,
        )
        joined["behavior_match_delta_ms"] = pd.to_numeric(joined["unix_ms"]) - pd.to_numeric(
            joined["absolute_onset_time"]
        )
        joined["behavior_match_phase"] = joined["phase"]
        joined["behavior_match_phase_segment"] = joined["phase_segment"]
        result_parts.append(joined)

    linked = pd.concat(result_parts, ignore_index=True, sort=False)
    visual_cols = _visual_columns(visual_properties)
    current = visual_properties[visual_cols].copy().rename(
        columns={name: f"current_{name}" for name in visual_cols}
    )
    linked["current_stimulus_name"] = linked.get("stimulus_name")
    linked["current_stimulus_size_pct"] = pd.to_numeric(
        linked.get("stimulus_size"), errors="coerce"
    )
    linked = linked.merge(
        current,
        on=["current_stimulus_name", "current_stimulus_size_pct"],
        how="left",
        suffixes=("", "_visual"),
        validate="many_to_one",
    )
    # If keys also appear in the visual payload, retain the join keys once.
    linked["current_visual_match_status"] = np.where(
        linked.get("current_stimulus_code").notna(), "matched", "missing"
    )
    linked["current_visual_failure_reason"] = np.where(
        linked["current_visual_match_status"].eq("matched"),
        pd.NA,
        np.where(
            linked["behavior_match_status"].ne("matched"),
            "behavior_not_matched",
            "visual_key_not_found",
        ),
    )

    previous = visual_properties[visual_cols].copy().rename(
        columns={name: f"previous_{name}" for name in visual_cols}
    )
    linked["previous_stimulus_name"] = linked.get("prev_stimulus_name")
    linked["previous_stimulus_size_pct"] = pd.to_numeric(
        linked.get("prev_stimulus_size"), errors="coerce"
    )
    linked = linked.merge(
        previous,
        on=["previous_stimulus_name", "previous_stimulus_size_pct"],
        how="left",
        suffixes=("", "_visual"),
        validate="many_to_one",
    )
    is_first = linked["previous_stimulus_name"].isna()
    previous_matched = linked.get("previous_stimulus_code").notna()
    linked["previous_visual_match_status"] = np.select(
        [is_first, previous_matched], ["not_applicable", "matched"], default="missing"
    )
    linked["previous_visual_failure_reason"] = np.select(
        [is_first, previous_matched], ["block_first_trial", pd.NA], default="visual_key_not_found"
    )
    linked["current_trial_onset_unix_ms"] = linked.get("absolute_onset_time")
    linked["current_next_trial_onset_unix_ms"] = linked.get("next_trial_onset_time")
    linked["previous_trial_onset_unix_ms"] = linked.get("_previous_onset_time")
    linked["visual_luminance_semantics"] = (
        "linear-sRGB relative luminance; digital relative metric; not physical cd/m²"
    )
    return linked.sort_values(PRIMARY_KEY).reset_index(drop=True)


def refuse_pir_without_iris_geometry(*_: Any, **__: Any) -> None:
    raise IrisGeometryUnavailableError(
        "PIR/OAR refused: fullclass-final provides pupil geometry and iris class "
        "fractions, but no independently validated iris geometry. "
        "hard_iris_fraction/soft_iris_fraction are not iris diameters."
    )


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))
