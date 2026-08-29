from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _numeric(value: object) -> float | None:
    x = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(x) or not np.isfinite(float(x)) else float(x)


def _frame_landmark(frame: pd.DataFrame, name: str) -> pd.Series | None:
    rows = frame[frame["landmark_name"].astype(str).str.lower().eq(name)]
    return None if rows.empty else rows.iloc[0]


def _quality_ok(row: pd.Series | None, *, min_visibility: float, min_presence: float) -> bool:
    if row is None:
        return False
    x, y = _numeric(row.get("x")), _numeric(row.get("y"))
    vis, pres = _numeric(row.get("visibility")), _numeric(row.get("presence"))
    return bool(
        x is not None and y is not None
        and vis is not None and vis >= min_visibility
        and pres is not None and pres >= min_presence
    )


def _bbox_area(first: pd.Series) -> float | None:
    vals = [_numeric(first.get(c)) for c in ("pose_bbox_xmin", "pose_bbox_ymin", "pose_bbox_xmax", "pose_bbox_ymax")]
    if any(v is None for v in vals):
        return None
    xmin, ymin, xmax, ymax = vals  # type: ignore[misc]
    area = (xmax - xmin) * (ymax - ymin)
    return float(area) if area > 0 else None


def _safe_log_ratio(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None or current <= 0 or previous <= 0:
        return None
    return float(np.log(current / previous))


def derive_pose_direction(
    pose: pd.DataFrame,
    *,
    min_visibility: float = 0.5,
    min_presence: float = 0.5,
    gap_reset_ms: float = 300.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Derive signed pose-confirmation candidates from preserved Pose landmarks.

    Positive ``pose_lateral_right_per_sec`` means movement toward increasing image x.
    Positive ``pose_vertical_up_per_sec`` means movement upward on screen (decreasing image y).
    ``pose_radial_proximity_candidate_per_sec`` is a non-physical composite candidate:
    positive values mean the available visual cues jointly move in a camera-proximity-like
    direction. It must never be interpreted as metric displacement or distance.
    """
    if pose.empty:
        return pd.DataFrame(), {"status": "not_estimable", "reason": "pose_source_empty"}
    required = {"unix_ms", "landmark_name", "x", "y"}
    missing = sorted(required - set(pose.columns))
    if missing:
        return pd.DataFrame(), {"status": "not_estimable", "reason": "pose_columns_missing:" + ",".join(missing)}

    group_key = "video_frame_position" if "video_frame_position" in pose.columns else "unix_ms"
    rows: list[dict[str, object]] = []
    prev_time: float | None = None
    prev_center: tuple[float, float] | None = None
    prev_width: float | None = None
    prev_world_z: float | None = None
    prev_bbox_area: float | None = None
    prev_valid = False

    for _, frame_all in pose.groupby(group_key, sort=True):
        first_all = frame_all.iloc[0]
        unix_ms = _numeric(first_all.get("unix_ms"))
        if unix_ms is None:
            continue
        pose_count = int(_numeric(first_all.get("pose_count")) or (1 if bool(first_all.get("pose_valid", True)) else 0))
        ambiguous = pose_count > 1
        if "pose_index" in frame_all.columns:
            frame = frame_all[pd.to_numeric(frame_all["pose_index"], errors="coerce").eq(0)].copy()
        else:
            frame = frame_all.copy()
        if "pose_valid" in frame.columns:
            frame = frame[frame["pose_valid"].fillna(False).astype(bool)]
        if ambiguous:
            frame = frame.iloc[0:0]

        left = _frame_landmark(frame, "left_shoulder")
        right = _frame_landmark(frame, "right_shoulder")
        shoulder_quality = _quality_ok(left, min_visibility=min_visibility, min_presence=min_presence) and _quality_ok(
            right, min_visibility=min_visibility, min_presence=min_presence
        )
        if shoulder_quality:
            lx, ly = _numeric(left.get("x")), _numeric(left.get("y"))  # type: ignore[union-attr]
            rx, ry = _numeric(right.get("x")), _numeric(right.get("y"))  # type: ignore[union-attr]
            assert lx is not None and ly is not None and rx is not None and ry is not None
            center = ((lx + rx) / 2.0, (ly + ry) / 2.0)
            width = float(np.hypot(rx - lx, ry - ly))
            wz_values = [_numeric(left.get("world_z")), _numeric(right.get("world_z"))]  # type: ignore[union-attr]
            wz_finite = [v for v in wz_values if v is not None]
            world_z = float(np.mean(wz_finite)) if wz_finite else None
            bbox_area = _bbox_area(first_all)
        else:
            center = None
            width = None
            world_z = None
            bbox_area = None

        dt_ms = unix_ms - prev_time if prev_time is not None else None
        gap_break = prev_time is None or dt_ms is None or dt_ms <= 0 or dt_ms > gap_reset_ms
        quality_break = not shoulder_quality or not prev_valid
        diff_reset = gap_break or quality_break
        dt_sec = (dt_ms / 1000.0) if dt_ms is not None and dt_ms > 0 else None

        lateral = vertical = np.nan
        world_z_rate = width_rate = bbox_rate = np.nan
        radial = np.nan
        radial_components = 0
        if not diff_reset and center is not None and prev_center is not None and dt_sec:
            lateral = (center[0] - prev_center[0]) / dt_sec
            vertical = -(center[1] - prev_center[1]) / dt_sec

            if world_z is not None and prev_world_z is not None:
                world_z_rate = -(world_z - prev_world_z) / dt_sec
                radial_components += 1
            width_delta = _safe_log_ratio(width, prev_width)
            if width_delta is not None:
                width_rate = width_delta / dt_sec
                radial_components += 1
            bbox_delta = _safe_log_ratio(bbox_area, prev_bbox_area)
            if bbox_delta is not None:
                bbox_rate = bbox_delta / dt_sec
                radial_components += 1
            component_values = [v for v in (world_z_rate, width_rate, bbox_rate) if np.isfinite(v)]
            if component_values:
                radial = float(np.mean(np.sign(component_values)))

        reason = ""
        if ambiguous:
            reason = "multi_pose_ambiguous"
        elif not shoulder_quality:
            reason = "shoulder_visibility_or_presence_below_gate"
        elif prev_time is None:
            reason = "analysis_start"
        elif dt_ms is not None and dt_ms <= 0:
            reason = "nonpositive_dt"
        elif dt_ms is not None and dt_ms > gap_reset_ms:
            reason = "pose_timestamp_gap"
        elif not prev_valid:
            reason = "previous_pose_not_observable"

        row: dict[str, object] = {
            "subject": first_all.get("subject"),
            "unix_ms": unix_ms,
            "video_frame_position": first_all.get("video_frame_position"),
            "capture_frame_idx": first_all.get("capture_frame_idx"),
            "phase": first_all.get("phase"),
            "block": first_all.get("block"),
            "trial_num": first_all.get("trial_num"),
            "pose_count": pose_count,
            "primary_pose_ambiguous": ambiguous,
            "pose_shoulders_observable": shoulder_quality,
            "pose_diff_reset": bool(diff_reset),
            "pose_diff_reset_reason": reason,
            "pose_dt_ms": dt_ms,
            "shoulder_center_x": center[0] if center else np.nan,
            "shoulder_center_y": center[1] if center else np.nan,
            "shoulder_width_norm": width,
            "pose_bbox_area_norm": bbox_area,
            "shoulder_world_z_mean": world_z,
            "pose_lateral_right_per_sec": lateral,
            "pose_vertical_up_per_sec": vertical,
            "radial_world_z_proximity_rate": world_z_rate,
            "radial_shoulder_width_log_rate": width_rate,
            "radial_bbox_area_log_rate": bbox_rate,
            "pose_radial_proximity_candidate_per_sec": radial,
            "pose_radial_component_n": radial_components,
            "pose_direction_interpretation": "auxiliary_qc_candidate_not_physical_displacement",
        }
        rows.append(row)

        prev_time = unix_ms
        if shoulder_quality and not ambiguous:
            prev_center = center
            prev_width = width
            prev_world_z = world_z
            prev_bbox_area = bbox_area
            prev_valid = True
        else:
            prev_center = None
            prev_width = None
            prev_world_z = None
            prev_bbox_area = None
            prev_valid = False

    out = pd.DataFrame(rows)
    valid_direction_rows = int(out["pose_lateral_right_per_sec"].notna().sum()) if not out.empty else 0
    radial_rows = int(out["pose_radial_proximity_candidate_per_sec"].notna().sum()) if not out.empty else 0
    return out, {
        "status": "generated" if not out.empty else "not_estimable",
        "reason": "" if not out.empty else "no_pose_frames_after_projection",
        "direction_valid_rows": valid_direction_rows,
        "radial_candidate_valid_rows": radial_rows,
        "gap_or_quality_reset_rows": int(out["pose_diff_reset"].sum()) if not out.empty else 0,
        "radial_interpretation": "auxiliary_qc_candidate_not_physical_displacement",
    }
