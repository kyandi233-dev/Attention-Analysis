from __future__ import annotations

import math

import numpy as np
import pandas as pd

from attention_pipeline.config import Config
from attention_pipeline.rgb.paths import RGBOutputLayout


POSE_FEATURE_SCHEMA_VERSION = "rgb-pose-features-v0.2"
KEY_LANDMARKS = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
]
MOTION_LANDMARKS = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]


def _dist(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float | None:
    if a is None or b is None:
        return None
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _midpoint(a: tuple[float, float] | None, b: tuple[float, float] | None) -> tuple[float, float] | None:
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _landmark_measurement(
    frame: pd.DataFrame,
    name: str,
    *,
    min_visibility: float,
    min_presence: float,
    require_in_frame: bool,
) -> tuple[tuple[float, float] | None, float | None, float | None, bool]:
    row = frame[frame["landmark_name"] == name]
    if row.empty:
        return None, None, None, False

    x = pd.to_numeric(row["x"], errors="coerce").iloc[0]
    y = pd.to_numeric(row["y"], errors="coerce").iloc[0]
    visibility = pd.to_numeric(row["visibility"], errors="coerce").iloc[0] if "visibility" in row else np.nan
    presence = pd.to_numeric(row["presence"], errors="coerce").iloc[0] if "presence" in row else np.nan

    vis = None if pd.isna(visibility) else float(visibility)
    pres = None if pd.isna(presence) else float(presence)
    coordinates_ok = not pd.isna(x) and not pd.isna(y) and np.isfinite(x) and np.isfinite(y)
    in_frame = bool(coordinates_ok and 0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0)
    quality_ok = (
        coordinates_ok
        and vis is not None and vis >= min_visibility
        and pres is not None and pres >= min_presence
        and (in_frame or not require_in_frame)
    )
    point = (float(x), float(y)) if quality_ok else None
    return point, vis, pres, quality_ok


def _safe_rate(distance: float | None, scale: float | None, dt_ms: float | None) -> float | None:
    if distance is None or scale is None or scale <= 0 or dt_ms is None or dt_ms <= 0:
        return None
    return float((distance / scale) / (dt_ms / 1000.0))


def derive_pose_features(
    table: pd.DataFrame,
    *,
    subject: str,
    min_visibility: float = 0.5,
    min_presence: float = 0.5,
    require_in_frame: bool = True,
    gap_reset_ms: float = 300.0,
) -> pd.DataFrame:
    if table.empty:
        raise ValueError("Pose parquet is empty")
    required = {
        "video_frame_position", "capture_frame_idx", "unix_ms", "phase", "block",
        "pose_valid", "pose_count", "pose_index", "landmark_name", "x", "y",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Pose parquet missing required columns: {missing}")

    results: list[dict[str, object]] = []
    previous_points: dict[str, tuple[float, float] | None] = {}
    previous_unix_ms: int | None = None
    previous_scale: float | None = None

    grouped = table.groupby("video_frame_position", sort=True)
    for video_frame_position, frame_all in grouped:
        first = frame_all.iloc[0]
        pose_count = int(pd.to_numeric(pd.Series([first.get("pose_count")]), errors="coerce").fillna(0).iloc[0])
        pose_valid = bool(first.get("pose_valid", False))
        ambiguous = pose_count > 1

        frame = frame_all[(frame_all["pose_valid"].fillna(False).astype(bool)) & (frame_all["pose_index"] == 0)].copy()
        if ambiguous or not pose_valid or frame.empty:
            frame = frame.iloc[0:0]

        unix_ms = int(first["unix_ms"])
        dt_ms = float(unix_ms - previous_unix_ms) if previous_unix_ms is not None else None
        capture_gap_before = bool(first.get("capture_gap_before", False))
        timestamp_gap_before = bool(first.get("timestamp_gap_before", False))
        gap_before = (
            previous_unix_ms is None or dt_ms is None or dt_ms <= 0
            or dt_ms > gap_reset_ms or capture_gap_before or timestamp_gap_before
        )
        if previous_unix_ms is None:
            gap_reason = "analysis_start"
        elif dt_ms is not None and dt_ms <= 0:
            gap_reason = "nonpositive_dt"
        elif dt_ms is not None and dt_ms > gap_reset_ms:
            gap_reason = "pose_timestamp_gap"
        elif capture_gap_before:
            gap_reason = "pose_capture_gap"
        elif timestamp_gap_before:
            gap_reason = "pose_timestamp_gap"
        else:
            gap_reason = ""

        measurements = {
            name: _landmark_measurement(
                frame,
                name,
                min_visibility=min_visibility,
                min_presence=min_presence,
                require_in_frame=require_in_frame,
            )
            for name in KEY_LANDMARKS
        }
        points = {name: measurements[name][0] for name in KEY_LANDMARKS}
        qualities = {name: (measurements[name][1], measurements[name][2]) for name in KEY_LANDMARKS}
        quality_valid = {name: measurements[name][3] for name in KEY_LANDMARKS}

        shoulder_width = _dist(points["left_shoulder"], points["right_shoulder"])
        shoulder_center = _midpoint(points["left_shoulder"], points["right_shoulder"])
        hip_center = _midpoint(points["left_hip"], points["right_hip"])
        scale = shoulder_width if shoulder_width is not None and shoulder_width > 0 else None

        if gap_before:
            rate_scale = None
        elif scale is not None and previous_scale is not None:
            rate_scale = (scale + previous_scale) / 2.0
        else:
            rate_scale = None

        rates: dict[str, float | None] = {}
        for name in MOTION_LANDMARKS:
            distance = None if gap_before else _dist(points[name], previous_points.get(name))
            rates[name] = _safe_rate(distance, rate_scale, dt_ms)

        def mean_available(names: list[str]) -> float | None:
            values = [rates[name] for name in names if rates.get(name) is not None]
            return float(np.mean(values)) if values else None

        available_motion_count = sum(rates[name] is not None for name in MOTION_LANDMARKS)
        arm_names = ["left_elbow", "right_elbow", "left_wrist", "right_wrist"]
        available_arm_count = sum(rates[name] is not None for name in arm_names)

        shoulder_line_angle = None
        if points["left_shoulder"] is not None and points["right_shoulder"] is not None:
            left = points["left_shoulder"]
            right = points["right_shoulder"]
            shoulder_line_angle = float(math.degrees(math.atan2(right[1] - left[1], right[0] - left[0])))

        trunk_angle = None
        if shoulder_center is not None and hip_center is not None:
            dx = shoulder_center[0] - hip_center[0]
            dy = shoulder_center[1] - hip_center[1]
            if dx != 0 or dy != 0:
                trunk_angle = float(math.degrees(math.atan2(dx, -dy)))

        vis_values = [qualities[name][0] for name in MOTION_LANDMARKS if qualities[name][0] is not None]
        pres_values = [qualities[name][1] for name in MOTION_LANDMARKS if qualities[name][1] is not None]

        row: dict[str, object] = {
            "schema_version": POSE_FEATURE_SCHEMA_VERSION,
            "subject": subject,
            "video_frame_position": int(video_frame_position),
            "capture_frame_idx": int(first["capture_frame_idx"]),
            "unix_ms": unix_ms,
            "dt_ms": dt_ms,
            "gap_before": gap_before,
            "gap_reason": gap_reason,
            "phase": first.get("phase"),
            "block": first.get("block"),
            "trial_num": first.get("trial_num"),
            "behavior_state": first.get("behavior_state"),
            "pose_valid": pose_valid,
            "pose_count": pose_count,
            "primary_pose_ambiguous": ambiguous,
            "feature_min_visibility": min_visibility,
            "feature_min_presence": min_presence,
            "feature_require_in_frame": require_in_frame,
            "shoulder_width_norm": shoulder_width,
            "shoulder_center_x": shoulder_center[0] if shoulder_center else None,
            "shoulder_center_y": shoulder_center[1] if shoulder_center else None,
            "shoulder_line_angle_deg": shoulder_line_angle,
            "hip_center_x": hip_center[0] if hip_center else None,
            "hip_center_y": hip_center[1] if hip_center else None,
            "trunk_angle_deg_from_vertical": trunk_angle,
            "left_wrist_quality_valid": quality_valid["left_wrist"],
            "right_wrist_quality_valid": quality_valid["right_wrist"],
            "left_elbow_quality_valid": quality_valid["left_elbow"],
            "right_elbow_quality_valid": quality_valid["right_elbow"],
            "left_shoulder_quality_valid": quality_valid["left_shoulder"],
            "right_shoulder_quality_valid": quality_valid["right_shoulder"],
            "left_hip_quality_valid": quality_valid["left_hip"],
            "right_hip_quality_valid": quality_valid["right_hip"],
            "left_wrist_motion_swidth_per_sec": rates["left_wrist"],
            "right_wrist_motion_swidth_per_sec": rates["right_wrist"],
            "wrist_motion_swidth_per_sec": mean_available(["left_wrist", "right_wrist"]),
            "left_elbow_motion_swidth_per_sec": rates["left_elbow"],
            "right_elbow_motion_swidth_per_sec": rates["right_elbow"],
            "elbow_motion_swidth_per_sec": mean_available(["left_elbow", "right_elbow"]),
            "arm_motion_swidth_per_sec": mean_available(arm_names),
            "arm_motion_landmark_count": available_arm_count,
            "left_shoulder_motion_swidth_per_sec": rates["left_shoulder"],
            "right_shoulder_motion_swidth_per_sec": rates["right_shoulder"],
            "shoulder_motion_swidth_per_sec": mean_available(["left_shoulder", "right_shoulder"]),
            "upper_body_motion_swidth_per_sec": mean_available(MOTION_LANDMARKS),
            "upper_body_motion_landmark_count": available_motion_count,
            "upper_body_min_visibility": min(vis_values) if vis_values else None,
            "upper_body_mean_visibility": float(np.mean(vis_values)) if vis_values else None,
            "upper_body_min_presence": min(pres_values) if pres_values else None,
            "upper_body_mean_presence": float(np.mean(pres_values)) if pres_values else None,
        }
        results.append(row)

        previous_points = points
        previous_unix_ms = unix_ms
        previous_scale = scale

    return pd.DataFrame(results)


def run_pose_features(config: Config, subject: str) -> dict[str, object]:
    layout = RGBOutputLayout.from_config(config)
    source = layout.test_file(f"{subject}_pose-test.parquet")
    if not source.exists():
        raise FileNotFoundError(f"Pose pilot output not found: {source}")
    table = pd.read_parquet(source)
    pose_cfg = config.section("pose")
    min_visibility = float(pose_cfg.get("feature_min_visibility", 0.5))
    min_presence = float(pose_cfg.get("feature_min_presence", 0.5))
    require_in_frame = bool(pose_cfg.get("feature_require_in_frame", True))
    gap_reset_ms = float(pose_cfg.get("feature_gap_reset_ms", 300.0))
    features = derive_pose_features(
        table,
        subject=subject,
        min_visibility=min_visibility,
        min_presence=min_presence,
        require_in_frame=require_in_frame,
        gap_reset_ms=gap_reset_ms,
    )
    output = layout.test_file(f"{subject}_pose-features-test.parquet")
    features.to_parquet(output, index=False, engine="pyarrow", compression="zstd")
    return {
        "schema_version": POSE_FEATURE_SCHEMA_VERSION,
        "subject": subject,
        "source_pose_parquet": str(source),
        "output": str(output),
        "rows": int(len(features)),
        "ambiguous_multi_pose_rows": int(features["primary_pose_ambiguous"].fillna(False).astype(bool).sum()),
        "gap_reset_rows": int(features["gap_before"].fillna(False).astype(bool).sum()) - 1,
        "shoulder_motion_valid_rows": int(features["shoulder_motion_swidth_per_sec"].notna().sum()),
        "elbow_motion_valid_rows": int(features["elbow_motion_swidth_per_sec"].notna().sum()),
        "wrist_motion_valid_rows": int(features["wrist_motion_swidth_per_sec"].notna().sum()),
        "upper_body_motion_valid_rows": int(features["upper_body_motion_swidth_per_sec"].notna().sum()),
        "trunk_angle_valid_rows": int(features["trunk_angle_deg_from_vertical"].notna().sum()),
        "parquet_size_bytes": int(output.stat().st_size),
        "quality_gate": {
            "min_visibility": min_visibility,
            "min_presence": min_presence,
            "require_in_frame": require_in_frame,
            "gap_reset_ms": gap_reset_ms,
        },
        "unit_note": "motion rates are quality-gated normalized-coordinate displacement / shoulder width / second",
    }
