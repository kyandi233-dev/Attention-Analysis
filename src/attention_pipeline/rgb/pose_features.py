from __future__ import annotations

import math

import numpy as np
import pandas as pd

from attention_pipeline.config import Config
from attention_pipeline.rgb.paths import RGBOutputLayout


POSE_FEATURE_SCHEMA_VERSION = "rgb-pose-features-v0.1"
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


def _dist(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float | None:
    if a is None or b is None:
        return None
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _midpoint(a: tuple[float, float] | None, b: tuple[float, float] | None) -> tuple[float, float] | None:
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _point(frame: pd.DataFrame, name: str) -> tuple[float, float] | None:
    row = frame[frame["landmark_name"] == name]
    if row.empty:
        return None
    x = pd.to_numeric(row["x"], errors="coerce").iloc[0]
    y = pd.to_numeric(row["y"], errors="coerce").iloc[0]
    if pd.isna(x) or pd.isna(y):
        return None
    return float(x), float(y)


def _quality(frame: pd.DataFrame, name: str) -> tuple[float | None, float | None]:
    row = frame[frame["landmark_name"] == name]
    if row.empty:
        return None, None
    visibility = pd.to_numeric(row["visibility"], errors="coerce").iloc[0] if "visibility" in row else np.nan
    presence = pd.to_numeric(row["presence"], errors="coerce").iloc[0] if "presence" in row else np.nan
    return (
        None if pd.isna(visibility) else float(visibility),
        None if pd.isna(presence) else float(presence),
    )


def _safe_rate(distance: float | None, scale: float | None, dt_ms: float | None) -> float | None:
    if distance is None or scale is None or scale <= 0 or dt_ms is None or dt_ms <= 0:
        return None
    return float((distance / scale) / (dt_ms / 1000.0))


def derive_pose_features(table: pd.DataFrame, *, subject: str) -> pd.DataFrame:
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
        points = {name: _point(frame, name) for name in KEY_LANDMARKS}
        qualities = {name: _quality(frame, name) for name in KEY_LANDMARKS}

        shoulder_width = _dist(points["left_shoulder"], points["right_shoulder"])
        shoulder_center = _midpoint(points["left_shoulder"], points["right_shoulder"])
        hip_center = _midpoint(points["left_hip"], points["right_hip"])
        scale = shoulder_width if shoulder_width is not None and shoulder_width > 0 else None
        rate_scale = None
        if scale is not None and previous_scale is not None:
            rate_scale = (scale + previous_scale) / 2.0
        elif scale is not None:
            rate_scale = scale
        elif previous_scale is not None:
            rate_scale = previous_scale

        rates: dict[str, float | None] = {}
        for name in [
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
        ]:
            rates[name] = _safe_rate(
                _dist(points[name], previous_points.get(name)),
                rate_scale,
                dt_ms,
            )

        def mean_available(names: list[str]) -> float | None:
            values = [rates[name] for name in names if rates.get(name) is not None]
            return float(np.mean(values)) if values else None

        trunk_angle = None
        if shoulder_center is not None and hip_center is not None:
            dx = shoulder_center[0] - hip_center[0]
            dy = shoulder_center[1] - hip_center[1]
            if dx != 0 or dy != 0:
                trunk_angle = float(math.degrees(math.atan2(dx, -dy)))

        core_names = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]
        vis_values = [qualities[name][0] for name in core_names if qualities[name][0] is not None]
        pres_values = [qualities[name][1] for name in core_names if qualities[name][1] is not None]

        row: dict[str, object] = {
            "schema_version": POSE_FEATURE_SCHEMA_VERSION,
            "subject": subject,
            "video_frame_position": int(video_frame_position),
            "capture_frame_idx": int(first["capture_frame_idx"]),
            "unix_ms": unix_ms,
            "dt_ms": dt_ms,
            "phase": first.get("phase"),
            "block": first.get("block"),
            "trial_num": first.get("trial_num"),
            "behavior_state": first.get("behavior_state"),
            "pose_valid": pose_valid,
            "pose_count": pose_count,
            "primary_pose_ambiguous": ambiguous,
            "shoulder_width_norm": shoulder_width,
            "shoulder_center_x": shoulder_center[0] if shoulder_center else None,
            "shoulder_center_y": shoulder_center[1] if shoulder_center else None,
            "hip_center_x": hip_center[0] if hip_center else None,
            "hip_center_y": hip_center[1] if hip_center else None,
            "trunk_angle_deg_from_vertical": trunk_angle,
            "left_wrist_motion_swidth_per_sec": rates["left_wrist"],
            "right_wrist_motion_swidth_per_sec": rates["right_wrist"],
            "wrist_motion_swidth_per_sec": mean_available(["left_wrist", "right_wrist"]),
            "left_elbow_motion_swidth_per_sec": rates["left_elbow"],
            "right_elbow_motion_swidth_per_sec": rates["right_elbow"],
            "elbow_motion_swidth_per_sec": mean_available(["left_elbow", "right_elbow"]),
            "left_shoulder_motion_swidth_per_sec": rates["left_shoulder"],
            "right_shoulder_motion_swidth_per_sec": rates["right_shoulder"],
            "shoulder_motion_swidth_per_sec": mean_available(["left_shoulder", "right_shoulder"]),
            "upper_body_motion_swidth_per_sec": mean_available(core_names),
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
    features = derive_pose_features(table, subject=subject)
    output = layout.test_file(f"{subject}_pose-features-test.parquet")
    features.to_parquet(output, index=False, engine="pyarrow", compression="zstd")
    return {
        "schema_version": POSE_FEATURE_SCHEMA_VERSION,
        "subject": subject,
        "source_pose_parquet": str(source),
        "output": str(output),
        "rows": int(len(features)),
        "ambiguous_multi_pose_rows": int(features["primary_pose_ambiguous"].fillna(False).astype(bool).sum()),
        "upper_body_motion_valid_rows": int(features["upper_body_motion_swidth_per_sec"].notna().sum()),
        "trunk_angle_valid_rows": int(features["trunk_angle_deg_from_vertical"].notna().sum()),
        "parquet_size_bytes": int(output.stat().st_size),
        "unit_note": "motion rates are normalized-coordinate displacement / shoulder width / second",
    }
