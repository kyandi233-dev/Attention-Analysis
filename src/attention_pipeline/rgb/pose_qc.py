from __future__ import annotations

import json

import numpy as np
import pandas as pd

from attention_pipeline.config import Config
from attention_pipeline.rgb.paths import RGBOutputLayout


POSE_QC_SCHEMA_VERSION = "rgb-pose-qc-v0.1"
DEFAULT_UPPER_BODY = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]
DEFAULT_OPTIONAL_TRUNK = ["left_hip", "right_hip"]


def _num(value: object) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _summary(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    values = values[np.isfinite(values)]
    if values.empty:
        return {"count": 0, "p10": None, "p50": None, "p90": None, "mean": None}
    q = values.quantile([0.10, 0.50, 0.90])
    return {
        "count": int(len(values)),
        "p10": _num(q.loc[0.10]),
        "p50": _num(q.loc[0.50]),
        "p90": _num(q.loc[0.90]),
        "mean": _num(values.mean()),
    }


def _landmark_qc(rows: pd.DataFrame, sampled_frames: int) -> dict[str, object]:
    vis = pd.to_numeric(rows.get("visibility"), errors="coerce")
    pres = pd.to_numeric(rows.get("presence"), errors="coerce")
    x = pd.to_numeric(rows.get("x"), errors="coerce")
    y = pd.to_numeric(rows.get("y"), errors="coerce")
    world_x = pd.to_numeric(rows.get("world_x"), errors="coerce")
    frame_count = int(rows["video_frame_position"].nunique()) if not rows.empty else 0
    denominator = sampled_frames if sampled_frames > 0 else 1
    in_frame = x.notna() & y.notna() & x.between(0.0, 1.0) & y.between(0.0, 1.0)
    return {
        "frames": frame_count,
        "frame_coverage": frame_count / denominator,
        "visibility": _summary(vis),
        "presence": _summary(pres),
        "visibility_ge_0_5_fraction": float((vis >= 0.5).mean()) if vis.notna().any() else None,
        "visibility_ge_0_8_fraction": float((vis >= 0.8).mean()) if vis.notna().any() else None,
        "presence_ge_0_5_fraction": float((pres >= 0.5).mean()) if pres.notna().any() else None,
        "presence_ge_0_8_fraction": float((pres >= 0.8).mean()) if pres.notna().any() else None,
        "in_frame_fraction": float(in_frame.mean()) if len(in_frame) else None,
        "world_coordinate_fraction": float(world_x.notna().mean()) if len(world_x) else None,
    }


def summarize_pose_table(
    table: pd.DataFrame,
    *,
    subject: str,
    upper_body_landmarks: list[str] | None = None,
    optional_trunk_landmarks: list[str] | None = None,
) -> dict[str, object]:
    if table.empty:
        raise ValueError("Pose parquet is empty")
    required = {"video_frame_position", "pose_valid", "pose_count", "pose_index", "landmark_name"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Pose parquet missing required columns: {missing}")

    upper = upper_body_landmarks or DEFAULT_UPPER_BODY
    optional = optional_trunk_landmarks or DEFAULT_OPTIONAL_TRUNK
    frame_table = table.drop_duplicates("video_frame_position")
    sampled_frames = int(frame_table["video_frame_position"].nunique())
    valid_frames = int(frame_table["pose_valid"].fillna(False).astype(bool).sum())
    multi_frames = int((pd.to_numeric(frame_table["pose_count"], errors="coerce") > 1).sum())

    valid_landmarks = table[table["pose_valid"].fillna(False).astype(bool)].copy()
    by_landmark: dict[str, object] = {}
    for name in [*upper, *optional]:
        rows = valid_landmarks[valid_landmarks["landmark_name"] == name]
        by_landmark[name] = _landmark_qc(rows, sampled_frames)

    upper_rows = valid_landmarks[valid_landmarks["landmark_name"].isin(upper)]
    optional_rows = valid_landmarks[valid_landmarks["landmark_name"].isin(optional)]
    phase_frames = {
        str(k): int(v)
        for k, v in frame_table["phase"].value_counts(dropna=False).to_dict().items()
    } if "phase" in frame_table else {}

    return {
        "schema_version": POSE_QC_SCHEMA_VERSION,
        "subject": subject,
        "sampled_frames": sampled_frames,
        "frames_with_pose": valid_frames,
        "pose_valid_fraction": valid_frames / sampled_frames if sampled_frames else None,
        "frames_with_multiple_poses": multi_frames,
        "phase_sampled_frames": phase_frames,
        "upper_body_landmarks": upper,
        "optional_trunk_landmarks": optional,
        "upper_body_group": {
            "mean_visibility": _num(pd.to_numeric(upper_rows.get("visibility"), errors="coerce").mean()),
            "mean_presence": _num(pd.to_numeric(upper_rows.get("presence"), errors="coerce").mean()),
        },
        "optional_trunk_group": {
            "mean_visibility": _num(pd.to_numeric(optional_rows.get("visibility"), errors="coerce").mean()),
            "mean_presence": _num(pd.to_numeric(optional_rows.get("presence"), errors="coerce").mean()),
        },
        "landmarks": by_landmark,
        "interpretation_rule": (
            "Do not judge upper-body quality from the 33-landmark grand mean. "
            "Inspect shoulders/elbows/wrists separately; hips are optional because the camera may crop the lower torso."
        ),
    }


def run_pose_qc(config: Config, subject: str) -> dict[str, object]:
    layout = RGBOutputLayout.from_config(config)
    source = layout.test_file(f"{subject}_pose-test.parquet")
    if not source.exists():
        raise FileNotFoundError(f"Pose pilot output not found: {source}")
    table = pd.read_parquet(source)
    pose_cfg = config.section("pose")
    upper = [str(v) for v in pose_cfg.get("upper_body_landmarks", DEFAULT_UPPER_BODY)]
    optional = [str(v) for v in pose_cfg.get("optional_trunk_landmarks", DEFAULT_OPTIONAL_TRUNK)]
    summary = summarize_pose_table(
        table,
        subject=subject,
        upper_body_landmarks=upper,
        optional_trunk_landmarks=optional,
    )
    summary["source_parquet"] = str(source)
    output = layout.test_file(f"{subject}_pose-qc.json")
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["output"] = str(output)
    return summary
