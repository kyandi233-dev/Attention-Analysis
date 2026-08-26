from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from attention_pipeline.config import Config
from attention_pipeline.rgb.discover import RGBSubjectFiles, discover_rgb_subjects
from attention_pipeline.rgb.paths import RGBOutputLayout


MOTION_REVIEW_SCHEMA_VERSION = "rgb-motion-review-v0.1"


def _find_subject(config: Config, subject: str) -> RGBSubjectFiles:
    records, duplicates = discover_rgb_subjects(config)
    if subject in duplicates:
        raise RuntimeError(f"Subject {subject} is duplicated across data roots: {duplicates[subject]}")
    for record in records:
        if record.subject == subject:
            return record
    raise FileNotFoundError(f"RGB subject not discovered: {subject}")


def _nearest_index(series: pd.Series, target: float) -> int:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        raise ValueError("No valid values available for representative selection")
    return int((valid - float(target)).abs().idxmin())


def select_motion_review_rows(table: pd.DataFrame) -> list[dict[str, object]]:
    """Choose a small, deduplicated set of rows for manual video spot-checking.

    Selection deliberately covers both extreme and ordinary Motion values. It is a
    development aid only and never filters the raw Motion table.
    """
    required = {
        "video_frame_position",
        "motion_valid",
        "global_motion_energy",
        "gray_mean_delta",
        "dt_ms",
        "phase",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Motion parquet missing required columns: {missing}")

    valid = table.loc[table["motion_valid"].fillna(False).astype(bool)].copy()
    if valid.empty:
        raise ValueError("Motion parquet has no valid motion rows")

    valid["_motion"] = pd.to_numeric(valid["global_motion_energy"], errors="coerce")
    valid["_abs_brightness"] = pd.to_numeric(valid["gray_mean_delta"], errors="coerce").abs()
    valid = valid.dropna(subset=["_motion", "_abs_brightness"])
    if valid.empty:
        raise ValueError("Motion parquet has no finite review values")

    chosen: list[tuple[str, int]] = []
    used_positions: set[int] = set()

    def add(label: str, index: int) -> None:
        position = int(valid.loc[index, "video_frame_position"])
        if position in used_positions:
            return
        chosen.append((label, index))
        used_positions.add(position)

    for index in valid.nlargest(4, "_motion").index:
        add("highest_motion", int(index))

    for index in valid.nlargest(4, "_abs_brightness").index:
        add("largest_brightness_change", int(index))

    for quantile, label in ((0.50, "motion_p50"), (0.90, "motion_p90"), (0.99, "motion_p99")):
        target = float(valid["_motion"].quantile(quantile))
        add(label, _nearest_index(valid["_motion"], target))

    columns = [
        "video_frame_position",
        "capture_frame_idx",
        "unix_ms",
        "dt_ms",
        "phase",
        "block",
        "trial_num",
        "behavior_state",
        "global_motion_energy",
        "global_motion_energy_per_sec",
        "changed_pixel_ratio",
        "gray_mean_delta",
        "gray_mean",
        "irregular_dt",
    ]
    available = [column for column in columns if column in valid.columns]
    output: list[dict[str, object]] = []
    for label, index in chosen:
        record = valid.loc[index, available].to_dict()
        clean: dict[str, object] = {"review_category": label}
        for key, value in record.items():
            if isinstance(value, (np.integer,)):
                clean[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                clean[key] = None if pd.isna(value) or not np.isfinite(float(value)) else float(value)
            elif pd.isna(value):
                clean[key] = None
            else:
                clean[key] = value
        output.append(clean)
    return output


def _read_frame_pair(video: Path, position: int) -> tuple[np.ndarray, np.ndarray]:
    if position <= 0:
        raise ValueError(f"Cannot review previous frame for position {position}")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open RGB video: {video}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(position - 1))
        ok_prev, previous = cap.read()
        ok_curr, current = cap.read()
        if not ok_prev or previous is None or not ok_curr or current is None:
            raise RuntimeError(f"Cannot read frame pair around position {position}: {video}")
        return previous, current
    finally:
        cap.release()


def _fit_frame(frame: np.ndarray, width: int = 480) -> np.ndarray:
    height = max(1, int(round(frame.shape[0] * width / frame.shape[1])))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _format_label(record: dict[str, object]) -> str:
    motion = record.get("global_motion_energy")
    brightness = record.get("gray_mean_delta")
    dt = record.get("dt_ms")
    phase = record.get("phase")
    position = record.get("video_frame_position")
    return (
        f"{record.get('review_category')} | pos={position} | {phase} | "
        f"dt={dt} ms | motion={float(motion):.6f} | dGray={float(brightness):+.3f}"
    )


def _contact_sheet(video: Path, records: list[dict[str, object]]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for record in records:
        position = int(record["video_frame_position"])
        previous, current = _read_frame_pair(video, position)
        previous = _fit_frame(previous)
        current = _fit_frame(current)
        pair = np.hstack([previous, current])
        header = np.full((52, pair.shape[1], 3), 245, dtype=np.uint8)
        cv2.putText(
            header,
            _format_label(record),
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            header,
            "left=previous frame | right=current frame",
            (10, 43),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        rows.append(np.vstack([header, pair]))
    if not rows:
        raise ValueError("No representative Motion rows selected")
    return np.vstack(rows)


def run_motion_review(config: Config, subject: str) -> dict[str, object]:
    """Create one compact contact sheet for manual Motion/brightness spot-checking."""
    layout = RGBOutputLayout.from_config(config)
    source_parquet = layout.test_file(f"{subject}_motion-test.parquet")
    if not source_parquet.exists():
        raise FileNotFoundError(f"Motion pilot output not found: {source_parquet}")

    files = _find_subject(config, subject)
    table = pd.read_parquet(source_parquet)
    records = select_motion_review_rows(table)
    sheet = _contact_sheet(files.video, records)

    image_path = layout.test_file(f"{subject}_motion-review.png")
    if not cv2.imwrite(str(image_path), sheet):
        raise RuntimeError(f"Failed to write Motion review image: {image_path}")

    manifest = {
        "schema_version": MOTION_REVIEW_SCHEMA_VERSION,
        "subject": subject,
        "purpose": "manual spot-check only; does not filter or alter Motion raw output",
        "source_video": str(files.video),
        "source_parquet": str(source_parquet),
        "contact_sheet": str(image_path),
        "left_panel": "previous AVI frame",
        "right_panel": "current AVI frame",
        "selected_events": records,
    }
    json_path = layout.test_file(f"{subject}_motion-review.json")
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest | {"output_json": str(json_path)}
