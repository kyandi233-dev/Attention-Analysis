"""Formal production-evidence input layer for the seven-algorithm benchmark.

This module is deliberately separate from the detector adapters.  It owns the
scientific input contract:

``complete production run -> eyes.csv evidence -> deterministic sample plan ->
source-video frame -> 1:1 source-pixel crop -> detector rows -> agreement-only
RITnet comparison``.

It never reruns YOLO or RITnet and never treats RITnet as ground truth.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .core import normalize_phase, normalize_subject, safe_ratio
from .runner import assemble_row, run_crop_list
from .schema import ALGORITHM_SPECS, RESULT_COLUMNS


EYES_REQUIRED_COLUMNS: tuple[str, ...] = (
    "subject", "video", "phase", "phase_segment", "frame_idx", "eye",
    "frame_status", "status", "anchor_yolo_confidence",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "roi_x1", "roi_y1", "roi_x2", "roi_y2", "roi_clipped",
    "ritnet_found", "pupil_center_x", "pupil_center_y",
    "pupil_axis_a", "pupil_axis_b", "pupil_angle_deg", "pupil_confidence",
)

SAMPLE_COLUMNS: tuple[str, ...] = (
    "subject", "production_run", "production_run_id", "source_video",
    "phase", "phase_segment", "frame_idx", "eye", "sample_role",
    "sequence_id", "input_kind", "input_status", "crop_path",
    "bbox_float_x1", "bbox_float_y1", "bbox_float_x2", "bbox_float_y2",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "input_width", "input_height",
    "ritnet_found", "ritnet_source_center_x", "ritnet_source_center_y",
    "ritnet_source_major_axis", "ritnet_source_minor_axis",
    "ritnet_source_angle_deg", "ritnet_source_diameter_geom",
    "ritnet_pupil_confidence", "sampling_evidence",
)


@dataclass(frozen=True)
class ProductionRun:
    subject: str
    path: Path
    completion: dict[str, Any]
    eyes_csv: Path
    run_manifest: Path
    source_video: Path

    @property
    def run_id(self) -> str:
        return str(self.completion.get("run_id", ""))


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"benchmark config must be a mapping: {path}")
    return config


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def atomic_write_json(value: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    os.replace(temporary, path)


def discover_production_run(
    production_root: str | Path,
    subject: str,
    *,
    required_status: str = "complete",
    preferred_name_token: str = "yolo-b8_ritnet-b16",
    required_files: Sequence[str] = ("completion.json", "eyes.csv", "run_manifest.json"),
) -> ProductionRun:
    """Select exactly one complete production run without silent ambiguity."""
    production_root = Path(production_root)
    subject = normalize_subject(subject)
    candidates: list[ProductionRun] = []
    rejected: list[str] = []
    for directory in sorted(production_root.glob(f"{subject}_formal_*")):
        if not directory.is_dir():
            continue
        completion_path = directory / "completion.json"
        if not completion_path.is_file():
            rejected.append(f"{directory.name}:missing_completion")
            continue
        completion = _read_json(completion_path)
        if str(completion.get("status")) != required_status:
            rejected.append(f"{directory.name}:status={completion.get('status')}")
            continue
        if normalize_subject(completion.get("subject", "")) != subject:
            raise ValueError(f"completion subject mismatch: {completion_path}")
        missing = [name for name in required_files if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(f"complete run missing {missing}: {directory}")
        video = Path(str(completion.get("video", "")))
        if not video.is_file():
            raise FileNotFoundError(f"source video from completion does not exist: {video}")
        candidates.append(
            ProductionRun(
                subject=subject,
                path=directory,
                completion=completion,
                eyes_csv=directory / "eyes.csv",
                run_manifest=directory / "run_manifest.json",
                source_video=video,
            )
        )
    if not candidates:
        raise FileNotFoundError(
            f"no complete production run for {subject} under {production_root}; rejected={rejected}"
        )
    preferred = [run for run in candidates if preferred_name_token in run.path.name]
    pool = preferred or candidates
    if len(pool) != 1:
        raise RuntimeError(
            f"ambiguous complete production runs for {subject}: "
            + ", ".join(run.path.name for run in pool)
        )
    return pool[0]


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalize_eye(value: Any) -> str:
    raw = str(value).strip().lower().replace("-", "_")
    aliases = {
        "frame_left": "eye_left", "left": "eye_left", "eye_left": "eye_left",
        "frame_right": "eye_right", "right": "eye_right", "eye_right": "eye_right",
    }
    if raw not in aliases:
        raise ValueError(f"unknown eye value: {value!r}")
    return aliases[raw]


def load_production_eyes(run: ProductionRun) -> pd.DataFrame:
    header = pd.read_csv(run.eyes_csv, nrows=0)
    missing = [column for column in EYES_REQUIRED_COLUMNS if column not in header.columns]
    if missing:
        raise KeyError(f"production eyes.csv missing required columns {missing}: {run.eyes_csv}")
    frame = pd.read_csv(run.eyes_csv, usecols=list(EYES_REQUIRED_COLUMNS), low_memory=False)
    frame["subject"] = frame["subject"].map(normalize_subject)
    if set(frame["subject"].dropna().unique()) != {run.subject}:
        raise ValueError(f"eyes.csv subject identity mismatch: {run.eyes_csv}")
    videos = set(frame["video"].dropna().astype(str).unique())
    if videos != {str(run.source_video)}:
        raise ValueError(
            f"eyes.csv video identity differs from completion: {videos} vs {run.source_video}"
        )
    frame["phase"] = frame["phase"].map(normalize_phase)
    frame["phase_segment"] = pd.to_numeric(frame["phase_segment"], errors="raise").astype("int64")
    frame["frame_idx"] = pd.to_numeric(frame["frame_idx"], errors="raise").astype("int64")
    frame["eye"] = frame["eye"].map(normalize_eye)
    key = ["phase", "phase_segment", "frame_idx", "eye"]
    duplicates = frame.duplicated(key, keep=False)
    if duplicates.any():
        example = frame.loc[duplicates, key].head(5).to_dict("records")
        raise ValueError(f"duplicate production eye keys: {example}")
    numeric = [
        "anchor_yolo_confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "roi_x1", "roi_y1", "roi_x2", "roi_y2", "pupil_center_x",
        "pupil_center_y", "pupil_axis_a", "pupil_axis_b", "pupil_angle_deg",
        "pupil_confidence",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["roi_clipped"] = frame["roi_clipped"].map(_bool_value)
    frame["ritnet_found"] = frame["ritnet_found"].map(_bool_value)
    return frame.sort_values(key).reset_index(drop=True)


def transform_ellipse_affine(
    center_x: float,
    center_y: float,
    axis_a: float,
    axis_b: float,
    angle_deg: float,
    *,
    scale_x: float,
    scale_y: float,
    translate_x: float = 0.0,
    translate_y: float = 0.0,
) -> dict[str, float]:
    """Apply anisotropic scale + translation to an ellipse exactly.

    ``axis_a`` and ``axis_b`` are full axis lengths in the coordinate system
    described by OpenCV's RotatedRect angle.  Eigen-decomposition of the
    transformed quadratic form is required; multiplying both axes by one scale
    is incorrect when ROI aspect ratio differs from the analysis canvas.
    """
    values = np.asarray(
        [center_x, center_y, axis_a, axis_b, angle_deg, scale_x, scale_y], dtype=float
    )
    if not np.isfinite(values).all() or axis_a <= 0 or axis_b <= 0 or scale_x <= 0 or scale_y <= 0:
        return {key: float("nan") for key in (
            "center_x", "center_y", "major_axis", "minor_axis", "angle_deg", "diameter_geom"
        )}
    theta = math.radians(float(angle_deg))
    rotation = np.asarray(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=float,
    )
    semi = np.diag([(float(axis_a) / 2.0) ** 2, (float(axis_b) / 2.0) ** 2])
    shape = rotation @ semi @ rotation.T
    scale = np.diag([float(scale_x), float(scale_y)])
    transformed = scale @ shape @ scale.T
    eigenvalues, eigenvectors = np.linalg.eigh(transformed)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    major = 2.0 * math.sqrt(max(float(eigenvalues[0]), 0.0))
    minor = 2.0 * math.sqrt(max(float(eigenvalues[1]), 0.0))
    vector = eigenvectors[:, 0]
    major_angle = math.degrees(math.atan2(float(vector[1]), float(vector[0]))) % 180.0
    return {
        "center_x": float(translate_x) + float(scale_x) * float(center_x),
        "center_y": float(translate_y) + float(scale_y) * float(center_y),
        "major_axis": major,
        "minor_axis": minor,
        "angle_deg": major_angle,
        "diameter_geom": math.sqrt(major * minor),
    }


def map_ritnet_to_source(
    row: Mapping[str, Any], analysis_size: tuple[int, int] = (320, 160)
) -> dict[str, Any]:
    empty = {
        "ritnet_found": False,
        "ritnet_source_center_x": np.nan, "ritnet_source_center_y": np.nan,
        "ritnet_source_major_axis": np.nan, "ritnet_source_minor_axis": np.nan,
        "ritnet_source_angle_deg": np.nan, "ritnet_source_diameter_geom": np.nan,
    }
    if not _bool_value(row.get("ritnet_found")):
        return empty
    roi = [float(row.get(key, np.nan)) for key in ("roi_x1", "roi_y1", "roi_x2", "roi_y2")]
    if not np.isfinite(roi).all() or roi[2] <= roi[0] or roi[3] <= roi[1]:
        return empty
    analysis_w, analysis_h = map(float, analysis_size)
    mapped = transform_ellipse_affine(
        float(row.get("pupil_center_x", np.nan)),
        float(row.get("pupil_center_y", np.nan)),
        float(row.get("pupil_axis_a", np.nan)),
        float(row.get("pupil_axis_b", np.nan)),
        float(row.get("pupil_angle_deg", np.nan)),
        scale_x=(roi[2] - roi[0]) / analysis_w,
        scale_y=(roi[3] - roi[1]) / analysis_h,
        translate_x=roi[0],
        translate_y=roi[1],
    )
    if not np.isfinite(mapped["center_x"]):
        return empty
    return {
        "ritnet_found": True,
        "ritnet_source_center_x": mapped["center_x"],
        "ritnet_source_center_y": mapped["center_y"],
        "ritnet_source_major_axis": mapped["major_axis"],
        "ritnet_source_minor_axis": mapped["minor_axis"],
        "ritnet_source_angle_deg": mapped["angle_deg"],
        "ritnet_source_diameter_geom": mapped["diameter_geom"],
    }


def _uniform_pick(values: Sequence[int], n: int) -> list[int]:
    unique = sorted(set(int(value) for value in values))
    if len(unique) < n:
        raise ValueError(f"insufficient frames for uniform sample: need {n}, have {len(unique)}")
    positions = np.rint(np.linspace(0, len(unique) - 1, n)).astype(int)
    picked = [unique[int(position)] for position in positions]
    if len(set(picked)) != n:
        raise AssertionError("uniform sampling produced duplicate positions")
    return picked


def _frame_quality_table(eyes: pd.DataFrame) -> pd.DataFrame:
    task = eyes[eyes["phase"].isin(["block1", "block2"])].copy()
    task["bbox_ready"] = (
        task[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].notna().all(axis=1)
        & task["bbox_x2"].gt(task["bbox_x1"])
        & task["bbox_y2"].gt(task["bbox_y1"])
    )
    analysis_w, analysis_h = 320.0, 160.0
    task["edge_margin_norm"] = np.nan
    valid_center = task[["pupil_center_x", "pupil_center_y"]].notna().all(axis=1)
    task.loc[valid_center, "edge_margin_norm"] = np.minimum.reduce(
        [
            task.loc[valid_center, "pupil_center_x"] / analysis_w,
            (analysis_w - task.loc[valid_center, "pupil_center_x"]) / analysis_w,
            task.loc[valid_center, "pupil_center_y"] / analysis_h,
            (analysis_h - task.loc[valid_center, "pupil_center_y"]) / analysis_h,
        ]
    )
    keys = ["phase", "phase_segment", "frame_idx"]
    rows: list[dict[str, Any]] = []
    for key, group in task.groupby(keys, sort=True):
        reasons: list[str] = []
        if len(group) < 2:
            reasons.append("missing_eye_row")
        if int(group["bbox_ready"].sum()) < 2:
            reasons.append("bbox_unavailable")
        if int(group["ritnet_found"].sum()) < len(group):
            reasons.append("ritnet_missing")
        if bool(group["roi_clipped"].any()):
            reasons.append("roi_clipped")
        if not group["status"].astype(str).eq("observed").all():
            reasons.append("non_observed_status")
        rows.append(
            {
                **dict(zip(keys, key)),
                "eye_rows": int(len(group)),
                "bbox_ready_count": int(group["bbox_ready"].sum()),
                "ritnet_found_count": int(group["ritnet_found"].sum()),
                "observed_count": int(group["status"].astype(str).eq("observed").sum()),
                "roi_clipped_count": int(group["roi_clipped"].sum()),
                "min_pupil_confidence": float(group["pupil_confidence"].min(skipna=True)),
                "min_anchor_confidence": float(group["anchor_yolo_confidence"].min(skipna=True)),
                "min_edge_margin_norm": float(group["edge_margin_norm"].min(skipna=True)),
                "difficulty_reasons": ";".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def _key_tuple(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return str(row["phase"]), int(row["phase_segment"]), int(row["frame_idx"])


def _choose_temporal_frames(quality: pd.DataFrame, n: int, preferred_phase: str) -> pd.DataFrame:
    eligible = quality[
        quality["eye_rows"].eq(2) & quality["bbox_ready_count"].eq(2)
    ].copy()
    phases = [preferred_phase] + [value for value in ("block1", "block2") if value != preferred_phase]
    for phase in phases:
        phase_rows = eligible[eligible["phase"].eq(phase)]
        for segment, group in phase_rows.groupby("phase_segment", sort=True):
            values = np.sort(group["frame_idx"].unique().astype(np.int64))
            if len(values) < n:
                continue
            runs = np.split(values, np.flatnonzero(np.diff(values) != 1) + 1)
            runs = [run for run in runs if len(run) >= n]
            if not runs:
                continue
            phase_mid = float(np.median(values))
            options: list[tuple[float, int, np.ndarray]] = []
            for run in runs:
                max_start = len(run) - n
                ideal = int(np.clip(round(phase_mid - run[0] - (n - 1) / 2), 0, max_start))
                selected = run[ideal:ideal + n]
                center = (float(selected[0]) + float(selected[-1])) / 2.0
                options.append((abs(center - phase_mid), int(selected[0]), selected))
            selected = min(options, key=lambda item: (item[0], item[1]))[2]
            return pd.DataFrame(
                {
                    "phase": phase,
                    "phase_segment": int(segment),
                    "frame_idx": selected,
                    "sample_role": "temporal",
                }
            )
    raise ValueError(f"no consecutive {n}-frame two-eye interval in {preferred_phase}/fallback")


def build_sample_plan(
    eyes: pd.DataFrame,
    *,
    block_uniform_n: int,
    ritnet_high_quality_n: int,
    ritnet_difficult_n: int,
    temporal_n: int,
    temporal_preferred_phase: str = "block1",
    full_video: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build detector-outcome-independent tight and temporal frame samples.

    ``full_video=True`` selects every frame of block1+block2 for the tight
    sample (no uniform subsampling, no RITnet strata) and is intended for the
    full-video benchmark profile. ``temporal_n <= 0`` skips the temporal window.
    """
    quality = _frame_quality_table(eyes)
    selected: dict[tuple[str, int, int], set[str]] = {}
    for phase in ("block1", "block2"):
        phase_rows = quality[quality["phase"].eq(phase)]
        if full_video:
            values = sorted(phase_rows["frame_idx"].unique().tolist())
            role = f"{phase}_full"
        else:
            values = _uniform_pick(phase_rows["frame_idx"].tolist(), int(block_uniform_n))
            role = f"{phase}_uniform"
        segment_by_frame = phase_rows.set_index("frame_idx")["phase_segment"].to_dict()
        for frame_idx in values:
            key = (phase, int(segment_by_frame[frame_idx]), int(frame_idx))
            selected.setdefault(key, set()).add(role)

    if not full_video:
        used = set(selected)
        high = quality[
            quality["eye_rows"].eq(2)
            & quality["bbox_ready_count"].eq(2)
            & quality["ritnet_found_count"].eq(2)
            & quality["observed_count"].eq(2)
            & quality["roi_clipped_count"].eq(0)
            & quality["min_pupil_confidence"].notna()
        ].copy()
        high["_key"] = high.apply(lambda row: _key_tuple(row), axis=1)
        high = high[~high["_key"].isin(used)].sort_values(
            ["min_pupil_confidence", "min_anchor_confidence", "min_edge_margin_norm",
             "phase", "phase_segment", "frame_idx"],
            ascending=[False, False, False, True, True, True],
            na_position="last",
        )
        if len(high) < ritnet_high_quality_n:
            raise ValueError(
                f"insufficient RITnet high-quality frames: need {ritnet_high_quality_n}, have {len(high)}"
            )
        for row in high.head(ritnet_high_quality_n).to_dict("records"):
            key = _key_tuple(row)
            selected.setdefault(key, set()).add("ritnet_high_quality")
            used.add(key)

        difficult = quality.copy()
        difficult["_key"] = difficult.apply(lambda row: _key_tuple(row), axis=1)
        difficult = difficult[~difficult["_key"].isin(used)].copy()
        difficult["_hard_failure"] = (
            difficult["eye_rows"].lt(2)
            | difficult["bbox_ready_count"].lt(2)
            | difficult["ritnet_found_count"].lt(difficult["eye_rows"])
            | difficult["observed_count"].lt(difficult["eye_rows"])
        ).astype(int)
        difficult = difficult.sort_values(
            ["_hard_failure", "roi_clipped_count", "min_pupil_confidence",
             "min_edge_margin_norm", "min_anchor_confidence", "phase", "phase_segment", "frame_idx"],
            ascending=[False, False, True, True, True, True, True, True],
            na_position="first",
        )
        if len(difficult) < ritnet_difficult_n:
            raise ValueError(
                f"insufficient RITnet difficult frames: need {ritnet_difficult_n}, have {len(difficult)}"
            )
        for row in difficult.head(ritnet_difficult_n).to_dict("records"):
            selected.setdefault(_key_tuple(row), set()).add("ritnet_difficult")

    tight = pd.DataFrame(
        [
            {
                "phase": phase,
                "phase_segment": segment,
                "frame_idx": frame_idx,
                "sample_role": ";".join(sorted(roles)),
            }
            for (phase, segment, frame_idx), roles in sorted(selected.items())
        ]
    )
    if int(temporal_n) <= 0:
        temporal = pd.DataFrame(columns=["phase", "phase_segment", "frame_idx", "sample_role"])
    else:
        temporal = _choose_temporal_frames(quality, int(temporal_n), temporal_preferred_phase)
    return tight, temporal


def _bbox_from_row(row: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    values = tuple(float(row.get(key, np.nan)) for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))
    if not np.isfinite(values).all() or values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values


def _integer_bbox(
    bbox: tuple[float, float, float, float], frame_width: int, frame_height: int
) -> tuple[int, int, int, int] | None:
    x1 = max(0, min(frame_width, math.floor(bbox[0])))
    y1 = max(0, min(frame_height, math.floor(bbox[1])))
    x2 = max(0, min(frame_width, math.ceil(bbox[2])))
    y2 = max(0, min(frame_height, math.ceil(bbox[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _video_geometry(video: Path) -> tuple[int, int, int]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open source video: {video}")
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        cap.release()
    if width <= 0 or height <= 0 or count <= 0:
        raise ValueError(f"invalid video geometry for {video}: {width}x{height}, n={count}")
    return width, height, count


def make_sample_manifest(
    run: ProductionRun,
    eyes: pd.DataFrame,
    tight_frames: pd.DataFrame,
    temporal_frames: pd.DataFrame,
    *,
    crop_root: str | Path,
    analysis_size: tuple[int, int] = (320, 160),
    min_crop_width: int = 20,
    min_crop_height: int = 12,
) -> pd.DataFrame:
    """Expand selected frames to eye inputs and derive fixed temporal canvases."""
    crop_root = Path(crop_root)
    frame_width, frame_height, frame_count = _video_geometry(run.source_video)
    evidence = eyes.set_index(["phase", "phase_segment", "frame_idx", "eye"], drop=False)
    rows: list[dict[str, Any]] = []

    def base_identity(
        frame_row: Mapping[str, Any], eye: str, input_kind: str, sequence_id: str
    ) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
        key = (
            str(frame_row["phase"]), int(frame_row["phase_segment"]),
            int(frame_row["frame_idx"]), eye,
        )
        source = evidence.loc[key].to_dict() if key in evidence.index else None
        identity: dict[str, Any] = {
            "subject": run.subject,
            "production_run": str(run.path),
            "production_run_id": run.run_id,
            "source_video": str(run.source_video),
            "phase": key[0], "phase_segment": key[1], "frame_idx": key[2], "eye": eye,
            "sample_role": str(frame_row["sample_role"]),
            "sequence_id": sequence_id,
            "input_kind": input_kind,
            "input_status": "pending",
            "crop_path": None,
        }
        if source is None:
            identity["input_status"] = "missing_production_eye"
        else:
            identity.update(map_ritnet_to_source(source, analysis_size))
            identity["ritnet_pupil_confidence"] = source.get("pupil_confidence")
            identity["sampling_evidence"] = json.dumps(
                {
                    "production_status": source.get("status"),
                    "frame_status": source.get("frame_status"),
                    "roi_clipped": bool(source.get("roi_clipped")),
                    "ritnet_found": bool(source.get("ritnet_found")),
                    "pupil_confidence": source.get("pupil_confidence"),
                    "anchor_yolo_confidence": source.get("anchor_yolo_confidence"),
                },
                ensure_ascii=False,
            )
        return identity, source

    for frame_row in tight_frames.to_dict("records"):
        for eye in ("eye_left", "eye_right"):
            identity, source = base_identity(frame_row, eye, "production_tight_bbox", "")
            if source is not None:
                bbox = _bbox_from_row(source)
                if bbox is None:
                    identity["input_status"] = "invalid_production_bbox"
                else:
                    integer = _integer_bbox(bbox, frame_width, frame_height)
                    if integer is None:
                        identity["input_status"] = "bbox_outside_video"
                    else:
                        x1, y1, x2, y2 = integer
                        identity.update(
                            dict(zip(
                                ("bbox_float_x1", "bbox_float_y1", "bbox_float_x2", "bbox_float_y2"),
                                bbox,
                            ))
                        )
                        identity.update(
                            {"bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                             "input_width": x2 - x1, "input_height": y2 - y1}
                        )
                        if x2 - x1 < min_crop_width or y2 - y1 < min_crop_height:
                            identity["input_status"] = "crop_too_small"
                        else:
                            relative = Path("crops") / run.subject / "tight" / (
                                f"{frame_row['phase']}_s{int(frame_row['phase_segment'])}_"
                                f"f{int(frame_row['frame_idx']):08d}_{eye}.png"
                            )
                            identity["crop_path"] = relative.as_posix()
            rows.append(identity)

    temporal_sequence = (
        f"{run.subject}_{temporal_frames.iloc[0]['phase']}_s"
        f"{int(temporal_frames.iloc[0]['phase_segment'])}_"
        f"f{int(temporal_frames['frame_idx'].min()):08d}-"
        f"f{int(temporal_frames['frame_idx'].max()):08d}"
    )
    temporal_sources: dict[str, list[tuple[float, float, float, float]]] = {
        "eye_left": [], "eye_right": []
    }
    for frame_row in temporal_frames.to_dict("records"):
        for eye in temporal_sources:
            key = (
                str(frame_row["phase"]), int(frame_row["phase_segment"]),
                int(frame_row["frame_idx"]), eye,
            )
            if key not in evidence.index:
                raise ValueError(f"temporal sequence missing production eye row: {key}")
            bbox = _bbox_from_row(evidence.loc[key].to_dict())
            if bbox is None:
                raise ValueError(f"temporal sequence has invalid bbox: {key}")
            temporal_sources[eye].append(bbox)
    fixed: dict[str, tuple[int, int, int, int]] = {}
    for eye, boxes in temporal_sources.items():
        union = (
            min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes),
        )
        integer = _integer_bbox(union, frame_width, frame_height)
        if integer is None:
            raise ValueError(f"invalid temporal fixed canvas for {eye}: {union}")
        fixed[eye] = integer

    for frame_row in temporal_frames.to_dict("records"):
        for eye in ("eye_left", "eye_right"):
            identity, source = base_identity(
                frame_row,
                eye,
                "fixed_source_canvas_from_temporal_tight_bbox_union",
                temporal_sequence,
            )
            if source is None:
                raise AssertionError("temporal source was validated but disappeared")
            source_bbox = _bbox_from_row(source)
            x1, y1, x2, y2 = fixed[eye]
            identity.update(
                {
                    "bbox_float_x1": source_bbox[0], "bbox_float_y1": source_bbox[1],
                    "bbox_float_x2": source_bbox[2], "bbox_float_y2": source_bbox[3],
                    "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                    "input_width": x2 - x1, "input_height": y2 - y1,
                    "crop_path": (
                        Path("crops") / run.subject / "continuous" / temporal_sequence /
                        f"f{int(frame_row['frame_idx']):08d}_{eye}.png"
                    ).as_posix(),
                }
            )
            rows.append(identity)

    manifest = pd.DataFrame(rows)
    if manifest["frame_idx"].ge(frame_count).any() or manifest["frame_idx"].lt(0).any():
        raise ValueError("sample plan contains frame outside source video")
    return manifest.reindex(columns=SAMPLE_COLUMNS)


def materialize_crops(manifest: pd.DataFrame, run_dir: str | Path) -> pd.DataFrame:
    """Decode requested frames and write lossless grayscale PNG crops."""
    import cv2

    run_dir = Path(run_dir)
    output = manifest.copy()
    ready_indices = output.index[output["input_status"].eq("pending")].tolist()
    if not ready_indices:
        return output
    video_values = output.loc[ready_indices, "source_video"].dropna().unique()
    if len(video_values) != 1:
        raise ValueError(f"one materialization call requires one source video, got {video_values}")
    video = Path(str(video_values[0]))
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open source video: {video}")
        for frame_idx, group in output.loc[ready_indices].groupby("frame_idx", sort=True):
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx)):
                output.loc[group.index, "input_status"] = "video_seek_failed"
                continue
            ok, frame = cap.read()
            actual = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1
            if not ok or frame is None:
                output.loc[group.index, "input_status"] = "video_read_failed"
                continue
            if actual != int(frame_idx):
                output.loc[group.index, "input_status"] = f"video_frame_mismatch:{actual}"
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            for index in group.index:
                row = output.loc[index]
                x1, y1, x2, y2 = (int(row[key]) for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))
                crop = gray[y1:y2, x1:x2]
                expected = (int(row["input_height"]), int(row["input_width"]))
                if crop.shape[:2] != expected:
                    output.at[index, "input_status"] = f"crop_shape_mismatch:{crop.shape[:2]}"
                    continue
                path = run_dir / str(row["crop_path"])
                if path.exists():
                    raise FileExistsError(f"refusing to overwrite existing crop: {path}")
                path.parent.mkdir(parents=True, exist_ok=True)
                encoded_ok, encoded = cv2.imencode(".png", crop)
                if not encoded_ok:
                    output.at[index, "input_status"] = "png_encode_failed"
                    continue
                encoded.tofile(str(path))
                output.at[index, "input_status"] = "ready"
    finally:
        cap.release()
    return output


def _unavailable_results(
    rows: Sequence[Mapping[str, Any]], algorithms: Sequence[str]
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for identity in rows:
        for algorithm in algorithms:
            detection = {
                "algorithm": algorithm,
                "algorithm_returned": False,
                "official_valid": False,
                "geometry_sane": False,
                "failure": f"input_unavailable:{identity.get('input_status')}",
            }
            records.append(assemble_row(identity, detection))
    return pd.DataFrame(records, columns=RESULT_COLUMNS)


def enrich_source_and_agreement(results: pd.DataFrame) -> pd.DataFrame:
    output = results.copy()
    numeric = [
        "center_x", "center_y", "major_axis", "minor_axis", "diameter_geom",
        "bbox_x1", "bbox_y1", "ritnet_source_center_x", "ritnet_source_center_y",
        "ritnet_source_major_axis", "ritnet_source_minor_axis",
        "ritnet_source_diameter_geom",
    ]
    for column in numeric:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["source_center_x"] = output["bbox_x1"] + output["center_x"]
    output["source_center_y"] = output["bbox_y1"] + output["center_y"]
    output["source_major_axis"] = output["major_axis"]
    output["source_minor_axis"] = output["minor_axis"]
    output["source_angle_deg"] = output["angle_deg"]
    output["source_diameter_geom"] = output["diameter_geom"]
    output["agreement_center_distance_px"] = np.hypot(
        output["source_center_x"] - output["ritnet_source_center_x"],
        output["source_center_y"] - output["ritnet_source_center_y"],
    )
    output["agreement_diameter_ratio"] = [
        safe_ratio(a, b) for a, b in zip(
            output["source_diameter_geom"], output["ritnet_source_diameter_geom"]
        )
    ]
    output["agreement_major_axis_ratio"] = [
        safe_ratio(a, b) for a, b in zip(
            output["source_major_axis"], output["ritnet_source_major_axis"]
        )
    ]
    output["agreement_minor_axis_ratio"] = [
        safe_ratio(a, b) for a, b in zip(
            output["source_minor_axis"], output["ritnet_source_minor_axis"]
        )
    ]
    return output.reindex(columns=RESULT_COLUMNS)


def execute_manifest(
    manifest: pd.DataFrame,
    algorithms: Sequence[str],
    *,
    run_dir: str | Path,
    run_confidence: bool = False,
) -> pd.DataFrame:
    for algorithm in algorithms:
        if algorithm not in ALGORITHM_SPECS:
            raise KeyError(f"unknown algorithm: {algorithm}")
    ready = manifest[manifest["input_status"].eq("ready")]
    tight = ready[ready["input_kind"].eq("production_tight_bbox")]
    continuous = ready[
        ready["input_kind"].eq("fixed_source_canvas_from_temporal_tight_bbox_union")
    ]
    parts: list[pd.DataFrame] = []
    if not tight.empty:
        parts.append(
            run_crop_list(
                tight.to_dict("records"), algorithms, crop_root=run_dir,
                run_confidence=run_confidence, mode="independent",
            )
        )
    if not continuous.empty:
        parts.append(
            run_crop_list(
                continuous.to_dict("records"), algorithms, crop_root=run_dir,
                run_confidence=run_confidence, mode="continuous",
            )
        )
    unavailable = manifest[~manifest["input_status"].eq("ready")]
    if not unavailable.empty:
        parts.append(_unavailable_results(unavailable.to_dict("records"), algorithms))
    if not parts:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return enrich_source_and_agreement(pd.concat(parts, ignore_index=True))


def subject_algorithm_summary(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["subject", "input_kind", "phase", "eye", "algorithm"]
    rows: list[dict[str, Any]] = []
    for key, group in results.groupby(keys, dropna=False, sort=True):
        runtime = pd.to_numeric(group["runtime_ms"], errors="coerce")
        center = pd.to_numeric(group["agreement_center_distance_px"], errors="coerce")
        ratio = pd.to_numeric(group["agreement_diameter_ratio"], errors="coerce")
        rows.append(
            {
                **dict(zip(keys, key)),
                "n_expected": int(len(group)),
                "n_input_ready": int(group["input_status"].eq("ready").sum()),
                "n_returned": int(group["algorithm_returned"].fillna(False).astype(bool).sum()),
                "return_rate": float(group["algorithm_returned"].fillna(False).astype(bool).mean()),
                "n_official_valid": int(group["official_valid"].fillna(False).astype(bool).sum()),
                "n_geometry_sane": int(group["geometry_sane"].fillna(False).astype(bool).sum()),
                "runtime_median_ms": _series_stat(runtime, "median"),
                "runtime_p90_ms": _series_stat(runtime, "quantile", 0.90),
                "ritnet_center_distance_median_px": _series_stat(center, "median"),
                "ritnet_center_distance_p90_px": _series_stat(center, "quantile", 0.90),
                "ritnet_diameter_ratio_median": _series_stat(ratio, "median"),
                "ritnet_diameter_ratio_p10": _series_stat(ratio, "quantile", 0.10),
                "ritnet_diameter_ratio_p90": _series_stat(ratio, "quantile", 0.90),
            }
        )
    return pd.DataFrame(rows)


def _series_stat(series: pd.Series, method: str, *args: Any) -> float:
    finite = pd.to_numeric(series, errors="coerce").dropna()
    if finite.empty:
        return float("nan")
    return float(getattr(finite, method)(*args))


def algorithm_pairwise_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Descriptive pairwise center/diameter agreement among classical methods."""
    from itertools import combinations

    identity = [
        "subject", "input_kind", "phase", "phase_segment", "frame_idx", "eye", "sequence_id"
    ]
    rows: list[dict[str, Any]] = []
    for key, group in results.groupby(identity, dropna=False, sort=True):
        by_algorithm = {str(row["algorithm"]): row for row in group.to_dict("records")}
        for algorithm_a, algorithm_b in combinations(sorted(by_algorithm), 2):
            a, b = by_algorithm[algorithm_a], by_algorithm[algorithm_b]
            both_sane = bool(a.get("geometry_sane")) and bool(b.get("geometry_sane"))
            center_distance = float("nan")
            diameter_ratio = float("nan")
            if both_sane:
                values = np.asarray(
                    [a.get("source_center_x"), a.get("source_center_y"),
                     b.get("source_center_x"), b.get("source_center_y")],
                    dtype=float,
                )
                if np.isfinite(values).all():
                    center_distance = float(np.hypot(values[0] - values[2], values[1] - values[3]))
                    diameter_ratio = safe_ratio(
                        a.get("source_diameter_geom"), b.get("source_diameter_geom")
                    )
            rows.append(
                {
                    **dict(zip(identity, key)),
                    "algorithm_a": algorithm_a,
                    "algorithm_b": algorithm_b,
                    "both_geometry_sane": both_sane,
                    "center_distance_px": center_distance,
                    "diameter_ratio_a_over_b": diameter_ratio,
                }
            )
    long = pd.DataFrame(rows)
    if long.empty:
        return long
    summary_keys = ["subject", "input_kind", "phase", "eye", "algorithm_a", "algorithm_b"]
    summaries: list[dict[str, Any]] = []
    for key, group in long.groupby(summary_keys, dropna=False, sort=True):
        center = group["center_distance_px"]
        ratio = group["diameter_ratio_a_over_b"]
        summaries.append(
            {
                **dict(zip(summary_keys, key)),
                "n_expected_pairs": int(len(group)),
                "n_both_geometry_sane": int(group["both_geometry_sane"].sum()),
                "center_distance_median_px": _series_stat(center, "median"),
                "center_distance_p90_px": _series_stat(center, "quantile", 0.90),
                "diameter_ratio_median": _series_stat(ratio, "median"),
                "diameter_ratio_p10": _series_stat(ratio, "quantile", 0.10),
                "diameter_ratio_p90": _series_stat(ratio, "quantile", 0.90),
            }
        )
    return pd.DataFrame(summaries)


def temporal_summary(results: pd.DataFrame) -> pd.DataFrame:
    work = results[
        results["input_kind"].eq("fixed_source_canvas_from_temporal_tight_bbox_union")
    ].copy()
    rows: list[dict[str, Any]] = []
    keys = ["subject", "sequence_id", "eye", "algorithm"]
    for key, group in work.groupby(keys, sort=True):
        group = group.sort_values("frame_idx")
        sane = group[group["geometry_sane"].fillna(False).astype(bool)].copy()
        consecutive = sane["frame_idx"].diff().eq(1)
        center_step = np.hypot(
            sane["source_center_x"].diff()[consecutive],
            sane["source_center_y"].diff()[consecutive],
        )
        diameter_step = sane["source_diameter_geom"].diff().abs()[consecutive]
        runtime = pd.to_numeric(group["runtime_ms"], errors="coerce")
        rows.append(
            {
                **dict(zip(keys, key)),
                "n_expected": int(len(group)),
                "n_input_ready": int(group["input_status"].eq("ready").sum()),
                "n_returned": int(group["algorithm_returned"].fillna(False).astype(bool).sum()),
                "n_geometry_sane": int(len(sane)),
                "geometry_sane_coverage": float(len(sane) / len(group)) if len(group) else np.nan,
                "n_consecutive_pairs": int(consecutive.sum()),
                "center_step_median_px": _series_stat(center_step, "median"),
                "center_step_p90_px": _series_stat(center_step, "quantile", 0.90),
                "diameter_step_median_px": _series_stat(diameter_step, "median"),
                "diameter_step_p90_px": _series_stat(diameter_step, "quantile", 0.90),
                "runtime_median_ms": _series_stat(runtime, "median"),
                "estimated_fps_from_median_runtime": safe_ratio(
                    1000.0, _series_stat(runtime, "median")
                ),
            }
        )
    return pd.DataFrame(rows)


def write_manual_qc_montages(
    results: pd.DataFrame,
    run_dir: str | Path,
    *,
    n_frames_per_subject: int,
) -> pd.DataFrame:
    """Write local frame montages: original + RITnet + all algorithms, both eyes.

    The returned CSV template intentionally leaves the human label blank.  A
    montage is review evidence, not an automatic correctness decision.
    """
    import cv2

    from .overlay import draw_detection

    run_dir = Path(run_dir)
    tight = results[results["input_kind"].eq("production_tight_bbox")].copy()
    if tight.empty or n_frames_per_subject <= 0:
        return pd.DataFrame()
    algorithm_order = [name for name in ALGORITHM_SPECS if name in set(tight["algorithm"])]
    label_categories = (
        "credible_pupil;eyelid_or_edge_false_positive;reflection_false_positive;"
        "other_false_positive;miss;uncertain"
    )
    template_rows: list[dict[str, Any]] = []

    for subject, subject_rows in tight.groupby("subject", sort=True):
        identity_columns = ["phase", "phase_segment", "frame_idx", "sample_role"]
        candidate_inputs = subject_rows[
            [*identity_columns, "eye", "input_status", "crop_path"]
        ].drop_duplicates()
        candidate_inputs = candidate_inputs[
            candidate_inputs["input_status"].eq("ready")
            & candidate_inputs["crop_path"].notna()
        ]
        ready_eye_counts = candidate_inputs.groupby(
            identity_columns, dropna=False, sort=True
        )["eye"].nunique()
        reviewable = ready_eye_counts[ready_eye_counts.eq(2)].reset_index()[identity_columns]
        identities = reviewable.drop_duplicates()
        if len(identities) < n_frames_per_subject:
            raise ValueError(
                f"manual QC needs {n_frames_per_subject} two-eye ready frames for "
                f"{subject}, only {len(identities)} are reviewable"
            )
        identities["priority"] = identities["sample_role"].astype(str).map(
            lambda value: 0 if "ritnet_difficult" in value else (
                1 if "ritnet_high_quality" in value else 2
            )
        )
        identities = identities.sort_values(
            ["priority", "phase", "phase_segment", "frame_idx"]
        )
        # Deterministic stratification: alternate difficult/high/uniform pools,
        # then fill any remainder in stable frame order.
        pools = [group for _, group in identities.groupby("priority", sort=True)]
        selected_rows: list[pd.Series] = []
        while len(selected_rows) < n_frames_per_subject and any(len(pool) for pool in pools):
            for index, pool in enumerate(pools):
                if len(selected_rows) >= n_frames_per_subject:
                    break
                if pool.empty:
                    continue
                selected_rows.append(pool.iloc[0])
                pools[index] = pool.iloc[1:]

        for selected in selected_rows:
            key_mask = (
                subject_rows["phase"].eq(selected["phase"])
                & subject_rows["phase_segment"].eq(selected["phase_segment"])
                & subject_rows["frame_idx"].eq(selected["frame_idx"])
            )
            frame_rows = subject_rows[key_mask]
            tile_w, tile_h = 240, 120
            columns = ["original", "RITnet", *algorithm_order]
            canvas = np.zeros((2 * tile_h, len(columns) * tile_w, 3), dtype=np.uint8)
            montage_relative = (
                Path("manual_qc") / str(subject) /
                f"{selected['phase']}_s{int(selected['phase_segment'])}_"
                f"f{int(selected['frame_idx']):08d}.png"
            )
            for eye_index, eye in enumerate(("eye_left", "eye_right")):
                eye_rows = frame_rows[frame_rows["eye"].eq(eye)]
                if eye_rows.empty:
                    continue
                first = eye_rows.iloc[0]
                crop_path = run_dir / str(first["crop_path"])
                image = cv2.imdecode(np.fromfile(str(crop_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"manual QC could not decode crop: {crop_path}")

                tiles: list[tuple[str, np.ndarray]] = [("original", image.copy())]
                ritnet_row = {
                    "center_x": float(first["ritnet_source_center_x"]) - float(first["bbox_x1"]),
                    "center_y": float(first["ritnet_source_center_y"]) - float(first["bbox_y1"]),
                    "major_axis": first["ritnet_source_major_axis"],
                    "minor_axis": first["ritnet_source_minor_axis"],
                    "angle_deg": first["ritnet_source_angle_deg"],
                }
                tiles.append(("RITnet", draw_detection(image, ritnet_row, color=(0, 255, 255))))
                for algorithm in algorithm_order:
                    algorithm_rows = eye_rows[eye_rows["algorithm"].eq(algorithm)]
                    if algorithm_rows.empty:
                        tiles.append((f"{algorithm}:NO_ROW", image.copy()))
                        continue
                    row = algorithm_rows.iloc[0].to_dict()
                    status = "RET" if bool(row.get("algorithm_returned")) else "MISS"
                    if bool(row.get("geometry_sane")):
                        status = "GEOM"
                    tiles.append(
                        (f"{algorithm}:{status}", draw_detection(image, row, color=(0, 0, 255)))
                    )
                    template_rows.append(
                        {
                            "subject": subject,
                            "phase": selected["phase"],
                            "phase_segment": int(selected["phase_segment"]),
                            "frame_idx": int(selected["frame_idx"]),
                            "eye": eye,
                            "algorithm": algorithm,
                            "sample_role": selected["sample_role"],
                            "montage_path": montage_relative.as_posix(),
                            "human_label": "",
                            "allowed_labels": label_categories,
                            "reviewer": "",
                            "reviewed_at": "",
                            "notes": "",
                        }
                    )

                for column_index, (label, tile) in enumerate(tiles):
                    resized = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
                    cv2.putText(resized, f"{eye} {label}", (4, 17), cv2.FONT_HERSHEY_SIMPLEX,
                                0.42, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(resized, f"{eye} {label}", (4, 17), cv2.FONT_HERSHEY_SIMPLEX,
                                0.42, (255, 255, 255), 1, cv2.LINE_AA)
                    y0 = eye_index * tile_h
                    x0 = column_index * tile_w
                    canvas[y0:y0 + tile_h, x0:x0 + tile_w] = resized
            montage_path = run_dir / montage_relative
            montage_path.parent.mkdir(parents=True, exist_ok=True)
            ok, encoded = cv2.imencode(".png", canvas)
            if not ok:
                raise RuntimeError(f"could not encode manual QC montage: {montage_path}")
            encoded.tofile(str(montage_path))
    return pd.DataFrame(template_rows)


def validate_result_contract(
    manifest: pd.DataFrame, results: pd.DataFrame, algorithms: Sequence[str]
) -> dict[str, Any]:
    expected = int(len(manifest) * len(algorithms))
    identity_key = [
        "subject", "phase", "phase_segment", "frame_idx", "eye", "input_kind", "sequence_id"
    ]
    result_key = [*identity_key, "algorithm"]
    missing_columns = [column for column in RESULT_COLUMNS if column not in results.columns]
    missing_manifest_keys = [column for column in identity_key if column not in manifest.columns]
    if missing_manifest_keys:
        raise KeyError(f"manifest missing identity columns: {missing_manifest_keys}")

    duplicate_manifest_keys = int(manifest.duplicated(identity_key).sum())
    duplicate_count = (
        int(results.duplicated(result_key).sum())
        if not any(column not in results.columns for column in result_key)
        else -1
    )

    def key_tuples(frame: pd.DataFrame, columns: Sequence[str]) -> set[tuple[Any, ...]]:
        tuples: set[tuple[Any, ...]] = set()
        for row in frame.loc[:, list(columns)].itertuples(index=False, name=None):
            normalized: list[Any] = []
            for column, value in zip(columns, row):
                if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
                    normalized.append("")
                elif column in {"phase_segment", "frame_idx"}:
                    normalized.append(int(value))
                else:
                    normalized.append(str(value))
            tuples.add(tuple(normalized))
        return tuples

    expected_frame = manifest.loc[:, identity_key].copy()
    expected_frame = expected_frame.assign(_join_key=1).merge(
        pd.DataFrame({"algorithm": list(algorithms), "_join_key": 1}),
        on="_join_key",
        how="inner",
    ).drop(columns="_join_key")
    expected_keys = key_tuples(expected_frame, result_key)
    if any(column not in results.columns for column in result_key):
        actual_keys: set[tuple[Any, ...]] = set()
    else:
        actual_keys = key_tuples(results, result_key)
    missing_keys = expected_keys - actual_keys
    unexpected_keys = actual_keys - expected_keys
    checks = {
        "expected_result_rows": expected,
        "actual_result_rows": int(len(results)),
        "duplicate_manifest_keys": duplicate_manifest_keys,
        "duplicate_result_keys": duplicate_count,
        "missing_result_keys": int(len(missing_keys)),
        "unexpected_result_keys": int(len(unexpected_keys)),
        "missing_result_key_examples": [list(value) for value in sorted(missing_keys)[:5]],
        "unexpected_result_key_examples": [list(value) for value in sorted(unexpected_keys)[:5]],
        "missing_result_columns": missing_columns,
        "manifest_rows": int(len(manifest)),
        "ready_manifest_rows": int(manifest["input_status"].eq("ready").sum()),
        "input_unavailable_rows": int((~manifest["input_status"].eq("ready")).sum()),
        "algorithm_count": int(len(algorithms)),
    }
    checks["valid"] = bool(
        expected == len(results)
        and duplicate_manifest_keys == 0
        and duplicate_count == 0
        and not missing_keys
        and not unexpected_keys
        and not missing_columns
    )
    if not checks["valid"]:
        raise AssertionError(f"formal result contract failed: {checks}")
    return checks


def environment_manifest() -> dict[str, Any]:
    import cv2
    import importlib.metadata as metadata

    packages = {}
    for name in ("numpy", "opencv-python", "pandas", "PyYAML", "pytest", "PyPupilEXT", "pupil-detectors"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "opencv_runtime_version": cv2.__version__,
        "packages": packages,
    }


def production_provenance(run: ProductionRun, config_path: Path, hash_evidence: bool) -> dict[str, Any]:
    stat = run.source_video.stat()
    evidence = {
        "completion_json": str(run.path / "completion.json"),
        "eyes_csv": str(run.eyes_csv),
        "run_manifest_json": str(run.run_manifest),
    }
    hashes = {key: sha256_file(path) for key, path in evidence.items()} if hash_evidence else {}
    return {
        "subject": run.subject,
        "production_run": str(run.path),
        "production_run_id": run.run_id,
        "source_video": str(run.source_video),
        "source_video_size_bytes": int(stat.st_size),
        "source_video_mtime_ns": int(stat.st_mtime_ns),
        "evidence_files": evidence,
        "evidence_sha256": hashes,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment_manifest(),
        "scientific_boundary": "RITnet comparison is agreement only, not accuracy or ground truth",
    }
