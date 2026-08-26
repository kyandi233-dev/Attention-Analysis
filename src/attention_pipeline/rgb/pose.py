from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
import urllib.request
import ctypes
from bisect import bisect_left, bisect_right
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from attention_pipeline.config import Config
from attention_pipeline.rgb.audit import read_rgb_timestamps, video_metadata
from attention_pipeline.rgb.behavior import BehaviorIndex, empty_behavior_context
from attention_pipeline.rgb.discover import RGBSubjectFiles, discover_rgb_subjects
from attention_pipeline.rgb.paths import RGBOutputLayout
from attention_pipeline.rgb.timeline import detailed_rgb_intervals, formal_analysis_span


POSE_SCHEMA_VERSION = "rgb-pose-landmarks-v0.1"
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index",
    "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
]


def _configured_exclusion(config: Config, subject: str) -> tuple[bool, str]:
    raw = config.section("data").get("exclude", {})
    if isinstance(raw, dict):
        if subject in raw:
            return True, str(raw[subject])
        return False, ""
    if isinstance(raw, list):
        return subject in {str(value) for value in raw}, "configured exclusion"
    return False, ""


def _find_subject(config: Config, subject: str) -> RGBSubjectFiles:
    records, duplicates = discover_rgb_subjects(config)
    if subject in duplicates:
        raise RuntimeError(f"Subject {subject} is duplicated across data roots: {duplicates[subject]}")
    for record in records:
        if record.subject == subject:
            return record
    raise FileNotFoundError(f"RGB subject not discovered: {subject}")


def _phase_at(unix_ms: int, intervals) -> str:
    for interval in intervals:
        if interval.start_unix_ms <= unix_ms < interval.end_unix_ms:
            return interval.phase
    if intervals and unix_ms == intervals[-1].end_unix_ms:
        return intervals[-1].phase
    return "outside_analysis_span"


def _block_from_phase(phase: str) -> int | None:
    if phase.startswith("block") and phase[5:].isdigit():
        return int(phase[5:])
    return None


def _git_commit(config: Config) -> str | None:
    repo_root = config.path.parent.parent
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_model(path: Path, url: str) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[pose] downloading MediaPipe model -> {path}")
    urllib.request.urlretrieve(url, path)
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"Failed to download MediaPipe pose model: {url}")
    return path


def _native_model_path(path: Path) -> str:
    """Return a MediaPipe-readable path on Windows even under a Chinese output root."""
    if not path.exists() or os.name != "nt":
        return str(path)
    get_short = ctypes.windll.kernel32.GetShortPathNameW
    buffer = ctypes.create_unicode_buffer(32768)
    length = get_short(str(path), buffer, len(buffer))
    if length and buffer.value != str(path):
        return buffer.value
    # MediaPipe's native loader may reject Unicode paths even when the file is
    # present. Keep the configured model in the output root, and expose an
    # identical ASCII-path cache only for the native reader.
    ascii_cache = Path("D:/AttentionAnalysis_models") / path.name
    ascii_cache.parent.mkdir(parents=True, exist_ok=True)
    if not ascii_cache.exists() or ascii_cache.stat().st_size != path.stat().st_size:
        shutil.copy2(path, ascii_cache)
    return str(ascii_cache)


def _bbox(landmarks) -> tuple[float | None, float | None, float | None, float | None]:
    xs = [float(item.x) for item in landmarks if item.x is not None and np.isfinite(item.x)]
    ys = [float(item.y) for item in landmarks if item.y is not None and np.isfinite(item.y)]
    if not xs or not ys:
        return None, None, None, None
    return min(xs), min(ys), max(xs), max(ys)


def pose_result_rows(
    result,
    *,
    base: dict[str, object],
) -> list[dict[str, object]]:
    """Flatten every returned pose/landmark without silently dropping extra poses."""
    pose_landmarks = list(getattr(result, "pose_landmarks", []) or [])
    world_landmarks = list(getattr(result, "pose_world_landmarks", []) or [])
    if not pose_landmarks:
        row = dict(base)
        row.update(
            {
                "pose_valid": False,
                "pose_count": 0,
                "pose_index": None,
                "landmark_index": None,
                "landmark_name": None,
                "x": None, "y": None, "z": None,
                "visibility": None, "presence": None,
                "world_x": None, "world_y": None, "world_z": None,
                "world_visibility": None, "world_presence": None,
                "pose_bbox_xmin": None, "pose_bbox_ymin": None,
                "pose_bbox_xmax": None, "pose_bbox_ymax": None,
            }
        )
        return [row]

    rows: list[dict[str, object]] = []
    pose_count = len(pose_landmarks)
    for pose_index, landmarks in enumerate(pose_landmarks):
        worlds = world_landmarks[pose_index] if pose_index < len(world_landmarks) else []
        xmin, ymin, xmax, ymax = _bbox(landmarks)
        for landmark_index, landmark in enumerate(landmarks):
            world = worlds[landmark_index] if landmark_index < len(worlds) else None
            row = dict(base)
            row.update(
                {
                    "pose_valid": True,
                    "pose_count": pose_count,
                    "pose_index": pose_index,
                    "landmark_index": landmark_index,
                    "landmark_name": LANDMARK_NAMES[landmark_index] if landmark_index < len(LANDMARK_NAMES) else f"landmark_{landmark_index}",
                    "x": float(landmark.x) if landmark.x is not None else None,
                    "y": float(landmark.y) if landmark.y is not None else None,
                    "z": float(landmark.z) if landmark.z is not None else None,
                    "visibility": float(landmark.visibility) if getattr(landmark, "visibility", None) is not None else None,
                    "presence": float(landmark.presence) if getattr(landmark, "presence", None) is not None else None,
                    "world_x": float(world.x) if world is not None and world.x is not None else None,
                    "world_y": float(world.y) if world is not None and world.y is not None else None,
                    "world_z": float(world.z) if world is not None and world.z is not None else None,
                    "world_visibility": float(world.visibility) if world is not None and getattr(world, "visibility", None) is not None else None,
                    "world_presence": float(world.presence) if world is not None and getattr(world, "presence", None) is not None else None,
                    "pose_bbox_xmin": xmin,
                    "pose_bbox_ymin": ymin,
                    "pose_bbox_xmax": xmax,
                    "pose_bbox_ymax": ymax,
                }
            )
            rows.append(row)
    return rows


def run_pose_test(config: Config, subject: str) -> dict[str, object]:
    """Run MediaPipe Pose on the real formal RGB analysis span into _test."""
    excluded, reason = _configured_exclusion(config, subject)
    if excluded:
        raise ValueError(f"Subject {subject} is excluded from RGB analysis: {reason}")

    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            'MediaPipe is not installed. Run: python -m pip install -e ".[test,rgb]"'
        ) from exc

    files = _find_subject(config, subject)
    timestamps = read_rgb_timestamps(files.timestamps)
    metadata = video_metadata(files.video)
    if not metadata["video_open_ok"]:
        raise RuntimeError(f"RGB video cannot be opened: {files.video}")
    if int(metadata["video_frame_count_nominal"]) != len(timestamps):
        raise ValueError(f"AVI/timestamp row mismatch for {subject}")

    focuswave = config.section("focuswave")
    pose_cfg = config.section("pose")
    baseline_duration_sec = float(focuswave.get("baseline_duration_sec", 180))
    expected_blocks = int(focuswave.get("expected_blocks", 2))
    trial_duration_ms = int(focuswave.get("trial_duration_ms", 1150))
    inference_fps = float(pose_cfg.get("inference_fps", 5.0) or 5.0)
    if inference_fps <= 0:
        raise ValueError("pose.inference_fps must be > 0")
    inference_interval_ms = 1000.0 / inference_fps
    decode_sampled_frames_only = bool(pose_cfg.get("decode_sampled_frames_only", True))

    layout = RGBOutputLayout.from_config(config)
    model_path_raw = str(pose_cfg.get("model_path", "_test/pose_landmarker_lite.task"))
    model_path = Path(model_path_raw)
    if not model_path.is_absolute():
        model_path = layout.root / model_path
    model_url = str(pose_cfg.get("model_url", DEFAULT_MODEL_URL))
    model_path = _ensure_model(model_path, model_url)

    intervals = detailed_rgb_intervals(
        files.master_timeline,
        baseline_duration_sec=baseline_duration_sec,
        expected_blocks=expected_blocks,
    )
    analysis_start, analysis_end = formal_analysis_span(
        files.master_timeline,
        baseline_duration_sec=baseline_duration_sec,
        expected_blocks=expected_blocks,
    )
    all_times = [row[1] for row in timestamps]
    start_position = bisect_left(all_times, analysis_start)
    end_position = bisect_right(all_times, analysis_end) - 1
    if end_position < start_position:
        raise ValueError(f"No RGB frames inside formal analysis span for {subject}")

    behavior_indexes = {
        1: BehaviorIndex.from_csv(files.block1_behavior),
        2: BehaviorIndex.from_csv(files.block2_behavior),
    }

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_native_model_path(model_path)),
        running_mode=RunningMode.VIDEO,
        num_poses=int(pose_cfg.get("num_poses", 2)),
        min_pose_detection_confidence=float(pose_cfg.get("min_pose_detection_confidence", 0.5)),
        min_pose_presence_confidence=float(pose_cfg.get("min_pose_presence_confidence", 0.5)),
        min_tracking_confidence=float(pose_cfg.get("min_tracking_confidence", 0.5)),
        output_segmentation_masks=False,
    )

    cap = cv2.VideoCapture(str(files.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open RGB video: {files.video}")
    if start_position > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_position))

    output_path = layout.test_file(f"{subject}_pose-test.parquet")
    manifest_path = layout.test_file(f"{subject}_pose-test_manifest.json")
    started_utc = datetime.now(timezone.utc).isoformat()
    started_clock = time.perf_counter()
    rows: list[dict[str, object]] = []
    sampled_frames = 0
    valid_frames = 0
    multi_pose_frames = 0
    next_sample_unix_ms = float(analysis_start)
    previous_sample_capture_idx: int | None = None
    previous_sample_unix_ms: int | None = None
    expected_capture_step = max(
        1, int(round(float(metadata.get("video_fps_nominal", 30.0)) / inference_fps))
    )

    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            for video_position in range(start_position, end_position + 1):
                capture_idx, unix_ms = timestamps[video_position]

                if decode_sampled_frames_only:
                    if not cap.grab():
                        raise RuntimeError(f"Failed to grab {subject} RGB frame at {video_position}")
                    if unix_ms + 1e-9 < next_sample_unix_ms:
                        continue
                    ok, frame = cap.retrieve()
                else:
                    ok, frame = cap.read()
                    if unix_ms + 1e-9 < next_sample_unix_ms:
                        continue
                if not ok or frame is None:
                    raise RuntimeError(f"Failed to decode {subject} RGB frame at {video_position}")

                while next_sample_unix_ms <= unix_ms:
                    next_sample_unix_ms += inference_interval_ms

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                relative_timestamp_ms = int(unix_ms - analysis_start)
                result = landmarker.detect_for_video(mp_image, relative_timestamp_ms)

                phase = _phase_at(unix_ms, intervals)
                block = _block_from_phase(phase)
                behavior = empty_behavior_context()
                if block is not None and behavior_indexes.get(block) is not None:
                    behavior = behavior_indexes[block].context_at(unix_ms, trial_duration_ms=trial_duration_ms)
                base: dict[str, object] = {
                    "subject": subject,
                    "video_frame_position": video_position,
                    "capture_frame_idx": capture_idx,
                    "unix_ms": unix_ms,
                    "pose_timestamp_ms": relative_timestamp_ms,
                    "phase": phase,
                    "block": block,
                    "capture_gap_before": bool(
                        previous_sample_capture_idx is not None
                        and capture_idx - previous_sample_capture_idx > expected_capture_step + 1
                    ),
                    "timestamp_gap_before": bool(
                        previous_sample_unix_ms is not None
                        and unix_ms - previous_sample_unix_ms > int(pose_cfg.get("feature_gap_reset_ms", 300))
                    ),
                }
                base.update(behavior)
                flattened = pose_result_rows(result, base=base)
                rows.extend(flattened)
                sampled_frames += 1
                pose_count = len(getattr(result, "pose_landmarks", []) or [])
                if pose_count > 0:
                    valid_frames += 1
                if pose_count > 1:
                    multi_pose_frames += 1
                previous_sample_capture_idx = capture_idx
                previous_sample_unix_ms = unix_ms
    finally:
        cap.release()

    table = pd.DataFrame(rows)
    table.to_parquet(output_path, index=False, engine="pyarrow", compression="zstd")
    elapsed_sec = time.perf_counter() - started_clock
    finished_utc = datetime.now(timezone.utc).isoformat()

    pose_rows = table[table["pose_valid"].fillna(False).astype(bool)] if not table.empty else table
    visibility = pd.to_numeric(pose_rows.get("visibility"), errors="coerce") if not pose_rows.empty else pd.Series(dtype=float)
    presence = pd.to_numeric(pose_rows.get("presence"), errors="coerce") if not pose_rows.empty else pd.Series(dtype=float)
    phase_frame_counts = (
        table.drop_duplicates("video_frame_position")["phase"].value_counts().to_dict()
        if not table.empty else {}
    )
    try:
        mp_version = str(mp.__version__)
    except Exception:
        mp_version = None

    manifest = {
        "schema_version": str(pose_cfg.get("schema_version", POSE_SCHEMA_VERSION)),
        "stage": "pose-test",
        "subject": subject,
        "output_mode": "test",
        "run_started_utc": started_utc,
        "run_finished_utc": finished_utc,
        "elapsed_sec": elapsed_sec,
        "processing_inference_fps": sampled_frames / elapsed_sec if elapsed_sec > 0 else None,
        "attention_analysis_git_commit": _git_commit(config),
        "model": {
            "backend": "mediapipe_tasks_pose_landmarker",
            "path": str(model_path),
            "url": model_url,
            "sha256": _sha256(model_path),
            "size_bytes": int(model_path.stat().st_size),
            "num_poses": int(pose_cfg.get("num_poses", 2)),
            "output_segmentation_masks": False,
        },
        "parameters": {
            "requested_inference_fps": inference_fps,
            "decode_sampled_frames_only": decode_sampled_frames_only,
            "formal_landmark_scope": str(pose_cfg.get("formal_landmark_scope", "upper_body")),
            "min_pose_detection_confidence": float(pose_cfg.get("min_pose_detection_confidence", 0.5)),
            "min_pose_presence_confidence": float(pose_cfg.get("min_pose_presence_confidence", 0.5)),
            "min_tracking_confidence": float(pose_cfg.get("min_tracking_confidence", 0.5)),
            "trial_duration_ms": trial_duration_ms,
            "raw_retention": "all returned poses; 33 normalized + world landmarks with visibility/presence",
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "mediapipe": mp_version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "video_decode": "grab_all_retrieve_sampled" if decode_sampled_frames_only else "decode_all",
        },
        "source": {
            "video": str(files.video),
            "timestamps": str(files.timestamps),
            "master_timeline": str(files.master_timeline),
            "block1_behavior": str(files.block1_behavior),
            "block2_behavior": str(files.block2_behavior),
        },
        "video_metadata": metadata,
        "analysis_span": {
            "requested_start_unix_ms": analysis_start,
            "requested_end_unix_ms": analysis_end,
            "first_video_frame_position": start_position,
            "last_video_frame_position": end_position,
        },
        "output": {
            "parquet": str(output_path),
            "manifest": str(manifest_path),
            "sampled_frames": sampled_frames,
            "frames_with_pose": valid_frames,
            "pose_valid_fraction": valid_frames / sampled_frames if sampled_frames else None,
            "frames_with_multiple_poses": multi_pose_frames,
            "landmark_rows": int(len(table)),
            "mean_visibility": float(visibility.mean()) if not visibility.empty else None,
            "mean_presence": float(presence.mean()) if not presence.empty else None,
            "phase_sampled_frames": {str(k): int(v) for k, v in phase_frame_counts.items()},
            "parquet_size_bytes": int(output_path.stat().st_size),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
