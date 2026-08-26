from __future__ import annotations

import argparse
import json
import queue
import struct
import subprocess
import threading
import time
from bisect import bisect_left, bisect_right
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from attention_pipeline.config import Config, load_config
from attention_pipeline.rgb.audit import read_rgb_timestamps, video_metadata
from attention_pipeline.rgb.behavior import BehaviorIndex, empty_behavior_context
from attention_pipeline.rgb.paths import RGBOutputLayout
from attention_pipeline.rgb.timeline import detailed_rgb_intervals, formal_analysis_span
from attention_pipeline.rgb import motion as motion_mod
from attention_pipeline.rgb import pose as pose_mod


def _complete_manifest(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("completion_status") == "complete"
    except Exception:
        return False


def _stage_paths(layout: RGBOutputLayout, subject: str, stage: str) -> tuple[Path, Path]:
    if stage == "motion":
        return (
            layout.subject_file(subject, "motion_raw.parquet"),
            layout.subject_file(subject, "motion_manifest.json"),
        )
    if stage == "pose":
        return (
            layout.subject_file(subject, "pose_landmarks.parquet"),
            layout.subject_file(subject, "pose_manifest.json"),
        )
    if stage == "face":
        return (
            layout.subject_file(subject, "face_raw.parquet"),
            layout.subject_file(subject, "face_raw_manifest.json"),
        )
    raise ValueError(stage)


def _stage_complete(layout: RGBOutputLayout, subject: str, stage: str) -> bool:
    raw, manifest = _stage_paths(layout, subject, stage)
    return raw.is_file() and _complete_manifest(manifest)


def _guard_partial(layout: RGBOutputLayout, subject: str, stage: str, *, force: bool) -> None:
    if force:
        return
    raw, manifest = _stage_paths(layout, subject, stage)
    if _stage_complete(layout, subject, stage):
        return
    existing = [str(path) for path in (raw, manifest) if path.exists()]
    if existing:
        raise RuntimeError(
            f"{stage} has partial formal output. Inspect first or rerun with --force: {existing}"
        )


def _context(
    unix_ms: int,
    *,
    intervals,
    behavior_indexes: dict[int, BehaviorIndex | None],
    trial_duration_ms: int,
) -> tuple[str, int | None, dict[str, object]]:
    phase = motion_mod._phase_at(unix_ms, intervals)
    block = motion_mod._block_from_phase(phase)
    behavior = empty_behavior_context()
    if block is not None and behavior_indexes.get(block) is not None:
        behavior = behavior_indexes[block].context_at(
            unix_ms, trial_duration_ms=trial_duration_ms
        )
    return phase, block, behavior


class FacePipe:
    def __init__(
        self,
        *,
        face_python: Path,
        repo_root: Path,
        config_path: str,
        subject: str,
        model_dir: str,
        height: int,
        width: int,
        force: bool,
        log_dir: Path,
        queue_size: int = 8,
    ) -> None:
        self.q: queue.Queue = queue.Queue(maxsize=max(2, int(queue_size)))
        self.error: BaseException | None = None
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.stdout_path = log_dir / f"{stamp}-shared-face.stdout.log"
        self.stderr_path = log_dir / f"{stamp}-shared-face.stderr.log"
        self.stdout_handle = self.stdout_path.open("w", encoding="utf-8")
        self.stderr_handle = self.stderr_path.open("w", encoding="utf-8")
        cmd = [
            str(face_python),
            "scripts/face_formal_stream_worker.py",
            "--config", config_path,
            "--subject", subject,
            "--model-dir", model_dir,
            "--height", str(height),
            "--width", str(width),
        ]
        if force:
            cmd.append("--force")
        self.proc = subprocess.Popen(
            cmd,
            cwd=repo_root,
            stdin=subprocess.PIPE,
            stdout=self.stdout_handle,
            stderr=self.stderr_handle,
        )
        if self.proc.stdin is None:
            raise RuntimeError("Face worker stdin unavailable")
        self.thread = threading.Thread(target=self._sender, daemon=True)
        self.thread.start()

    def _sender(self) -> None:
        try:
            assert self.proc.stdin is not None
            while True:
                item = self.q.get()
                if item is None:
                    break
                meta, frame = item
                meta_bytes = json.dumps(
                    meta, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
                self.proc.stdin.write(struct.pack("<I", len(meta_bytes)))
                self.proc.stdin.write(meta_bytes)
                self.proc.stdin.write(memoryview(np.ascontiguousarray(frame)))
            self.proc.stdin.close()
        except BaseException as exc:
            self.error = exc
            try:
                if self.proc.stdin is not None:
                    self.proc.stdin.close()
            except Exception:
                pass

    def send(self, meta: dict[str, Any], frame: np.ndarray) -> None:
        if self.error is not None:
            raise RuntimeError(f"Face stream sender failed: {self.error}")
        self.q.put((meta, frame))

    def finish(self) -> None:
        self.q.put(None)
        self.thread.join()
        if self.error is not None:
            raise RuntimeError(f"Face stream sender failed: {self.error}")
        code = self.proc.wait()
        self.stdout_handle.close()
        self.stderr_handle.close()
        if code != 0:
            stderr_tail = ""
            try:
                stderr_tail = "\n".join(
                    self.stderr_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()[-40:]
                )
            except Exception:
                pass
            raise RuntimeError(
                f"Face shared-decode worker failed with exit code {code}\n{stderr_tail}"
            )


class PoseThread:
    def __init__(
        self,
        *,
        config: Config,
        subject: str,
        analysis_start: int,
        intervals,
        behavior_indexes: dict[int, BehaviorIndex | None],
        trial_duration_ms: int,
        output_path: Path,
    ) -> None:
        self.config = config
        self.subject = subject
        self.analysis_start = int(analysis_start)
        self.intervals = intervals
        self.behavior_indexes = behavior_indexes
        self.trial_duration_ms = int(trial_duration_ms)
        self.output_path = output_path
        self.q: queue.Queue = queue.Queue(maxsize=8)
        self.error: BaseException | None = None
        self.rows: list[dict[str, object]] = []
        self.sampled_frames = 0
        self.valid_frames = 0
        self.multi_pose_frames = 0
        self.started_utc: str | None = None
        self.finished_utc: str | None = None
        self.elapsed_sec: float | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def put(self, item: tuple[int, int, int, np.ndarray]) -> None:
        if self.error is not None:
            raise RuntimeError(f"Pose worker failed: {self.error}")
        self.q.put(item)

    def _run(self) -> None:
        started_clock = time.perf_counter()
        self.started_utc = datetime.now(timezone.utc).isoformat()
        try:
            import mediapipe as mp

            pose_cfg = self.config.section("pose")
            layout = RGBOutputLayout.from_config(self.config)
            model_path_raw = str(
                pose_cfg.get("model_path", "_test/pose_landmarker_lite.task")
            )
            model_path = Path(model_path_raw)
            if not model_path.is_absolute():
                model_path = layout.root / model_path
            model_path = pose_mod._ensure_model(
                model_path,
                str(pose_cfg.get("model_url", pose_mod.DEFAULT_MODEL_URL)),
            )

            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_poses=int(pose_cfg.get("num_poses", 2)),
                min_pose_detection_confidence=float(
                    pose_cfg.get("min_pose_detection_confidence", 0.5)
                ),
                min_pose_presence_confidence=float(
                    pose_cfg.get("min_pose_presence_confidence", 0.5)
                ),
                min_tracking_confidence=float(
                    pose_cfg.get("min_tracking_confidence", 0.5)
                ),
                output_segmentation_masks=False,
            )

            with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
                while True:
                    item = self.q.get()
                    if item is None:
                        break
                    video_position, capture_idx, unix_ms, frame = item
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=np.ascontiguousarray(rgb),
                    )
                    relative_timestamp_ms = int(unix_ms - self.analysis_start)
                    result = landmarker.detect_for_video(
                        mp_image, relative_timestamp_ms
                    )
                    phase, block, behavior = _context(
                        unix_ms,
                        intervals=self.intervals,
                        behavior_indexes=self.behavior_indexes,
                        trial_duration_ms=self.trial_duration_ms,
                    )
                    base: dict[str, object] = {
                        "subject": self.subject,
                        "video_frame_position": video_position,
                        "capture_frame_idx": capture_idx,
                        "unix_ms": unix_ms,
                        "pose_timestamp_ms": relative_timestamp_ms,
                        "phase": phase,
                        "block": block,
                    }
                    base.update(behavior)
                    self.rows.extend(pose_mod.pose_result_rows(result, base=base))
                    self.sampled_frames += 1
                    pose_count = len(getattr(result, "pose_landmarks", []) or [])
                    if pose_count > 0:
                        self.valid_frames += 1
                    if pose_count > 1:
                        self.multi_pose_frames += 1

            table = pd.DataFrame(self.rows)
            table.to_parquet(
                self.output_path,
                index=False,
                engine="pyarrow",
                compression="zstd",
            )
            self.finished_utc = datetime.now(timezone.utc).isoformat()
            self.elapsed_sec = time.perf_counter() - started_clock
        except BaseException as exc:
            self.error = exc

    def finish(self) -> None:
        self.q.put(None)
        self.thread.join()
        if self.error is not None:
            raise RuntimeError(f"Pose shared-decode worker failed: {self.error}")


def _write_motion_manifest(
    *,
    config: Config,
    subject: str,
    files,
    metadata: dict[str, Any],
    table: pd.DataFrame,
    output_path: Path,
    manifest_path: Path,
    analysis_start: int,
    analysis_end: int,
    start_position: int,
    end_position: int,
    median_interval_ms: float | None,
    elapsed_sec: float,
    started_utc: str,
    finished_utc: str,
    gap_reset_ms: int,
    irregular_dt_multiple: float,
    pixel_diff_threshold: int,
    trial_duration_ms: int,
) -> None:
    motion_cfg = config.section("motion")
    focuswave = config.section("focuswave")
    manifest = {
        "schema_version": str(
            motion_cfg.get("schema_version", motion_mod.MOTION_SCHEMA_VERSION)
        ),
        "stage": "motion-formal",
        "subject": subject,
        "output_mode": "formal",
        "run_started_utc": started_utc,
        "run_finished_utc": finished_utc,
        "elapsed_sec": elapsed_sec,
        "processing_fps": len(table) / elapsed_sec if elapsed_sec > 0 else None,
        "attention_analysis_git_commit": motion_mod._git_commit(config),
        "config_path": str(config.path),
        "config_sha256": motion_mod._config_digest(config),
        "focuswave_provenance": {
            "repository": focuswave.get("repository"),
            "branch": focuswave.get("branch"),
            "accepted_formal_versions": focuswave.get("accepted_formal_versions"),
            "formal_structure": focuswave.get("formal_structure"),
        },
        "source": {
            "video": motion_mod._file_info(files.video),
            "timestamps": motion_mod._file_info(files.timestamps),
            "master_timeline": motion_mod._file_info(files.master_timeline),
            "block1_behavior": motion_mod._file_info(files.block1_behavior),
            "block2_behavior": motion_mod._file_info(files.block2_behavior),
        },
        "video_metadata": metadata,
        "analysis_span": {
            "requested_start_unix_ms": analysis_start,
            "requested_end_unix_ms": analysis_end,
            "first_output_unix_ms": int(table["unix_ms"].iloc[0]) if not table.empty else None,
            "last_output_unix_ms": int(table["unix_ms"].iloc[-1]) if not table.empty else None,
            "first_video_frame_position": start_position,
            "last_video_frame_position": end_position,
            "median_interval_ms": median_interval_ms,
        },
        "parameters": {
            "process_full_fps": True,
            "gap_reset_ms": gap_reset_ms,
            "irregular_dt_multiple": irregular_dt_multiple,
            "pixel_diff_threshold": pixel_diff_threshold,
            "trial_duration_ms": trial_duration_ms,
            "motion_definition": "mean absolute grayscale frame difference / 255",
            "motion_rate_definition": "global_motion_energy / dt_seconds",
            "gap_policy": "retain current frame; adjacent-frame metrics missing after timestamp/capture gap",
        },
        "runtime": {
            "python": __import__("platform").python_version(),
            "platform": __import__("platform").platform(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "video_seek_mode": "opencv_seek",
            "parquet_engine": "pyarrow",
            "parquet_compression": "zstd",
            "decode_mode": "single_shared_decode",
        },
        "output": {
            "parquet": str(output_path),
            "manifest": str(manifest_path),
            "rows": int(len(table)),
            "motion_valid_rows": int(
                table["motion_valid"].fillna(False).astype(bool).sum()
            ) if not table.empty else 0,
            "gap_reset_rows": int(
                table["gap_before"].fillna(False).astype(bool).sum()
            ) if not table.empty else 0,
            "irregular_dt_rows": int(
                table["irregular_dt"].fillna(False).astype(bool).sum()
            ) if not table.empty else 0,
            "phase_rows": {
                str(k): int(v) for k, v in table["phase"].value_counts().to_dict().items()
            } if not table.empty else {},
            "parquet_size_bytes": int(output_path.stat().st_size),
        },
        "completion_status": "complete",
        "engine_origin": "single shared AVI decode; motion measurement uses validated measure_motion_pair implementation",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_pose_manifest(
    *,
    config: Config,
    subject: str,
    files,
    metadata: dict[str, Any],
    output_path: Path,
    manifest_path: Path,
    analysis_start: int,
    analysis_end: int,
    start_position: int,
    end_position: int,
    worker: PoseThread,
) -> None:
    pose_cfg = config.section("pose")
    table = pd.read_parquet(
        output_path,
        columns=["video_frame_position", "pose_valid", "visibility", "presence", "phase"],
    )
    pose_rows = (
        table[table["pose_valid"].fillna(False).astype(bool)]
        if not table.empty else table
    )
    visibility = (
        pd.to_numeric(pose_rows.get("visibility"), errors="coerce")
        if not pose_rows.empty else pd.Series(dtype=float)
    )
    presence = (
        pd.to_numeric(pose_rows.get("presence"), errors="coerce")
        if not pose_rows.empty else pd.Series(dtype=float)
    )
    phase_frame_counts = (
        table.drop_duplicates("video_frame_position")["phase"].value_counts().to_dict()
        if not table.empty else {}
    )
    model_path_raw = str(pose_cfg.get("model_path", "_test/pose_landmarker_lite.task"))
    model_path = Path(model_path_raw)
    layout = RGBOutputLayout.from_config(config)
    if not model_path.is_absolute():
        model_path = layout.root / model_path
    manifest = {
        "schema_version": str(pose_cfg.get("schema_version", pose_mod.POSE_SCHEMA_VERSION)),
        "stage": "pose-formal",
        "subject": subject,
        "output_mode": "formal",
        "run_started_utc": worker.started_utc,
        "run_finished_utc": worker.finished_utc,
        "elapsed_sec": worker.elapsed_sec,
        "processing_inference_fps": (
            worker.sampled_frames / worker.elapsed_sec if worker.elapsed_sec else None
        ),
        "attention_analysis_git_commit": pose_mod._git_commit(config),
        "model": {
            "backend": "mediapipe_tasks_pose_landmarker",
            "path": str(model_path),
            "url": str(pose_cfg.get("model_url", pose_mod.DEFAULT_MODEL_URL)),
            "sha256": pose_mod._sha256(model_path),
            "size_bytes": int(model_path.stat().st_size),
            "num_poses": int(pose_cfg.get("num_poses", 2)),
            "output_segmentation_masks": False,
        },
        "parameters": {
            "requested_inference_fps": float(pose_cfg.get("inference_fps", 10.0)),
            "decode_sampled_frames_only": True,
            "formal_landmark_scope": str(pose_cfg.get("formal_landmark_scope", "upper_body")),
            "min_pose_detection_confidence": float(
                pose_cfg.get("min_pose_detection_confidence", 0.5)
            ),
            "min_pose_presence_confidence": float(
                pose_cfg.get("min_pose_presence_confidence", 0.5)
            ),
            "min_tracking_confidence": float(
                pose_cfg.get("min_tracking_confidence", 0.5)
            ),
            "trial_duration_ms": int(
                config.section("focuswave").get("trial_duration_ms", 1150)
            ),
            "raw_retention": "all returned poses; 33 normalized + world landmarks with visibility/presence",
        },
        "runtime": {
            "python": __import__("platform").python_version(),
            "platform": __import__("platform").platform(),
            "opencv": cv2.__version__,
            "mediapipe": None,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "video_decode": "single_shared_decode",
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
            "sampled_frames": worker.sampled_frames,
            "frames_with_pose": worker.valid_frames,
            "pose_valid_fraction": (
                worker.valid_frames / worker.sampled_frames
                if worker.sampled_frames else None
            ),
            "frames_with_multiple_poses": worker.multi_pose_frames,
            "landmark_rows": int(len(table)),
            "mean_visibility": float(visibility.mean()) if not visibility.empty else None,
            "mean_presence": float(presence.mean()) if not presence.empty else None,
            "phase_sampled_frames": {
                str(k): int(v) for k, v in phase_frame_counts.items()
            },
            "parquet_size_bytes": int(output_path.stat().st_size),
        },
        "completion_status": "complete",
        "derived_features_deferred": True,
        "engine_origin": "single shared AVI decode; validated MediaPipe Pose Landmarker and pose_result_rows implementation",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_shared(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    layout = RGBOutputLayout.from_config(config)
    files = motion_mod._find_subject(config, args.subject)
    timestamps = read_rgb_timestamps(files.timestamps)
    metadata = video_metadata(files.video)
    if not metadata["video_open_ok"]:
        raise RuntimeError(f"RGB video cannot be opened: {files.video}")
    if int(metadata["video_frame_count_nominal"]) != len(timestamps):
        raise ValueError(
            f"AVI/timestamp row mismatch for {args.subject}: "
            f"video={metadata['video_frame_count_nominal']}, timestamps={len(timestamps)}"
        )

    focuswave = config.section("focuswave")
    baseline_duration_sec = float(focuswave.get("baseline_duration_sec", 180))
    expected_blocks = int(focuswave.get("expected_blocks", 2))
    trial_duration_ms = int(focuswave.get("trial_duration_ms", 1150))
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
        raise ValueError(f"No RGB frames inside formal analysis span for {args.subject}")

    behavior_indexes = {
        1: BehaviorIndex.from_csv(files.block1_behavior),
        2: BehaviorIndex.from_csv(files.block2_behavior),
    }

    for stage in ("motion", "pose", "face"):
        _guard_partial(layout, args.subject, stage, force=args.force)

    face_needed = args.force or not _stage_complete(layout, args.subject, "face")
    pose_needed = args.force or not _stage_complete(layout, args.subject, "pose")
    motion_needed = args.force or not _stage_complete(layout, args.subject, "motion")

    if not any((face_needed, pose_needed, motion_needed)):
        return {"subject": args.subject, "status": "all_raw_complete"}

    face_schedule: dict[int, dict[str, Any]] = {}
    if face_needed:
        face_frames_path = layout.subject_file(args.subject, "face_frames.csv")
        frame_table = pd.read_csv(face_frames_path)
        for row in frame_table.to_dict(orient="records"):
            cleaned: dict[str, Any] = {}
            for key, value in row.items():
                if pd.isna(value):
                    cleaned[key] = None
                elif isinstance(value, np.generic):
                    cleaned[key] = value.item()
                else:
                    cleaned[key] = value
            face_schedule[int(cleaned["video_frame_position"])] = cleaned

    pose_cfg = config.section("pose")
    pose_interval_ms = 1000.0 / float(pose_cfg.get("inference_fps", 10.0))
    next_pose_unix_ms = float(analysis_start)

    subject_dir = layout.subject_dir(args.subject)
    log_dir = subject_dir / "_runlogs"
    log_dir.mkdir(parents=True, exist_ok=True)

    face_pipe: FacePipe | None = None
    if face_needed:
        face_pipe = FacePipe(
            face_python=Path(args.face_python),
            repo_root=config.path.parent.parent,
            config_path=args.config,
            subject=args.subject,
            model_dir=args.model_dir,
            height=int(metadata["video_height"]),
            width=int(metadata["video_width"]),
            force=args.force,
            log_dir=log_dir,
        )

    pose_worker: PoseThread | None = None
    pose_output, pose_manifest = _stage_paths(layout, args.subject, "pose")
    if pose_needed:
        if args.force:
            for path in (pose_output, pose_manifest):
                if path.exists():
                    path.unlink()
        pose_worker = PoseThread(
            config=config,
            subject=args.subject,
            analysis_start=analysis_start,
            intervals=intervals,
            behavior_indexes=behavior_indexes,
            trial_duration_ms=trial_duration_ms,
            output_path=pose_output,
        )
        pose_worker.start()

    motion_output, motion_manifest = _stage_paths(layout, args.subject, "motion")
    if motion_needed and args.force:
        for path in (motion_output, motion_manifest):
            if path.exists():
                path.unlink()

    motion_cfg = config.section("motion")
    gap_reset_ms = int(motion_cfg.get("gap_reset_ms", 100))
    irregular_dt_multiple = float(motion_cfg.get("irregular_dt_multiple", 1.5))
    pixel_diff_threshold = int(motion_cfg.get("pixel_diff_threshold", 15))
    analysis_rows = timestamps[start_position:end_position + 1]
    positive_dt = [
        b[1] - a[1]
        for a, b in zip(analysis_rows, analysis_rows[1:])
        if b[1] > a[1]
    ]
    median_interval_ms = motion_mod._median(positive_dt)

    cap = cv2.VideoCapture(str(files.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open RGB video: {files.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_position))

    motion_rows: list[dict[str, object]] = []
    previous_gray = None
    previous_gray_mean = None
    previous_capture_idx = None
    previous_unix_ms = None
    motion_started_utc = datetime.now(timezone.utc).isoformat()
    motion_started_clock = time.perf_counter()
    shared_started = time.perf_counter()

    try:
        for video_position in range(start_position, end_position + 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"Failed to read {args.subject} RGB frame at video position {video_position}"
                )
            capture_idx, unix_ms = timestamps[video_position]

            if motion_needed:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                dt_ms = (
                    unix_ms - previous_unix_ms
                    if previous_unix_ms is not None else None
                )
                measurement = motion_mod.measure_motion_pair(
                    gray,
                    previous_gray,
                    dt_ms=dt_ms,
                    median_interval_ms=median_interval_ms,
                    previous_capture_idx=previous_capture_idx,
                    current_capture_idx=capture_idx,
                    previous_gray_mean=previous_gray_mean,
                    gap_reset_ms=gap_reset_ms,
                    irregular_dt_multiple=irregular_dt_multiple,
                    pixel_diff_threshold=pixel_diff_threshold,
                )
                phase, block, behavior = _context(
                    unix_ms,
                    intervals=intervals,
                    behavior_indexes=behavior_indexes,
                    trial_duration_ms=trial_duration_ms,
                )
                row: dict[str, object] = {
                    "subject": args.subject,
                    "video_frame_position": video_position,
                    "capture_frame_idx": capture_idx,
                    "unix_ms": unix_ms,
                    "dt_ms": dt_ms,
                    "phase": phase,
                    "block": block,
                }
                row.update(behavior)
                row.update(measurement)
                motion_rows.append(row)
                previous_gray = gray
                previous_gray_mean = float(measurement["gray_mean"])
                previous_capture_idx = capture_idx
                previous_unix_ms = unix_ms

            if pose_worker is not None and unix_ms + 1e-9 >= next_pose_unix_ms:
                pose_worker.put((video_position, capture_idx, unix_ms, frame))
                while next_pose_unix_ms <= unix_ms:
                    next_pose_unix_ms += pose_interval_ms

            if face_pipe is not None:
                face_meta = face_schedule.get(video_position)
                if face_meta is not None:
                    face_pipe.send(face_meta, frame)

            if (video_position - start_position + 1) % 3000 == 0:
                elapsed = time.perf_counter() - shared_started
                print(
                    f"[shared-decode] {video_position - start_position + 1}/"
                    f"{end_position - start_position + 1} frames | {elapsed:.1f}s",
                    flush=True,
                )
    finally:
        cap.release()

    if motion_needed:
        table = pd.DataFrame(motion_rows)
        table.to_parquet(
            motion_output, index=False, engine="pyarrow", compression="zstd"
        )
        motion_elapsed = time.perf_counter() - motion_started_clock
        _write_motion_manifest(
            config=config,
            subject=args.subject,
            files=files,
            metadata=metadata,
            table=table,
            output_path=motion_output,
            manifest_path=motion_manifest,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
            start_position=start_position,
            end_position=end_position,
            median_interval_ms=median_interval_ms,
            elapsed_sec=motion_elapsed,
            started_utc=motion_started_utc,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            gap_reset_ms=gap_reset_ms,
            irregular_dt_multiple=irregular_dt_multiple,
            pixel_diff_threshold=pixel_diff_threshold,
            trial_duration_ms=trial_duration_ms,
        )

    if pose_worker is not None:
        pose_worker.finish()
        _write_pose_manifest(
            config=config,
            subject=args.subject,
            files=files,
            metadata=metadata,
            output_path=pose_output,
            manifest_path=pose_manifest,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
            start_position=start_position,
            end_position=end_position,
            worker=pose_worker,
        )

    if face_pipe is not None:
        face_pipe.finish()

    return {
        "subject": args.subject,
        "status": "complete",
        "shared_decode_wall_sec": time.perf_counter() - shared_started,
        "motion": _stage_complete(layout, args.subject, "motion"),
        "pose": _stage_complete(layout, args.subject, "pose"),
        "face": _stage_complete(layout, args.subject, "face"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experimental single-AVI-decode formal RGB raw extractor"
    )
    parser.add_argument("--config", default="configs/rgb_analysis.yaml")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--face-python", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = run_shared(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") == "complete" and not all(
        result.get(name, False) for name in ("motion", "pose", "face")
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
