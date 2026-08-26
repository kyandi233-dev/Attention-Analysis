"""Single-pass formal RGB runner for Motion, Pose (10 Hz), and Face (15 Hz CUDA).

The runner decodes the original AVI once over the formal analysis span. Motion is
computed on every decoded frame; sampled Pose and Face frames are dispatched from
the same pass. Raw model outputs are retained and gap/QC flags are written before
any downstream filtering.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from bisect import bisect_left, bisect_right
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from attention_pipeline.config import load_config
from attention_pipeline.rgb.audit import read_rgb_timestamps, video_metadata
from attention_pipeline.rgb.behavior import BehaviorIndex, empty_behavior_context
from attention_pipeline.rgb.discover import discover_rgb_subjects
from attention_pipeline.rgb.motion import measure_motion_pair
from attention_pipeline.rgb.paths import RGBOutputLayout
from attention_pipeline.rgb.pose import (
    DEFAULT_MODEL_URL, LANDMARK_NAMES, _ensure_model, _find_subject as _find_pose_subject,
    pose_result_rows,
)
from attention_pipeline.rgb.timeline import detailed_rgb_intervals, formal_analysis_span


def _phase_at(unix_ms: int, intervals) -> str:
    for interval in intervals:
        if interval.start_unix_ms <= unix_ms < interval.end_unix_ms:
            return interval.phase
    return intervals[-1].phase if intervals and unix_ms == intervals[-1].end_unix_ms else "outside_analysis_span"


def _block(phase: str) -> int | None:
    return int(phase[5:]) if phase.startswith("block") and phase[5:].isdigit() else None


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_subject(config, subject):
    records, duplicates = discover_rgb_subjects(config)
    if subject in duplicates:
        raise RuntimeError(f"duplicate subject across roots: {subject}")
    for record in records:
        if record.subject == subject:
            return record
    raise FileNotFoundError(subject)


_MOTION_FLOAT_COLUMNS = {
    "dt_ms", "gray_mean", "gray_std", "gray_min", "gray_max", "gray_mean_delta",
    "dt_multiple_of_median", "gap_duration_ms", "mean_abs_difference", "std_abs_difference",
    "changed_pixel_ratio", "global_motion_energy", "global_motion_energy_per_sec",
    "stimulus_size", "rt", "probe_rt", "probe_vigilance_rt", "block_onset_time", "rest_duration",
}
_MOTION_INT_COLUMNS = {
    "video_frame_position", "capture_frame_idx", "unix_ms", "capture_missing_frame_indices_before",
    "pixel_diff_threshold", "block", "trial_num", "cycle_num", "position_in_cycle", "is_no_go",
    "response", "correct", "commission", "omission", "is_probe", "probe_response", "probe_vigilance",
    "absolute_onset_time", "response_time", "probe_onset_time", "probe_response_time",
    "trial_onset_unix_ms", "time_from_trial_onset_ms", "next_trial_onset_unix_ms",
    "time_to_next_trial_onset_ms", "probe_onset_unix_ms", "probe_response_unix_ms",
}
_MOTION_BOOL_COLUMNS = {"irregular_dt", "gap_before", "motion_valid", "trial_active", "probe_active"}
_STRING_COLUMNS = {"subject", "phase", "condition", "stimulus_name", "gap_reason", "behavior_state", "landmark_name"}


def _canonicalize_dataframe(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Apply stable nullable dtypes before the first Parquet schema is created."""
    frame = frame.copy()
    for column in frame.columns:
        name = str(column)
        if name in _STRING_COLUMNS:
            frame[column] = frame[column].astype("string")
        elif kind == "motion" and name in _MOTION_BOOL_COLUMNS:
            frame[column] = frame[column].astype("boolean")
        elif kind == "motion" and name in _MOTION_INT_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
        elif kind == "motion" and name in _MOTION_FLOAT_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Float64")
        elif kind == "pose":
            if name in {"video_frame_position", "capture_frame_idx", "unix_ms", "pose_timestamp_ms", "block", "pose_count", "pose_index", "landmark_index"}:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
            elif name == "pose_valid":
                frame[column] = frame[column].astype("boolean")
            else:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Float64")
        elif kind == "face":
            if name in {"frame", "local_sample_index", "video_frame_position", "capture_frame_idx", "unix_ms", "block", "face_rank"}:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
            elif name == "detected":
                frame[column] = frame[column].astype("boolean")
            else:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Float64")
    return frame


def _face_flush(detector, tensors, metas, writer_state):
    if not tensors:
        return
    import torch
    batch = torch.stack(tensors, dim=0)
    fex = detector.detect(batch, data_type="tensor", batch_size=len(tensors), num_workers=0,
                          pin_memory=False, face_detection_threshold=0.5, progress_bar=False)
    native = pd.DataFrame(fex).copy()
    native = native.drop(columns=[c for c in native.columns if str(c).startswith("Identity")], errors="ignore")
    native["local_sample_index"] = pd.to_numeric(native["frame"], errors="raise").astype(int)
    native["video_frame_position"] = native["local_sample_index"].map(lambda i: metas[i]["video_frame_position"])
    native["capture_frame_idx"] = native["local_sample_index"].map(lambda i: metas[i]["capture_frame_idx"])
    native["unix_ms"] = native["local_sample_index"].map(lambda i: metas[i]["unix_ms"])
    native["subject"] = metas[0]["subject"]
    native["phase"] = native["local_sample_index"].map(lambda i: metas[i]["phase"])
    native["block"] = pd.array(
        [metas[i]["block"] for i in native["local_sample_index"]], dtype="Int64"
    )
    native["face_rank"] = native.groupby("video_frame_position", sort=False).cumcount().astype(int)
    for col in ["FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight"]:
        native[col] = pd.to_numeric(native[col], errors="coerce")
    native["detected"] = native[["FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight"]].notna().all(axis=1)
    native["rf_bbox_x1"] = native["FaceRectX"]
    native["rf_bbox_y1"] = native["FaceRectY"]
    native["rf_bbox_x2"] = native["FaceRectX"] + native["FaceRectWidth"]
    native["rf_bbox_y2"] = native["FaceRectY"] + native["FaceRectHeight"]
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(_canonicalize_dataframe(native, "face"), preserve_index=False)
    if writer_state["writer"] is None:
        writer_state["writer"] = pq.ParquetWriter(writer_state["path"], table.schema, compression="zstd")
    writer_state["writer"].write_table(table)
    writer_state["rows"] += len(native)
    tensors.clear(); metas.clear()


def _table_flush(rows, writer_state, kind, chunk_rows=2048):
    if not rows:
        return 0
    table = pa.Table.from_pandas(_canonicalize_dataframe(pd.DataFrame(rows), kind), preserve_index=False)
    if writer_state["writer"] is None:
        writer_state["schema"] = table.schema
        writer_state["writer"] = pq.ParquetWriter(writer_state["path"], table.schema, compression="zstd")
    else:
        table = table.cast(writer_state["schema"], safe=False)
    writer_state["writer"].write_table(table)
    n = len(rows)
    rows.clear()
    return n


def run(config, subject: str, device: str, face_batch: int) -> dict[str, object]:
    if not device.lower().startswith("cuda"):
        raise ValueError("formal NVIDIA runner requires CUDA; CPU fallback is forbidden")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent CPU fallback")
    from feat import Detectorv2
    import mediapipe as mp

    files = _find_subject(config, subject)
    timestamps = read_rgb_timestamps(files.timestamps)
    metadata = video_metadata(files.video)
    if not metadata["video_open_ok"] or int(metadata["video_frame_count_nominal"]) != len(timestamps):
        raise RuntimeError("AVI/timestamp structural gate failed")
    layout = RGBOutputLayout.from_config(config)
    out_dir = layout.root / subject
    out_dir.mkdir(parents=True, exist_ok=True)
    focus = config.section("focuswave"); pose_cfg = config.section("pose"); motion_cfg = config.section("motion")
    intervals = detailed_rgb_intervals(
        files.master_timeline,
        baseline_duration_sec=float(focus.get("baseline_duration_sec", 180)),
        expected_blocks=int(focus.get("expected_blocks", 2)),
    )
    start_ms, end_ms = formal_analysis_span(
        files.master_timeline,
        baseline_duration_sec=float(focus.get("baseline_duration_sec", 180)),
        expected_blocks=int(focus.get("expected_blocks", 2)),
    )
    all_times = [x[1] for x in timestamps]; start = bisect_left(all_times, start_ms); end = bisect_right(all_times, end_ms) - 1
    if end < start: raise RuntimeError("no formal RGB frames")
    positive = [b[1] - a[1] for a, b in zip(timestamps[start:end+1], timestamps[start+1:end+1]) if b[1] > a[1]]
    median_dt = float(np.median(positive)) if positive else None
    behaviors = {1: BehaviorIndex.from_csv(files.block1_behavior), 2: BehaviorIndex.from_csv(files.block2_behavior)}
    detector = Detectorv2(device=device, identity_model=None)
    pose_model = Path(str(pose_cfg.get("model_path", "_test/pose_landmarker_lite.task")))
    if not pose_model.is_absolute(): pose_model = layout.root / pose_model
    pose_model = _ensure_model(pose_model, str(pose_cfg.get("model_url", DEFAULT_MODEL_URL)))
    from attention_pipeline.rgb.pose import _native_model_path
    BaseOptions = mp.tasks.BaseOptions; PoseLandmarker = mp.tasks.vision.PoseLandmarker
    Options = mp.tasks.vision.PoseLandmarkerOptions; RunningMode = mp.tasks.vision.RunningMode
    options = Options(base_options=BaseOptions(model_asset_path=_native_model_path(pose_model)), running_mode=RunningMode.VIDEO,
        num_poses=int(pose_cfg.get("num_poses", 2)), min_pose_detection_confidence=float(pose_cfg.get("min_pose_detection_confidence", .5)),
        min_pose_presence_confidence=float(pose_cfg.get("min_pose_presence_confidence", .5)), min_tracking_confidence=float(pose_cfg.get("min_tracking_confidence", .5)), output_segmentation_masks=False)
    motion_rows=[]; pose_rows=[]; tensors=[]; metas=[]
    motion_path = out_dir/f"{subject}_motion_raw.parquet"; pose_path = out_dir/f"{subject}_pose_landmarks.parquet"; face_path = out_dir/f"{subject}_face_raw.parquet"
    motion_state = {"path": str(motion_path), "writer": None, "schema": None, "rows": 0}
    pose_state = {"path": str(pose_path), "writer": None, "schema": None, "rows": 0}
    face_state = {"path": str(face_path), "writer": None, "rows": 0}
    previous_gray=None; previous_mean=None; previous_capture=None; previous_ms=None
    next_pose=float(start_ms); next_face=float(start_ms); pose_step=1000.0/float(pose_cfg.get("inference_fps",10.0)); face_step=1000.0/15.0
    cap=cv2.VideoCapture(str(files.video));
    if not cap.isOpened(): raise RuntimeError(f"cannot open {files.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))
    started=time.perf_counter()
    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            for position in range(start, end+1):
                ok, frame=cap.read()
                if not ok or frame is None: raise RuntimeError(f"decode failed at {position}")
                capture_idx, unix_ms=timestamps[position]; phase=_phase_at(unix_ms, intervals); block=_block(phase)
                behavior=behaviors[block].context_at(unix_ms, trial_duration_ms=int(focus.get("trial_duration_ms",1150))) if block in behaviors else empty_behavior_context()
                gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); dt=unix_ms-previous_ms if previous_ms is not None else None
                measure=measure_motion_pair(gray, previous_gray, dt_ms=dt, median_interval_ms=median_dt, previous_capture_idx=previous_capture, current_capture_idx=capture_idx, previous_gray_mean=previous_mean, gap_reset_ms=int(motion_cfg.get("gap_reset_ms",100)), irregular_dt_multiple=float(motion_cfg.get("irregular_dt_multiple",1.5)), pixel_diff_threshold=int(motion_cfg.get("pixel_diff_threshold",15)))
                base={"subject":subject,"video_frame_position":position,"capture_frame_idx":capture_idx,"unix_ms":unix_ms,"dt_ms":dt,"phase":phase,"block":block}; base.update(behavior); base.update(measure); motion_rows.append(base)
                if len(motion_rows) >= 2048: motion_state["rows"] += _table_flush(motion_rows, motion_state, "motion")
                if unix_ms + 1e-9 >= next_pose:
                    rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); result=landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb)), int(unix_ms-start_ms)); pbase={"subject":subject,"video_frame_position":position,"capture_frame_idx":capture_idx,"unix_ms":unix_ms,"pose_timestamp_ms":int(unix_ms-start_ms),"phase":phase,"block":block}; pbase.update(behavior); pose_rows.extend(pose_result_rows(result,base=pbase)); next_pose += pose_step
                    if len(pose_rows) >= 2048: pose_state["rows"] += _table_flush(pose_rows, pose_state, "pose")
                if unix_ms + 1e-9 >= next_face:
                    tensors.append(torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).transpose(2,0,1).copy())); metas.append({"subject":subject,"video_frame_position":position,"capture_frame_idx":capture_idx,"unix_ms":unix_ms,"phase":phase,"block":block}); next_face += face_step
                    if len(tensors) >= face_batch: _face_flush(detector,tensors,metas,face_state)
                previous_gray=gray; previous_mean=float(measure["gray_mean"]); previous_capture=capture_idx; previous_ms=unix_ms
    finally:
        cap.release()
    _face_flush(detector,tensors,metas,face_state)
    if face_state["writer"] is not None:
        face_state["writer"].close()
    motion_state["rows"] += _table_flush(motion_rows, motion_state, "motion"); pose_state["rows"] += _table_flush(pose_rows, pose_state, "pose")
    for state in (motion_state, pose_state):
        if state["writer"] is not None: state["writer"].close()
    manifest={"schema_version":"rgb-formal-full-runner-v1","subject":subject,"status":"complete","source_video":str(files.video),"source_timestamps":str(files.timestamps),"analysis_span":{"start_unix_ms":start_ms,"end_unix_ms":end_ms,"first_video_frame_position":start,"last_video_frame_position":end},"single_pass_decode":True,"cadence":{"motion":"full_fps","pose_hz":float(pose_cfg.get("inference_fps",10.0)),"face_hz":15.0},"gap_policy":"retain raw rows and reset temporal derived values after timestamp/capture gaps","runtime":{"torch":torch.__version__,"torch_cuda":torch.version.cuda,"cuda_available":bool(torch.cuda.is_available()),"gpu":torch.cuda.get_device_name(0),"py_feat":"2.1.1","mediapipe":mp.__version__},"outputs":{"motion":str(motion_path),"pose":str(pose_path),"face":str(face_path),"motion_rows":motion_state["rows"],"pose_rows":pose_state["rows"],"face_rows":face_state["rows"]},"elapsed_sec":time.perf_counter()-started}
    (out_dir/f"{subject}_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8"); return manifest


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/rgb_analysis.yaml"); p.add_argument("--subject",required=True); p.add_argument("--device",default="cuda"); p.add_argument("--face-batch",type=int,default=8); a=p.parse_args(); print(json.dumps(run(load_config(a.config),a.subject,a.device,a.face_batch),ensure_ascii=False,indent=2))
