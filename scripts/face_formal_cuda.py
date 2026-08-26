from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import queue
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from attention_pipeline.config import load_config
from attention_pipeline.rgb.paths import RGBOutputLayout


SCHEMA_VERSION = "rgb-face-formal-pyfeat-cuda-v1.0"
REQUIRED_PYFEAT_VERSION = "2.1.1"

_STRING_COLUMNS = {
    "schema_version", "subject", "phase", "condition", "stimulus_name",
    "behavior_state", "input",
}
_BOOL_COLUMNS = {
    "detected", "capture_gap_before", "temporal_gap", "trial_active", "probe_active",
}
_INT_COLUMNS = {
    "frame", "local_sample_index", "sample_index", "benchmark_index",
    "video_frame_position", "capture_frame_idx", "unix_ms", "target_unix_ms",
    "sample_error_ms", "block", "trial_num", "cycle_num", "position_in_cycle",
    "is_no_go", "response", "correct", "commission", "omission", "is_probe",
    "probe_response", "probe_vigilance", "absolute_onset_time", "response_time",
    "probe_onset_time", "probe_response_time", "trial_onset_unix_ms",
    "time_from_trial_onset_ms", "next_trial_onset_unix_ms",
    "time_to_next_trial_onset_ms", "probe_onset_unix_ms", "probe_response_unix_ms",
    "face_rank", "FrameHeight", "FrameWidth",
}


def _safe_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except Exception:
        return None


def _git_commit(repo_root: Path) -> str | None:
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


def _runtime_info(torch: Any) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    devices: list[dict[str, Any]] = []
    if cuda_available:
        for idx in range(int(torch.cuda.device_count())):
            props = torch.cuda.get_device_properties(idx)
            devices.append(
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "total_memory_bytes": int(props.total_memory),
                    "compute_capability": [int(props.major), int(props.minor)],
                }
            )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "py_feat": _safe_version("py-feat"),
        "torch": _safe_version("torch"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": int(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None,
        "cuda_available": cuda_available,
        "devices": devices,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
    }


def _canonicalize_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Force stable nullable dtypes before establishing a streaming Parquet schema."""
    frame = frame.copy()
    for column in frame.columns:
        name = str(column)
        if name in _STRING_COLUMNS:
            frame[column] = frame[column].astype("string")
        elif name in _BOOL_COLUMNS:
            frame[column] = frame[column].astype("boolean")
        elif name in _INT_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Float64")
    return frame


def _append_parquet_chunk(frame: pd.DataFrame, state: dict[str, Any]) -> int:
    if frame.empty:
        return 0
    canonical = _canonicalize_dataframe(frame)
    if state["writer"] is None:
        table = pa.Table.from_pandas(canonical, preserve_index=False)
        state["schema"] = table.schema
        state["columns"] = list(canonical.columns)
        state["writer"] = pq.ParquetWriter(
            state["path"], table.schema, compression="zstd"
        )
    else:
        expected = list(state["columns"])
        extras = [c for c in canonical.columns if c not in expected]
        if extras:
            raise RuntimeError(
                f"Py-Feat native schema changed after Parquet writer initialization; new columns={extras}"
            )
        canonical = canonical.reindex(columns=expected)
        table = pa.Table.from_pandas(canonical, preserve_index=False)
        table = table.cast(state["schema"], safe=False)
    state["writer"].write_table(table)
    state["rows"] += int(len(canonical))
    return int(len(canonical))


def _load_inputs(config_path: str, subject: str):
    config = load_config(config_path)
    layout = RGBOutputLayout.from_config(config)
    frames_csv = layout.subject_file(subject, "face_frames.csv")
    prepare_manifest = layout.subject_file(subject, "face_prepare_manifest.json")
    if not frames_csv.is_file():
        raise FileNotFoundError(
            f"Formal Face frame manifest not found: {frames_csv}. Run scripts/face_formal_prepare.py first."
        )
    if not prepare_manifest.is_file():
        raise FileNotFoundError(prepare_manifest)
    frames = pd.read_csv(frames_csv).sort_values("benchmark_index").reset_index(drop=True)
    required = {
        "subject", "benchmark_index", "video_frame_position", "capture_frame_idx",
        "unix_ms", "target_unix_ms", "sample_error_ms", "phase",
    }
    if frames.empty or not required.issubset(frames.columns):
        raise ValueError(
            f"Invalid formal Face frame manifest {frames_csv}; missing={sorted(required - set(frames.columns))}"
        )
    if set(frames["subject"].astype(str).unique()) != {subject}:
        raise ValueError(f"Frame manifest contains a different subject: {frames_csv}")
    meta = json.loads(prepare_manifest.read_text(encoding="utf-8"))
    source_video = Path(str(meta["source_video"])).expanduser().resolve()
    if not source_video.is_file():
        raise FileNotFoundError(source_video)
    return config, layout, frames, source_video, frames_csv, prepare_manifest


def _complete_manifest(raw_path: Path, manifest_path: Path) -> dict[str, Any] | None:
    if not raw_path.is_file() or not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return manifest if manifest.get("completion_status") == "complete" else None


def _archive_existing(paths: list[Path], subject_dir: Path, label: str) -> None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive = subject_dir / "_superseded" / f"{stamp}-{label}"
    archive.mkdir(parents=True, exist_ok=True)
    for path in existing:
        destination = archive / path.name
        suffix = 1
        while destination.exists():
            destination = archive / f"{path.stem}-{suffix}{path.suffix}"
            suffix += 1
        path.replace(destination)


def _reader_worker(
    frames: pd.DataFrame,
    source_video: Path,
    batch_size: int,
    out_queue: queue.Queue,
    timing: dict[str, float],
    stop_event: threading.Event,
    *,
    seek_threshold_frames: int,
) -> None:
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        out_queue.put(RuntimeError(f"Cannot open RGB video: {source_video}"))
        return
    last_target: int | None = None
    started = time.perf_counter()
    try:
        for start in range(0, len(frames), batch_size):
            if stop_event.is_set():
                break
            batch_df = frames.iloc[start:start + batch_size].copy()
            t0 = time.perf_counter()
            rgb_list: list[np.ndarray] = []
            for row in batch_df.itertuples(index=False):
                target = int(row.video_frame_position)
                if (
                    last_target is None
                    or target <= last_target
                    or target - last_target > seek_threshold_frames
                ):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, float(target))
                else:
                    for _ in range(target - last_target - 1):
                        if not cap.grab():
                            raise RuntimeError(f"Failed to advance RGB video before frame {target}")
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    raise RuntimeError(f"Failed to decode RGB frame {target}")
                rgb_list.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                last_target = target
            timing["decode_preprocess_cpu_sec"] += time.perf_counter() - t0
            while not stop_event.is_set():
                try:
                    out_queue.put((batch_df, rgb_list), timeout=0.2)
                    break
                except queue.Full:
                    continue
    except BaseException as exc:
        out_queue.put(exc)
    finally:
        cap.release()
        timing["reader_thread_wall_sec"] = time.perf_counter() - started
        out_queue.put(None)


def _attach_context(native: pd.DataFrame, batch_df: pd.DataFrame, frame_h: int, frame_w: int) -> tuple[pd.DataFrame, int]:
    out = native.copy()
    identity_cols = [c for c in out.columns if str(c).startswith("Identity")]
    if identity_cols:
        out = out.drop(columns=identity_cols)
    if "frame" not in out.columns:
        if out.empty:
            out["frame"] = pd.Series(dtype="Int64")
        else:
            raise ValueError("Py-Feat tensor output does not contain the expected 'frame' column")

    local = pd.to_numeric(out["frame"], errors="coerce")
    if local.notna().any():
        local_int = local.dropna().astype(int)
        if local_int.min() < 0 or local_int.max() >= len(batch_df):
            raise ValueError(
                f"Py-Feat tensor frame index outside current batch: min={local_int.min()}, max={local_int.max()}, batch={len(batch_df)}"
            )
    out["local_sample_index"] = local

    contexts = batch_df.reset_index(drop=True)
    for column in contexts.columns:
        values = contexts[column]
        out[column] = out["local_sample_index"].map(
            lambda value, series=values: series.iloc[int(value)] if pd.notna(value) else None
        )

    out["FrameHeight"] = int(frame_h)
    out["FrameWidth"] = int(frame_w)
    x = pd.to_numeric(out.get("FaceRectX"), errors="coerce")
    y = pd.to_numeric(out.get("FaceRectY"), errors="coerce")
    w = pd.to_numeric(out.get("FaceRectWidth"), errors="coerce")
    h = pd.to_numeric(out.get("FaceRectHeight"), errors="coerce")
    if isinstance(x, pd.Series) and isinstance(y, pd.Series) and isinstance(w, pd.Series) and isinstance(h, pd.Series):
        out["detected"] = x.notna() & y.notna() & w.gt(0) & h.gt(0)
        out["rf_bbox_x1"] = x
        out["rf_bbox_y1"] = y
        out["rf_bbox_x2"] = x + w
        out["rf_bbox_y2"] = y + h
    else:
        out["detected"] = False
        out["rf_bbox_x1"] = np.nan
        out["rf_bbox_y1"] = np.nan
        out["rf_bbox_x2"] = np.nan
        out["rf_bbox_y2"] = np.nan

    represented = set(
        pd.to_numeric(out["local_sample_index"], errors="coerce").dropna().astype(int).tolist()
    )
    missing_rows: list[dict[str, Any]] = []
    for local_idx in range(len(contexts)):
        if local_idx in represented:
            continue
        row = {column: contexts.iloc[local_idx][column] for column in contexts.columns}
        row.update(
            {
                "frame": local_idx,
                "local_sample_index": local_idx,
                "FrameHeight": int(frame_h),
                "FrameWidth": int(frame_w),
                "detected": False,
                "rf_bbox_x1": None,
                "rf_bbox_y1": None,
                "rf_bbox_x2": None,
                "rf_bbox_y2": None,
            }
        )
        missing_rows.append(row)
    if missing_rows:
        out = pd.concat([out, pd.DataFrame(missing_rows)], ignore_index=True, sort=False)

    out["face_rank"] = out.groupby("benchmark_index", sort=False).cumcount()
    detected_rows = int(out["detected"].fillna(False).astype(bool).sum())
    out = out.sort_values(["benchmark_index", "face_rank"]).reset_index(drop=True)
    return out, detected_rows


def run_formal(args: argparse.Namespace) -> dict[str, Any]:
    config, layout, frames, source_video, frames_csv, prepare_manifest = _load_inputs(
        args.config, args.subject
    )
    out_path = layout.subject_file(args.subject, "face_raw.parquet")
    manifest_path = layout.subject_file(args.subject, "face_raw_manifest.json")
    subject_dir = layout.subject_dir(args.subject)

    if not args.force:
        existing = _complete_manifest(out_path, manifest_path)
        if existing is not None:
            print(json.dumps(
                {
                    "status": "skipped_complete",
                    "subject": args.subject,
                    "raw_output": str(out_path),
                    "manifest": str(manifest_path),
                },
                ensure_ascii=False,
                indent=2,
            ))
            return existing
        if out_path.exists() or manifest_path.exists():
            raise RuntimeError(
                "Partial formal Face output exists. Inspect it first or rerun with --force; "
                "--force archives prior files instead of deleting them."
            )
    else:
        _archive_existing([out_path, manifest_path], subject_dir, "face-formal")

    pyfeat_version = _safe_version("py-feat")
    if pyfeat_version != REQUIRED_PYFEAT_VERSION and not args.allow_pyfeat_version_mismatch:
        raise RuntimeError(
            f"Formal NVIDIA Face freezes py-feat {REQUIRED_PYFEAT_VERSION}; found {pyfeat_version!r}."
        )

    import torch
    from feat import Detectorv2

    device = str(args.device)
    if not device.lower().startswith("cuda"):
        raise ValueError("NVIDIA formal Face requires --device cuda or cuda:<index>")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False; refusing silent CPU fallback")

    face_cfg = config.section("face")
    batch_size = int(
        args.batch_size if args.batch_size is not None else face_cfg.get("native_cuda_batch", 16)
    )
    prefetch_batches = int(
        args.prefetch_batches if args.prefetch_batches is not None else face_cfg.get("native_cuda_prefetch_batches", 2)
    )
    seek_threshold_frames = int(
        args.seek_threshold_frames if args.seek_threshold_frames is not None else face_cfg.get("seek_threshold_frames", 120)
    )
    threshold = float(face_cfg.get("detection_threshold", 0.5))
    if batch_size <= 0 or prefetch_batches <= 0:
        raise ValueError("CUDA batch and prefetch sizes must be positive")

    torch.cuda.synchronize()
    init_started = time.perf_counter()
    detector = Detectorv2(device=device, identity_model=None)
    torch.cuda.synchronize()
    init_sec = time.perf_counter() - init_started
    torch.cuda.reset_peak_memory_stats()

    timing = {
        "decode_preprocess_cpu_sec": 0.0,
        "reader_thread_wall_sec": 0.0,
        "detector_detect_cuda_sec": 0.0,
        "attach_context_cpu_sec": 0.0,
        "parquet_write_sec": 0.0,
    }
    counters = {
        "detector_calls": 0,
        "detected_rows": 0,
        "output_rows": 0,
    }
    output_frame_ids: set[int] = set()
    writer_state: dict[str, Any] = {
        "path": str(out_path), "writer": None, "schema": None, "columns": None, "rows": 0,
    }
    initial_buffer: list[pd.DataFrame] = []

    q: queue.Queue = queue.Queue(maxsize=prefetch_batches)
    stop_event = threading.Event()
    reader = threading.Thread(
        target=_reader_worker,
        args=(frames, source_video, batch_size, q, timing, stop_event),
        kwargs={"seek_threshold_frames": seek_threshold_frames},
        daemon=True,
    )

    wall_started = time.perf_counter()
    reader.start()
    try:
        with torch.inference_mode():
            while True:
                payload = q.get()
                if payload is None:
                    break
                if isinstance(payload, BaseException):
                    raise payload
                batch_df, rgb_list = payload
                batch = torch.stack(
                    [torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))) for rgb in rgb_list],
                    dim=0,
                )
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fex = detector.detect(
                    batch,
                    data_type="tensor",
                    batch_size=int(len(batch_df)),
                    num_workers=0,
                    pin_memory=False,
                    face_detection_threshold=threshold,
                    progress_bar=False,
                )
                torch.cuda.synchronize()
                timing["detector_detect_cuda_sec"] += time.perf_counter() - t0
                counters["detector_calls"] += 1

                t0 = time.perf_counter()
                processed, detected_rows = _attach_context(
                    pd.DataFrame(fex).copy(),
                    batch_df,
                    frame_h=int(rgb_list[0].shape[0]),
                    frame_w=int(rgb_list[0].shape[1]),
                )
                timing["attach_context_cpu_sec"] += time.perf_counter() - t0
                counters["detected_rows"] += detected_rows
                output_frame_ids.update(
                    pd.to_numeric(processed["benchmark_index"], errors="coerce").dropna().astype(int).tolist()
                )

                if writer_state["writer"] is None:
                    initial_buffer.append(processed)
                    if detected_rows > 0:
                        merged = pd.concat(initial_buffer, ignore_index=True, sort=False)
                        t0 = time.perf_counter()
                        _append_parquet_chunk(merged, writer_state)
                        timing["parquet_write_sec"] += time.perf_counter() - t0
                        initial_buffer.clear()
                else:
                    t0 = time.perf_counter()
                    _append_parquet_chunk(processed, writer_state)
                    timing["parquet_write_sec"] += time.perf_counter() - t0
    finally:
        stop_event.set()
        reader.join(timeout=15.0)
        if reader.is_alive():
            raise RuntimeError("Face CUDA reader thread did not stop cleanly")

    if initial_buffer:
        merged = pd.concat(initial_buffer, ignore_index=True, sort=False)
        t0 = time.perf_counter()
        _append_parquet_chunk(merged, writer_state)
        timing["parquet_write_sec"] += time.perf_counter() - t0
        initial_buffer.clear()

    if writer_state["writer"] is not None:
        writer_state["writer"].close()
    if not out_path.is_file():
        raise RuntimeError("CUDA Face runner produced no parquet output")

    expected_ids = set(pd.to_numeric(frames["benchmark_index"], errors="raise").astype(int).tolist())
    missing_ids = expected_ids - output_frame_ids
    if missing_ids:
        raise RuntimeError(f"CUDA Face raw is missing {len(missing_ids)} planned 15 Hz frame ids")

    total_wall = time.perf_counter() - wall_started
    peak_memory = int(torch.cuda.max_memory_allocated())
    counters["output_rows"] = int(writer_state["rows"])
    runtime = _runtime_info(torch)
    repo_root = config.path.parent.parent

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "face-formal-cuda",
        "output_mode": "formal",
        "completion_status": "complete",
        "subject": args.subject,
        "candidate": "pyfeat211_detectorv2_native_pytorch_cuda",
        "scientific_core": "Py-Feat 2.1.1 Detectorv2; identity_model=None",
        "execution_backend": "pytorch_cuda",
        "device": device,
        "source_video": str(source_video),
        "source_frame_manifest": str(frames_csv),
        "source_prepare_manifest": str(prepare_manifest),
        "expected_input_frames": int(len(frames)),
        "unique_output_frames": int(len(output_frame_ids)),
        "output_rows": int(writer_state["rows"]),
        "detected_rows": int(counters["detected_rows"]),
        "native_cuda_batch": batch_size,
        "native_cuda_prefetch_batches": prefetch_batches,
        "seek_threshold_frames": seek_threshold_frames,
        "face_detection_threshold": threshold,
        "requested_inference_fps": float(face_cfg.get("inference_fps", 15.0)),
        "identity_model": None,
        "timing_sec": {
            "model_initialization": init_sec,
            **timing,
            "total_wall_with_parquet_write": total_wall,
        },
        "input_frames_per_sec_total": float(len(frames) / total_wall) if total_wall > 0 else None,
        "cuda_peak_memory_allocated_bytes": peak_memory,
        "counters": counters,
        "runtime": runtime,
        "attention_analysis_git_commit": _git_commit(repo_root),
        "config_path": str(config.path),
        "config_digest": config.digest,
        "raw_output": str(out_path),
        "parquet_size_bytes": int(out_path.stat().st_size),
        "stable_nullable_streaming_schema": True,
        "notes": [
            "Native Py-Feat/PyTorch CUDA path; no ONNX Runtime or DirectML calls are used.",
            "Formal frames are decoded directly from the original AVI using the shared 15 Hz timestamp grid.",
            "All native non-identity Detectorv2 scientific columns are retained; identity_model=None is fixed.",
            "No-face planned samples receive explicit placeholder rows so formal frame coverage remains reconstructable.",
            "Tracking, primary-face selection, eyelid metrics, blink/PERCLOS and QC are downstream and do not block raw extraction.",
            "Streaming Parquet uses explicit nullable dtypes to prevent first-chunk null schema failures (ported-from: 51d17c9a6b7db7a1114380910bb111db38293512).",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full-span Py-Feat 2.1.1 native PyTorch/CUDA formal Face runner"
    )
    parser.add_argument("--config", default="configs/rgb_analysis.yaml")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--prefetch-batches", type=int)
    parser.add_argument("--seek-threshold-frames", type=int)
    parser.add_argument("--allow-pyfeat-version-mismatch", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_formal(args)


if __name__ == "__main__":
    main()
