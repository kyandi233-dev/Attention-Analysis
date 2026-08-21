"""正式实验 ROI 提取公共工具。

> 08-16（Asia/Shanghai）｜三个 ROI 算法脚本（roi_mediapipe/roi_yunet/roi_yolo）共用的
> 公共部分：抽样段、AVI 读取、manifest 写入、几何门、ASCII 模型路径、venv 检测调度、代理指标统计。

跳转关系：roi_<算法>.py → build_roi() → roi_common；build 产出 manifest + ROI PNG，
run_detect() 调 venv 的 nir_sequence_detect.py，compute_stats() 汇总代理指标。
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from attention_pipeline.config import load_config
from attention_pipeline.nir.formal import formal_subject_paths, load_nir_timestamps, locate_block_segment
from attention_pipeline.nir.review import _imwrite

DEFAULT_ROOT = "E:/正式实验"
DEFAULT_VENV = "D:/aaawork/07-竞赛/厚璨杯/venv-pupil/Scripts/python.exe"
DEFAULT_OUT = "artifacts/formal-validate"
DEFAULT_FACE_MODEL = "D:/aaawork/07-竞赛/厚璨杯/021-analysisplan/NIR/models/face_landmarker.task"
DEFAULT_YUNET = "D:/aaawork/07-竞赛/厚璨杯/021-analysisplan/NIR/models/yunet_2023mar.onnx"
DEFAULT_YOLO = "D:/aaawork/07-竞赛/厚璨杯/021-analysisplan/NIR/models/yolov8n-face.onnx"

EYE_KEYS = ["eye_right", "eye_left"]  # 与 contracts.EYE_RIGHT/EYE_LEFT 一致


def ascii_model_path(model_path: str) -> str:
    """cv2.dnn 读 ONNX 不支持非 ASCII 路径；必要时复制到临时 ASCII 路径（同 FaceLandmarkerSession 处理）。"""
    p = Path(model_path)
    if str(p).isascii():
        return str(p)
    dst = Path(tempfile.gettempdir()) / p.name
    if not dst.exists() or dst.stat().st_size != p.stat().st_size:
        shutil.copy2(p, dst)
    return str(dst)


def crop_resize_gray(frame: np.ndarray, box, out_w: int, out_h: int) -> np.ndarray | None:
    """按源图坐标 box=(x0,y0,x1,y1) 裁矩形并 resize 到 out_w×out_h 灰度。越界 clip；太小返回 None。"""
    h, w = frame.shape[:2]
    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
    x1, y1 = min(w, int(box[2])), min(h, int(box[3]))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    roi = cv2.resize(frame[y0:y1, x0:x1], (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))


def _serialize_points(values) -> str:
    if values is None:
        return ""
    return json.dumps(np.asarray(values, dtype=float).tolist(), ensure_ascii=False, separators=(",", ":"))


def roi_payload(image: np.ndarray, source_to_roi: np.ndarray, canthi_source=None, normalized_canthi=None, *, reference_kind="unknown", locator_box_source=None, reference_points_source=None, normalized_reference_points=None) -> dict:
    """Package an ROI with reversible source mapping and optional real canthi landmarks."""
    matrix = np.asarray(source_to_roi, dtype=float).reshape(2, 3)
    inverse = cv2.invertAffineTransform(matrix)
    if normalized_canthi is None and canthi_source is not None:
        normalized_canthi = cv2.transform(
            np.asarray(canthi_source, dtype=np.float32)[None, :, :], matrix
        )[0]
    if normalized_reference_points is None and reference_points_source is not None:
        normalized_reference_points = cv2.transform(
            np.asarray(reference_points_source, dtype=np.float32)[None, :, :], matrix
        )[0]
    height, width = image.shape[:2]
    roi_corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    source_corners = cv2.transform(roi_corners[None, :, :], inverse)[0]
    return {
        "image": np.ascontiguousarray(image),
        "source_to_roi_affine": matrix,
        "roi_to_source_affine": inverse,
        "roi_corners_source": source_corners,
        "canthi_source": canthi_source,
        "normalized_canthi": normalized_canthi,
        "reference_kind": reference_kind,
        "locator_box_source": locator_box_source,
        "reference_points_source": reference_points_source,
        "normalized_reference_points": normalized_reference_points,
    }


def crop_resize_gray_payload(frame: np.ndarray, box, out_w: int, out_h: int, *, reference_points=None, reference_kind="estimated_eye_center") -> dict | None:
    """Axis-aligned crop/resize plus an exact reversible affine map."""
    height, width = frame.shape[:2]
    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
    x1, y1 = min(width, int(box[2])), min(height, int(box[3]))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    image = cv2.resize(frame[y0:y1, x0:x1], (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    image = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    source_to_roi = np.array([
        [out_w / (x1 - x0), 0.0, -x0 * out_w / (x1 - x0)],
        [0.0, out_h / (y1 - y0), -y0 * out_h / (y1 - y0)],
    ], dtype=float)
    normalized = None
    if reference_points is not None:
        normalized = cv2.transform(np.asarray(reference_points, dtype=np.float32)[None, :, :], source_to_roi)[0]
    return roi_payload(image, source_to_roi, reference_kind=reference_kind, locator_box_source=[x0, y0, x1, y1], reference_points_source=reference_points, normalized_reference_points=normalized)

def geometry_ok(center, major, minor, imsize=(320, 160)) -> bool:
    """阶段4 几何门口径：中心在画内、3≤minor≤major≤0.65·min(h,w)、aspect≥0.25。"""
    width, height = imsize
    cx, cy = center
    if not (0 <= cx < width and 0 <= cy < height):
        return False
    major, minor = abs(float(major)), abs(float(minor))
    if major < 3 or minor < 3:
        return False
    if major > 0.65 * min(width, height):
        return False
    if minor / major < 0.25:
        return False
    return True


def sample_starts(total: int, n_segments: int, frames_per_seg: int) -> list[tuple[int, int]]:
    """按时间均匀取 n 段，返回 [(seq_no, start_frame), ...]。n<=1 时取中段。"""
    if n_segments <= 1:
        return [(1, (total - frames_per_seg) // 2)]
    span = total - frames_per_seg
    return [(i + 1, round(i * span / (n_segments - 1))) for i in range(n_segments)]


def build_roi(provider, args, outdir: Path) -> tuple[Path, int]:
    """Sequentially read sampled segments and preserve one manifest row per frame and eye."""
    subject_paths = formal_subject_paths(args.root, args.subject)
    avi = subject_paths["nir_video"]
    cap = cv2.VideoCapture(str(avi))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {avi}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows: list[dict] = []
    success_count = 0
    timing_rows: list[dict] = []
    try:
        import psutil
        process = psutil.Process()
        initial_rss_mb = process.memory_info().rss / (1024 ** 2)
        peak_rss_mb = initial_rss_mb
    except Exception:
        process = None
        initial_rss_mb = float("nan")
        peak_rss_mb = float("nan")
    t0 = time.perf_counter()
    timestamps = None
    if subject_paths["nir_timestamps"].exists():
        timestamps = load_nir_timestamps(subject_paths["nir_timestamps"])
    window = None
    if getattr(args, "seg_starts", "") and args.seg_starts.strip():
        starts = [(i + 1, int(v)) for i, v in enumerate(args.seg_starts.split(",")) if v.strip()]
    elif getattr(args, "block", None) is not None:
        window, timestamps, _ = locate_block_segment(
            args.root, args.subject, args.block, args.duration_sec
        )
        args.frames_per_seg = int(window["n_frames"])
        starts = [(int(args.block), int(window["start_frame_idx"]))]
        (outdir / "selection_window.json").write_text(
            json.dumps(window, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        starts = sample_starts(total, args.n_segments, args.frames_per_seg)
    unix_by_frame = (
        dict(zip(timestamps["frame_idx"], timestamps["unix_ms"]))
        if timestamps is not None else {}
    )

    fields = [
        "subject", "block", "backend", "sequence_id", "eye", "frame_offset",
        "frame_idx", "unix_ms", "face_detected", "face_status", "roi_status",
        "image_status", "error_code", "roi_path", "source_width", "source_height",
        "source_to_roi_affine", "roi_to_source_affine", "roi_corners_source",
        "normalized_canthi", "canthi_source", "reference_kind", "locator_box_source", "reference_points_source", "normalized_reference_points", "ear", "baseline_open_ear", "openness", "visible_proxy", "p80_closed_proxy", "visibility_source",
    ]
    try:
        for seg, segment_start in starts:
            seq_id = f"{args.subject}-s{seg}"
            cap.set(cv2.CAP_PROP_POS_FRAMES, segment_start)
            for off in range(args.frames_per_seg):
                frame_idx = segment_start + off
                frame_started = time.perf_counter()
                decode_started = time.perf_counter()
                ok, frame = cap.read()
                decode_ms = (time.perf_counter() - decode_started) * 1000.0
                if not ok or frame is None:
                    for eye in EYE_KEYS:
                        rows.append({
                            "subject": args.subject, "block": getattr(args, "block", ""),
                            "backend": provider.name, "sequence_id": seq_id, "eye": eye,
                            "frame_offset": off, "frame_idx": frame_idx, "unix_ms": unix_by_frame.get(frame_idx, ""),
                            "face_detected": 0, "face_status": "unknown",
                            "roi_status": "roi_missing", "image_status": "read_failed",
                            "error_code": "image_read_failed", "roi_path": "",
                        })
                    timing_rows.append({
                        "sequence_id": seq_id, "frame_offset": off, "frame_idx": frame_idx,
                        "unix_ms": unix_by_frame.get(frame_idx, ""), "decode_ms": decode_ms,
                        "inference_ms": float("nan"), "crop_normalize_ms": float("nan"),
                        "roi_only_ms": float("nan"),
                        "end_to_end_ms": (time.perf_counter() - frame_started) * 1000.0,
                        "valid_eye_rois": 0, "image_status": "read_failed",
                    })
                    continue
                height, width = frame.shape[:2]
                provider_started = time.perf_counter()
                try:
                    eyes = provider.eyes(frame)
                    provider_error = ""
                except Exception as exc:
                    eyes = None
                    provider_error = f"{type(exc).__name__}:{exc}"
                    provider.last_timing = {}
                for eye in EYE_KEYS:
                    base = {
                        "subject": args.subject, "block": getattr(args, "block", ""),
                        "backend": provider.name, "sequence_id": seq_id, "eye": eye,
                        "frame_offset": off, "frame_idx": frame_idx, "unix_ms": unix_by_frame.get(frame_idx, ""),
                        "source_width": width, "source_height": height,
                        "image_status": "ok",
                    }
                    payload = eyes.get(eye) if eyes else None
                    if payload is None:
                        base.update({
                            "face_detected": 0 if eyes is None else 1,
                            "face_status": "no_face" if eyes is None else "ok",
                            "roi_status": "roi_missing",
                            "error_code": provider_error or "roi_missing",
                            "roi_path": "",
                        })
                    else:
                        if isinstance(payload, np.ndarray):
                            # Compatibility for tests; production backends return mapping metadata.
                            roi = payload
                            metadata = {}
                        else:
                            roi = payload["image"]
                            metadata = payload
                        rel = f"segments/{seq_id}/{eye}/f{off:04d}.png"
                        _imwrite(outdir / rel, roi)
                        base.update({
                            "face_detected": 1, "face_status": "ok", "roi_status": "ok",
                            "error_code": "", "roi_path": rel,
                            "source_to_roi_affine": _serialize_points(metadata.get("source_to_roi_affine")),
                            "roi_to_source_affine": _serialize_points(metadata.get("roi_to_source_affine")),
                            "roi_corners_source": _serialize_points(metadata.get("roi_corners_source")),
                            "normalized_canthi": _serialize_points(metadata.get("normalized_canthi")),
                            "canthi_source": _serialize_points(metadata.get("canthi_source")),
                            "reference_kind": metadata.get("reference_kind", "unknown"),
                            "locator_box_source": _serialize_points(metadata.get("locator_box_source")),
                            "reference_points_source": _serialize_points(metadata.get("reference_points_source")),
                            "normalized_reference_points": _serialize_points(metadata.get("normalized_reference_points")),
                            "ear": metadata.get("ear", ""),
                        })
                        success_count += 1
                    rows.append(base)
                provider_ms = (time.perf_counter() - provider_started) * 1000.0
                detail = getattr(provider, "last_timing", {}) or {}
                inference_ms = float(detail.get("inference_ms", provider_ms))
                crop_ms = float(detail.get("crop_normalize_ms", max(0.0, provider_ms - inference_ms)))
                if process is not None:
                    peak_rss_mb = max(peak_rss_mb, process.memory_info().rss / (1024 ** 2))
                timing_rows.append({
                    "sequence_id": seq_id, "frame_offset": off, "frame_idx": frame_idx,
                    "unix_ms": unix_by_frame.get(frame_idx, ""), "decode_ms": decode_ms,
                    "inference_ms": inference_ms, "crop_normalize_ms": crop_ms,
                    "roi_only_ms": inference_ms + crop_ms,
                    "end_to_end_ms": (time.perf_counter() - frame_started) * 1000.0,
                    "valid_eye_rois": sum(1 for eye in EYE_KEYS if eyes and eyes.get(eye) is not None),
                    "image_status": "ok",
                })
            print(f"[build] {seq_id} {segment_start}..{segment_start + args.frames_per_seg - 1}", file=sys.stderr)
    finally:
        cap.release()
        provider.close()

    timing_path = outdir / "timing_frames.csv"
    timing_fields = [
        "sequence_id", "frame_offset", "frame_idx", "unix_ms", "decode_ms",
        "inference_ms", "crop_normalize_ms", "roi_only_ms", "end_to_end_ms",
        "valid_eye_rois", "image_status",
    ]
    with timing_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=timing_fields)
        writer.writeheader()
        writer.writerows(timing_rows)
    wall_seconds = max(time.perf_counter() - t0, 1e-9)
    def timing_values(name):
        values = np.asarray([row[name] for row in timing_rows], dtype=float)
        return values[np.isfinite(values)]
    def timing_stat(name, percentile):
        values = timing_values(name)
        return float(np.percentile(values, percentile)) if len(values) else float("nan")
    def timing_mean(name):
        values = timing_values(name)
        return float(np.mean(values)) if len(values) else float("nan")
    roi_seconds = sum(row["roi_only_ms"] for row in timing_rows if np.isfinite(row["roi_only_ms"])) / 1000.0
    speed = {
        "backend": provider.name,
        "frames": len(timing_rows),
        "valid_eye_rois": success_count,
        "model_load_ms": float(getattr(provider, "model_load_ms", float("nan"))),
        "decode_mean_ms": timing_mean("decode_ms"),
        "decode_p50_ms": timing_stat("decode_ms", 50),
        "decode_p95_ms": timing_stat("decode_ms", 95),
        "inference_mean_ms": timing_mean("inference_ms"),
        "inference_p50_ms": timing_stat("inference_ms", 50),
        "inference_p95_ms": timing_stat("inference_ms", 95),
        "crop_mean_ms": timing_mean("crop_normalize_ms"),
        "crop_p50_ms": timing_stat("crop_normalize_ms", 50),
        "crop_p95_ms": timing_stat("crop_normalize_ms", 95),
        "roi_only_mean_ms": timing_mean("roi_only_ms"),
        "roi_only_p50_ms": timing_stat("roi_only_ms", 50),
        "roi_only_p95_ms": timing_stat("roi_only_ms", 95),
        "end_to_end_mean_ms": timing_mean("end_to_end_ms"),
        "end_to_end_p50_ms": timing_stat("end_to_end_ms", 50),
        "end_to_end_p95_ms": timing_stat("end_to_end_ms", 95),
        "roi_only_valid_eye_rois_per_sec": success_count / max(roi_seconds, 1e-9),
        "end_to_end_valid_eye_rois_per_sec": success_count / wall_seconds,
        "initial_rss_mb": initial_rss_mb,
        "peak_rss_mb": peak_rss_mb,
        "peak_rss_delta_mb": peak_rss_mb - initial_rss_mb,
    }
    with (outdir / "speed_roi_only.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(speed))
        writer.writeheader()
        writer.writerow(speed)

    manifest = outdir / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"[build] {success_count}/{len(rows)} successful eye ROIs -> {manifest} "
        f"({time.perf_counter() - t0:.1f}s)", file=sys.stderr
    )
    return manifest, success_count

def apply_openness_gate(manifest: Path, calibration_frames: int, visible_threshold: float,
                        closed_threshold: float, source_manifest: Path | None = None) -> dict:
    """Attach one shared MediaPipe-EAR gate to a candidate manifest.

    The baseline is eye-specific P95 over the first calibration_frames. Alternative
    ROI manifests receive EAR by exact frame_idx×eye join; they do not rerun or fake
    an eyelid model.
    """
    import pandas as pd
    target = pd.read_csv(manifest)
    source = pd.read_csv(source_manifest or manifest)
    source["ear"] = pd.to_numeric(source.get("ear"), errors="coerce")
    if source_manifest is not None:
        gate = source[["frame_idx", "eye", "ear"]].drop_duplicates(["frame_idx", "eye"])
        target = target.drop(columns=["ear"], errors="ignore").merge(
            gate, on=["frame_idx", "eye"], how="left", validate="many_to_one"
        )
        visibility_source = "mediapipe_shared"
    else:
        target["ear"] = pd.to_numeric(target.get("ear"), errors="coerce")
        visibility_source = "mediapipe_self"
    calibration = source[
        (pd.to_numeric(source["frame_offset"], errors="coerce") < calibration_frames)
        & source["ear"].notna()
    ]
    baselines = calibration.groupby("eye")["ear"].quantile(0.95).to_dict()
    missing = [eye for eye in EYE_KEYS if eye not in baselines or not np.isfinite(baselines[eye]) or baselines[eye] <= 0]
    if missing:
        raise RuntimeError(
            "shared MediaPipe EAR gate unavailable for: " + ",".join(missing)
            + "; do not run PuReST ranking without a visibility gate"
        )
    target["baseline_open_ear"] = target["eye"].map(baselines)
    target["openness"] = target["ear"] / target["baseline_open_ear"]
    finite = np.isfinite(pd.to_numeric(target["openness"], errors="coerce"))
    target["visible_proxy"] = np.where(finite, (target["openness"] >= visible_threshold).astype(int), np.nan)
    target["p80_closed_proxy"] = np.where(finite, (target["openness"] < closed_threshold).astype(int), np.nan)
    target["visibility_source"] = visibility_source
    target.to_csv(manifest, index=False)
    return {"baselines": baselines, "source": visibility_source, "rows": len(target)}

def run_detect(venv_python: str, manifest: Path, outdir: Path,
               px_min: int, px_max: int, pupil_min_mm: float, openness_visible: float,
               max_session_gap_ms: float, reset_after_quality_rejects: int) -> Path:
    """调用连续适配器；PuReST内部宽松搜索，Python层严格执行px接受门。"""
    det_csv = outdir / "detections_sequence.csv"
    cmd = [
        venv_python, str(Path(__file__).resolve().parent / "nir_sequence_detect.py"),
        "--manifest", str(manifest), "--roi-root", str(outdir), "--out", str(det_csv),
        "--px-min", str(px_min), "--px-max", str(px_max), "--pupil-min-mm", str(pupil_min_mm),
        "--openness-visible", str(openness_visible),
        "--max-session-gap-ms", str(max_session_gap_ms),
        "--reset-after-quality-rejects", str(reset_after_quality_rejects),
    ]
    print(f"[detect] venv 子进程：{Path(cmd[1]).name} --manifest {manifest.name} ...", file=sys.stderr)
    started = time.perf_counter()
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )
    peak_rss_mb = float("nan")
    try:
        import psutil
        child = psutil.Process(process.pid)
        peak_rss_mb = 0.0
        while process.poll() is None:
            try:
                rss = child.memory_info().rss + sum(
                    p.memory_info().rss for p in child.children(recursive=True)
                )
                peak_rss_mb = max(peak_rss_mb, rss / (1024 ** 2))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(0.05)
    except ImportError:
        pass
    stdout, stderr = process.communicate()
    wall_seconds = max(time.perf_counter() - started, 1e-9)
    if stdout:
        print(stdout, end="", file=sys.stdout)
    if stderr:
        print(stderr, end="", file=sys.stderr)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, cmd, stdout, stderr)
    with manifest.open(encoding="utf-8") as handle:
        eye_rows = sum(1 for _ in csv.DictReader(handle))
    row = {
        "wall_seconds": wall_seconds,
        "eye_rows": eye_rows,
        "eye_rows_per_sec": eye_rows / wall_seconds,
        "peak_child_rss_mb": peak_rss_mb,
    }
    with (outdir / "speed_detector.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return det_csv


def compute_stats(args, det_csv: Path, outdir: Path, n_eye_frames: int) -> None:
    """代理指标（正式实验无人工真值，仅供定稿参考）：返回率/几何门/outline/光度/直径分布 + ROI 样本图。"""
    import pandas as pd
    df = pd.read_csv(det_csv)
    df["geometry_ok"] = df.apply(
        lambda r: geometry_ok((r["center_x"], r["center_y"]), r["major_diameter"], r["minor_diameter"])
        if r["returned"] == 1 else False, axis=1)
    summary = []
    print("\n=== 代理指标（无真值，仅供定稿参考）===")
    print(f"ROI 眼帧总数: {n_eye_frames}")
    for algo in df["algorithm"].unique():
        sub = df[df["algorithm"] == algo]
        algorithm_returned = sub[sub["algorithm_returned"] == 1]
        accepted = sub[sub["returned"] == 1]
        plausible = accepted[accepted["geometry_ok"]]
        algorithm_rate = len(algorithm_returned) / len(sub) if len(sub) else float("nan")
        accepted_rate = len(accepted) / len(sub) if len(sub) else float("nan")
        p_rate = len(plausible) / len(accepted) if len(accepted) else float("nan")
        outline = plausible["outline_confidence"].median() if len(plausible) else float("nan")
        contrast = plausible["photometric_contrast"].median() if len(plausible) else float("nan")
        diam = plausible["major_diameter"] if len(plausible) else pd.Series(dtype=float)
        row = {
            "algorithm": algo, "frames": len(sub),
            "algorithm_return_rate": round(algorithm_rate, 4),
            "accepted_rate": round(accepted_rate, 4),
            "geom_pass_of_accepted": round(p_rate, 4),
            "outline_median": round(outline, 3) if not np.isnan(outline) else None,
            "contrast_median": round(contrast, 4) if not np.isnan(contrast) else None,
            "diam_median": round(float(diam.median()), 1) if len(diam) else None,
            "diam_p10": round(float(diam.quantile(0.10)), 1) if len(diam) else None,
            "diam_p90": round(float(diam.quantile(0.90)), 1) if len(diam) else None,
        }
        summary.append(row)
        print(f"  {algo:8s} 算法返回 {algorithm_rate:.3f}  生产接受 {accepted_rate:.3f}  "
              f"几何通过(占接受) {p_rate:.3f}  outline {row['outline_median']}  "
              f"contrast {row['contrast_median']}  直径中位{row['diam_median']} "
              f"p10{row['diam_p10']} p90{row['diam_p90']}px")
    pd.DataFrame(summary).to_csv(outdir / "summary.csv", index=False)

    # This is an explicit compute-time estimate. Subprocess startup and PNG transport are
    # captured separately by speed_detector.csv and are not hidden in this value.
    timing_path = outdir / "timing_frames.csv"
    if timing_path.exists() and "runtime_ms" in df.columns:
        timing = pd.read_csv(timing_path)
        purest = df[df["algorithm"] == "PuReST"].copy()
        per_frame_detector = purest.groupby(
            ["sequence_id", "frame_offset"], as_index=False
        )["runtime_ms"].sum().rename(columns={"runtime_ms": "purest_detector_ms"})
        merged = timing.merge(per_frame_detector, on=["sequence_id", "frame_offset"], how="left")
        merged["purest_detector_ms"] = merged["purest_detector_ms"].fillna(0.0)
        merged["full_chain_estimated_ms"] = merged["end_to_end_ms"] + merged["purest_detector_ms"]
        values = merged["full_chain_estimated_ms"].dropna().to_numpy(dtype=float)
        speed_row = {
            "timing_kind": "estimated_decode_roi_plus_purest_compute",
            "frames": len(values),
            "mean_ms": float(np.mean(values)) if len(values) else float("nan"),
            "p50_ms": float(np.percentile(values, 50)) if len(values) else float("nan"),
            "p95_ms": float(np.percentile(values, 95)) if len(values) else float("nan"),
            "note": "Excludes subprocess startup and PNG transport; see speed_detector.csv.",
        }
        pd.DataFrame([speed_row]).to_csv(outdir / "speed_full_chain.csv", index=False)
        merged.to_csv(outdir / "timing_full_chain_frames.csv", index=False)

    sample_dir = outdir / "roi_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(outdir / "manifest.csv")
    for seq in rows["sequence_id"].unique():
        for off in [0, args.frames_per_seg // 2, args.frames_per_seg - 1]:
            pair = {}
            for eye in EYE_KEYS:
                m = rows[(rows["sequence_id"] == seq) & (rows["frame_offset"] == off) & (rows["eye"] == eye)]
                if not len(m) or pd.isna(m.iloc[0]["roi_path"]) or not str(m.iloc[0]["roi_path"]).strip():
                    continue
                pair[eye] = cv2.imdecode(np.fromfile(outdir / m.iloc[0]["roi_path"], dtype=np.uint8),
                                         cv2.IMREAD_GRAYSCALE)
            if len(pair) == 2:
                vis = np.hstack([pair["eye_right"], np.full((args.roi_h, 4), 255, np.uint8), pair["eye_left"]])
                cv2.imencode(".png", vis)[1].tofile(str(sample_dir / f"{seq}_f{off:04d}_rl.png"))
    print(f"[stats] summary.csv + {len(list(sample_dir.glob('*.png')))} 张 ROI 样本 → {outdir}", file=sys.stderr)


def resolve_common_args(args, model_path_key: str, model_arg_name: str | None = None) -> str:
    """Fill CLI omissions from the shared formal config and return the selected model path."""
    config = load_config(args.config)
    selection = config.section("nir")["roi_selection"]
    sequence = config.section("nir")["sequence"]
    args.root = args.root or str(config.path_value("formal_data_root"))
    args.out = args.out or str(config.path_value("roi_selection_artifact_root"))
    args.venv_python = args.venv_python or str(config.section("runtimes")["pypupilext_python"])
    args.block = args.block if args.block is not None else int(selection["block"])
    args.n_segments = args.n_segments if args.n_segments is not None else 1
    args.duration_sec = (
        getattr(args, "duration_sec", None) if getattr(args, "duration_sec", None) is not None
        else float(selection["duration_sec"])
    )
    args.fps_nominal = float(config.section("nir")["fps_nominal"])
    args.frames_per_seg = (
        args.frames_per_seg if args.frames_per_seg is not None
        else int(selection["duration_sec"] * config.section("nir")["fps_nominal"])
    )
    args.roi_w = args.roi_w if args.roi_w is not None else int(selection["roi_size"][0])
    args.roi_h = args.roi_h if args.roi_h is not None else int(selection["roi_size"][1])
    args.corner_span = (
        getattr(args, "corner_span", None) if getattr(args, "corner_span", None) is not None
        else float(selection["corner_span_candidates"][0])
    )
    args.px_min = args.px_min if args.px_min is not None else int(sequence["px_min"])
    args.px_max = args.px_max if args.px_max is not None else int(sequence["px_max"])
    args.pupil_min_mm = (
        args.pupil_min_mm if args.pupil_min_mm is not None
        else float(sequence["pupil_min_mm_compat"])
    )
    args.openness_visible = (
        args.openness_visible if args.openness_visible is not None
        else float(sequence["openness_visible"])
    )
    args.max_session_gap_ms = (
        args.max_session_gap_ms if args.max_session_gap_ms is not None
        else float(sequence["max_session_gap_ms"])
    )
    args.reset_after_quality_rejects = (
        args.reset_after_quality_rejects if args.reset_after_quality_rejects is not None
        else int(sequence["reset_after_quality_rejects"])
    )
    model_override = getattr(args, model_arg_name or model_path_key, None)
    return str(model_override or config.path_value(model_path_key))


def add_common_args(parser) -> None:
    parser.add_argument("--config", default="configs/formal.yaml")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--block", type=int, default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--n-segments", type=int, default=None)
    parser.add_argument("--seg-starts", default="", help="逗号分隔段起点帧号，覆盖均匀抽样（如 2000,39000,65250）")
    parser.add_argument("--frames-per-seg", type=int, default=None)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--roi-w", type=int, default=None)
    parser.add_argument("--roi-h", type=int, default=None)
    parser.add_argument("--corner-span", type=float, default=None,
                        help="三种后端统一使用的眼部参考尺度占320px画布宽度比例")
    parser.add_argument("--venv-python", default=None)
    parser.add_argument("--px-min", type=int, default=None)
    parser.add_argument("--px-max", type=int, default=None)
    parser.add_argument("--pupil-min-mm", type=float, default=None)
    parser.add_argument("--openness-visible", type=float, default=None)
    parser.add_argument("--max-session-gap-ms", type=float, default=None)
    parser.add_argument("--reset-after-quality-rejects", type=int, default=None)
    parser.add_argument("--skip-detect", action="store_true", help="只 build，不调 venv 检测")


def finish(args, provider) -> int:
    """Shared build/detect/stats flow with a backend-specific output directory."""
    outdir = Path(args.out) / args.subject / provider.name
    outdir.mkdir(parents=True, exist_ok=True)
    manifest, n_eye_frames = build_roi(provider, args, outdir)
    if args.skip_detect:
        print("[skip-detect] detection was not run", file=sys.stderr)
        return 0
    det_csv = run_detect(
        args.venv_python, manifest, outdir, args.px_min, args.px_max, args.pupil_min_mm,
        args.openness_visible, args.max_session_gap_ms, args.reset_after_quality_rejects,
    )
    compute_stats(args, det_csv, outdir, n_eye_frames)
    return 0














