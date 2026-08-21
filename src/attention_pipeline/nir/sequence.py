"""阶段5：连续序列构建 + 眼睑 EAR/openness + 门控校准（主环境 3.13）。

44 段 × 121 帧 ≈ 4s 的固定 ROI 连续序列（Block1/3/6 均匀 + 眨眼转换段），逐帧
MediaPipe 人脸 → 320×160 眼角 ROI → EAR（逐眼）→ openness（个体内 P95 基线）。

EAR 索引（被试解剖眼别，画面左=右眼）：
  eye_right = [33,133,160,144,158,153]；eye_left = [362,263,385,380,387,373]
  EAR = (|p2−p3| + |p4−p5|) / (2·|p0−p1|)，p0,p1=水平眼角。

门控校准：对 528 眼所在帧算 openness，与 ground_truth_528 人工可见性对照，
t=0.45/0.50/0.55/0.60 报告准确率/灵敏/特异，只报告不静默重调。
"""
from __future__ import annotations

import csv
import math
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import pandas as pd

from ..config import Config
from ..contracts import EYE_LEFT, EYE_RIGHT, EYES
from ..io import block_windows, load_timestamps, subject_paths
from .review import _read_frame
from .roi import EYE_CORNERS, normalized_eye_roi

EAR_INDICES = {
    EYE_RIGHT: [33, 133, 160, 144, 158, 153],
    EYE_LEFT: [362, 263, 385, 380, 387, 373],
}
BLINK_EAR_THRESHOLD = 0.18  # 仅用于眨眼段扫描的粗判（绝对值），不是评估门控
BLINK_SCAN_STRIDE = 15
FRAMES_PER_SEQUENCE = 121


def compute_ear(points_xy: np.ndarray, indices: list[int]) -> float:
    """Soukupova & Cech EAR；h_dist<1e-6 返回 0。"""
    p = [points_xy[i] for i in indices]
    d01 = float(np.linalg.norm(p[1] - p[0]))
    d23 = float(np.linalg.norm(p[3] - p[2]))
    d45 = float(np.linalg.norm(p[5] - p[4]))
    if d01 < 1e-6:
        return 0.0
    return (d23 + d45) / (2.0 * d01)


def ear_for_eyes(points_xy: np.ndarray) -> dict:
    return {eye: compute_ear(points_xy, EAR_INDICES[eye]) for eye in EYES}


class FaceLandmarkerSession:
    """跨帧复用同一个 MediaPipe FaceLandmarker（IMAGE 模式无状态），避免每帧重建。"""

    def __init__(self, model_path: Path, confidence: float = 0.5):
        """复用 detector。confidence 同时作用于 detection/presence/tracking 三阈值（默认 0.5）。"""
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        self._mp = mp
        self._vision = vision
        usable_model = model_path
        if not str(model_path).isascii():
            usable_model = Path(tempfile.gettempdir()) / "attention_pipeline_v2_face_landmarker.task"
            if not usable_model.exists() or usable_model.stat().st_size != model_path.stat().st_size:
                shutil.copy2(model_path, usable_model)
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(usable_model)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=confidence,
            min_face_presence_confidence=confidence,
            min_tracking_confidence=confidence,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._detector.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        if not result.face_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        return np.array([[point.x * w, point.y * h] for point in result.face_landmarks[0]], dtype=np.float32)

    def close(self):
        self._detector.close()


def resolve_sequence_dir(config: Config, tag: str | None = None, force: bool = False) -> Path:
    """序列目录：artifacts/sequence-44x121/（固定，单层）。"""
    output = config.path_value("sequence_artifact_root")
    if output.exists() and not force:
        raise RuntimeError(f"序列目录已存在: {output}（如需覆盖请显式加 --force）")
    return output


def _frame_window(center_avi: int, count: int = FRAMES_PER_SEQUENCE) -> tuple[int, int]:
    half = count // 2
    return center_avi - half, center_avi + half


def _scan_ear_in_block(session: FaceLandmarkerSession, config: Config, subject: str, block_num: int, stride: int = BLINK_SCAN_STRIDE) -> list[dict]:
    """粗扫一个 block 的 EAR，找闭眼事件（用于眨眼转换段定位）。"""
    paths = subject_paths(config.path_value("raw_root"), subject)
    timestamps = load_timestamps(paths["nir_timestamps"])
    window = next(w for w in block_windows(paths["master_timeline"]) if w["block_num"] == block_num)
    rows = timestamps.loc[
        ~timestamps["is_dropped"]
        & timestamps["unix_ms"].notna()
        & timestamps["unix_ms"].between(window["start_ms"], window["end_ms"])
    ].copy()
    frames = rows["avi_frame_idx"].astype(int).to_numpy()
    samples = frames[::stride]
    results = []
    for avi in samples:
        frame = _read_frame(paths["nir_video"], int(avi))
        points = session.detect(frame)
        if points is None:
            results.append({"avi_frame_idx": int(avi), "ear_right": np.nan, "ear_left": np.nan})
            continue
        ear = ear_for_eyes(points)
        results.append({"avi_frame_idx": int(avi), "ear_right": ear[EYE_RIGHT], "ear_left": ear[EYE_LEFT]})
    return results


def _find_blink_center(session: FaceLandmarkerSession, config: Config, subject: str, scan_block: int = 4) -> dict:
    """返回 {block_num, center_avi_idx, kind}；无闭眼事件回退 block 中点。"""
    raw_root = config.path_value("raw_root")
    paths = subject_paths(raw_root, subject)
    timestamps = load_timestamps(paths["nir_timestamps"])
    window = next(w for w in block_windows(paths["master_timeline"]) if w["block_num"] == scan_block)
    candidates = _scan_ear_in_block(session, config, subject, scan_block)
    blink = next(
        (c for c in candidates if min(c["ear_right"], c["ear_left"]) < BLINK_EAR_THRESHOLD),
        None,
    )
    if blink is not None:
        return {"block_num": scan_block, "center_avi_idx": int(blink["avi_frame_idx"]), "kind": "blink_transition"}
    center_ms = (window["start_ms"] + window["end_ms"]) / 2
    nearest = timestamps.loc[
        ~timestamps["is_dropped"] & timestamps["unix_ms"].notna()
    ]
    center = int(nearest.iloc[(nearest["unix_ms"] - center_ms).abs().argmin()]["avi_frame_idx"])
    return {"block_num": scan_block, "center_avi_idx": center, "kind": "block4_uniform"}


def sequence_plan(config: Config) -> list[dict]:
    """44 段：每被试 Block1/3/6 均匀 + 1 段眨眼转换。均匀段用 block 窗中点。"""
    raw_root = config.path_value("raw_root")
    plan = []
    for subject in config.data["subjects"]["include"]:
        paths = subject_paths(raw_root, subject)
        timestamps = load_timestamps(paths["nir_timestamps"])
        windows = block_windows(paths["master_timeline"])
        for block_num in (1, 3, 6):
            window = next(w for w in windows if w["block_num"] == block_num)
            center_ms = (window["start_ms"] + window["end_ms"]) / 2
            valid = timestamps.loc[~timestamps["is_dropped"] & timestamps["unix_ms"].notna()]
            center = int(valid.iloc[(valid["unix_ms"] - center_ms).abs().argmin()]["avi_frame_idx"])
            plan.append({
                "subject": subject, "kind": f"block{block_num}_uniform",
                "block_num": block_num, "center_avi_idx": center,
            })
    # 眨眼段需要 MediaPipe 扫，统一在 build 阶段补（避免 plan 阶段开检测器）
    return plan


def compute_ear_baselines(config: Config) -> dict:
    """逐被试逐眼基线 EAR = Block1 任务窗的 EAR P95（含三均匀段可并）。返回 {subject: {eye: baseline}}。"""
    # 基线用 Block1 均匀段：先算出该段的 EAR（复用 build 的逐帧逻辑）
    # 这里用一个独立实现：读 Block1 均匀段中心 ±60 帧算 EAR 的 P95
    raw_root = config.path_value("raw_root")
    baselines = {}
    session = FaceLandmarkerSession(config.path_value("face_landmarker_model"))
    try:
        for subject in config.data["subjects"]["include"]:
            paths = subject_paths(raw_root, subject)
            timestamps = load_timestamps(paths["nir_timestamps"])
            window = next(w for w in block_windows(paths["master_timeline"]) if w["block_num"] == 1)
            center_ms = (window["start_ms"] + window["end_ms"]) / 2
            valid = timestamps.loc[~timestamps["is_dropped"] & timestamps["unix_ms"].notna()]
            center = int(valid.iloc[(valid["unix_ms"] - center_ms).abs().argmin()]["avi_frame_idx"])
            start, end = _frame_window(center)
            ears = {EYE_RIGHT: [], EYE_LEFT: []}
            for avi in range(start, end + 1):
                frame = _read_frame(paths["nir_video"], avi)
                points = session.detect(frame)
                if points is None:
                    continue
                for eye, ear in ear_for_eyes(points).items():
                    if math.isfinite(ear):
                        ears[eye].append(ear)
            baselines[subject] = {
                eye: float(np.percentile(values, 95)) if len(values) else np.nan
                for eye, values in ears.items()
            }
    finally:
        session.close()
    return baselines


def build_sequences(config: Config, tag: str | None, force: bool = False, subjects: list[str] | None = None) -> dict:
    """构建 44 段序列：逐帧 MediaPipe→ROI→EAR→openness，写 ROI PNG + manifest。subjects 可限定（冒烟）。"""
    output = resolve_sequence_dir(config, tag, force)
    output.mkdir(parents=True, exist_ok=True)
    raw_root = config.path_value("raw_root")
    roi_size = tuple(config.section("nir")["review"]["roi_size"])
    span = float(config.section("nir")["review"]["corner_span_fraction"])
    seq_cfg = config.section("nir")["sequence"]
    frames_per = int(seq_cfg.get("frames_per_sequence", FRAMES_PER_SEQUENCE))
    session = FaceLandmarkerSession(config.path_value("face_landmarker_model"))
    plan = sequence_plan(config)
    baselines = compute_ear_baselines(config)
    manifest_rows = []
    sequence_count = 0
    subject_list = subjects or list(config.data["subjects"]["include"])
    try:
        for subject in subject_list:
            paths = subject_paths(raw_root, subject)
            timestamps = load_timestamps(paths["nir_timestamps"])
            segments = [s for s in plan if s["subject"] == subject]
            # 补眨眼段
            blink = _find_blink_center(session, config, subject)
            segments.append(blink)
            for seg in segments:
                sequence_id = f"{subject}_{seg['kind']}"
                center = seg["center_avi_idx"]
                start, end = _frame_window(center, frames_per)
                seq_dir = output / "segments" / sequence_id
                for eye in EYES:
                    (seq_dir / eye).mkdir(parents=True, exist_ok=True)
                for offset, avi in enumerate(range(start, end + 1)):
                    frame = _read_frame(paths["nir_video"], avi)
                    ts_row = timestamps.loc[timestamps["avi_frame_idx"].eq(avi)]
                    unix_ms = float(ts_row["unix_ms"].iloc[0]) if len(ts_row) else np.nan
                    points = session.detect(frame)
                    ears = ear_for_eyes(points) if points is not None else {e: np.nan for e in EYES}
                    baseline = baselines[subject]
                    for eye in EYES:
                        ear = ears[eye]
                        openness = float(ear / baseline[eye]) if math.isfinite(ear) and baseline[eye] and baseline[eye] > 0 else np.nan
                        roi_rel = ""
                        affine = None
                        if points is not None:
                            roi, affine, _ = normalized_eye_roi(frame, points, EYE_CORNERS[eye], roi_size, span)
                            if roi is not None:
                                roi_path = seq_dir / eye / f"frame_{offset:03d}.png"
                                ok, encoded = cv2.imencode(".png", roi)
                                if ok:
                                    roi_path.write_bytes(encoded.tobytes())
                                    roi_rel = roi_path.relative_to(output).as_posix()
                        source_to_roi = (
                            ",".join(f"{float(v):.9g}" for v in np.asarray(affine, dtype=float).reshape(-1))
                            if roi_rel and affine is not None else ""
                        )
                        roi_to_source = (
                            ",".join(f"{float(v):.9g}" for v in cv2.invertAffineTransform(np.asarray(affine, dtype=float)).reshape(-1))
                            if roi_rel and affine is not None else ""
                        )
                        manifest_rows.append({
                            "sequence_id": sequence_id,
                            "subject": subject,
                            "kind": seg["kind"],
                            "block_num": seg.get("block_num", ""),
                            "eye": eye,
                            "frame_offset": offset,
                            "avi_frame_idx": int(avi),
                            "unix_ms": unix_ms,
                            "face_detected": int(points is not None),
                            "ear": ear,
                            "openness": openness,
                            "visible_proxy": int(openness >= seq_cfg["openness_visible"]) if math.isfinite(openness) else np.nan,
                            "p80_closed_proxy": int(openness < seq_cfg["openness_closed"]) if math.isfinite(openness) else np.nan,
                                                        "roi_path": roi_rel,
                            "roi_status": "ready" if roi_rel else ("missing" if points is None else "degenerate"),
                            "source_to_roi_affine": source_to_roi,
                            "roi_to_source_affine": roi_to_source,
                        })
                sequence_count += 1
    finally:
        session.close()
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output / "sequence_manifest.csv", index=False, encoding="utf-8-sig")
    (output / "sequence_plan.json").write_text(
        pd.DataFrame(plan).to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "output_dir": str(output),
        "segments": sequence_count,
        "rows": len(manifest),
        "roi_frames": int(manifest["roi_path"].fillna("").astype(str).str.strip().ne("").sum()),
    }


def run_sequence_detect(config: Config, tag: str, force: bool = False) -> dict:
    """调用 venv-pupil 序列检测适配器，产出 detections_sequence.csv。"""
    import subprocess
    sequence_dir = config.path_value("sequence_artifact_root")
    manifest = sequence_dir / "sequence_manifest.csv"
    det_path = sequence_dir / "detections_sequence.csv"
    if det_path.exists() and not force:
        raise RuntimeError(f"检测结果已存在: {det_path}（如需覆盖请显式加 --force）")
    adapter = Path(__file__).resolve().parents[3] / "scripts" / "nir_sequence_detect.py"
    python = config.section("runtimes")["pypupilext_python"]
    seq_cfg = config.section("nir")["sequence"]
    cmd = [
        str(python), str(adapter),
        "--manifest", str(manifest),
        "--roi-root", str(sequence_dir),
        "--out", str(det_path),
        "--px-min", str(seq_cfg["px_min"]),
        "--px-max", str(seq_cfg["px_max"]),
        "--pupil-min-mm", str(seq_cfg["pupil_min_mm"]),
        "--openness-visible", str(seq_cfg["openness_visible"]),
        "--max-session-gap-ms", str(seq_cfg.get("max_session_gap_ms", 200.0)),
        "--reset-after-quality-rejects", str(seq_cfg.get("reset_after_quality_rejects", 3)),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)
    if completed.returncode != 0:
        raise RuntimeError(f"序列检测适配器失败:\n{completed.stderr[-2000:]}")
    import pandas as _pd
    det = _pd.read_csv(det_path)
    return {"detections": str(det_path), "rows": len(det)}


def gate_calibration(config: Config, output: Path | None = None) -> pd.DataFrame:
    """用 528 眼人工可见性校准 openness 阈值：openness≥t 预测"可见"的准确率/灵敏/特异。

    只报告 t=0.45/0.50/0.55/0.60，不静默重调；"不确定"样本排除，按被试拆。
    """
    review_dir = config.path_value("truth_artifact_root")
    gt = pd.read_csv(review_dir / "ground_truth_528.csv")
    man = pd.read_csv(review_dir / "review_manifest.csv")
    merged = gt.merge(man[["sample_id", "context_path", "subject", "eye"]], on="sample_id", how="left")
    baselines = compute_ear_baselines(config)

    session = FaceLandmarkerSession(config.path_value("face_landmarker_model"))
    ear_cache: dict[str, dict] = {}
    try:
        for context_path in merged["context_path"].dropna().unique():
            full = cv2.imdecode(np.fromfile(str(review_dir / context_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if full is None:
                continue
            points = session.detect(full)
            ear_cache[str(context_path)] = ear_for_eyes(points) if points is not None else {e: np.nan for e in EYES}
    finally:
        session.close()

    def openness_for(row) -> float:
        ear = ear_cache.get(str(row["context_path"]), {e: np.nan for e in EYES})[row["eye"]]
        baseline = baselines.get(row["subject"], {}).get(row["eye"], np.nan)
        return float(ear / baseline) if math.isfinite(ear) and baseline and baseline > 0 else np.nan

    merged["openness"] = merged.apply(openness_for, axis=1)
    binary = merged[merged["visibility"].isin(["可见", "不可见"])].copy()
    binary["human_visible"] = (binary["visibility"] == "可见").astype(int)

    rows = []
    for t in (0.45, 0.50, 0.55, 0.60):
        valid = binary[binary["openness"].notna()]
        pred = (valid["openness"] >= t).astype(int)
        y = valid["human_visible"].astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        rows.append({
            "threshold": t,
            "n": len(valid),
            "accuracy": (tp + tn) / len(valid) if len(valid) else np.nan,
            "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
            "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })
    summary = pd.DataFrame(rows)
    per_subject = []
    for t in (0.50, 0.55):
        valid = binary[binary["openness"].notna()]
        for subject, group in valid.groupby("subject"):
            pred = (group["openness"] >= t).astype(int)
            y = group["human_visible"].astype(int)
            acc = float((pred == y).mean()) if len(group) else np.nan
            per_subject.append({"threshold": t, "subject": subject, "n": len(group), "accuracy": acc})
    per_subject_df = pd.DataFrame(per_subject)

    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output / "gate_calibration.csv", index=False, encoding="utf-8-sig")
        per_subject_df.to_csv(output / "gate_calibration_per_subject.csv", index=False, encoding="utf-8-sig")
    return summary

