from __future__ import annotations

import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from attention_pipeline.config import Config
from attention_pipeline.rgb.paths import RGBOutputLayout


FACE_VISUAL_REVIEW_SCHEMA = "rgb-face-visual-review-v0.1"
PYFEAT_AU_RE = re.compile(r"AU(\d+)$")


def _safe_float(value):
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _primary_pyfeat(raw: pd.DataFrame) -> pd.DataFrame:
    input_col = next((c for c in ("input", "image_path", "file", "filename") if c in raw.columns), None)
    if input_col is None:
        raise ValueError("Py-Feat raw output has no recognizable input-path column")
    work = raw.copy()
    work["_image_name"] = work[input_col].astype(str).map(lambda p: Path(p).name)
    if "FaceScore" in work.columns:
        work["_score"] = pd.to_numeric(work["FaceScore"], errors="coerce").fillna(-np.inf)
        work = work.sort_values(["_image_name", "_score"], ascending=[True, False])
    return work.groupby("_image_name", as_index=False).first()


def _parse_libreface_landmarks(value, width: int, height: int) -> list[tuple[int, int]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    try:
        obj = json.loads(value) if isinstance(value, str) else value
    except Exception:
        return []
    if not isinstance(obj, dict):
        return []
    points: list[tuple[int, int]] = []
    i = 0
    while True:
        kx = f"lm_mp_{i}_x"
        ky = f"lm_mp_{i}_y"
        if kx not in obj or ky not in obj:
            break
        x = _safe_float(obj.get(kx))
        y = _safe_float(obj.get(ky))
        if x is not None and y is not None:
            points.append((int(round(x * width)), int(round(y * height))))
        i += 1
    return points


def _parse_headpose(value) -> tuple[float | None, float | None, float | None]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None, None, None
    try:
        obj = json.loads(value) if isinstance(value, str) else value
    except Exception:
        obj = value
    if isinstance(obj, dict):
        return _safe_float(obj.get("pitch")), _safe_float(obj.get("yaw")), _safe_float(obj.get("roll"))
    text = str(obj)
    vals = []
    for key in ("pitch", "yaw", "roll"):
        m = re.search(rf"{key}\s*:\s*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
        vals.append(float(m.group(1)) if m else None)
    return vals[0], vals[1], vals[2]


def _put_lines(frame: np.ndarray, title: str, lines: list[str]) -> None:
    cv2.rectangle(frame, (8, 8), (520, 30 + 23 * len(lines)), (0, 0, 0), -1)
    cv2.putText(frame, title, (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    y = 52
    for line in lines:
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
        y += 23


def _draw_gaze(frame: np.ndarray, center: tuple[int, int], pitch_rad: float | None, yaw_rad: float | None, scale: int = 90) -> None:
    if pitch_rad is None or yaw_rad is None:
        return
    dx = int(round(math.sin(yaw_rad) * scale))
    dy = int(round(-math.sin(pitch_rad) * scale))
    cv2.arrowedLine(frame, center, (center[0] + dx, center[1] + dy), (0, 255, 255), 3, cv2.LINE_AA, tipLength=0.2)


def _top_values(row: pd.Series, columns: list[str], n: int = 5) -> list[str]:
    pairs = []
    for col in columns:
        val = _safe_float(row.get(col))
        if val is not None:
            pairs.append((abs(val), col, val))
    pairs.sort(reverse=True)
    return [f"{col}={val:.3f}" for _, col, val in pairs[:n]]


def _draw_pyfeat(frame: np.ndarray, row: pd.Series) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    x = _safe_float(row.get("FaceRectX"))
    y = _safe_float(row.get("FaceRectY"))
    bw = _safe_float(row.get("FaceRectWidth"))
    bh = _safe_float(row.get("FaceRectHeight"))
    if None not in (x, y, bw, bh):
        p1 = (int(round(x)), int(round(y)))
        p2 = (int(round(x + bw)), int(round(y + bh)))
        cv2.rectangle(out, p1, p2, (0, 255, 0), 2)
        center = (int(round(x + bw / 2)), int(round(y + bh / 2)))
    else:
        center = (w // 2, h // 2)

    for i in range(68):
        px = _safe_float(row.get(f"x_{i}"))
        py = _safe_float(row.get(f"y_{i}"))
        if px is not None and py is not None:
            cv2.circle(out, (int(round(px)), int(round(py))), 1, (255, 255, 0), -1)

    gp = _safe_float(row.get("gaze_pitch"))
    gy = _safe_float(row.get("gaze_yaw"))
    _draw_gaze(out, center, gp, gy)

    au_cols = [c for c in row.index if PYFEAT_AU_RE.fullmatch(str(c))]
    lines = [
        f"FaceScore={_safe_float(row.get('FaceScore')) or float('nan'):.4f}",
        f"Pose P/R/Y={_safe_float(row.get('Pitch')) or 0:.3f}/{_safe_float(row.get('Roll')) or 0:.3f}/{_safe_float(row.get('Yaw')) or 0:.3f}",
        f"Gaze p/y={gp if gp is not None else float('nan'):.3f}/{gy if gy is not None else float('nan'):.3f} rad",
        "Top AU(native): " + ", ".join(_top_values(row, au_cols, 5)),
    ]
    _put_lines(out, "Py-Feat Detectorv2", lines)
    return out


def _draw_libreface(frame: np.ndarray, row: pd.Series) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    points = _parse_libreface_landmarks(row.get("landmarks_json"), w, h)
    for i, (px, py) in enumerate(points):
        if i % 4 == 0:
            cv2.circle(out, (px, py), 1, (255, 255, 0), -1)
    if points:
        center = (int(np.mean([p[0] for p in points])), int(np.mean([p[1] for p in points])))
    else:
        center = (w // 2, h // 2)

    gp = _safe_float(row.get("gaze__gaze_pitch"))
    gy = _safe_float(row.get("gaze__gaze_yaw"))
    # LibreFace reports gaze in degrees in its current examples/output semantics.
    _draw_gaze(out, center, math.radians(gp) if gp is not None else None, math.radians(gy) if gy is not None else None)

    pitch, yaw, roll = _parse_headpose(row.get("headpose_json"))
    au_cols = [c for c in row.index if str(c).startswith("au_intensity__") or (str(c).startswith("int__") and str(c).endswith("_intensity"))]
    lines = [
        f"Alignment={bool(row.get('alignment_success', False))}",
        f"Pose P/R/Y={pitch if pitch is not None else float('nan'):.3f}/{roll if roll is not None else float('nan'):.3f}/{yaw if yaw is not None else float('nan'):.3f}",
        f"Gaze p/y={gp if gp is not None else float('nan'):.2f}/{gy if gy is not None else float('nan'):.2f} deg",
        "Top AU intensity: " + ", ".join(_top_values(row, au_cols, 5)),
    ]
    _put_lines(out, "LibreFace 2.0", lines)
    return out


def run_face_visual_review(config: Config, subject: str) -> dict[str, object]:
    layout = RGBOutputLayout.from_config(config)
    root = layout.test_dir() / "face-continuous" / subject
    manifests = sorted(root.glob("*_face-continuous_frames.csv"))
    if len(manifests) != 1:
        raise RuntimeError(f"Expected one continuous frame manifest in {root}, found {len(manifests)}")
    sample = pd.read_csv(manifests[0])
    py_path = root / "pyfeat_raw.parquet"
    lf_path = root / "libreface_raw.parquet"
    if not py_path.exists() or not lf_path.exists():
        raise FileNotFoundError("Both pyfeat_raw.parquet and libreface_raw.parquet are required")

    py = _primary_pyfeat(pd.read_parquet(py_path))
    py["_image_name"] = py["_image_name"].astype(str)
    lf = pd.read_parquet(lf_path)

    work = sample.copy()
    work["_image_name"] = work["image_path"].astype(str).map(lambda p: Path(p).name)
    work = work.merge(py, on="_image_name", how="left", suffixes=("", "__pyfeat"))
    lf_keep = [c for c in lf.columns if c == "benchmark_index" or c not in work.columns]
    work = work.merge(lf[lf_keep], on="benchmark_index", how="left")

    first = cv2.imread(str(work.iloc[0]["image_path"]))
    if first is None:
        raise RuntimeError("Could not read first continuous review image")
    h, w = first.shape[:2]
    fps = 10.0
    if "dt_ms" in sample.columns:
        med = pd.to_numeric(sample["dt_ms"], errors="coerce").dropna().median()
        if pd.notna(med) and med > 0:
            fps = 1000.0 / float(med)

    output = root / f"{subject}_face-visual-review.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 2, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output}")
    try:
        for row in work.itertuples(index=False):
            s = pd.Series(row._asdict())
            image = cv2.imread(str(s["image_path"]))
            if image is None:
                continue
            left = _draw_pyfeat(image, s)
            right = _draw_libreface(image, s)
            writer.write(np.concatenate([left, right], axis=1))
    finally:
        writer.release()

    summary = {
        "schema_version": FACE_VISUAL_REVIEW_SCHEMA,
        "subject": subject,
        "source_manifest": str(manifests[0]),
        "frames_requested": int(len(work)),
        "video_fps": float(fps),
        "output": str(output),
        "left_panel": "Py-Feat primary face: bbox, 68 landmarks, gaze direction cue, pose and top native AU values",
        "right_panel": "LibreFace: MediaPipe landmarks when parseable, gaze direction cue, pose and top AU intensity values",
        "warning": "This is model-prediction visualization, not human ground-truth annotation. Gaze arrows are qualitative direction cues, not calibrated screen-point estimates.",
    }
    manifest = root / f"{subject}_face-visual-review_manifest.json"
    manifest.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["manifest"] = str(manifest)
    return summary
